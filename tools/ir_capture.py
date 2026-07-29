#!/usr/bin/env python3
"""ir_capture — orchestrates tools/wave_inject to measure three fidelity
gaps using the injected-impulse/sine excitation wave_inject.cpp provides,
which a musical note excitation could not measure cleanly. All emulator
interaction (ROM loading/mutation, patch construction, rendering) lives in
tools/wave_inject.cpp; this script shells out to the compiled `wave_inject`
binary and does the signal analysis.

--stage reverb (Gap 1 -- room reverb fidelity):
  PROOF -> render a synthetic single-sample impulse dry, measure spectral
  flatness; refuses to proceed if it doesn't exceed the prior best (wave 17
  musical note, 0.552).
  CAPTURE -> sweep reverb type 0-5 x time (step 16, 9 steps), pure-wet
  (drylevel=0, reverbsendlevel=127), pure-impulse excitation; each render
  already IS the impulse response, no deconvolution needed. Trim where the
  smoothed envelope falls IR_TRIM_DB below peak (60dB -- see that constant's
  own comment for why, and what it replaced), fade out, resample to 48kHz,
  write under calib/ir_synth/. THIS is now the bank calibration.json's
  "reverb_ir" points at (superseding the old deconvolution-based
  calib/ir/*.wav bank the very first version of this tool deliberately
  avoided touching).
  VALIDATE (cmd_groundtruth_multi) -> for every reverb type, against SEVERAL
  real patches discovered from the whole internal ROM (a single named patch
  was shown to be misleading), reconstruct dry*(1-mix)+convolved*mix (using
  the real per-patch mix tools/emit_presets.py would compute) and compare to
  the real hardware's wet render over the decay region, with a reliability
  guard skipping patches whose post-note-off decay is too short to be a
  meaningful test at all.

--stage chorus-depth (Gap 2 -- CHORUS_MAX_MOD_DEPTH's shape):
  Pure-wet chorus (dry=0, chorussend=127) excited by an injected SINE (not
  the impulse -- depth needs a sustained signal to see modulation over
  multiple LFO cycles), sweeping chorusdepth. Measures stereo decorrelation
  (RMS(L-R)/RMS(L+R)) -- the metric that finally measured cleanly after a
  lag-tracking attempt (also tried here) failed like the two prior musical-
  note attempts did. See cmd_chorus_depth's own module-level comment block.

--stage delay (Gap 3 -- delay time/feedback/pan/stereoOffset):
  Delay/Pan-Dly (reverb types 6-7) STAY PARAMETRIC (explicit product
  decision: playability over freezing time/feedback/mix into an IR) but are
  now measured from a pure-wet impulse response directly (the render's own
  taps ARE the echo train -- no cross-correlation against a smeared note
  attack needed) instead of the old note-based method, including a measured
  (not analogised) DecentSampler stereoOffset. See cmd_delay_capture,
  find_stereo_offset, cmd_delay_validate.

Usage:
  python3 tools/ir_capture.py --roms "<rom dir>" [--stage all|reverb|chorus-depth|delay]
                               [--wave-inject build/wave_inject]
                               [--out calib/ir_synth] [--tmp /tmp/ir_capture]
                               [--calibration-json calib/calibration.json] [--no-merge]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, find_peaks, resample_poly

SR = 64000
IR_TARGET_SR = 48000
IR_FADE_MS = 15
# Room-fidelity investigation (2026-07-29): the original 28dB trim was
# measured, via a multi-patch reliability-filtered groundtruth sweep (see
# cmd_groundtruth_multi below), to systematically starve LONGER reverb
# tails: type4/Hall1 improved 0.7713->0.9249 and type5/Hall2 improved
# 0.8739->0.9640 (mean decay-envelope correlation across several real
# patches each) simply by raising this one number to 60dB, while the two
# ROOM types -- which were hypothesised to be the ones hurt by early
# trimming -- were UNAFFECTED or slightly improved (Room1 0.9635->0.9760,
# Room2 0.9281->0.9307), never made worse. A sweep from 28 to 80dB (plus an
# untrimmed control) shows correlation rising monotonically up to ~60dB and
# then plateauing (60/70/80/untrimmed are all within +/-0.001 of each
# other) -- 60dB captures essentially all of the achievable benefit without
# shipping needlessly long IR files. See docs/ in the room-trim report for
# the full sweep table.
#
# The reason this ended up being about DURATION, not proportional bite: the
# validation window itself is anchored to where the REAL hardware's own
# post-note-off decay first crosses -45dB (see cmd_groundtruth_multi's
# DECAY_FLOOR_DB) -- a threshold materially DEEPER than the old 28dB IR
# trim. Every captured IR was therefore missing real, audible decay content
# in the -28..-45dB range, and a slow (long-RT60) reverb takes far longer
# in absolute time to traverse that range than a fast one, so it lost
# proportionally more of its true tail. Raising the trim past the
# validation floor (with margin) fixes this for every type at once.
IR_TRIM_DB = 60.0
PRIOR_BEST_FLATNESS = 0.552   # wave 17, multi-pitch averaged + deconvolved (task brief)
REVERB_TYPES = range(6)
REVERB_TIME_STEPS = [0, 16, 32, 48, 64, 80, 96, 112, 127]
REVERB_NAMES = ["Room1", "Room2", "Stage1", "Stage2", "Hall1", "Hall2"]
# Full 8-entry name table (indices 6/7 are the delay types) -- matches
# tools/emit_presets.py's own REVERB_NAMES exactly. Kept as a separate list
# rather than extending REVERB_NAMES itself: several loops above iterate
# `range(6)`/REVERB_TYPES against REVERB_NAMES and rely on it staying
# reverb-only length 6.
ALL_TYPE_NAMES = REVERB_NAMES + ["Delay", "Pan-Dly"]

# Multi-patch groundtruth validation set for cmd_groundtruth_multi (gap 1).
# One "named" representative patch per type (from the task brief) plus every
# other internal patch that happens to use that reverb type, discovered at
# runtime via `wave_inject effects` -- a single-patch test was already shown
# to be misleading (task brief's own warning, confirmed here too: e.g. Hall1
# alone via A.Piano 1 measures ~0.96-0.99, hiding a real, separate problem on
# other Hall1 patches like Vibrobell). Capped per type to keep validation run
# time reasonable.
NAMED_REVERB_PATCH = {0: 3, 1: 4, 2: 38, 3: 9, 4: 0, 5: 1}
MAX_PATCHES_PER_TYPE = 8


# =============================================================================
# subprocess helpers

def run_wave_inject(binary, args, cwd=None):
    cmd = [str(binary)] + args
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(f"wave_inject {' '.join(args)} failed (exit {p.returncode})")
    return p.stdout, p.stderr


# =============================================================================
# signal analysis (self-contained -- deliberately not importing from
# tools/analyze_calibration.py, which this task does not touch and whose
# behavior must stay exactly as-is for the existing pipeline)

def spectral_flatness(x, fmin=20.0, fmax=None, sr=SR):
    """Geometric mean / arithmetic mean of the power spectrum in [fmin,fmax].
    1.0 == perfectly flat (white); near 0 == strongly peaky/tonal."""
    n = len(x)
    if n < 8:
        return 0.0
    X = np.fft.rfft(np.asarray(x, dtype=np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    if fmax is None:
        fmax = sr / 2.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    power = np.abs(X[mask]) ** 2
    power = power[power > 0]
    if len(power) == 0:
        return 0.0
    gm = np.exp(np.mean(np.log(power)))
    am = np.mean(power)
    return float(gm / am)


def peak_frequency(seg, sr=SR, skip_bins=3):
    """Parabolic-interpolated FFT peak frequency of a windowed segment,
    skipping the first `skip_bins` (DC / near-DC) so a residual offset
    doesn't get reported as "the" frequency."""
    seg = np.asarray(seg, dtype=np.float64)
    win = np.hanning(len(seg))
    X = np.fft.rfft(seg * win)
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    mag = np.abs(X)
    if len(mag) <= skip_bins + 1:
        return 0.0
    k = int(np.argmax(mag[skip_bins:-1])) + skip_bins
    if 1 <= k < len(mag) - 1:
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return float(freqs[k] + delta * (freqs[1] - freqs[0]))


def env_max(x, hop):
    n = len(x) // hop
    if n == 0:
        return np.array([np.max(np.abs(x))]) if len(x) else np.array([0.0])
    return np.array([np.max(np.abs(x[i * hop:(i + 1) * hop])) for i in range(n)])


def smoothed_db_envelope(mono, hop=64, smooth=5):
    env = env_max(mono, hop) + 1e-9
    if len(env) >= smooth:
        kernel = np.ones(smooth) / smooth
        env = np.convolve(env, kernel, mode="same")
    peak = float(env.max())
    if peak <= 1e-9:
        return env, -np.inf * np.ones_like(env)
    db = 20 * np.log10(env / peak)
    return env, db


def strip_leading_silence(x):
    """Drop leading samples that are EXACTLY zero.

    Exact zero, not a threshold: the gap this removes is literal digital
    silence (0 non-zero samples in the first 15 ms of Room1), whereas a
    diffuse reverb's genuine early buildup is quiet but never exactly zero.
    Testing for zero therefore removes the capture artifact without ever
    clipping the front off a real response.
    """
    if x.ndim > 1:
        nz = np.nonzero(np.abs(x).sum(axis=1))[0]
    else:
        nz = np.nonzero(x)[0]
    return x[nz[0]:] if len(nz) else x


def _pink_reference(n=1 << 17, seed=20260729):
    """Deterministic pink-noise probe used to normalise IR gain.

    Pink rather than white because it is far closer to the long-term spectrum
    of musical material, and these IRs differ in spectral shape as well as
    energy -- normalising against white leaves the perceived wet level
    type-dependent even when the maths says the gain is equal.
    """
    rng = np.random.default_rng(seed)
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.arange(len(spec))
    spec[1:] /= np.sqrt(freqs[1:])      # -3 dB/octave
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return x / np.abs(x).max()


def normalize_ir(x, reference=None):
    """Scale an IR so that convolving program material with it is gain-neutral.

    Without this the wet level depends on whatever total energy the captured
    IR happens to hold, which varies with reverb type and decay time. The
    previously shipped bank measured -13.6 dB of convolution gain for Hall1
    and -9.2 dB for Room1, so a preset asking for mix=0.28 got its wet signal
    21.8 dB under the dry -- audible only as "is there actually reverb on
    this?", which is exactly how it was reported.

    Normalising by measured gain against a fixed pink-noise reference, rather
    than by the IR's own energy: unit energy is only gain-neutral for a white
    input, and against real material it still left a 4.6 dB spread across the
    bank (+0.9 dB for Hall1 vs +5.5 dB for Room1). With this, `mix` means the
    same thing for every type and time step, and how much reverb a patch gets
    is decided solely by the mix value emit_presets derives from its own
    reverb level and sends.

    Returns the input unchanged if it is all zero (see the underflow note in
    the caller).
    """
    if not np.abs(x).any():
        return x
    if reference is None:
        reference = _pink_reference()
    ref_rms = np.sqrt(np.mean(reference ** 2))
    gains = []
    for c in range(x.shape[1] if x.ndim > 1 else 1):
        ch = x[:, c] if x.ndim > 1 else x
        wet = fftconvolve(reference, ch)[:len(reference)]
        gains.append(np.sqrt(np.mean(wet ** 2)) / ref_rms)
    gain = float(np.mean(gains))
    if gain <= 0:
        return x
    scaled = x / gain
    peak = np.abs(scaled).max()
    if peak >= 1.0:
        # A gain-neutral IR peaks well below full scale in this bank, but a
        # future capture must never ship a clipped IR.
        scaled = scaled / peak * 0.99
    return scaled


def trim_and_fade(mono_or_stereo, sr, trim_db=IR_TRIM_DB, fade_ms=IR_FADE_MS, hop=64):
    """Trims a captured IR where its smoothed envelope falls trim_db below
    peak, then applies a short linear fade-out over the last fade_ms so the
    hard trim point isn't an audible click. mono_or_stereo: (n,) or (n,2)."""
    x = np.asarray(mono_or_stereo, dtype=np.float64)
    mono = x if x.ndim == 1 else x.mean(axis=1)
    _, db = smoothed_db_envelope(mono, hop=hop)
    # Search AFTER the peak, not from sample 0: a pure-wet reverb capture
    # starts with a stretch of true silence (the note-on/voice-allocation
    # latency before the excitation reaches the reverb algorithm at all,
    # ~130-260 raw samples => -inf dB in the very first hop bins), which is
    # trivially "below -28dB" and would otherwise make trim_hop_idx land at
    # index 0 on every single capture (confirmed: this was firing on every
    # file except the type-5/Delay-ish ones whose peak happens to sit in
    # the first hop bin). We want the point the DECAY crosses threshold,
    # which only makes sense measured forward from the envelope's own peak.
    peak_idx = int(np.argmax(db))
    below = np.where(db[peak_idx:] <= -trim_db)[0]
    if len(below) == 0:
        trim_hop_idx = len(db)   # never decayed that far -- keep the whole render
    else:
        trim_hop_idx = peak_idx + int(below[0]) + 1   # a little past the first post-peak crossing
    trim_sample = min(len(x), trim_hop_idx * hop)
    trim_sample = max(trim_sample, int(0.01 * sr))   # never trim to nothing
    out = x[:trim_sample].copy()
    fade_n = min(len(out), int(fade_ms / 1000.0 * sr))
    if fade_n > 1:
        fade = np.linspace(1.0, 0.0, fade_n)
        if out.ndim == 1:
            out[-fade_n:] *= fade
        else:
            out[-fade_n:] *= fade[:, None]
    return out


def measure_rt60(mono, tail_start, sr=SR):
    tail = mono[tail_start:]
    if len(tail) < int(0.2 * sr):
        return None
    hop = 64
    env, _ = smoothed_db_envelope(tail, hop=hop)
    fs_env = sr / hop
    ref_window = max(1, int(0.05 * fs_env))
    ref = float(env[:ref_window].max())
    if ref <= 1e-9:
        return None
    db = 20 * np.log10(env / ref)
    t = np.arange(len(env)) / fs_env
    mask = (db <= -5) & (db >= -35)
    if int(mask.sum()) < 5:
        return None
    slope = np.polyfit(t[mask], db[mask], 1)[0]
    if slope >= -0.01:
        return None
    return float(-60.0 / slope)


def band_energy(x, sr, f_lo, f_hi):
    n = len(x)
    if n < 8:
        return 0.0
    X = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(np.abs(X[mask]) ** 2))


def envelope_correlation(a, b, hop=64):
    """Pearson correlation of two dB envelopes, resampled onto the shorter
    signal's own grid so unequal lengths don't bias the comparison."""
    ea, _ = smoothed_db_envelope(a, hop=hop)
    eb, _ = smoothed_db_envelope(b, hop=hop)
    n = min(len(ea), len(eb))
    if n < 8:
        return 0.0
    ea, eb = ea[:n], eb[:n]
    dba = 20 * np.log10(ea / (ea.max() + 1e-12) + 1e-12)
    dbb = 20 * np.log10(eb / (eb.max() + 1e-12) + 1e-12)
    dba = np.clip(dba, -80, 0)
    dbb = np.clip(dbb, -80, 0)
    if np.std(dba) < 1e-9 or np.std(dbb) < 1e-9:
        return 0.0
    return float(np.corrcoef(dba, dbb)[0, 1])


# =============================================================================

def cmd_proof(wave_inject, roms, tmp, args):
    print("\n=== PROOF: impulse spectral flatness ===")
    results = []
    for wn, key in [(args.host_wavenumber, args.host_key), (17, args.host_key)]:
        out = tmp / f"proof_impulse_wn{wn}_key{key}.wav"
        run_wave_inject(wave_inject, [
            "impulse", "--roms", roms, "--wavegroup", "0", "--wavenumber", str(wn),
            "--key", str(key), "--n", "8", "--amp-i", "127", "--out", str(out)])
        x, sr = sf.read(str(out))
        mono = x.mean(axis=1)
        nz = np.where(np.abs(mono) > 1e-6)[0]
        flat_full = spectral_flatness(mono, 20, sr / 2, sr)
        flat_audio = spectral_flatness(mono, 20, 20000, sr)
        width = int(nz.max() - nz.min() + 1) if len(nz) else 0
        outside = np.abs(mono).copy()
        if len(nz):
            outside[nz.min():nz.max() + 1] = 0
        leak_db = 20 * np.log10((outside.max() + 1e-12) / (np.abs(mono).max() + 1e-12))
        print(f"  wave {wn} @key{key}: width={width} samples, "
              f"flatness(full Nyquist)={flat_full:.4f}, flatness(20Hz-20kHz)={flat_audio:.4f}, "
              f"isolation={leak_db:.1f}dB (energy outside the pulse, relative to peak)")
        results.append({"wavenumber": wn, "key": key, "width_samples": width,
                        "flatness_full_nyquist": flat_full, "flatness_20_20k": flat_audio,
                        "isolation_db": leak_db, "onset_sample": int(nz.min()) if len(nz) else 0})

    print("\n  NOTE: full-Nyquist flatness is structurally capped near 0.50 by a hard "
          "comb null at 16kHz. This is confirmed INHERENT to the emulator's own DAC "
          "oversampling (pcm.cpp posts 2 output frames per internal DSP tick -- "
          "confirmed independently by inspecting calib/dry.wav, an EXISTING baseline "
          "render with no injection involved: 77% of its consecutive sample pairs are "
          "bit-identical). It affects every render equally, injected or not. Measured "
          "over the standard 20Hz-20kHz audio band, where that null sits just outside, "
          "flatness reaches ~0.95.")

    print("\n=== PROOF: sine frequency accuracy ===")
    sine_results = []
    for freq in (300.0, 1000.0, 2000.0):
        out = tmp / f"proof_sine_{int(freq)}hz.wav"
        _, err = run_wave_inject(wave_inject, [
            "sine", "--roms", roms, "--wavegroup", "0", "--wavenumber", str(args.host_wavenumber),
            "--key", "60", "--n", "3000", "--freq", str(freq), "--amp", "100000",
            "--out", str(out)])
        # pull the encoder's own effective-rate line back out for the analysis window
        eff = None
        for line in err.splitlines():
            if line.startswith("sine:"):
                eff = line
        x, sr = sf.read(str(out))
        mono = x.mean(axis=1)
        nz = np.where(np.abs(mono) > 1e-6)[0]
        seg = mono[nz.min():nz.max() + 1] if len(nz) else mono
        measured = peak_frequency(seg, sr)
        err_pct = 100.0 * (measured - freq) / freq
        print(f"  target={freq:.0f}Hz measured={measured:.1f}Hz error={err_pct:+.2f}%  ({eff})")
        sine_results.append({"target_hz": freq, "measured_hz": measured, "error_pct": err_pct})

    best_flatness = max(r["flatness_20_20k"] for r in results)
    passed = best_flatness > 0.9 and best_flatness > PRIOR_BEST_FLATNESS
    print(f"\n=== PROOF VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(best flatness {best_flatness:.4f}, need >0.9 and >{PRIOR_BEST_FLATNESS} prior best) ===")
    # The capture-irs host wave/key's own onset latency (note-on -> first
    # nonzero output sample), measured with NO reverb involved (drylevel=127,
    # reverbsendlevel=0). This is pure emulator/voice-trigger pipeline
    # latency -- constant regardless of reverb type/time -- and is reused by
    # cmd_capture/cmd_groundtruth to strip it from the FRONT of every
    # captured IR before it's used for anything. Left un-stripped, it would
    # add a fixed ~130-260-sample "fake pre-delay" to every IR that has
    # nothing to do with the reverb algorithm's own (real, type-dependent)
    # pre-delay character, and misaligns the dry*IR convolution against a
    # real wet render by that same amount (confirmed: this was the actual
    # cause of the first groundtruth run's low correlation and the
    # implausible mid-band ratio).
    excitation_latency = results[0]["onset_sample"]
    print(f"  excitation latency (note-on -> first audible sample, no reverb): "
          f"{excitation_latency} samples ({1000*excitation_latency/SR:.2f} ms) -- "
          "stripped from every captured IR's leading edge before use")
    return {"impulse": results, "sine": sine_results, "passed": passed,
            "best_flatness_20_20k": best_flatness, "excitation_latency_samples": excitation_latency}


def cmd_capture(wave_inject, roms, tmp, out_dir, args, excitation_latency):
    print("\n=== CAPTURE: reverb IRs (types 0-5, time step 16) ===")
    raw_dir = tmp / "ir_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_wave_inject(wave_inject, [
        "capture-irs", "--roms", roms, "--wavegroup", "0",
        "--wavenumber", str(args.host_wavenumber), "--key", str(args.host_key),
        "--n", "8", "--amp-i", "127", "--out-dir", str(raw_dir)])

    out_dir.mkdir(parents=True, exist_ok=True)
    bank = {}
    for t in REVERB_TYPES:
        bank[t] = {}
        for raw in REVERB_TIME_STEPS:
            src = raw_dir / f"ir_t{t}_time_{raw:03d}.wav"
            if not src.exists():
                print(f"  MISSING: {src.name}")
                continue
            x, sr = sf.read(str(src))
            assert sr == SR
            # Strip the fixed excitation latency (see cmd_proof) so sample 0
            # of the shipped IR is "the excitation just arrived", not "the
            # excitation is still 130-260 samples from arriving".
            x = x[excitation_latency:]
            # Then strip whatever leading silence REMAINS. An earlier version
            # kept it, on the theory that it was the reverb algorithm's own
            # type-dependent pre-delay. It is not. Measured against the plugin
            # -- reverb-only residual after least-squares removal of the dry,
            # so a dry-level difference between the two renders cannot fake an
            # instant arrival -- the JV's reverb starts within -0.3..+3.6 ms of
            # the note, while the captured IRs carried gaps of 1.1 ms (Hall1)
            # to 17.3 ms (Stage2). So the gap is latency in the injection path,
            # not the algorithm, and it was audible as the reverb arriving late
            # and sounding detached from the note.
            x = strip_leading_silence(x)
            mono = x.mean(axis=1)
            rt60 = measure_rt60(mono, tail_start=0, sr=SR)
            trimmed = trim_and_fade(x, SR)
            resampled = resample_poly(trimmed, up=3, down=4, axis=0)   # 64000 -> 48000
            dst = out_dir / f"reverb_t{t}_time_{raw:03d}.wav"
            normalized = normalize_ir(resampled)
            # 24-bit, not 16. A unit-energy IR peaks around -17 dBFS (the
            # energy constraint sets the level, so the peak lands where it
            # lands), and the raw emulator capture is itself only about
            # -39 dBFS -- in 16 bits that leaves the tail sitting on the
            # quantisation floor. Also note soundfile.read() returns floats
            # in [-1,1] regardless of the source's bit depth: writing such an
            # array to an int-subtype file WITHOUT rescaling rounds every
            # sample to zero, which is exactly how the previously committed
            # bank came to be entirely silent.
            sf.write(str(dst), normalized, IR_TARGET_SR, subtype="PCM_24")
            dur_raw = len(x) / SR
            dur_trim = len(trimmed) / SR
            # A captured IR can come back all-zero. This happens at reverbtime
            # 0 for every type, and it is an UNDERFLOW, not an absence: the JV
            # does produce wet signal there (measured against a real patch's
            # own dry render at 0.9x dry RMS), but a single-sample impulse into
            # a near-zero decay time lands below the fixed-point reverb's own
            # resolution. Record it so the emitter can ship mix=0 rather than
            # convolving against silence, which at mix>0 attenuates the dry
            # signal instead of leaving it alone.
            bank[t][raw] = {"file": str(dst.relative_to(out_dir.parent)),
                            "raw_duration_s": dur_raw, "trimmed_duration_s": dur_trim,
                            "rt60_s": rt60, "silent": not bool(np.abs(resampled).any())}
            print(f"  t{t} time{raw:3d}: raw={dur_raw:6.3f}s -> trimmed={dur_trim:6.3f}s "
                  f"({100*dur_trim/dur_raw:5.1f}%), rt60={rt60}")
    return bank


def cmd_sanity_vs_baseline(bank, calibration_json):
    print("\n=== SANITY: synthetic-IR RT60 vs existing calib/calibration.json ===")
    if not calibration_json.exists():
        print("  (no calibration.json found -- skipping)")
        return
    existing = json.loads(calibration_json.read_text())
    existing_rt60 = existing.get("reverb_rt60", {})
    rows = []
    for t in REVERB_TYPES:
        for raw in REVERB_TIME_STEPS:
            ours = bank.get(t, {}).get(raw, {}).get("rt60_s")
            theirs = existing_rt60.get(str(t), {}).get(str(raw))
            if ours is None or theirs is None:
                continue
            ratio = ours / theirs if theirs else None
            rows.append((t, raw, ours, theirs, ratio))
            print(f"  type{t} time{raw:3d}: synthetic={ours:6.3f}s  "
                  f"existing(note-based)={theirs:6.3f}s  ratio={ratio:.2f}x" if ratio else
                  f"  type{t} time{raw:3d}: synthetic={ours}  existing={theirs}")
    if rows:
        ratios = [r[4] for r in rows if r[4] is not None]
        print(f"\n  {len(rows)} comparable points, median ratio={np.median(ratios):.2f}x, "
              f"mean ratio={np.mean(ratios):.2f}x (1.0x == identical decay time; the two "
              "methods excite the reverb completely differently -- a musical note's own "
              "hold+release vs. a single impulse -- so a broad match in RT60 trend, not "
              "an exact match, is the meaningful check here)")


DECAY_FLOOR_DB = -45.0   # per the task's validation method: decay region ends here
GROUNDTRUTH_HOLD_SECONDS = 3.5   # matches wave_inject.cpp's groundtruth GridSpec.hold_seconds
EFFECT_MAX_MIX = 0.5   # must match tools/emit_presets.py's own constant -- see its comment
# A patch whose render_note tail exits within this long of note-off has
# (almost always) an amplitude envelope that releases to silence almost
# instantly for BOTH the wet and dry groundtruth render -- confirmed on
# several concrete patches (Woody Bass 1/2, Clav 1, MIDI EPiano's own
# Marimba SW cousin): render length settles at hold+~0.1s regardless of
# reverb type, i.e. render_note's own quiet-run detector fired almost
# immediately. There is then no real post-note-off decay left to compare at
# all -- what's left is two near-silent, noise-floor-dominated signals whose
# correlation is essentially meaningless (measured as low as -0.33 on such a
# patch even though the SAME IR/type scored >0.97 on every patch with a real
# tail). This is exactly the "record unmeasurable points as null" case, not
# a genuine reconstruction failure -- skip them rather than let a handful of
# degenerate patches dominate a type's reported mean.
MIN_RELIABLE_DECAY_S = 0.15


def _parse_last_json_line(stdout_text):
    obj = None
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                pass
    return obj


def run_groundtruth_render(wave_inject, roms, tmp, patch_index):
    """Runs `wave_inject groundtruth --patch-index N`, returns
    (wet_mono, dry_mono, effects_dict). effects_dict is the patch's own
    native reverb_type/level/time/feedback plus per-tone tone_level/
    reverb_send (see wave_inject.cpp's groundtruth stdout JSON) -- enough to
    reproduce tools/emit_presets.py's own effective_send()/mix computation
    exactly, so this validates what would actually ship, not an arbitrary
    stand-in mix fraction."""
    out_dir = tmp / f"groundtruth_{patch_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout, _ = run_wave_inject(wave_inject, [
        "groundtruth", "--roms", roms, "--patch-index", str(patch_index),
        "--out-dir", str(out_dir)])
    effects = _parse_last_json_line(stdout)
    if effects is None:
        raise SystemExit(f"groundtruth --patch-index {patch_index}: no effects JSON on stdout")
    wet, sr1 = sf.read(str(out_dir / "groundtruth_wet.wav"))
    dry, sr2 = sf.read(str(out_dir / "groundtruth_dry.wav"))
    assert sr1 == SR and sr2 == SR
    return wet.mean(axis=1), dry.mean(axis=1), effects


def effective_send_from_effects(effects, which="reverb_send"):
    """Mirrors tools/emit_presets.py's effective_send(): average `which`
    across tones whose tone_level > 0 only."""
    levels = effects["tone_level"]
    sends = effects[which]
    active = [s for s, lv in zip(sends, levels) if lv > 0]
    return (sum(active) / len(active)) if active else 0.0


def _amount(raw, ceiling=1.0):
    return max(0.0, min(1.0, max(0.0, min(127, float(raw))) / 127.0 * ceiling))


def reverb_mix_from_effects(effects):
    """The convolution `mix` tools/emit_presets.py would actually emit for
    this patch: amount(reverb_level) * effective_send/127, capped at the
    same EFFECT_MAX_MIX equal-blend ceiling (see that module's own
    docstring on why a full JV send maps to an EQUAL blend, not mix=1.0)."""
    wet = _amount(effects["reverb_level"]) * (effective_send_from_effects(effects) / 127.0)
    return max(0.0, min(1.0, _amount(wet) * EFFECT_MAX_MIX))


def compare_decay_region(wet_mono, dry_mono, ir_mono, mix,
                          hold_seconds=GROUNDTRUTH_HOLD_SECONDS,
                          decay_floor_db=DECAY_FLOOR_DB,
                          min_reliable_s=MIN_RELIABLE_DECAY_S):
    """Reconstructs `dry*(1-mix) + convolved*mix` (the task's stated method,
    using the SAME mix fraction emit_presets.py would ship) and compares it
    against the real hardware's wet render over the decay region: from
    note-off until the wet render's own envelope falls decay_floor_db below
    its post-note-off peak (see the original single-patch version's comment
    on why a fixed floor beats a fixed-duration window -- this reverb's tail
    has a nonlinear noise floor around -47dB that a linear convolution can
    never reproduce, so comparing past it measures noise-vs-noise, not
    reconstruction accuracy).

    Returns None (not a fabricated number) when the reference decay region
    is too short to be a meaningful test at all -- see MIN_RELIABLE_DECAY_S's
    comment above."""
    note_off = int(hold_seconds * SR)
    if len(wet_mono) < note_off + int(min_reliable_s * SR):
        return None   # render_note's own tail exited almost immediately -- nothing to compare

    conv = fftconvolve(dry_mono, ir_mono, mode="full")
    n = max(len(dry_mono), len(conv))
    dry_pad = np.zeros(n)
    dry_pad[:len(dry_mono)] = dry_mono
    conv_pad = np.zeros(n)
    conv_pad[:len(conv)] = conv
    recon = dry_pad * (1 - mix) + conv_pad * mix

    wet_env, wet_db = smoothed_db_envelope(wet_mono[note_off:note_off + int(6.0 * SR)])
    below = np.where(wet_db <= decay_floor_db)[0]
    hop = 64
    window_samples = (int(below[0]) * hop) if len(below) else int(6.0 * SR)
    window_samples = max(window_samples, int(0.3 * SR))
    if window_samples < int(min_reliable_s * SR):
        return None   # degenerate: reference decayed below the floor almost immediately

    def norm_tail(x, start, n):
        n = min(n, max(0, len(x) - start))
        t = x[start:start + n]
        peak = np.max(np.abs(t)) + 1e-12
        return t / peak

    wet_tail = norm_tail(wet_mono, note_off, window_samples)
    recon_tail = norm_tail(recon, note_off, window_samples)
    corr = envelope_correlation(wet_tail, recon_tail)

    bands = [("low_80_250", 80, 250), ("mid_250_2000", 250, 2000), ("high_2000_8000", 2000, 8000)]
    band_ratios = {}
    for name, lo, hi in bands:
        e_wet = band_energy(wet_tail, SR, lo, hi)
        e_recon = band_energy(recon_tail, SR, lo, hi)
        band_ratios[name] = (e_recon / e_wet) if e_wet > 0 else None

    return {"correlation": corr, "band_ratios": band_ratios, "window_s": window_samples / SR}


def _nearest_step(raw_time):
    return min(REVERB_TIME_STEPS, key=lambda k: abs(k - raw_time))


def _interp_table(table, raw):
    """Linear interpolation over a {str(raw): value} table, skipping None
    entries -- a deliberate copy of tools/emit_presets.py's own interp_table
    (not an import: this module treats emit_presets.py as a black box its
    own tests exercise, and duplicating ~15 lines here is cheaper than a
    cross-module coupling). Used by cmd_delay_validate/find_stereo_offset so
    the delay-reconstruction validation looks up delay_time_s/delay_feedback
    the SAME way (true interpolation, not nearest-step snapping) the shipped
    preset actually would -- nearest-step snapping is fine for reverb_ir
    (there is no choice, IRs are discrete files) but would be a needlessly
    pessimistic stand-in here, where the real code path interpolates."""
    pts = sorted((int(k), float(v)) for k, v in table.items() if v is not None)
    if not pts:
        raise ValueError("empty/all-null calibration table")
    keys = [k for k, _ in pts]
    vals = [v for _, v in pts]
    if raw <= keys[0]:
        return vals[0]
    if raw >= keys[-1]:
        return vals[-1]
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k0 <= raw <= k1:
            if k1 == k0:
                return vals[i]
            t = (raw - k0) / (k1 - k0)
            return vals[i] + t * (vals[i + 1] - vals[i])
    return vals[-1]


def _discover_test_patches(wave_inject, roms, tmp):
    """Every internal patch's own native reverb type, via `wave_inject
    effects` (one process call for the whole ROM) -- used to build a
    multi-patch-per-type validation set rather than trusting a single named
    patch, which the task brief explicitly warns is misleading."""
    stdout, _ = run_wave_inject(wave_inject, ["effects", "--roms", roms])
    by_type = {t: [] for t in range(8)}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        by_type[row["reverb_type"]].append(row["index"])
    test_set = {}
    for t in range(6):
        named = NAMED_REVERB_PATCH[t]
        rest = [i for i in by_type.get(t, []) if i != named]
        test_set[t] = ([named] + rest)[:MAX_PATCHES_PER_TYPE]
    return test_set


def cmd_groundtruth_multi(wave_inject, roms, tmp, ir_bank_raw_dir, args, excitation_latency,
                          trim_db_values=(28.0, IR_TRIM_DB)):
    """Gap 1 validation: for every reverb type (0-5), against every
    discovered patch that actually uses it, reconstruct `dry*(1-mix) +
    convolved(dry, ir)*mix` and compare to the real hardware's wet render
    over the decay region (see compare_decay_region). Reports per-type,
    reliability-filtered means at each trim_db in `trim_db_values` (default:
    the OLD 28dB default vs. the new IR_TRIM_DB), so before/after is a
    genuine same-methodology comparison, not two different runs.
    """
    print("\n=== VALIDATE (multi-patch): reverb types 0-5 vs real hardware ===")
    test_set = _discover_test_patches(wave_inject, roms, tmp)

    gt_cache = {}
    for t in range(6):
        for idx in test_set[t]:
            if idx in gt_cache:
                continue
            wet, dry, effects = run_groundtruth_render(wave_inject, roms, tmp, idx)
            gt_cache[idx] = (wet, dry, effects, reverb_mix_from_effects(effects))

    report = {}
    for trim_db in trim_db_values:
        print(f"\n--- trim_db={trim_db:.0f} ---")
        type_report = {}
        for t in range(6):
            per_patch = []
            n_skipped = 0
            for idx in test_set[t]:
                wet, dry, effects, mix = gt_cache[idx]
                step = _nearest_step(effects["reverb_time"])
                src = ir_bank_raw_dir / f"ir_t{t}_time_{step:03d}.wav"
                if not src.exists():
                    continue
                raw_ir, sr = sf.read(str(src))
                assert sr == SR
                raw_ir = raw_ir[excitation_latency:]
                ir_mono = trim_and_fade(raw_ir, SR, trim_db=trim_db).mean(axis=1)
                result = compare_decay_region(wet, dry, ir_mono, mix)
                if result is None:
                    n_skipped += 1
                    continue
                per_patch.append((idx, effects["name"], result["correlation"]))
            mean_corr = float(np.mean([c for _, _, c in per_patch])) if per_patch else None
            type_report[REVERB_NAMES[t]] = {
                "mean_correlation": mean_corr,
                "n_reliable": len(per_patch),
                "n_skipped_unreliable": n_skipped,
                "per_patch": [{"index": i, "name": n, "correlation": round(c, 4)}
                              for i, n, c in per_patch],
            }
            corr_str = f"{mean_corr:.4f}" if mean_corr is not None else "n/a"
            print(f"  {REVERB_NAMES[t]:8s}  mean={corr_str}  "
                  f"n_reliable={len(per_patch)} n_skipped={n_skipped}")
        overall = [v["mean_correlation"] for v in type_report.values()
                   if v["mean_correlation"] is not None]
        print(f"  overall mean = {np.mean(overall):.4f}" if overall else "  overall mean = n/a")
        report[f"trim_db_{trim_db:.0f}"] = type_report

    return report


# =============================================================================
# GAP 2: chorus depth (CHORUS_MAX_MOD_DEPTH)
#
# Two prior attempts (documented in tools/emit_presets.py) tried to measure
# how much chorusdepth (patch byte 17) actually modulates the chorus's delay
# line, using a musical note as excitation, and both produced non-monotonic
# garbage. With a clean injected sine (wave_inject's `sine` machinery, proven
# accurate to <0.4% in the impulse-proof step) held through a PURE-WET chorus
# (dry=0, chorussend=127, chorusfeedback=0), a direct lag-tracking attempt
# (cross-correlating short windows of wet against dry to recover the
# instantaneous delay) was tried FIRST here too, and also failed cleanly: the
# recovered "lag" curve is dominated by a slow, non-periodic drift of
# unknown origin (tens to hundreds of samples/second, varying between
# renders) that swamps any real LFO-periodic signal -- fitting a sinusoid at
# the KNOWN chorus rate (from chorus_rate_hz) to the detrended lag curve
# gives a flat ~1-2 samples of "amplitude" at EVERY depth setting including
# 0 and 127, i.e. no signal, matching the two prior failures' character.
#
# What DOES measure cleanly: stereo decorrelation. This chorus is a stereo
# effect (independently modulated L/R delay lines -- confirmed by
# tools/analyze_calibration.py's own docstring, "the JV-880 chorus is a
# STEREO effect"), so a deeper LFO swing should measurably decorrelate L
# from R even when the underlying carrier tone's own timbre/pitch is held
# fixed. RMS(L-R)/RMS(L+R) over the pure-wet chorus render is a simple,
# ENERGY-domain metric (no phase/cross-correlation involved, so none of the
# periodicity-aliasing failure modes above apply) that turns out to be
# cleanly, strongly MONOTONIC in raw chorusdepth, and reproduces almost
# identically (normalized-shape correlation 0.999) across two independent,
# unrelated carrier frequencies (50Hz and 200Hz) -- strong cross-validation
# that this is a real, physical signal, not an artifact of one specific test
# tone. See docs/ in the chorus-depth report for the full sweep.
CHORUS_DEPTH_STEPS = [0, 16, 32, 48, 64, 80, 96, 112, 127]
CHORUS_DEPTH_CARRIER_HZ = 200.0   # mid-range: strong, well-separated signal (see above)
CHORUS_DEPTH_ANALYSIS_SKIP_S = 0.02   # skip the onset transient
# A moderate MIDI key, NOT args.host_key (127): 127 is tuned for the IMPULSE
# captures (minimal excitation WIDTH is what matters there), which is the
# wrong criterion for a sustained sine at a specific target Hz -- confirmed
# empirically, key=127 measured a noisy, non-monotonic width curve (3/8
# violations) where key=60 (this constant) reproduces the clean, strongly
# monotonic curve found during investigation (cross-validated at 0.999
# shape-correlation between two independent carrier frequencies).
CHORUS_DEPTH_KEY = 60


def measure_stereo_width(stereo, skip_s=CHORUS_DEPTH_ANALYSIS_SKIP_S, max_s=None, sr=SR):
    n0 = int(skip_s * sr)
    n1 = len(stereo) if max_s is None else min(len(stereo), int(max_s * sr))
    if n1 - n0 < int(0.01 * sr):
        return None
    l, r = stereo[n0:n1, 0], stereo[n0:n1, 1]
    diff, ssum = l - r, l + r
    denom = np.sqrt(np.mean(ssum ** 2)) + 1e-12
    return float(np.sqrt(np.mean(diff ** 2)) / denom)


def normalize_0_1(values):
    """Linear min-clip-then-scale to [0, 1] by the sweep's own maximum --
    matches tools/analyze_calibration.py's own helper of the same name and
    the same convention already used by chorus_mix/reverb_wet: raw=0 is NOT
    forced to exactly 0.0 if it genuinely measures a nonzero floor (a real,
    small STATIC L/R offset this chorus implementation has even with zero
    LFO swing -- confirmed present, and roughly proportional to carrier
    frequency, at both tested carriers -- not fabricated away)."""
    arr = np.clip(np.array(values, dtype=np.float64), 0.0, None)
    mx = arr.max() if len(arr) else 0.0
    if mx <= 1e-9:
        return [0.0] * len(values)
    return (arr / mx).tolist()


def cmd_chorus_depth(wave_inject, roms, tmp, args):
    print("\n=== GAP 2: chorus depth (stereo decorrelation vs raw chorusdepth) ===")
    out_dir = tmp / "chorus_depth"
    run_wave_inject(wave_inject, [
        "capture-chorus-depth", "--roms", roms, "--wavegroup", "0",
        "--wavenumber", str(args.host_wavenumber), "--key", str(CHORUS_DEPTH_KEY),
        "--freq", str(CHORUS_DEPTH_CARRIER_HZ), "--amp", "100000", "--out-dir", str(out_dir)])

    widths = {}
    for raw in CHORUS_DEPTH_STEPS:
        wet, sr = sf.read(str(out_dir / f"chorus_depth_wet_{raw:03d}.wav"))
        assert sr == SR
        w = measure_stereo_width(wet)
        widths[raw] = w
        print(f"  raw={raw:3d}  stereo_width={w}")

    measured = [raw for raw in CHORUS_DEPTH_STEPS if widths[raw] is not None]
    if len(measured) < len(CHORUS_DEPTH_STEPS):
        print(f"  WARNING: {len(CHORUS_DEPTH_STEPS) - len(measured)} depth steps unmeasurable "
              f"(render too short) -- omitting, not fabricating")

    norm_values = normalize_0_1([widths[raw] for raw in measured])
    chorus_depth_norm = {str(raw): round(v, 4) for raw, v in zip(measured, norm_values)}

    # Monotonicity check (printed, not enforced here -- tests/test_calibration.py
    # asserts on the shape of whatever this writes to calibration.json).
    vals = [chorus_depth_norm[str(raw)] for raw in measured]
    violations = sum(1 for a, b in zip(vals, vals[1:]) if b < a - 1e-6)
    print(f"  normalized (0..1): {chorus_depth_norm}")
    print(f"  monotonicity violations: {violations} / {len(vals) - 1} transitions")
    print(f"  raw=0 floor (static L/R offset, not fabricated to 0): "
          f"{chorus_depth_norm.get('0')}")

    return {"chorus_depth_norm": chorus_depth_norm, "raw_stereo_width": widths,
            "carrier_hz": CHORUS_DEPTH_CARRIER_HZ, "monotonicity_violations": violations}


# =============================================================================
# GAP 3: delay (types 6/7) -- stays PARAMETRIC per an explicit product
# decision (playability: a convolution IR freezes time/feedback/mix, a
# parametric delay stays adjustable in DecentSampler, and a delay genuinely
# IS time+feedback+pan -- only the NUMBERS were ever unvalidated). This
# section re-measures delay_time_s/delay_feedback/delay_pan_alternation
# using the clean injected impulse (pure-wet, so the render IS the echo
# train directly -- no cross-correlation against a smeared note attack
# needed), and derives a measured stereoOffset instead of a reasoned guess.
DELAY_TYPES = (6, 7)
DELAY_TIME_STEPS = [0, 16, 32, 48, 64, 80, 96, 112, 127]
DELAY_FIXED_TIME_RAW = 64   # matches tools/calibrate.cpp's own feedback-sweep convention
# 2% of the render's own peak: comfortably above the render's actual noise
# floor (measured ~-78dB below peak on these synthetic-impulse renders --
# there is essentially no noise at all in this path, see the flatness
# proof's -227dB isolation figure) but low enough to recover genuine,
# audible low-amplitude repeats. An earlier, more conservative 0.06 (-24dB)
# missed real decaying taps on several patches (e.g. type6 raw=32 has a
# clean, real ~0.5 per-repeat decay visible down to -34dB and beyond) and
# fed measure_feedback_gain_from_taps too few points, collapsing several
# genuinely nonzero feedback settings to a fabricated-looking 0.0 floor and
# visibly hurting the delay-reconstruction validation (patch "B Analog Seq"
# dropped to a NEGATIVE decay-region correlation when its real ~0.5
# per-repeat decay was floored away).
DELAY_TAP_PROMINENCE_FRAC = 0.02
DELAY_TAP_MERGE_SAMPLES = 8   # merges a multi-sample-wide single spike into one tap


def find_delay_taps(stereo, prominence_frac=DELAY_TAP_PROMINENCE_FRAC,
                    merge_samples=DELAY_TAP_MERGE_SAMPLES):
    """Peak-picks discrete echo taps directly from a pure-wet, impulse-
    excited Delay/Pan-Dly render: since the excitation IS a single impulse,
    the render's own combined |L|,|R| envelope is literally the echo train
    (a spike at each repeat), so no matched filtering/cross-correlation
    against a note's own attack shape is needed (contrast the old note-based
    method in tools/analyze_calibration.py). Returns a list of
    (sample_index, amplitude, l_amplitude, r_amplitude), earliest first."""
    combined = np.maximum(np.abs(stereo[:, 0]), np.abs(stereo[:, 1]))
    peak_amp = combined.max()
    if peak_amp < 1e-9:
        return []
    peaks, _ = find_peaks(combined, prominence=peak_amp * prominence_frac)
    merged = []
    for p in peaks:
        if merged and p - merged[-1][0] <= merge_samples:
            if combined[p] > merged[-1][1]:
                merged[-1] = (p, combined[p])
        else:
            merged.append((p, combined[p]))
    return [(int(p), float(a), float(np.abs(stereo[p, 0])), float(np.abs(stereo[p, 1])))
            for p, a in merged]


def measure_feedback_gain_from_taps(taps, skip_first=0):
    """Per-repeat gain g via a log-linear fit of tap amplitude vs. tap
    index, skipping the first `skip_first` taps -- Pan-Dly's first "there
    and back" pair is a fixed, feedback-INDEPENDENT artifact of its ping-
    pong routing (see the old analyze_calibration.py's own note on this,
    confirmed there via a full feedback sweep, and cmd_delay_capture below
    which only applies skip_first=2 for type 7 -- plain Delay/type 6 has no
    such quirk and every tap is a genuine repeat).

    Fewer than 3 taps surviving past the skipped ones is a REAL, physically
    meaningful floor (no reliable additional-repeat trend is audible within
    a 6-second capture at this feedback setting), reported as 0.0 -- not a
    fabricated number, the genuine "no additional repeats" case the old
    windowed-energy-ratio method's own docstring already described. A 2-point
    "fit" was tried and rejected: on type 7 at low feedback it produced
    wildly unstable, NON-monotonic readings (0.999 at raw16/32, dropping to
    0.664 at raw48) because two isolated points either side of the noise
    floor give an arbitrary slope, not a trend -- 3 points is the minimum for
    a fit that isn't just connecting two dots."""
    n_after = len(taps) - skip_first
    if n_after <= 2:
        return 0.0
    idx = np.arange(len(taps))[skip_first:]
    amps = np.array([a for _, a, _, _ in taps])[skip_first:]
    amps = np.clip(amps, 1e-6, None)
    slope, _ = np.polyfit(idx, np.log10(amps), 1)
    g = 10.0 ** slope
    return float(np.clip(g, 0.0, 0.999))


def measure_pan_alternation_from_taps(taps, skip_first=1):
    """max-min of each tap's L fraction (L/(L+R)) -- the exact-position
    version of the old analyze_calibration.py's measure_pan_alternation,
    which had to assume repeats land at exact multiples of a separately
    measured period; here each tap's own detected sample position is used
    directly."""
    fracs = []
    for _, _, l, r in taps[skip_first:]:
        tot = l + r
        if tot < 1e-6:
            continue
        fracs.append(l / tot)
    if len(fracs) < 2:
        return None
    return float(max(fracs) - min(fracs))


def cmd_delay_capture(wave_inject, roms, tmp, args, excitation_latency):
    print("\n=== GAP 3: delay time/feedback/pan (direct peak-picking on a pure-wet impulse) ===")
    out_dir = tmp / "delay_raw"
    run_wave_inject(wave_inject, [
        "capture-delay", "--roms", roms, "--wavegroup", "0",
        "--wavenumber", str(args.host_wavenumber), "--key", str(args.host_key),
        "--n", "8", "--amp-i", "127", "--out-dir", str(out_dir)])

    delay_time_s = {}
    delay_feedback = {}
    delay_pan_alternation = {}
    for t in DELAY_TYPES:
        time_table = {}
        for raw in DELAY_TIME_STEPS:
            stereo, sr = sf.read(str(out_dir / f"delay_t{t}_time_{raw:03d}.wav"))
            assert sr == SR
            taps = find_delay_taps(stereo[excitation_latency:])
            if not taps:
                print(f"  WARNING: type={t} raw={raw}: delay time unmeasurable "
                      f"(no tap found) -- recording null, not a fabricated number")
                time_table[str(raw)] = None
                continue
            time_table[str(raw)] = round(taps[0][0] / SR, 4)
        delay_time_s[str(t)] = time_table
        print(f"  type {t} delay_time_s: {time_table}")

        stereo_fb, sr = sf.read(str(out_dir / f"delay_t{t}_feedback_127.wav"))
        taps_fb127 = find_delay_taps(stereo_fb[excitation_latency:])
        pan_alt = measure_pan_alternation_from_taps(taps_fb127)
        delay_pan_alternation[str(t)] = round(pan_alt, 4) if pan_alt is not None else None
        print(f"  type {t} delay_pan_alternation (from {len(taps_fb127)} taps @ feedback=127): "
              f"{delay_pan_alternation[str(t)]}")

        # Pan-Dly's (type 7) first two taps are a fixed, feedback-independent
        # "there and back" ping-pong pair (see measure_feedback_gain_from_taps'
        # docstring); plain Delay (type 6) has no such quirk.
        skip_first = 2 if t == 7 else 0
        fb_table = {}
        for raw in DELAY_TIME_STEPS:
            stereo_fb, sr = sf.read(str(out_dir / f"delay_t{t}_feedback_{raw:03d}.wav"))
            taps = find_delay_taps(stereo_fb[excitation_latency:])
            g = measure_feedback_gain_from_taps(taps, skip_first=skip_first)
            fb_table[str(raw)] = round(g, 4)
        delay_feedback[str(t)] = fb_table
        print(f"  type {t} delay_feedback: {fb_table}")

    return {"delay_time_s": delay_time_s, "delay_feedback": delay_feedback,
            "delay_pan_alternation": delay_pan_alternation, "raw_dir": str(out_dir)}


def synth_delay_ir(delay_s, feedback, sr=SR, n_repeats=40):
    """A finite-length FIR approximation of a single-channel feedback delay
    (DecentSampler's own `delay` effect, per its developer guide) -- a spike
    at every multiple of delay_s, decaying by `feedback` per repeat. Used
    only to SIMULATE what DecentSampler's delay effect would sound like for
    a candidate stereoOffset, so its measured stereo width can be compared
    against the real hardware's -- not used anywhere in the shipped
    pipeline itself (emit_presets.py just sets the effect's own parameters;
    DecentSampler does this exact computation on playback)."""
    d = max(1, int(round(delay_s * sr)))
    length = d * n_repeats + 1
    ir = np.zeros(length)
    g = 1.0
    for k in range(n_repeats):
        idx = d * (k + 1)
        if idx >= length or g < 1e-4:
            break
        ir[idx] = g
        g *= feedback
    return ir


def simulate_ds_delay_width(dry_mono, delay_time_s, feedback, stereo_offset_s,
                            note_off_sample, window_samples, sr=SR):
    """Simulates DecentSampler's delay effect (independent per-channel taps,
    L = delayTime - stereoOffset/2, R = delayTime + stereoOffset/2 -- per
    the official developer guide's own worked example) applied to a mono
    dry signal, and measures the SAME stereo-width metric used for the
    chorus-depth measurement (RMS(L-R)/RMS(L+R)) over the same decay window
    used for groundtruth validation, so it is directly comparable to a
    measurement of the real hardware's own wet render."""
    l_delay = max(0.0003, delay_time_s - stereo_offset_s / 2.0)
    r_delay = max(0.0003, delay_time_s + stereo_offset_s / 2.0)
    ir_l = synth_delay_ir(l_delay, feedback, sr)
    ir_r = synth_delay_ir(r_delay, feedback, sr)
    wet_l = fftconvolve(dry_mono, ir_l, mode="full")
    wet_r = fftconvolve(dry_mono, ir_r, mode="full")
    n = min(len(wet_l), len(wet_r))
    stereo = np.stack([wet_l[:n], wet_r[:n]], axis=1)
    return measure_stereo_width(stereo, skip_s=note_off_sample / sr,
                                max_s=(note_off_sample + window_samples) / sr, sr=sr)


def find_stereo_offset(wave_inject, roms, tmp, pan_dly_patch_indices, delay_time_table,
                       feedback_table, candidates_s=None):
    """Measures the real hardware's own stereo width on Pan-Dly groundtruth
    renders (native reverb intact, chorus disabled -- same renders used for
    the parametric validation below), then finds which candidate DS
    stereoOffset value makes the SIMULATED parametric delay's width closest
    to that real measurement -- i.e. measured, not analogised from the
    developer guide's own generic example. DS's stereoOffset is a TIME
    offset between channels and structurally CANNOT reproduce true hard
    ping-pong alternation (a per-repeat amplitude pan), so this reports how
    close the best achievable value gets, honestly, rather than pretending
    a time offset can fully substitute for a pan swing.
    """
    if candidates_s is None:
        candidates_s = [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]
    print("\n--- stereoOffset search (type 7 / Pan-Dly) ---")
    results = []
    for idx in pan_dly_patch_indices:
        wet, dry, effects = run_groundtruth_render(wave_inject, roms, tmp, idx)
        note_off = int(GROUNDTRUTH_HOLD_SECONDS * SR)
        if len(wet) < note_off + int(MIN_RELIABLE_DECAY_S * SR):
            print(f"  patch {idx}: unreliable (render exited almost immediately) -- skipped")
            continue
        wet_env, wet_db = smoothed_db_envelope(wet[note_off:note_off + int(6.0 * SR)])
        below = np.where(wet_db <= DECAY_FLOOR_DB)[0]
        hop = 64
        window_samples = (int(below[0]) * hop) if len(below) else int(6.0 * SR)
        window_samples = max(window_samples, int(0.3 * SR))

        wet_stereo, _ = sf.read(str(tmp / f"groundtruth_{idx}" / "groundtruth_wet.wav"))
        real_width = measure_stereo_width(wet_stereo, skip_s=note_off / SR,
                                          max_s=(note_off + window_samples) / SR)
        if real_width is None:
            continue

        try:
            delay_time = _interp_table(delay_time_table, effects["reverb_time"])
            feedback = _interp_table(feedback_table, effects["reverb_feedback"])
        except ValueError:
            print(f"  patch {idx}: missing delay_time_s/delay_feedback lookup -- skipped")
            continue

        best_so, best_diff, best_width = None, None, None
        for so in candidates_s:
            sim_width = simulate_ds_delay_width(dry, delay_time, feedback, so,
                                                note_off, window_samples)
            if sim_width is None:
                continue
            diff = abs(sim_width - real_width)
            if best_diff is None or diff < best_diff:
                best_diff, best_so, best_width = diff, so, sim_width
        results.append({"patch": idx, "name": effects["name"], "real_width": real_width,
                        "best_stereo_offset_s": best_so, "sim_width_at_best": best_width,
                        "delay_time_s": delay_time, "feedback": feedback})
        print(f"  patch {idx} ({effects['name']}): real_width={real_width:.4f}  "
              f"best_stereoOffset={best_so}s (sim_width={best_width:.4f})  "
              f"delayTime={delay_time}s feedback={feedback}")
    return results


def compare_decay_region_delay(wet_mono, dry_mono, delay_time_s, feedback, wet_level,
                               stereo_offset_s=0.0, hold_seconds=GROUNDTRUTH_HOLD_SECONDS,
                               decay_floor_db=DECAY_FLOOR_DB, min_reliable_s=MIN_RELIABLE_DECAY_S):
    """Same decay-region comparison as compare_decay_region (reverb/
    convolution), but for the PARAMETRIC delay reconstruction: DecentSampler's
    `delay` effect return is ADDITIVE (wetLevel), like reverb's, not a BLEND
    like chorus/convolution's `mix` -- see tools/emit_presets.py's own
    comment on send-vs-mix semantics. Reconstruction is therefore
    `dry + delay_tap_signal * wet_level`, averaging the simulated L/R taps to
    mono for a like-for-like comparison against the (already mono-summed)
    wet reference."""
    note_off = int(hold_seconds * SR)
    if len(wet_mono) < note_off + int(min_reliable_s * SR):
        return None
    l_delay = max(0.0003, delay_time_s - stereo_offset_s / 2.0)
    r_delay = max(0.0003, delay_time_s + stereo_offset_s / 2.0)
    ir_l = synth_delay_ir(l_delay, feedback)
    ir_r = synth_delay_ir(r_delay, feedback)
    tap_l = fftconvolve(dry_mono, ir_l, mode="full")
    tap_r = fftconvolve(dry_mono, ir_r, mode="full")
    n = max(len(dry_mono), len(tap_l), len(tap_r))
    dry_pad = np.zeros(n)
    dry_pad[:len(dry_mono)] = dry_mono
    tap_mono = np.zeros(n)
    tap_mono[:len(tap_l)] += tap_l * 0.5
    tap_mono[:len(tap_r)] += tap_r * 0.5
    recon = dry_pad + tap_mono * wet_level

    wet_env, wet_db = smoothed_db_envelope(wet_mono[note_off:note_off + int(6.0 * SR)])
    below = np.where(wet_db <= decay_floor_db)[0]
    hop = 64
    window_samples = (int(below[0]) * hop) if len(below) else int(6.0 * SR)
    window_samples = max(window_samples, int(0.3 * SR))
    if window_samples < int(min_reliable_s * SR):
        return None

    def norm_tail(x, start, n):
        n = min(n, max(0, len(x) - start))
        t = x[start:start + n]
        peak = np.max(np.abs(t)) + 1e-12
        return t / peak

    wet_tail = norm_tail(wet_mono, note_off, window_samples)
    recon_tail = norm_tail(recon, note_off, window_samples)
    corr = envelope_correlation(wet_tail, recon_tail)
    bands = [("low_80_250", 80, 250), ("mid_250_2000", 250, 2000), ("high_2000_8000", 2000, 8000)]
    band_ratios = {}
    for name, lo, hi in bands:
        e_wet = band_energy(wet_tail, SR, lo, hi)
        e_recon = band_energy(recon_tail, SR, lo, hi)
        band_ratios[name] = (e_recon / e_wet) if e_wet > 0 else None
    return {"correlation": corr, "band_ratios": band_ratios, "window_s": window_samples / SR}


def _discover_delay_test_patches(wave_inject, roms, tmp, max_per_type=6):
    stdout, _ = run_wave_inject(wave_inject, ["effects", "--roms", roms])
    by_type = {6: [], 7: []}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row["reverb_type"] in by_type:
            by_type[row["reverb_type"]].append(row)
    # Prefer patches with SOME feedback (>0) so the reconstruction is
    # actually tested against multiple repeats, not just a single tap --
    # but keep a couple of feedback=0 patches too for coverage.
    out = {}
    for t in (6, 7):
        rows = sorted(by_type[t], key=lambda r: -r["reverb_feedback"])
        out[t] = [r["index"] for r in rows[:max_per_type]]
    return out


def cmd_delay_validate(wave_inject, roms, tmp, delay_time_s, delay_feedback,
                       stereo_offset_by_type):
    print("\n=== GAP 3 VALIDATE: parametric delay reconstruction vs real hardware ===")
    test_set = _discover_delay_test_patches(wave_inject, roms, tmp)
    report = {}
    for t in DELAY_TYPES:
        per_patch = []
        n_skipped = 0
        for idx in test_set[t]:
            wet, dry, effects = run_groundtruth_render(wave_inject, roms, tmp, idx)
            try:
                dt = _interp_table(delay_time_s.get(str(t), {}), effects["reverb_time"])
                fb = _interp_table(delay_feedback.get(str(t), {}), effects["reverb_feedback"])
            except ValueError:
                n_skipped += 1
                continue
            wet_level = max(0.0, min(1.0, _amount(effects["reverb_level"]) *
                                     (effective_send_from_effects(effects) / 127.0)))
            so = stereo_offset_by_type.get(t, 0.0)
            result = compare_decay_region_delay(wet, dry, dt, fb, wet_level, stereo_offset_s=so)
            if result is None:
                n_skipped += 1
                continue
            per_patch.append((idx, effects["name"], result["correlation"]))
        mean_corr = float(np.mean([c for _, _, c in per_patch])) if per_patch else None
        report[t] = {"mean_correlation": mean_corr, "n_reliable": len(per_patch),
                    "n_skipped": n_skipped,
                    "per_patch": [{"index": i, "name": n, "correlation": round(c, 4)}
                                  for i, n, c in per_patch]}
        corr_str = f"{mean_corr:.4f}" if mean_corr is not None else "n/a"
        print(f"  type {t} ({ALL_TYPE_NAMES[t]}): mean={corr_str}  "
              f"n_reliable={len(per_patch)} n_skipped={n_skipped}  "
              f"indiv={[(i, round(c, 3)) for i, _, c in per_patch]}")
    return report


def merge_into_calibration_json(calibration_json_path, report, out_dir):
    """Writes this tool's measured tables into calib/calibration.json,
    alongside (not replacing) the tables tools/analyze_calibration.py still
    owns (chorus_rate_hz, reverb_rt60, etc. -- unrelated calibrate.cpp-driven
    sweeps this task does not touch). Only overwrites the keys this tool
    actually produced this run, so a partial run (e.g. --stage chorus-depth
    alone) never clobbers tables it didn't recompute."""
    cal = json.loads(calibration_json_path.read_text()) if calibration_json_path.exists() else {}

    if "ir_bank" in report:
        reverb_ir, reverb_ir_silent = {}, {}
        for t in REVERB_TYPES:
            table, silent = {}, []
            for raw in REVERB_TIME_STEPS:
                entry = report["ir_bank"].get(t, {}).get(raw)
                if entry is None:
                    table[str(raw)] = None
                    continue
                rel = Path(entry["file"])   # e.g. "ir_synth/reverb_t0_time_000.wav"
                table[str(raw)] = str(rel)
                if entry.get("silent"):
                    silent.append(raw)
            reverb_ir[str(t)] = table
            reverb_ir_silent[str(t)] = silent
        cal["reverb_ir"] = reverb_ir
        cal["reverb_ir_silent"] = reverb_ir_silent

    if "chorus_depth" in report:
        cal["chorus_depth_norm"] = report["chorus_depth"]["chorus_depth_norm"]

    if "delay_capture" in report:
        cal["delay_time_s"] = report["delay_capture"]["delay_time_s"]
        cal["delay_feedback"] = report["delay_capture"]["delay_feedback"]
        cal["delay_pan_alternation"] = report["delay_capture"]["delay_pan_alternation"]
    if "chosen_pan_dly_stereo_offset_s" in report:
        cal["pan_dly_stereo_offset_s"] = round(report["chosen_pan_dly_stereo_offset_s"], 4)

    calibration_json_path.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"\nMerged into {calibration_json_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True, help="JV-880 ROM directory")
    ap.add_argument("--wave-inject", default="build/wave_inject")
    ap.add_argument("--out", default="calib/ir_synth")
    ap.add_argument("--tmp", default="/tmp/ir_capture")
    ap.add_argument("--calibration-json", default="calib/calibration.json")
    ap.add_argument("--host-wavenumber", type=int, default=50,
                    help="wavegroup-0 wave slot to host the injected content (default: 50, "
                         "length 8852 samples, comfortable headroom)")
    ap.add_argument("--host-key", type=int, default=127,
                    help="MIDI key for the impulse proof / IR capture (127: minimal-width "
                         "excitation -- see the proof step's own findings)")
    ap.add_argument("--stage", default="all",
                    choices=["all", "reverb", "chorus-depth", "delay"],
                    help="which gap to run (default: all three)")
    ap.add_argument("--skip-capture", action="store_true", help="reverb: skip re-capturing IRs")
    ap.add_argument("--skip-groundtruth", action="store_true", help="reverb: skip validation")
    ap.add_argument("--no-merge", action="store_true",
                    help="don't write results into calibration.json (report only)")
    args = ap.parse_args()

    wave_inject = Path(args.wave_inject).resolve()
    if not wave_inject.exists():
        raise SystemExit(f"wave_inject binary not found at {wave_inject} -- build it first")

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)

    report = {}
    report["proof"] = cmd_proof(wave_inject, args.roms, tmp, args)

    if not report["proof"]["passed"]:
        print("\nPROOF FAILED -- refusing to proceed (per task instructions: 'if you cannot "
              "exceed the current 0.552 baseline, the injection has failed and you should say "
              "so rather than proceed').")
        Path("calib").mkdir(exist_ok=True)
        (out_dir.parent / "ir_synth_report.json").write_text(json.dumps(report, indent=2, default=str))
        sys.exit(1)

    excitation_latency = report["proof"]["excitation_latency_samples"]

    if args.stage in ("all", "reverb"):
        if not args.skip_capture:
            bank = cmd_capture(wave_inject, args.roms, tmp, out_dir, args, excitation_latency)
            report["ir_bank"] = bank
            cmd_sanity_vs_baseline(bank, Path("calib/calibration.json"))
        else:
            bank = {}

        if not args.skip_groundtruth:
            raw_dir = tmp / "ir_raw"
            report["groundtruth_multi"] = cmd_groundtruth_multi(
                wave_inject, args.roms, tmp, raw_dir, args, excitation_latency)

    if args.stage in ("all", "chorus-depth"):
        report["chorus_depth"] = cmd_chorus_depth(wave_inject, args.roms, tmp, args)

    if args.stage in ("all", "delay"):
        delay_cap = cmd_delay_capture(wave_inject, args.roms, tmp, args, excitation_latency)
        report["delay_capture"] = delay_cap

        pan_dly_patches = _discover_delay_test_patches(wave_inject, args.roms, tmp, max_per_type=6)[7]
        so_results = find_stereo_offset(wave_inject, args.roms, tmp, pan_dly_patches,
                                        delay_cap["delay_time_s"]["7"], delay_cap["delay_feedback"]["7"])
        report["stereo_offset_search"] = so_results
        measured_so = [r["best_stereo_offset_s"] for r in so_results if r["best_stereo_offset_s"] is not None]
        chosen_so = float(np.median(measured_so)) if measured_so else 0.0
        report["chosen_pan_dly_stereo_offset_s"] = chosen_so
        print(f"\n  chosen Pan-Dly stereoOffset (median of {len(measured_so)} patches): "
              f"{chosen_so:.4f}s")

        report["delay_validate"] = cmd_delay_validate(
            wave_inject, args.roms, tmp, delay_cap["delay_time_s"], delay_cap["delay_feedback"],
            {6: 0.0, 7: chosen_so})

    if not args.no_merge:
        merge_into_calibration_json(Path(args.calibration_json), report, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
