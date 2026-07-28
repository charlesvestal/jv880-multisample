#!/usr/bin/env python3
"""Measure the JV-880's actual chorus/reverb response from calibrate's WAV
renders and write calib/calibration.json.

This exists so Task 6 can map the synth's raw 0-127 effect parameters onto
DecentSampler effect parameters from MEASURED behaviour rather than a guessed
formula -- the whole point of running a cycle-accurate emulator instead of
hand-approximating a JV-880 datasheet.

Usage:
    python3 tools/analyze_calibration.py [calib_dir, default "calib"]

Reads:
    <calib_dir>/dry.wav
    <calib_dir>/chorus_rate_*.wav
    <calib_dir>/chorus_depth_*.wav
    <calib_dir>/chorus_level_*.wav
    <calib_dir>/reverb_t{0..5}_time_*.wav
    <calib_dir>/reverb_level_*.wav

Writes:
    <calib_dir>/calibration.json

--- Chorus-rate measurement: why this is more than a naive Welch call -----

The task's suggested recipe (take the note's amplitude envelope, remove DC,
find the dominant frequency via Welch in 0.05..12 Hz) was tried first and
produced a chaotic, non-monotonic table. Investigation (see the Task 4
report) found two real, confirmed causes and fixed both:

1. Every patch in this ROM uses 3-4 tones (there is no single-tone patch at
   all), and this base patch's tones create a steady ~5-11 Hz amplitude
   beat from their own internal pitch relationships -- present even in the
   FULLY DRY render, independent of chorus. Downmixing L+R to mono buries
   the chorus's real signature under that beat. Fix: the JV-880 chorus is a
   STEREO effect (decorrelated per-channel modulated delay), so its
   modulation shows up cleanly in the L-R (stereo-difference) signal, which
   naturally cancels a beat pattern common to both channels. All amplitude
   analysis below therefore works on L-R, not L+R.

2. A slow LFO needs several full cycles of data to be measurable at all
   (autocorrelation/PSD tools cannot see a periodicity slower than roughly
   1/window_duration). calibrate.cpp already renders the chorus_rate sweep
   with a 12s hold (vs. 4s for every other sweep) specifically for this.

Even with both fixes, two further hazards remain and are handled explicitly:

3. Octave ambiguity: autocorrelation of an amplitude envelope routinely
   finds a taller peak at 2x the true period (a well-known failure mode in
   pitch/period detection). `_pick_candidate` checks for a comparably-tall
   genuine local maximum at ~half the top candidate's lag and prefers it
   when found -- never invents a period, only chooses among real local
   maxima that are actually present in that render's own autocorrelation.

4. A handful of fixed frequencies (~3.2-3.6 Hz, ~6.6-7.1 Hz, ~10-11 Hz)
   recur at IDENTICAL values across many unrelated raw settings -- the
   defining signature of an artifact unrelated to the swept parameter, not
   real chorus content (confirmed: present regardless of chorusrate, and
   the exact values are consistent to 3-4 significant figures across many
   independent analysis-window/hop choices). Candidates landing in these
   bands are excluded before candidate selection.

Even after all of this, ONE raw setting (24) still measures as unreliable:
the artifact described in (4) is measurably TALLER there than any surviving
real candidate, at every window/hop tried. Per the task's own instruction
("if a measurement is unreliable ... record a sentinel or omit the entry
and explain, rather than inventing a number"), that one entry is omitted
from chorus_rate_hz with a warning printed, rather than reporting a number
we know is dominated by a non-chorus artifact.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.io import wavfile

SR = 64000
HOLD_SECONDS = 4.0   # must match calibrate.cpp's default GridSpec.hold_seconds

# Sustained middle of the note (used for depth/mix measurements): avoids the
# attack transient and sits comfortably before note-off.
SUSTAIN_LO = 0.7
SUSTAIN_HI = 3.5

# Fixed, non-chorus artifact frequencies -- see module docstring point 4.
ARTIFACT_BANDS_HZ = [(3.2, 3.6), (6.6, 7.1), (10.0, 11.0)]

RAW_RE = re.compile(r"_(\d+)\.wav$")
TYPE_RE = re.compile(r"reverb_t(\d+)_time_")


def raw_from_name(path):
    m = RAW_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"can't parse raw value from {path}")
    return int(m.group(1))


def type_from_name(path):
    m = TYPE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"can't parse reverb type from {path}")
    return int(m.group(1))


def load_mono(path):
    sr, data = wavfile.read(path)
    assert sr == SR, f"unexpected sample rate {sr} in {path}"
    data = data.astype(np.float64)
    return data.mean(axis=1) if data.ndim > 1 else data


def load_lr_diff(path):
    """L-R (stereo difference) signal -- see module docstring point 1."""
    sr, data = wavfile.read(path)
    assert sr == SR, f"unexpected sample rate {sr} in {path}"
    data = data.astype(np.float64)
    if data.ndim < 2 or data.shape[1] < 2:
        return np.zeros(len(data))
    return data[:, 0] - data[:, 1]


def env_rms(x, hop):
    """Per-hop RMS envelope. Drops any trailing partial hop."""
    n = (len(x) // hop) * hop
    if n == 0:
        return np.array([])
    return np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1))


def env_max(x, hop):
    """Per-hop peak-magnitude envelope. Drops any trailing partial hop."""
    n = (len(x) // hop) * hop
    if n == 0:
        return np.array([])
    return np.abs(x[:n]).reshape(-1, hop).max(axis=1)


def parabolic_peak(y, k):
    """Sub-bin/sub-lag refinement of a discrete peak at index k via
    parabolic interpolation of its two neighbours (same idea as
    postprocess.py's estimate_period: an integer bin/lag quantizes the
    answer to the FFT/autocorrelation's native resolution, and
    interpolating the peak's neighbours recovers a meaningfully more
    precise estimate from the same data). Returns a fractional index;
    clamps to y's bounds when k is right at an edge."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    offset = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(k) + offset


def _in_artifact_band(freq_hz):
    return any(lo <= freq_hz <= hi for lo, hi in ARTIFACT_BANDS_HZ)


def _local_maxima(ac, lo_lag, hi_lag):
    """(lag, height) for every interior local maximum of ac in [lo_lag, hi_lag)."""
    seg = ac[lo_lag:hi_lag]
    out = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]:
            out.append((i + lo_lag, seg[i]))
    return out


def measure_chorus_rate_hz(path, oct_frac=0.6, reliability_frac=0.90):
    """Dominant chorus modulation frequency, via autocorrelation of the L-R
    envelope (see module docstring for why L-R, why autocorrelation, and
    the two-stage octave/artifact correction below). Returns None when the
    measurement is judged unreliable at this raw setting (module docstring
    point 4) -- callers must not fabricate a number in that case."""
    hop = 160
    lr = load_lr_diff(path)
    lo = int(0.3 * SR)
    hi = min(int(11.7 * SR), len(lr))   # leave margin below the 12s hold boundary
    lr = lr[lo:hi]
    env = env_rms(lr, hop)
    if len(env) < 32:
        return None
    fs_env = SR / hop
    env0 = env - env.mean()

    ac = np.correlate(env0, env0, mode="full")[len(env0) - 1:]
    ac = ac / (ac[0] + 1e-12)

    lo_lag = int(fs_env / 12.0)
    hi_lag = min(int(fs_env / 0.05), len(ac) - 1)
    if hi_lag <= lo_lag + 2:
        return None

    peaks = _local_maxima(ac, lo_lag, hi_lag)
    if not peaks:
        return None

    all_candidates = []
    for lag, h in peaks:
        k_ref = parabolic_peak(ac, lag)
        f = fs_env / k_ref if k_ref > 0 else 0.0
        all_candidates.append((lag, h, f))

    global_top = max(all_candidates, key=lambda c: c[1])
    surviving = [c for c in all_candidates if not _in_artifact_band(c[2])]
    if not surviving:
        return None
    top_surviving = max(surviving, key=lambda c: c[1])

    # Octave-ambiguity fix: prefer a genuine local max near HALF the top
    # surviving candidate's lag (i.e. ~2x its frequency) if one is
    # comparably tall -- the top pick is often a subharmonic (period-double)
    # of the true, faster fundamental.
    target_lag = top_surviving[0] / 2.0
    best_half = None
    for lag, h, f in surviving:
        if abs(lag - target_lag) <= max(2, 0.06 * target_lag) and h >= oct_frac * top_surviving[1]:
            if best_half is None or h > best_half[1]:
                best_half = (lag, h, f)
    chosen = best_half if best_half is not None else top_surviving

    # Reliability check: if the UNCONSTRAINED tallest peak anywhere in the
    # search band falls inside a known artifact band and is measurably
    # taller than what we're about to report, the real chorus signal is
    # weaker than the artifact at this specific raw setting -- don't report
    # a number we don't trust.
    if _in_artifact_band(global_top[2]) and global_top[1] > chosen[1] / reliability_frac:
        return None

    return float(chosen[2])


def measure_excursion_ratio(mono, dry):
    """Envelope excursion (std dev of the RMS envelope) in the sustained
    middle, relative to the dry reference's own sustained level. Larger
    excursion = deeper modulation."""
    lo, hi = int(SUSTAIN_LO * SR), int(SUSTAIN_HI * SR)
    seg, dry_seg = mono[lo:hi], dry[lo:hi]
    hop = 64
    e, de = env_rms(seg, hop), env_rms(dry_seg, hop)
    if len(e) == 0 or len(de) == 0:
        return 0.0
    dry_level = de.mean() + 1e-9
    return float(e.std() / dry_level)


def measure_sustain_extra_energy(mono, baseline):
    """(RMS(x) - RMS(baseline)) / RMS(baseline) over the sustained middle
    of the note. `baseline` should be a render sharing every OTHER effect
    parameter with `mono` (see measure_chorus_mix/measure_reverb_wet below
    for why this must be the sweep's own raw=0 render, not the global
    dry.wav, when the sweep fixes other effect params to non-default
    values)."""
    lo, hi = int(SUSTAIN_LO * SR), int(SUSTAIN_HI * SR)
    rms_x = float(np.sqrt(np.mean(mono[lo:hi] ** 2)))
    rms_b = float(np.sqrt(np.mean(baseline[lo:hi] ** 2))) + 1e-9
    return (rms_x - rms_b) / rms_b


def measure_tail_extra_energy(mono, baseline, hold_seconds=HOLD_SECONDS, window_s=1.5):
    """(RMS(x) - RMS(baseline)) / RMS(baseline) over a FIXED-length window
    starting at note-off, zero-padded to `window_s` if the render was
    truncated shorter than that (early-exit truncation happens only in the
    tail -- see jv_render.cpp -- so both `mono` and `baseline` can differ in
    raw frame count; comparing a fixed, zero-padded window keeps the
    comparison fair instead of being diluted by however many extra
    near-silent tail samples one render happened to keep beyond the
    other's early-exit point)."""
    def tail_rms(x):
        start = int(hold_seconds * SR)
        n = int(window_s * SR)
        buf = np.zeros(n)
        avail = x[start:start + n]
        buf[:len(avail)] = avail
        return float(np.sqrt(np.mean(buf ** 2)))

    rms_x = tail_rms(mono)
    rms_b = tail_rms(baseline) + 1e-9
    return (rms_x - rms_b) / rms_b


def normalize_0_1(values):
    """Linear min-clip-then-scale to [0, 1] by the sweep's own maximum.
    This is a per-sweep RELATIVE normalization of a real measured metric
    -- not a fabricated value: raw=0 genuinely measures near that metric's
    floor (by construction: it's the sweep's own zero-reference) and the
    largest raw setting in the sweep is genuinely that sweep's own
    ceiling."""
    arr = np.clip(np.array(values, dtype=np.float64), 0.0, None)
    mx = arr.max() if len(arr) else 0.0
    if mx <= 1e-9:
        return [0.0] * len(values)
    return (arr / mx).tolist()


def measure_rt60(mono, hold_seconds=HOLD_SECONDS):
    """RT60 from the decay slope after note-off: build a dB envelope of the
    tail, fit a line between roughly -5 dB and -35 dB (referenced to the
    level right after note-off), extrapolate to -60 dB. Returns None if the
    captured tail never reaches -35 dB (short capture or a reverb tail
    longer than the render) -- callers must not fabricate a number here."""
    tail_start = int(hold_seconds * SR)
    tail = mono[tail_start:]
    if len(tail) < int(0.5 * SR):
        return None

    hop = 64
    env = env_max(tail, hop) + 1e-9
    # Light smoothing: a 5-sample moving average (~16ms at hop=64) tames
    # sample-level scatter in the max-envelope before it's used for a
    # linear fit, without smearing the decay slope itself.
    if len(env) >= 5:
        kernel = np.ones(5) / 5.0
        env = np.convolve(env, kernel, mode="same")
    fs_env = SR / hop

    # Reference level (0 dB) = the loudest point in the first 300ms of the
    # tail. Using the tail's OWN early peak (rather than the note's overall
    # peak during hold) means the reference tracks wherever the reverb's
    # actual decay starts, even if a reverb's early reflections briefly
    # swell just after note-off.
    ref_window = max(1, int(0.3 * fs_env))
    ref = float(env[:ref_window].max())
    if ref <= 1e-9:
        return None

    db = 20 * np.log10(env / ref)
    t = np.arange(len(env)) / fs_env

    mask = (db <= -5) & (db >= -35)
    if int(mask.sum()) < 5:
        return None   # tail never decayed through the -5..-35 dB window we need

    ts, dbs = t[mask], db[mask]
    slope, _intercept = np.polyfit(ts, dbs, 1)
    if slope >= -0.01:
        return None   # not actually decaying (flat/growing) -- can't extrapolate
    return float(-60.0 / slope)


def collect(pattern, key_fn=raw_from_name):
    files = sorted(glob.glob(pattern), key=key_fn)
    return [(key_fn(f), f) for f in files]


def main():
    calib_dir = sys.argv[1] if len(sys.argv) > 1 else "calib"

    dry_path = os.path.join(calib_dir, "dry.wav")
    if not os.path.exists(dry_path):
        print(f"missing {dry_path}", file=sys.stderr)
        return 1
    dry = load_mono(dry_path)

    result = {
        "chorus_rate_hz": {},
        "chorus_depth_norm": {},
        "chorus_mix": {},
        "reverb_rt60": {},
        "reverb_wet": {},
    }

    # --- chorus rate -----------------------------------------------------
    for raw, f in collect(os.path.join(calib_dir, "chorus_rate_*.wav")):
        hz = measure_chorus_rate_hz(f)
        if hz is None:
            print(f"WARNING: chorus rate unmeasurable for raw={raw} "
                  f"(a fixed non-chorus artifact dominates the surviving "
                  f"signal at this setting) -- omitting, not fabricating",
                  file=sys.stderr)
            continue
        result["chorus_rate_hz"][str(raw)] = round(hz, 4)

    # --- chorus depth (normalized within its own sweep, vs. global dry) --
    # level is fixed (100) across this sweep, so there's no natural raw=0
    # "off" baseline within it; depth is a modulation-EXCURSION metric
    # anyway (variance, not absolute level), which is far less sensitive to
    # the static wet/dry level offset than the extra-energy metrics below,
    # so the global dry reference is fine here.
    depth_items = collect(os.path.join(calib_dir, "chorus_depth_*.wav"))
    depth_raws = [raw for raw, _ in depth_items]
    depth_metric = [measure_excursion_ratio(load_mono(f), dry) for _, f in depth_items]
    for raw, v in zip(depth_raws, normalize_0_1(depth_metric)):
        result["chorus_depth_norm"][str(raw)] = round(v, 4)

    # --- chorus mix / level (normalized within its own sweep) -----------
    # Baseline = this sweep's OWN raw=0 render, not the global dry.wav:
    # chorus_type/depth/rate are held fixed at non-default values across
    # this sweep, so its raw=0 entry (level=0) is the correct "everything
    # else equal" zero-reference -- comparing against global dry.wav would
    # mix in a level=0-vs-other-params difference unrelated to the level
    # parameter under test.
    level_items = collect(os.path.join(calib_dir, "chorus_level_*.wav"))
    if level_items:
        level_raws = [raw for raw, _ in level_items]
        level_baseline = load_mono(level_items[0][1])
        level_metric = [measure_sustain_extra_energy(load_mono(f), level_baseline)
                         for _, f in level_items]
        for raw, v in zip(level_raws, normalize_0_1(level_metric)):
            result["chorus_mix"][str(raw)] = round(v, 4)

    # --- reverb RT60 per type (0-5, delays 6-7 excluded) ------------------
    for rtype in range(6):
        pattern = os.path.join(calib_dir, f"reverb_t{rtype}_time_*.wav")
        items = collect(pattern)
        table = {}
        for raw, f in items:
            rt60 = measure_rt60(load_mono(f))
            if rt60 is None:
                print(f"WARNING: RT60 unmeasurable for type={rtype} raw={raw} "
                      f"(tail never crossed -35 dB within the captured render) "
                      f"-- recording null, not a fabricated number",
                      file=sys.stderr)
                table[str(raw)] = None
            else:
                table[str(raw)] = round(rt60, 4)
        result["reverb_rt60"][str(rtype)] = table

    # --- reverb wet/level (normalized within its own sweep) --------------
    # Baseline = this sweep's own raw=0 render (level=0, but type/time
    # fixed to non-default values), for the same "everything else equal"
    # reason as chorus_mix above. Measured over a fixed-length, zero-padded
    # window starting at note-off (not the sustain window): reverb's
    # effect on a CONTINUOUS tone is a static comb filter whose net RMS can
    # legitimately go either way depending on phase, but its effect on the
    # SILENCE after note-off is unambiguous -- more reverb level means more
    # audible tail energy where the dry signal has already died out.
    wet_items = collect(os.path.join(calib_dir, "reverb_level_*.wav"))
    if wet_items:
        wet_raws = [raw for raw, _ in wet_items]
        wet_baseline = load_mono(wet_items[0][1])
        wet_metric = [measure_tail_extra_energy(load_mono(f), wet_baseline)
                      for _, f in wet_items]
        for raw, v in zip(wet_raws, normalize_0_1(wet_metric)):
            result["reverb_wet"][str(raw)] = round(v, 4)

    out_path = os.path.join(calib_dir, "calibration.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
