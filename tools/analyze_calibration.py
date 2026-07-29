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

Even with both fixes, further hazards remain and are handled explicitly:

3. Octave ambiguity, generalised to a full subharmonic FAMILY: autocorrelation
   of an amplitude envelope routinely finds a taller peak at an integer
   multiple of the true period -- not just 2x, but 3x, 5x, 7x... (longer
   lags accumulate more correlation over a fixed-length window). Walk T/k for
   k=2..8 and collect every comparably-tall genuine peak found; report the
   TALLEST one across the whole family, not simply the last k examined.
   Never invents a period -- only chooses among real local maxima that are
   actually present in that render's own autocorrelation.

   (An earlier version of this fix only probed half the top lag, which
   cannot reach a 3rd- or 5th-order subharmonic, and separately, an earlier
   version of the family walk kept overwriting its answer with whichever k
   it examined LAST rather than the tallest -- both produced confirmed,
   measured misreads; see measure_chorus_rate_hz's inline comments for the
   specific raw settings and numbers each version got wrong.)

4. A handful of fixed frequencies (~3.2-3.6 Hz, ~6.6-7.1 Hz, ~10-11 Hz)
   recur at IDENTICAL values across many unrelated raw settings -- the
   defining signature of an artifact unrelated to the swept parameter, not
   real chorus content (confirmed: present regardless of chorusrate, and
   the exact values are consistent to 3-4 significant figures across many
   independent analysis-window/hop choices; also confirmed present, at
   comparable relative strength, in dry.wav -- i.e. with chorus fully off --
   so it is patch/render behaviour, not something the chorus effect itself
   generates). Candidates landing in these bands are excluded before
   candidate selection.

   (A dry.wav-envelope subtraction/gating approach was also tried, on the
   reasoning that anything present in the chorus-off reference can't be
   chorus. It does NOT generalise here: dry.wav's own window is only ~4.3s
   (vs. the chorus-rate sweep's 12s), and at the high end of the sweep
   (raw>=104, true rates ~6-9 Hz) dry.wav shows substantial apparent energy
   at the SAME frequencies as the genuine, confidently-measured chorus
   signal there -- a short reference window can't tell a few cycles of real
   high-rate modulation apart from a few cycles of its own short-window
   noise. Gating on it would have wrongly suppressed several of the
   cleanest points in the table. The fixed ARTIFACT_BANDS_HZ, characterised
   directly from repeated identical values across the WET sweep itself,
   remains the reliable signal here.)

Even after all of this, TWO raw settings (24 and 80) still measure as
unreliable: the artifact described in (4) is at least as tall there as any
surviving real candidate, at every window/hop tried. Per the task's own
instruction ("if a measurement is unreliable ... record a sentinel or omit
the entry and explain, rather than inventing a number"), those two entries
are omitted from chorus_rate_hz with a warning printed, rather than
reporting a number we know is dominated by a non-chorus artifact. (raw=80
was, before the reliability check's grace margin was removed, silently
reporting 1.6974 Hz -- its in-band artifact peak was already outright taller
than its own best surviving candidate, an even closer artifact-dominance
call than raw=24's; the old 10% grace margin was masking it.)

--- IMPORTANT for anything (Task 6 or otherwise) consuming calibration.json -

1. chorusrate raw=24 and raw=80 are intentional INTERPOLATION GAPS, not a
   zero and not "no chorus at this setting". A naive `table.get("24")` or
   `table[24]` read against chorus_rate_hz will come back missing/None -- a
   consumer that doesn't handle that could silently produce 0 Hz for a
   setting the real hardware clearly does modulate at some nonzero rate we
   just couldn't pin down cleanly (see point 4 above). The correct handling
   is to interpolate between the neighbouring present raw values (raw=24:
   16 -> 1.5573 Hz, 32 -> 1.8002 Hz; raw=80: 72 -> 2.9157 Hz, 88 -> 3.9109
   Hz), the same way every OTHER raw value in between the sampled
   step-8/step-16 points already has to be interpolated. Task 6's
   `interp_table` does this correctly by construction (it interpolates
   between whatever keys are present); this note exists so a future change
   to that interpolation logic can't silently start treating a missing key
   as 0 instead.

2. reverb_rt60 type 4 (Hall1) has a genuine, well-measured (R^2 = 0.88-0.96
   across 700-900 fit points, reproduced identically across many analysis
   parameter choices -- not a fitting artifact) local DIP: RT60 at raw=32
   (1.8122s) is lower than at raw=16 (2.0773s), a ~13% relative decrease.
   This is real device behaviour, not measurement noise to be smoothed
   away. Per-raw-value interpolation reproduces the actual hardware
   faithfully; fitting a smooth monotonic curve through the table instead
   would trade fidelity for tidiness and make the calibration LESS
   accurate, not more. Every other reverb type has similar small local
   dips (see tests/test_calibration.py's RT60_REL_TOL comment) -- none of
   this should be "corrected."

3. chorus_rate_hz has an analogous genuine local DIP: raw=120 (6.1005 Hz)
   measures LOWER than raw=112 (7.6893 Hz), by far more than plausible
   estimation noise. Unlike a fitting-precision question, this was checked
   directly: splitting raw=120's 12s render into independent first-half and
   second-half windows and re-measuring each gives 6.104 Hz and 6.099 Hz --
   tighter agreement than most other points in the table, i.e. a highly
   reproducible, stable read, not a fragile artifact lock (contrast raw=104
   and raw=127, whose "clean-looking" single dominant peaks actually
   DISAGREE by 0.3-3.2 Hz between their own first and second halves). Per
   the same principle as point 2 above, this is reported as measured, not
   interpolated past or smoothed into line with its neighbours.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly

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
DELAY_FB_TYPE_RE = re.compile(r"delay_t(\d+)_feedback_")
IR_TYPE_RE = re.compile(r"ir_t(\d+)_time_")

# --- Delay/Pan-Dly (reverb types 6-7) echo-time measurement ----------------
#
# calibrate.cpp renders these on the Marimba base patch (percussive, unlike
# the sustained Pipe Organ 3 used above) with the dry signal PRESENT, so the
# render contains both the original strike and its echo(es). The first echo
# is located by cross-correlating the render's envelope against a TEMPLATE
# -- the same patch's own dry attack shape, from delay_dry.wav -- rather
# than by peak-picking the render's own envelope directly.
#
# Peak-picking was tried first and failed: a percussive attack's natural
# decay is not smooth (harmonic beating in this 4-tone patch produces
# incidental bumps of up to ~30%, confirmed by direct inspection), so both
# "look for a local maximum" and "look for a rebound above the recent local
# minimum" produced false positives on that ripple, often well before the
# genuine echo, and were confused entirely once the echo grew TALLER than
# the original attack (which happens routinely here). Cross-correlating
# against the attack's own shape is far more selective: an unrelated ripple
# in the natural decay does not resemble a scaled copy of the attack ONSET,
# so it does not produce a comparably tall cross-correlation peak, while a
# genuine echo -- being an actual delayed, scaled copy of that same attack
# -- does, regardless of its absolute amplitude.
DELAY_HOP = 32                  # 0.5ms/bin @ 64kHz -- resolves the shortest
                                 # measured delay times (tens of ms) cleanly
DELAY_TEMPLATE_S = 0.025        # dry-attack template length for cross-correlation
DELAY_GUARD_S = 0.004           # skip this much of the cross-correlation's own
                                 # near-zero-lag self-similarity before searching
DELAY_MIN_PROM_FRAC = 0.12      # min peak prominence, as a fraction of the
                                 # template's own zero-lag autocorrelation height

# Windows for the feedback (per-repeat decay) measurement, in units of the
# repeat period T. Both windows are placed well past Pan-Dly's fixed,
# feedback-INDEPENDENT first "there and back" pair (2 repeats -- see
# measure_delay_feedback_gain's docstring), and are 2 periods wide so each
# one averages over multiple repeat cycles rather than depending on exactly
# where a single repeat lands.
DELAY_FB_EARLY_MULT = (3.0, 5.0)
DELAY_FB_LATE_MULT = (15.0, 17.0)

IR_TARGET_SR = 48000            # standard rate for a shipped IR file
IR_FADE_MS = 15                 # short linear fade-out to avoid a click at the
                                 # silence-threshold truncation point


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


def delay_fb_type_from_name(path):
    m = DELAY_FB_TYPE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"can't parse reverb type from {path}")
    return int(m.group(1))


def ir_type_from_name(path):
    m = IR_TYPE_RE.search(os.path.basename(path))
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
    # Bind shape to a local and check len(shape) rather than data.ndim: at
    # runtime these are equivalent, but pyright cannot narrow the union type
    # scipy.io.wavfile.read returns (which includes both a 1-D and a 2-D
    # ndarray variant) through a `data.ndim < 2 or data.shape[1] < 2`
    # comparison, so it flags data.shape[1] as indexing past a 1-element
    # tuple even though the `or` short-circuit makes that path unreachable
    # at runtime. len(shape) < 2 narrows shape's type directly, which
    # pyright does understand, so shape[1] below is verifiably safe rather
    # than merely "safe because of evaluation order nobody is guaranteed to
    # preserve on a future edit."
    shape = data.shape
    if len(shape) < 2 or shape[1] < 2:
        return np.zeros(len(data))
    return data[:, 0] - data[:, 1]


def load_stereo(path):
    """Raw stereo samples as float64, shape (n, 2). The delay/pan
    measurements below need L and R separately (to measure pan) as well as
    combined (to locate echoes), unlike the mono/L-R-difference helpers
    above."""
    sr, data = wavfile.read(path)
    assert sr == SR, f"unexpected sample rate {sr} in {path}"
    data = data.astype(np.float64)
    shape = data.shape
    if len(shape) < 2 or shape[1] < 2:
        mono = data if len(shape) == 1 else data[:, 0]
        return np.stack([mono, mono], axis=1)
    return data


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


def measure_chorus_rate_hz(path, oct_frac=0.6, reliability_frac=1.0):
    """Dominant chorus modulation frequency, via autocorrelation of the L-R
    envelope (see module docstring for why L-R, why autocorrelation, and
    the multi-stage octave/artifact correction below). Returns None when the
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

    # Subharmonic-family fix.  Autocorrelation of a periodic envelope peaks at
    # the true period T *and* at every integer multiple 2T, 3T, ...  Those
    # multiples are frequently TALLER than the fundamental (longer windows
    # accumulate more correlation), so simply taking the tallest peak reports
    # f/2, f/3, f/5 ... instead of f.
    #
    # An earlier version only probed HALF the top lag (one octave up), which
    # cannot reach a 3rd- or 5th-order subharmonic.  Measured failures were
    # exactly that: raw 48 reported 0.43 Hz where ~2.2 Hz was expected (~1/5),
    # raw 16 reported 0.52 Hz where ~1.5 Hz was expected (~1/3).
    #
    # Generalise: walk the family T/k for k = 2..MAX_SUBHARMONIC and, among
    # every genuine (comparably tall) peak found across the WHOLE family,
    # report the TALLEST -- not simply the last one examined. An earlier
    # version kept overwriting `chosen` with whichever k it found last
    # (reasoning "a higher k may be the true fundamental"), but that greedily
    # walks past a strong, confident mid-family peak to a much weaker,
    # barely-above-threshold peak at a higher k purely because it was
    # examined later. Measured failures were exactly that: raw 16 had a
    # confident k=3 match (h=0.464) overwritten by a weak k=4 match
    # (h=0.330, reporting 2.02 Hz instead of 1.56 Hz); raw 48 had a
    # confident k=5 match (h=0.422) overwritten by a weak k=7 match (h=0.310,
    # reporting 3.14 Hz instead of 2.14 Hz). The tallest family member is the
    # best-evidenced fundamental.
    #
    # This comparison is deliberately only AMONG the matched family members,
    # not against top_surviving's own height: top_surviving is by hypothesis
    # itself a subharmonic alias (that's the entire reason this search
    # exists), and the paragraph above already explains why the true
    # fundamental is routinely QUIETER than the alias it's found from
    # ("multiples are frequently TALLER than the fundamental") -- requiring
    # a family match to out-height top_surviving would silently disable the
    # fix for exactly the common case it exists to handle.
    MAX_SUBHARMONIC = 8
    family_matches = []
    for k in range(2, MAX_SUBHARMONIC + 1):
        target_lag = top_surviving[0] / k
        if target_lag < lo_lag:
            break
        tol = max(2.0, 0.06 * target_lag)
        best_k = None
        for lag, h, f in surviving:
            if abs(lag - target_lag) <= tol and h >= oct_frac * top_surviving[1]:
                if best_k is None or h > best_k[1]:
                    best_k = (lag, h, f)
        if best_k is not None:
            family_matches.append(best_k)
    chosen = max(family_matches, key=lambda c: c[1]) if family_matches else top_surviving

    # Reliability check: if the UNCONSTRAINED tallest peak anywhere in the
    # search band falls inside a known artifact band and is at least as tall
    # as what we're about to report, the real chorus signal is no stronger
    # than the artifact at this specific raw setting -- don't report a
    # number we don't trust. reliability_frac=1.0 means a straight height
    # contest (no grace margin): an earlier version divided by 0.90, giving
    # the reported candidate an unexplained 10% head start over the artifact
    # before distrusting it. That masked exactly the failure this check
    # exists to catch: raw 80's in-band artifact peak (h=0.4674) is OUTRIGHT
    # taller than its own best surviving candidate (h=0.4533) -- a closer
    # contest than raw 24's already-excluded case -- but the 10% margin let
    # it slide through and get reported as 1.70 Hz anyway. No other raw
    # setting in the measured sweep ever has its global (unconstrained) top
    # peak land in an artifact band at all, so this tightening changes
    # nothing else in the measured table.
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
    slope = np.polyfit(ts, dbs, 1)[0]   # degree-1 fit: only the slope is needed
    if slope >= -0.01:
        return None   # not actually decaying (flat/growing) -- can't extrapolate
    return float(-60.0 / slope)


# --- Delay/Pan-Dly (reverb types 6-7): echo time, feedback, pan ------------

def env_max_combined(stereo, hop):
    """Per-hop peak magnitude of max(|L|, |R|). Used (instead of a mono
    downmix) to locate delay/echo events: Pan-Dly's measured ping-pong
    behaviour (see measure_pan_alternation) puts successive repeats almost
    entirely in ONE channel at a time, and an L+R downmix would attenuate --
    or for a hard-panned repeat, nearly cancel -- whichever repeat lands
    opposite the previous one. Taking the max per channel never misses one."""
    combined = np.maximum(np.abs(stereo[:, 0]), np.abs(stereo[:, 1]))
    return env_max(combined, hop)


def delay_template(dry_path, hop=DELAY_HOP, template_s=DELAY_TEMPLATE_S):
    """The dry attack's own envelope shape (from delay_dry.wav), used as a
    cross-correlation template by measure_first_echo_time_s to locate
    copies of it later in a wet render -- see that function's docstring for
    why matched filtering, not peak-picking."""
    stereo = load_stereo(dry_path)
    env = env_max_combined(stereo, hop)
    n = max(4, int(template_s * SR / hop))
    return env[:n]


def measure_first_echo_time_s(wet_path, template, hop=DELAY_HOP,
                               guard_s=DELAY_GUARD_S,
                               min_prom_frac=DELAY_MIN_PROM_FRAC):
    """Locate the first echo of the dry attack in a Delay/Pan-Dly render
    (see the DELAY_HOP block's module-level comment for the full rationale).
    Cross-correlates the render's envelope against `template`; the dry
    attack itself produces a strong self-match at lag 0 (returned as
    `peak0` below), and the delay's first repeat produces a second, weaker
    match at lag = delay time.

    Returns None when no echo is confidently located -- observed at raw=0
    on both types, where the delay time is short enough that any echo is
    indistinguishable from the attack's own onset. Callers must not
    fabricate a number in that case."""
    stereo = load_stereo(wet_path)
    env = env_max_combined(stereo, hop)
    tpl = template / (np.linalg.norm(template) + 1e-9)
    n = len(tpl)
    if len(env) <= n:
        return None
    xc = np.correlate(env, tpl, mode="full")
    zero_lag = n - 1
    xc_pos = xc[zero_lag:]              # lag >= 0
    if len(xc_pos) < 3:
        return None
    peak0 = xc_pos[0]                    # the dry attack's own self-match
    if peak0 <= 0:
        return None
    guard_hops = max(1, int(guard_s * SR / hop))
    if guard_hops >= len(xc_pos):
        return None
    search = xc_pos[guard_hops:]
    pk, _ = find_peaks(search, prominence=peak0 * min_prom_frac)
    if len(pk) == 0:
        return None
    k = int(pk[0]) + guard_hops
    k_ref = parabolic_peak(xc_pos, k)
    return float(k_ref * hop / SR)


def _rms_window(stereo, t0, t1):
    """RMS over [t0, t1) seconds, zero-padded if the render is shorter than
    t1. A low-feedback render legitimately decays to true silence -- and
    can be truncated early by render_note()'s own tail early-exit -- well
    before a late window like this; that IS "no signal left here", not a
    measurement failure, so it must contribute true silence to the RMS, not
    be skipped (matching measure_tail_extra_energy's zero-padding
    approach above)."""
    n_total = int(round((t1 - t0) * SR))
    if n_total <= 0:
        return 0.0
    i0 = int(round(t0 * SR))
    i1 = i0 + n_total
    avail = stereo[max(0, i0):max(0, i1)]
    if len(avail) == 0:
        return 0.0
    acc = np.sum(avail[:, 0] ** 2) + np.sum(avail[:, 1] ** 2)
    return float(np.sqrt(acc / (2 * n_total)))


def measure_delay_feedback_gain(wet_path, repeat_period_s,
                                 early_mult=DELAY_FB_EARLY_MULT,
                                 late_mult=DELAY_FB_LATE_MULT):
    """Per-repeat gain (0-1) of a Delay/Pan-Dly's feedback: the ratio of
    echo-train energy far out (15-17 repeat periods after the strike) vs.
    only a few repeats out (3-5 periods), converted to an equivalent
    per-repeat gain g via g**(dt/T) = ratio, where T is the repeat period
    and dt is the gap between the two windows' centres (12T).

    Why a windowed energy ratio, not per-echo peak-picking: peak-picking
    each individual repeat (tried first, and used successfully by
    measure_first_echo_time_s to locate the FIRST echo) does not generalise
    to measuring feedback decay here, because Pan-Dly's ping-pong routing
    gives it one full "there and back" pair (2 repeats) at a FIXED,
    feedback-INDEPENDENT strength even at feedback=0 -- confirmed
    empirically: those two repeats' heights are identical to 3-4
    significant figures across the entire 0-127 feedback sweep -- before
    feedback-scaled decay takes over from the 3rd repeat onward. Treating
    those two fixed repeats as part of a single geometric decay biases a
    fitted decay rate. A ratio between two WIDE (2-period) windows, both
    placed well past that fixed pair, sidesteps needing to classify which
    individual repeat is "real" vs. "free" at all, and each window averages
    over multiple repeat cycles, which is robust to individual echoes
    landing at different absolute levels depending on which channel a
    ping-pong repeat happens to be panned to.

    Returns None if repeat_period_s is missing/non-positive (the raw=64
    delay-time point this sweep depends on was itself unmeasurable) --
    callers must not fabricate a period to divide by."""
    if repeat_period_s is None or repeat_period_s <= 0:
        return None
    stereo = load_stereo(wet_path)
    T = repeat_period_s
    e0, e1 = early_mult
    l0, l1 = late_mult
    early = _rms_window(stereo, e0 * T, e1 * T)
    late = _rms_window(stereo, l0 * T, l1 * T)
    if early <= 1e-9:
        return None
    ratio = late / early
    if ratio <= 0:
        return 0.0
    dt = ((l0 + l1) / 2.0 - (e0 + e1) / 2.0) * T
    if dt <= 0:
        return None
    g = ratio ** (T / dt)
    return float(np.clip(g, 0.0, 0.999))


def measure_pan_alternation(wet_path, repeat_period_s, n_repeats=8):
    """How strongly a Delay/Pan-Dly's successive echoes alternate between
    channels: the range (max - min) of each repeat's L fraction (L_peak /
    (L_peak + R_peak)), sampled at the ALREADY-measured repeat spacing (n *
    repeat_period_s for n=1..n_repeats). ~0 means every repeat is centred
    the same way (a genuinely mono/centred delay, as measured for type 6);
    a large range (measured ~0.5 for type 7) means repeats swing between
    mostly-L and mostly-R -- a real ping-pong.

    Repeat times are taken from the already-measured repeat period rather
    than re-detecting each one independently: this is meant to be run on a
    HIGH-feedback render (many repeats, needed to see an alternation
    pattern at all), where individual repeats increasingly overlap/smear
    together and independent per-repeat cross-correlation is unreliable --
    but the repeat SPACING itself is already known precisely from
    measure_first_echo_time_s, so re-deriving it isn't necessary."""
    if repeat_period_s is None or repeat_period_s <= 0:
        return None
    stereo = load_stereo(wet_path)
    win = max(1, int(0.006 * SR))   # +/-6ms peak window around each repeat
    fracs = []
    for n in range(1, n_repeats + 1):
        centre = int(n * repeat_period_s * SR)
        i0, i1 = centre - win, centre + win
        if i0 < 0 or i1 > len(stereo):
            break
        seg = stereo[i0:i1]
        l = np.abs(seg[:, 0]).max()
        r = np.abs(seg[:, 1]).max()
        tot = l + r
        if tot < 1e-6:
            continue
        fracs.append(l / tot)
    if len(fracs) < 2:
        return None
    return float(max(fracs) - min(fracs))


# --- Reverb impulse-response capture (types 0-5) ---------------------------

def write_ir_wav(src_path, dst_path, target_sr=IR_TARGET_SR, fade_ms=IR_FADE_MS):
    """Resample a captured pure-wet reverb render (calibrate.cpp's
    pure_wet_reverb sweep) to `target_sr` and write it as the final IR file
    DecentSampler's <effect type="convolution"> consumes, with a short
    linear fade-out at the very end: the render's own silence-threshold
    truncation (GridSpec.silence_db=-60 in calibrate.cpp) is an abrupt stop
    once the tail drops below that floor, not a natural decay to nothing,
    so a few milliseconds of fade avoids an audible click there.

    Returns the written file's duration in seconds, or None if the source
    render was too short to be a usable IR (shouldn't happen in practice --
    every captured type/raw combination produced a real decaying tail --
    but this is the same "don't fabricate, report the gap" policy as every
    other measurement in this file)."""
    sr, data = wavfile.read(src_path)
    assert sr == SR, f"unexpected sample rate {sr} in {src_path}"
    data = data.astype(np.float64)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    if len(data) < 8:
        return None
    resampled = resample_poly(data, target_sr, sr, axis=0)
    fade_n = min(len(resampled), int(fade_ms / 1000.0 * target_sr))
    if fade_n > 1:
        fade = np.linspace(1.0, 0.0, fade_n)[:, None]
        resampled[-fade_n:] *= fade
    dst_dir = os.path.dirname(dst_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    out16 = np.clip(np.round(resampled), -32768, 32767).astype(np.int16)
    wavfile.write(dst_path, target_sr, out16)
    return len(resampled) / float(target_sr)


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
        "delay_time_s": {},
        "delay_feedback": {},
        "delay_pan_alternation": {},
        "reverb_ir": {},
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

    # --- Delay/Pan-Dly (types 6-7): echo time --------------------------
    # See the DELAY_HOP block's module-level comment for the measurement
    # rationale (cross-correlation against the dry attack's own shape,
    # from delay_dry.wav).
    delay_dry_path = os.path.join(calib_dir, "delay_dry.wav")
    if os.path.exists(delay_dry_path):
        tpl = delay_template(delay_dry_path)
        for dtype in (6, 7):
            pattern = os.path.join(calib_dir, f"reverb_t{dtype}_time_*.wav")
            items = collect(pattern)
            table = {}
            for raw, f in items:
                t_s = measure_first_echo_time_s(f, tpl)
                if t_s is None:
                    print(f"WARNING: delay time unmeasurable for type={dtype} "
                          f"raw={raw} (no echo confidently distinguishable from "
                          f"the attack itself at this setting) -- recording "
                          f"null, not a fabricated number", file=sys.stderr)
                    table[str(raw)] = None
                else:
                    table[str(raw)] = round(t_s, 4)
            result["delay_time_s"][str(dtype)] = table
    else:
        print(f"WARNING: missing {delay_dry_path} -- skipping delay_time_s, "
              f"delay_feedback, delay_pan_alternation entirely", file=sys.stderr)

    # --- Delay/Pan-Dly (types 6-7): feedback (per-repeat gain) + pan ----
    # calibrate.cpp fixes time=64 for this sweep -- look up the repeat
    # period T for that exact raw value from delay_time_s (just measured
    # above) rather than re-deriving it, so the two measurements stay
    # self-consistent with whatever this run's own delay_time_s reported.
    for dtype in (6, 7):
        time_table = result["delay_time_s"].get(str(dtype), {})
        T = time_table.get("64")
        fb_table = {}
        items = collect(os.path.join(calib_dir, f"delay_t{dtype}_feedback_*.wav"))
        if T is None:
            if items:
                print(f"WARNING: delay_time_s[{dtype}]['64'] unmeasurable -- "
                      f"can't convert type={dtype}'s feedback sweep into a "
                      f"per-repeat gain (needs the repeat period); recording "
                      f"nulls, not fabricated numbers", file=sys.stderr)
            for raw, _ in items:
                fb_table[str(raw)] = None
        else:
            for raw, f in items:
                g = measure_delay_feedback_gain(f, T)
                if g is None:
                    print(f"WARNING: delay feedback gain unmeasurable for "
                          f"type={dtype} raw={raw} -- recording null, not a "
                          f"fabricated number", file=sys.stderr)
                    fb_table[str(raw)] = None
                else:
                    fb_table[str(raw)] = round(g, 4)
        result["delay_feedback"][str(dtype)] = fb_table

        # Pan alternation: measured on the SAME sweep's highest-feedback
        # render (most repeats -> most reliable), reusing T from above.
        fb127_path = os.path.join(calib_dir, f"delay_t{dtype}_feedback_127.wav")
        if T is not None and os.path.exists(fb127_path):
            alt = measure_pan_alternation(fb127_path, T)
            if alt is None:
                print(f"WARNING: pan alternation unmeasurable for type={dtype} "
                      f"-- recording null, not a fabricated number",
                      file=sys.stderr)
            result["delay_pan_alternation"][str(dtype)] = (
                round(alt, 4) if alt is not None else None)
        else:
            result["delay_pan_alternation"][str(dtype)] = None

    # --- Reverb impulse responses (types 0-5) ---------------------------
    # Resample each pure-wet capture to calib/ir/*.wav (see write_ir_wav),
    # record its path in reverb_ir, and cross-check its own RT60 (measured
    # directly on the IR, which -- being pure-wet -- IS essentially just a
    # decay tail) against the reverb_rt60 table above, computed completely
    # independently (dry+wet mixed, sustained organ source, hold=4s vs.
    # this capture's hold=0.1s). Agreement between the two is a genuine
    # cross-validation, not a tautology.
    ir_items_by_type = {}
    for rtype in range(6):
        items = collect(os.path.join(calib_dir, f"ir_t{rtype}_time_*.wav"))
        if items:
            ir_items_by_type[rtype] = items

    if ir_items_by_type:
        print("--- reverb IR bank: RT60 cross-check (independent measurement "
              "vs. the IR capture itself) ---", file=sys.stderr)
    for rtype, items in ir_items_by_type.items():
        table = {}
        for raw, src in items:
            dst = os.path.join(calib_dir, "ir", f"reverb_t{rtype}_time_{raw:03d}.wav")
            dur = write_ir_wav(src, dst)
            if dur is None:
                print(f"WARNING: IR unusable for type={rtype} raw={raw} "
                      f"(captured render too short) -- omitting, not "
                      f"fabricating a placeholder file", file=sys.stderr)
                table[str(raw)] = None
                continue
            table[str(raw)] = f"ir/reverb_t{rtype}_time_{raw:03d}.wav"

            ir_rt60 = measure_rt60(load_mono(src), hold_seconds=0.1)
            ref_rt60 = result["reverb_rt60"].get(str(rtype), {}).get(str(raw))
            if ir_rt60 is not None and ref_rt60 is not None:
                rel = abs(ir_rt60 - ref_rt60) / max(ref_rt60, 0.05)
                flag = "" if rel <= 0.25 else "  <-- DISAGREES (>25%)"
                print(f"  type={rtype} raw={raw:3d}  ir_rt60={ir_rt60:6.3f}s  "
                      f"reverb_rt60={ref_rt60:6.3f}s  dur={dur:5.2f}s{flag}",
                      file=sys.stderr)
            else:
                print(f"  type={rtype} raw={raw:3d}  ir_rt60={ir_rt60}  "
                      f"reverb_rt60={ref_rt60}  dur={dur:5.2f}s (no cross-check)",
                      file=sys.stderr)
        result["reverb_ir"][str(rtype)] = table

    out_path = os.path.join(calib_dir, "calibration.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
