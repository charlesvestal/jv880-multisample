import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import postprocess as pp  # noqa: E402

SR = pp.SR_OUT
HOLD = int(3.5 * SR)


def make_stereo(mono):
    return np.stack([mono, mono], axis=1)


def test_steady_sine_is_sustaining_and_loops():
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    mono[HOLD:] *= np.exp(-6.0 * (t[HOLD:] - t[HOLD]))
    x = make_stereo(mono)

    kind, _ = pp.classify(x, HOLD)
    assert kind == "sustaining"

    loop = pp.find_loop(x, SR, HOLD)
    assert loop and loop["enabled"]

    a = x[loop["start"], 0]
    b = x[loop["end"], 0]
    assert abs(a - b) < 0.01 * np.abs(x).max(), "loop endpoints discontinuous"


def test_decaying_burst_gets_no_loop():
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.8 * np.sin(2 * np.pi * 440.0 * t) * np.exp(-6.0 * t)
    x = make_stereo(mono)
    kind, _ = pp.classify(x, HOLD)
    assert kind == "decaying"


def test_crossfade_bounded():
    # Fixed numeric expectation for this exact (deterministic) synthetic
    # input, not a recomputation of the crossfade formula under test -- a
    # regression to e.g. `length // 3` would still satisfy a
    # self-referential "crossfade <= min(...)" check computed the same way
    # the implementation computes it.
    #
    # Values below are for the crossfade-region-correlation search (see
    # find_loop): a pure 220 Hz tone is periodic at every lag, so the
    # "prefer longer" tiebreak walks all the way out near the end of the
    # reachable region, not just to a small multiple of the pitch period.
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    x = make_stereo(mono)
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    assert loop["start"] == 48000
    assert loop["end"] == 165600
    assert loop["crossfade"] == 12000
    # This specific case exercises the loop_start//4 branch of the min():
    # loop_start//4 (12000) is smaller than MAX_XFADE (24000) and length//4
    # (29400), so loop_start//4 is the binding constraint here -- matching
    # the listening-test follow-up's own example (loop_start at 1s, a
    # crossfade landing around 250ms = 25% of loop_start).
    assert loop["crossfade"] == 48000 // 4
    # Independent sanity check against the documented bound, using the
    # loop's own reported values (not a copy of the implementation).
    length = loop["end"] - loop["start"]
    assert loop["crossfade"] <= min(pp.MAX_XFADE, loop["start"] // 4, length // 4)


def test_crossfade_bounded_by_length_for_short_loop():
    """test_crossfade_bounded above only exercises the loop_start//4
    branch of the crossfade formula (the common case, given the "prefer
    longer" bias). This exercises length//4 -- the bound that matters for
    a short loop, and specifically what protects against the DecentSampler
    silent-failure mode the raised MAX_XFADE ceiling could otherwise risk
    for short loops.

    Splices an exact repeat at the SHORTEST reachable lag (l_min ==
    xf_score, the minimum loop length find_loop's crossfade-region search
    will consider), forcing the search to accept a short loop. For that
    length, length//4 is far smaller than both MAX_XFADE and
    loop_start//4, so it must be the binding constraint.
    """
    from scipy.signal import butter, sosfiltfilt

    n_total = int(6.0 * SR)
    rng = np.random.default_rng(55)
    sos = butter(4, [900, 6000], btype="bandpass", fs=SR, output="sos")
    mono = sosfiltfilt(sos, rng.standard_normal(n_total))
    mono /= np.max(np.abs(mono)) + 1e-9
    mono *= 0.5

    start_lo = int(1.0 * SR)
    xf_score = min(pp.MAX_XFADE, start_lo // 4)
    l_target = xf_score  # the shortest loop length find_loop will consider
    idx = start_lo + l_target
    mono[idx - xf_score:idx] = mono[start_lo - xf_score:start_lo]

    x = make_stereo(mono)
    kind, _ = pp.classify(x, HOLD)
    assert kind == "sustaining"

    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    length = loop["end"] - loop["start"]
    assert length == l_target, (
        "expected the search to accept the shortest reachable loop -- if "
        "it found something longer instead, this test isn't exercising "
        "the length//4 branch"
    )
    assert loop["crossfade"] == length // 4, (
        f"expected length//4 ({length // 4}) to be the binding constraint "
        f"for this short loop, got crossfade={loop['crossfade']}"
    )
    assert loop["crossfade"] < start_lo // 4
    assert loop["crossfade"] < pp.MAX_XFADE
    # Independent sanity check against the documented bound.
    assert loop["crossfade"] <= min(pp.MAX_XFADE, loop["start"] // 4, length // 4)


def test_find_loop_prefers_longer_loop_when_endpoints_are_comparable():
    """A clean periodic signal has a near-perfect endpoint match at nearly
    every reachable lag -- the tiebreak must pull toward a long candidate
    (less perceptible repetition on pads/strings), not just whichever lag
    happens to have the single smallest endpoint discontinuity.
    """
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    x = make_stereo(mono)
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    length = loop["end"] - loop["start"]
    approx_period = SR / 220.0
    shortest_candidate = 8 * approx_period
    assert length > shortest_candidate * 2, (
        f"loop length {length} is too close to the shortest 8-period "
        f"candidate ({shortest_candidate:.0f}); longer-loop tiebreak does "
        "not appear to be working"
    )


def test_high_frequency_loop_beyond_old_period_multiple_ceiling():
    """Regression test for the exact pilot-render bug: the OLD find_loop
    constrained candidates to end = loop_start + period * n for n in
    8..60, so for a ~1200 Hz note (period ~40 frames) the search never
    reached beyond ~60*40 = 2400 frames (50ms). Real material (e.g. the
    pilot's "Wave Bells") has its actual repeat far beyond that -- e.g.
    D#6 at 1.9s, corr 0.996 -- so those zones classified sustaining but
    got no loop at all (pilot: 7,112 sustaining zones, only 1,284 looped,
    3.8%), silently looking like "this material isn't loopable" when it
    wasn't.

    This builds a textured, high-frequency, NON-tonal signal (bandpass
    noise -- no coherent short-period structure at all, standing in for
    inharmonic bell-like material) with an exact copy spliced in around
    L_target=1.9s. A single pure tone can't demonstrate this bug (every
    period multiple of a pure tone is trivially a perfect match, which is
    exactly why the bug passed unnoticed on this file's own 220 Hz test
    signal) -- the point is that old-style short-lag candidates must
    genuinely NOT correlate, only the engineered long lag should.

    The splice spans a wide radius straddling both loop_start and the
    target end point (not just forward from them), so the copied region
    covers BOTH what the current find_loop scores (the xf_score-frame
    window ENDING at each point -- the crossfade region it actually
    blends) and what the old-algorithm reconstruction below scores (a
    small window STARTING at each point, as the pre-length-search
    original code did).
    """
    from scipy.signal import butter, sosfiltfilt

    n_total = int(6.0 * SR)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n_total)
    sos = butter(4, [900, 6000], btype="bandpass", fs=SR, output="sos")
    texture = sosfiltfilt(sos, noise)
    texture /= np.max(np.abs(texture)) + 1e-9
    mono = 0.5 * texture

    start_lo = int(1.0 * SR)
    l_target = int(1.9 * SR)
    idx = start_lo + l_target
    xf_score = min(pp.MAX_XFADE, start_lo // 4)
    copy_radius = xf_score + 4096  # comfortably covers both window shapes
    mono[idx - copy_radius: idx + copy_radius] = \
        mono[start_lo - copy_radius: start_lo + copy_radius]
    x = make_stereo(mono)

    kind, ratio = pp.classify(x, HOLD)
    assert kind == "sustaining", f"test signal must classify sustaining (ratio={ratio})"

    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None, "the fixed length-based search must find the spliced-in repeat"
    assert loop["score"] >= 0.90
    assert loop["end"] - loop["start"] == l_target, (
        "expected the search to lock onto the exact engineered repeat point"
    )

    a = x[loop["start"], 0]
    b = x[loop["end"], 0]
    assert abs(a - b) < 0.01 * np.abs(x).max(), "loop endpoints discontinuous"

    # Direct, self-contained proof that the OLD period-multiple-constrained
    # search (n_per in 8..60, window = period*2) could never have found
    # this: reconstruct its exact reachable-candidate scoring and confirm
    # every one of the ~53 candidates falls short of the 0.90 gate --
    # matching the pilot's own diagnosis ("best achievable correlation ...
    # mostly 0.2-0.7").
    approx_period = SR / 1200.0
    old_ceiling = int(60 * approx_period)
    assert loop["end"] - loop["start"] > old_ceiling * 5, (
        "this test is supposed to exercise a repeat far beyond the old "
        "60-period ceiling"
    )

    mono64 = mono.astype(np.float64)
    period = round(approx_period)
    win = period * 2
    old_scores = []
    for n_per in range(8, 61):
        end = start_lo + period * n_per
        a_win = mono64[start_lo:start_lo + win]
        b_win = mono64[end:end + win]
        denom = (np.linalg.norm(a_win) * np.linalg.norm(b_win)) + 1e-12
        old_scores.append(float(np.dot(a_win, b_win) / denom))
    assert max(old_scores) < 0.90, (
        f"expected every old-algorithm candidate (n_per 8..60) to score "
        f"below the 0.90 gate, got max {max(old_scores):.3f} -- if this "
        "signal doesn't reproduce the old failure mode, the regression "
        "test isn't testing what it claims to"
    )


def test_stereo_divergent_crossfade_region_rejected_even_when_mono_cancels():
    """Regression test for a bug found in pilot validation: scoring on
    x.mean(axis=1) -- the MONO mix -- can hide a real per-channel problem,
    because L and R divergence can be near-exact opposites that CANCEL in
    the average. This was originally found (and fixed) for the old
    single-sample endpoint metric; the coordinator's later listening-test
    follow-up replaced that metric with crossfade-REGION correlation, so
    this proves the per-channel principle -- explicitly kept across that
    rewrite -- still holds for the new metric too: cancellation across an
    entire crossfade window, not just one sample, must still not fool the
    real per-channel-aware search.

    Builds a clean, stereo-identical sine (per-channel-identical
    candidates exist naturally almost everywhere), finds where the
    unperturbed search naturally lands, then overwrites the ENTIRE
    crossfade-region window ending at that exact point with a strong,
    opposite-signed perturbation on L vs R (so the mono average across
    that whole window is untouched -- the errors cancel -- while each
    channel individually diverges sharply from the reference). A
    reconstruction of mono-mix scoring (same crossfade-region metric,
    computed on the mono signal instead of per channel) is shown to still
    pick that exact poisoned lag; the real, per-channel-aware find_loop
    must not, and whatever it does pick must have a genuinely good
    per-channel score.
    """
    t = np.arange(int(6.0 * SR)) / SR
    base = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    peak = 0.5
    start_lo = int(1.0 * SR)
    xf_score = min(pp.MAX_XFADE, start_lo // 4)

    # Establish where the (correct) search naturally lands on the clean,
    # stereo-identical signal, so we know exactly which crossfade window
    # to poison.
    baseline = pp.find_loop(make_stereo(base), SR, HOLD)
    assert baseline is not None
    l_bad = baseline["end"] - baseline["start"]
    idx = start_lo + l_bad

    # A strong (2x peak), opposite-signed perturbation across the WHOLE
    # crossfade-region window ending at idx: L gets +divergence, R gets
    # -divergence, so (L+R)/2 over that window is untouched (still
    # exactly `base`), while each channel individually is dominated by
    # the divergence rather than the reference tone.
    rng = np.random.default_rng(3)
    divergence = rng.standard_normal(xf_score)
    divergence /= np.max(np.abs(divergence)) + 1e-9
    divergence *= 2.0 * peak

    left = base.copy()
    right = base.copy()
    left[idx - xf_score:idx] += divergence
    right[idx - xf_score:idx] -= divergence
    x = np.stack([left, right], axis=1)

    kind, ratio = pp.classify(x, HOLD)
    assert kind == "sustaining", f"test signal must classify sustaining (ratio={ratio})"

    # Self-contained proof mono-mix scoring would still pick this exact
    # bad lag: reconstruct find_loop's crossfade-region metric on the mono
    # mix instead of per-channel.
    def find_loop_mono_reconstruction(stereo, sr, hold_frames):
        m = stereo.mean(axis=1)
        s_lo = int(1.0 * sr)
        r_hi = min(hold_frames, len(m)) - int(0.05 * sr)
        xf = min(pp.MAX_XFADE, s_lo // 4)
        ext = m[s_lo - xf:r_hi].astype(np.float64)
        ref = ext[:xf]
        ref_norm = float(np.linalg.norm(ref))
        numerator = pp.fftconvolve(ext, ref[::-1], mode="valid")
        csq = np.cumsum(np.concatenate(([0.0], ext ** 2)))
        win_energy = csq[xf:] - csq[:-xf]
        win_norm = np.sqrt(np.maximum(win_energy, 0.0))
        score = numerator / (ref_norm * win_norm + 1e-12)
        l_min = max(int(pp.MIN_LOOP_SECONDS * sr), xf)
        l_max = len(ext) - xf
        lags = np.arange(len(score))
        mask = (lags >= l_min) & (lags <= l_max) & np.isfinite(score)
        cand_lags, cand_scores = lags[mask], score[mask]
        best = float(np.max(cand_scores))
        near = cand_lags[cand_scores >= max(best - pp.SCORE_TOL, pp.MIN_CROSSFADE_SCORE)]
        chosen = int(near[np.argmax(near)])
        return s_lo, s_lo + chosen

    mono_start, mono_end = find_loop_mono_reconstruction(x, SR, HOLD)
    assert mono_end - mono_start == l_bad, (
        "expected the mono-mix reconstruction to pick the poisoned lag "
        "(that's the bug being reproduced) -- if it picked something "
        "else, this test isn't exercising the cancellation case"
    )

    # The actual behavior under test: per-channel scoring must NOT pick
    # l_bad, and whatever it picks must have a genuinely good per-channel
    # crossfade-region score.
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    chosen_length = loop["end"] - loop["start"]
    assert chosen_length != l_bad, (
        "find_loop picked the same poisoned lag the mono-mix scoring "
        "did -- per-channel matching isn't actually being applied"
    )
    assert loop["score"] > 0.9, (
        f"find_loop's chosen candidate scored only {loop['score']} -- "
        "should have found one of the many genuinely clean candidates "
        "on this otherwise-clean signal"
    )


def test_find_loop_declines_when_best_candidate_is_still_bad():
    """The metric changed (crossfade-region correlation instead of a
    single-sample endpoint delta), and the decision changed too ("prefer
    to loop" -- see find_loop's docstring), but there's still a floor:
    when the crossfade region is genuinely uncorrelated everywhere in the
    reachable range, find_loop must still decline rather than fabricate a
    loop from nothing.

    A linear chirp (steadily rising frequency) is periodic nowhere. Over
    the wide (xf_score, ~250ms) window this now scores with, a chirp's
    instantaneous frequency has drifted enough that even the best
    achievable crossfade-region correlation is close to zero -- nothing
    here would blend acceptably at any candidate length.
    """
    t = np.arange(int(6.0 * SR)) / SR
    f0, f1 = 2000.0, 2600.0
    k = (f1 - f0) / t[-1]
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    mono = 0.4 * np.sin(phase)
    x = make_stereo(mono)

    kind, _ = pp.classify(x, HOLD)
    assert kind == "sustaining"

    # Confirm this signal actually exercises the floor: with it disabled,
    # find_loop still returns SOME candidate, but its score is close to
    # zero -- genuinely uncorrelated, not just "imperfect".
    original_floor = pp.MIN_CROSSFADE_SCORE
    try:
        pp.MIN_CROSSFADE_SCORE = -1.0
        uncapped = pp.find_loop(x, SR, HOLD)
    finally:
        pp.MIN_CROSSFADE_SCORE = original_floor

    assert uncapped is not None, (
        "expected find_loop to still find *a* best-available candidate "
        "with the floor disabled -- if nothing is found at all, this "
        "test isn't exercising the floor specifically"
    )
    assert uncapped["score"] < original_floor, (
        f"the uncapped candidate's score ({uncapped['score']}) should be "
        f"clearly below MIN_CROSSFADE_SCORE ({original_floor}) -- "
        "otherwise this isn't the 'genuinely uncorrelated' case"
    )

    # With the real floor active, find_loop must decline entirely.
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is None, (
        "find_loop returned a loop whose only available candidates are "
        "genuinely uncorrelated -- it should decline instead"
    )


def test_prefer_longer_tolerance_never_drops_below_crossfade_floor():
    """Analogous to a bug found (and fixed) in the previous, dv-based
    version of this function: the "prefer longer among comparable
    candidates" tolerance window (SCORE_TOL below the best score) can
    itself dip below MIN_CROSSFADE_SCORE even when best_score clears it --
    e.g. best_score at 0.38 (comfortably above a 0.35 floor) gives an
    unclamped tolerance floor of 0.38-0.05=0.33, which would let a LONGER
    candidate at 0.34 (itself below MIN_CROSSFADE_SCORE) through. Found
    while writing this test, not pilot validation, but it's the exact
    same shape of bug and needed the exact same fix: clamp the tolerance
    window to never go below the floor.

    Two candidates are built by blending a fixed reference texture with
    independent noise at precisely calibrated ratios (found via binary
    search) to hit exact target crossfade-region correlations: a SHORT
    one at ~0.38 (the best score) and a LONGER one at ~0.34 (below the
    0.35 floor, but within the unclamped 0.05 tolerance of 0.38). Without
    clamping, "prefer longer" picks the far, sub-floor candidate; with
    the clamp, it must pick the short, floor-clearing one instead.
    """
    from scipy.signal import butter, sosfiltfilt

    n_total = int(6.0 * SR)
    rng = np.random.default_rng(11)
    sos = butter(4, [900, 6000], btype="bandpass", fs=SR, output="sos")
    ref_texture = sosfiltfilt(sos, rng.standard_normal(n_total))
    ref_texture /= np.max(np.abs(ref_texture)) + 1e-9
    peak = 0.5
    mono = ref_texture * peak

    start_lo = int(1.0 * SR)
    xf_score = min(pp.MAX_XFADE, start_lo // 4)
    ref_win = mono[start_lo - xf_score:start_lo]

    def make_candidate(alpha, seed):
        r = np.random.default_rng(int(seed))
        noise = sosfiltfilt(sos, r.standard_normal(xf_score))
        noise /= np.max(np.abs(noise)) + 1e-9
        return alpha * ref_win + np.sqrt(max(0.0, 1 - alpha ** 2)) * noise * peak

    def measured_corr(cand):
        denom = (np.linalg.norm(ref_win) * np.linalg.norm(cand)) + 1e-12
        return float(np.dot(ref_win, cand) / denom)

    def find_alpha(target, seed, lo=0.0, hi=1.0, iters=40):
        for _ in range(iters):
            mid = (lo + hi) / 2
            if measured_corr(make_candidate(mid, seed)) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # best_score - SCORE_TOL = 0.38 - 0.05 = 0.33, which sits BELOW
    # MIN_CROSSFADE_SCORE (0.35) -- so unclamped, a candidate scoring
    # 0.34 (itself below the floor) would still fall inside the
    # tolerance window.
    a_short = find_alpha(0.38, seed=101)
    a_long = find_alpha(0.34, seed=202)
    l_short = int(1.0 * SR)
    l_long = int(2.0 * SR)
    idx_short = start_lo + l_short
    idx_long = start_lo + l_long
    mono[idx_short - xf_score:idx_short] = make_candidate(a_short, 101)
    mono[idx_long - xf_score:idx_long] = make_candidate(a_long, 202)

    x = make_stereo(mono)
    kind, _ = pp.classify(x, HOLD)
    assert kind == "sustaining"

    # Self-contained proof of the setup: reconstruct find_loop's
    # crossfade-region score array (identical computation, exposed here
    # for direct inspection) and confirm the UNCLAMPED tolerance window
    # would pick the far, sub-floor candidate.
    def find_loop_scores(stereo, sr, hold_frames):
        start = int(1.0 * sr)
        region_hi = min(hold_frames, len(stereo)) - int(0.05 * sr)
        xf = min(pp.MAX_XFADE, start // 4)
        ext = stereo[start - xf:region_hi].astype(np.float64)
        combined = None
        for c in range(ext.shape[1]):
            chan = ext[:, c]
            ref = chan[:xf]
            ref_norm = float(np.linalg.norm(ref))
            numerator = pp.fftconvolve(chan, ref[::-1], mode="valid")
            csq = np.cumsum(np.concatenate(([0.0], chan ** 2)))
            we = csq[xf:] - csq[:-xf]
            wn = np.sqrt(np.maximum(we, 0.0))
            score = numerator / (ref_norm * wn + 1e-12)
            combined = score if combined is None else np.minimum(combined, score)
        l_min = max(int(pp.MIN_LOOP_SECONDS * sr), xf)
        l_max = len(ext) - xf
        lags = np.arange(len(combined))
        mask = (lags >= l_min) & (lags <= l_max) & np.isfinite(combined)
        return lags[mask], combined[mask]

    cand_lags, cand_scores = find_loop_scores(x, SR, HOLD)
    best_score = float(np.max(cand_scores))
    assert best_score - pp.SCORE_TOL < pp.MIN_CROSSFADE_SCORE <= best_score, (
        "setup check: expected the unclamped tolerance floor to fall "
        "below MIN_CROSSFADE_SCORE while best_score itself clears it -- "
        "otherwise this isn't exercising the gap this test targets"
    )
    unclamped_near = cand_lags[cand_scores >= (best_score - pp.SCORE_TOL)]
    unclamped_pick = int(unclamped_near[np.argmax(unclamped_near)])
    assert unclamped_pick == l_long, (
        "expected the unclamped tiebreak to pick the far, sub-floor "
        "candidate -- if it picked something else, this test isn't "
        "exercising the tolerance-window gap"
    )

    # With the real (clamped) logic, it must pick the short, genuinely
    # acceptable candidate instead.
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    assert loop["end"] - loop["start"] == l_short, (
        "expected the clamped tiebreak to pick the short candidate "
        "within the floor, not the longer one that falls below it"
    )
    assert loop["score"] >= pp.MIN_CROSSFADE_SCORE


def test_metric_prefers_good_crossfade_region_over_good_endpoint_sample():
    """The whole point of the metric rewrite: a listening test on the real
    pilot found two candidates with nearly identical raw single-sample
    endpoint deltas (JP-8 Strings 40.9%, ChuChu Vox 42.0%) that got
    OPPOSITE verdicts -- JP-8 Strings sounded fine (dense detuned content
    masks a seam), ChuChu Vox/Whistle pulsed audibly (an exposed tone does
    not). The single-sample metric can't tell those apart; the crossfade-
    region metric must.

    Builds two candidates with the SAME large single-sample endpoint
    delta (~112% of peak -- deliberately much worse than either real
    example, to make the point starkly): candidate A's crossfade REGION
    (the xf-frame window before it) is an exact copy of the reference,
    so DecentSampler's actual blend would sound seamless despite the
    wild single-sample jump right at the boundary; candidate B's
    crossfade region is unrelated noise, so the blend would sound
    obviously wrong, even though its single boundary sample was tuned to
    match A's delta exactly. find_loop must pick A over B.
    """
    n_total = int(6.0 * SR)
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(21)
    sos = butter(4, [900, 6000], btype="bandpass", fs=SR, output="sos")
    base = sosfiltfilt(sos, rng.standard_normal(n_total))
    base /= np.max(np.abs(base)) + 1e-9
    peak = 0.5
    mono = base * peak

    start_lo = int(1.0 * SR)
    xf_score = min(pp.MAX_XFADE, start_lo // 4)
    ref_win = mono[start_lo - xf_score:start_lo].copy()

    # Candidate A ("JP-8 Strings"): crossfade region is an exact copy of
    # the reference (perfect blend), but the single SAMPLE right at the
    # loop-end point is a wild outlier -- a single-sample metric would
    # flag this as terrible.
    l_good_region = int(1.0 * SR)
    idx_good = start_lo + l_good_region
    mono[idx_good - xf_score:idx_good] = ref_win
    mono[idx_good] = -mono[start_lo] * 5.0

    # Candidate B ("Whistle"/"ChuChu Vox"): crossfade region is unrelated
    # noise (a bad blend), with its single endpoint SAMPLE tuned to match
    # candidate A's delta exactly -- so a single-sample metric can't tell
    # these apart, but a crossfade-region metric clearly can.
    l_bad_region = int(2.0 * SR)
    idx_bad = start_lo + l_bad_region
    unrelated = sosfiltfilt(sos, np.random.default_rng(99).standard_normal(xf_score))
    unrelated /= np.max(np.abs(unrelated)) + 1e-9
    mono[idx_bad - xf_score:idx_bad] = unrelated * peak
    mono[idx_bad] = mono[idx_good]  # same single-sample delta as A

    x = make_stereo(mono)
    kind, ratio = pp.classify(x, HOLD)
    assert kind == "sustaining", f"test signal must classify sustaining (ratio={ratio})"

    dv_good = abs(mono[start_lo] - mono[idx_good])
    dv_bad = abs(mono[start_lo] - mono[idx_bad])
    assert dv_good == dv_bad, "both candidates must share the same single-sample delta"
    assert dv_good > 0.5 * peak, (
        "the shared single-sample delta should be dramatically bad -- a "
        "single-sample metric would have rejected BOTH of these"
    )

    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None, "the good-crossfade-region candidate should still be accepted"
    chosen_length = loop["end"] - loop["start"]
    assert chosen_length == l_good_region, (
        f"expected find_loop to pick the good-crossfade-region candidate "
        f"({l_good_region}) over the bad one ({l_bad_region}) despite "
        f"identical single-sample deltas; got length {chosen_length}"
    )
    assert loop["score"] > 0.9, "the accepted candidate's crossfade region should score very well"


def test_resample_ratio():
    x = np.zeros((pp.SR_IN, 2))
    y = pp.resample_to_48k(x)
    assert abs(len(y) - pp.SR_OUT) <= 2


def test_release_measured():
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    mono[HOLD:] *= np.exp(-10.0 * (t[HOLD:] - t[HOLD]))
    r = pp.measure_release(make_stereo(mono), SR, HOLD)
    assert 0.1 < r < 2.0


# ---------------------------------------------------------------------------
# Additional tests: close real gaps not covered by the plan's synthetic tests.
# ---------------------------------------------------------------------------


def _make_patch_dir(tmp_path, zones_audio, hold_seconds=3.5, extra_top_level=None):
    """Write a synthetic patch.json + 64 kHz stereo 16-bit WAVs into tmp_path.

    zones_audio: list of (key, velocity, layer, mono_64k_array) tuples.
    Returns the patch dict that was written (pre-processing).
    """
    zones = []
    for i, (key, vel, layer, mono64) in enumerate(zones_audio):
        fname = f"zone{i}.wav"
        stereo64 = np.stack([mono64, mono64], axis=1).astype(np.float32)
        sf.write(str(tmp_path / fname), stereo64, pp.SR_IN, subtype="PCM_16")
        zones.append({
            "key": key,
            "velocity": vel,
            "layer": layer,
            "frames": len(mono64),
            "file": fname,
        })

    patch = {
        "name": "Test Patch",
        "bank": "A",
        "index": 3,
        "sample_rate": pp.SR_IN,
        "effects": {
            "reverb": {"type": 1, "time": 5},
            "chorus": {"rate": 3},
            "reverb_send": [10, 20],
            "chorus_send": [5, 6],
            "tone_level": [100, 90],
            "bend_up": 2,
            "bend_down": 2,
        },
        "lfo1": {"waveform": "triangle", "rate": 3},
        "lfo2": {"waveform": "sine", "rate": 1},
        "zones": zones,
    }
    if extra_top_level:
        patch.update(extra_top_level)

    (tmp_path / "patch.json").write_text(json.dumps(patch, indent=2))
    return patch


def _sustaining_zone_audio(hold_seconds=3.5, total_seconds=6.0, freq=220.0):
    t = np.arange(int(total_seconds * pp.SR_IN)) / pp.SR_IN
    hold_in = int(hold_seconds * pp.SR_IN)
    mono = 0.5 * np.sin(2 * np.pi * freq * t)
    mono[hold_in:] *= np.exp(-6.0 * (t[hold_in:] - t[hold_in]))
    return mono


def _decaying_zone_audio(total_seconds=6.0, freq=440.0):
    t = np.arange(int(total_seconds * pp.SR_IN)) / pp.SR_IN
    mono = 0.8 * np.sin(2 * np.pi * freq * t) * np.exp(-6.0 * t)
    return mono


def test_process_patch_end_to_end(tmp_path):
    zones_audio = [
        (60, 100, 1, _sustaining_zone_audio(freq=220.0)),
        (60, 64, 1, _decaying_zone_audio(freq=440.0)),
        (72, 100, 1, _sustaining_zone_audio(freq=330.0)),
    ]
    _make_patch_dir(tmp_path, zones_audio)

    meta = pp.process_patch(tmp_path)

    # Top-level pre-existing keys survive untouched.
    assert meta["name"] == "Test Patch"
    assert meta["bank"] == "A"
    assert meta["index"] == 3
    assert meta["effects"]["reverb"] == {"type": 1, "time": 5}
    assert meta["effects"]["chorus"] == {"rate": 3}
    assert meta["effects"]["reverb_send"] == [10, 20]
    assert meta["effects"]["chorus_send"] == [5, 6]
    assert meta["effects"]["tone_level"] == [100, 90]
    assert meta["effects"]["bend_up"] == 2
    assert meta["effects"]["bend_down"] == 2
    assert meta["lfo1"] == {"waveform": "triangle", "rate": 3}
    assert meta["lfo2"] == {"waveform": "sine", "rate": 1}

    # sample_rate updated to 48000.
    assert meta["sample_rate"] == pp.SR_OUT

    assert len(meta["zones"]) == 3
    for z in meta["zones"]:
        # file now points at a .flac, and the .wav is gone.
        assert z["file"].endswith(".flac")
        wav_path = tmp_path / (Path(z["file"]).stem + ".wav")
        flac_path = tmp_path / z["file"]
        assert not wav_path.exists()
        assert flac_path.exists()

        info = sf.info(str(flac_path))
        assert info.samplerate == pp.SR_OUT
        assert info.channels == 2
        assert "24" in info.subtype

        assert "loop" in z
        assert "release" in z
        assert "kind" in z
        assert "enabled" in z["loop"]

    # patch.json on disk reflects the same content returned in-memory.
    on_disk = json.loads((tmp_path / "patch.json").read_text())
    assert on_disk == meta


def test_process_patch_is_idempotent(tmp_path):
    zones_audio = [(60, 100, 1, _sustaining_zone_audio(freq=220.0))]
    _make_patch_dir(tmp_path, zones_audio)

    first = pp.process_patch(tmp_path)
    # Second pass: source .wav files are gone (replaced by .flac), so the
    # src.exists() guard should skip every zone cleanly instead of crashing.
    second = pp.process_patch(tmp_path)

    assert first == second
    for z in second["zones"]:
        assert (tmp_path / z["file"]).exists()


def test_silent_zone_is_safe():
    silent = np.zeros(int(6.0 * SR))
    x = make_stereo(silent)

    kind, ratio = pp.classify(x, HOLD)
    assert kind == "silent"
    assert ratio == 0.0

    loop = pp.find_loop(x, SR, HOLD)
    assert loop is None

    release = pp.measure_release(x, SR, HOLD)
    assert np.isfinite(release)
    assert release >= 0.0


def test_silent_zone_end_to_end_no_loop(tmp_path):
    zones_audio = [(60, 100, 1, np.zeros(int(6.0 * pp.SR_IN)))]
    _make_patch_dir(tmp_path, zones_audio)

    meta = pp.process_patch(tmp_path)
    z = meta["zones"][0]
    assert z["kind"] == "silent"
    assert z["loop"]["enabled"] is False


# ---------------------------------------------------------------------------
# Batch-safety tests (spec-review follow-up): a missing render, a
# zero-length WAV, or a mono WAV must never crash the whole 4,197-patch
# batch, and every zone must end up with the full schema regardless.
# ---------------------------------------------------------------------------


def _minimal_patch_dict(zones):
    return {
        "name": "Batch Safety Patch", "bank": "A", "index": 0,
        "sample_rate": pp.SR_IN,
        "effects": {}, "lfo1": {}, "lfo2": {},
        "zones": zones,
    }


def test_missing_zone_file_gets_safe_defaults(tmp_path):
    # No .wav is ever written for this zone -- simulates a render that the
    # C++ renderer (Task 3) never produced.
    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": 0,
         "file": "does_not_exist.wav"},
    ])
    (tmp_path / "patch.json").write_text(json.dumps(patch))

    meta = pp.process_patch(tmp_path)  # must not raise
    z = meta["zones"][0]

    assert z["kind"] == "missing"
    assert z["loop"] == {"enabled": False}
    assert "release" in z and isinstance(z["release"], (int, float))
    assert z["sustain_ratio"] == 0.0
    # Nothing was produced to point the file reference at; left as-is
    # rather than invented.
    assert z["file"] == "does_not_exist.wav"


def test_empty_array_does_not_crash_classify_find_loop_or_release():
    empty = np.zeros((0, 2))

    kind, ratio = pp.classify(empty, HOLD)
    assert kind == "silent"
    assert ratio == 0.0

    assert pp.find_loop(empty, SR, HOLD) is None

    r = pp.measure_release(empty, SR, HOLD)
    assert np.isfinite(r)


def test_process_patch_survives_zero_length_zone(tmp_path):
    good_audio = _sustaining_zone_audio(freq=220.0)
    good_stereo = np.stack([good_audio, good_audio], axis=1).astype(np.float32)
    sf.write(str(tmp_path / "good.wav"), good_stereo, pp.SR_IN, subtype="PCM_16")

    empty_stereo = np.zeros((0, 2), dtype=np.float32)
    sf.write(str(tmp_path / "empty.wav"), empty_stereo, pp.SR_IN, subtype="PCM_16")

    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": len(good_audio),
         "file": "good.wav"},
        {"key": 61, "velocity": 100, "layer": 1, "frames": 0,
         "file": "empty.wav"},
    ])
    (tmp_path / "patch.json").write_text(json.dumps(patch))

    meta = pp.process_patch(tmp_path)  # must not raise -- one bad zone
    # must not take down the other zones in this patch.

    good_zone = next(z for z in meta["zones"] if z["key"] == 60)
    bad_zone = next(z for z in meta["zones"] if z["key"] == 61)

    assert good_zone["kind"] == "sustaining"
    assert good_zone["file"].endswith(".flac")
    assert (tmp_path / good_zone["file"]).exists()

    assert bad_zone["kind"] == "error"
    assert bad_zone["loop"] == {"enabled": False}
    # The zero-length source is left in place untouched, rather than
    # deleted in favour of a FLAC that soundfile can't reopen (verified
    # separately: writing a zero-frame FLAC produces a file libsndfile
    # reports as "Format not recognised").
    assert (tmp_path / "empty.wav").exists()
    assert bad_zone["file"] == "empty.wav"


def test_mono_zone_fails_explicitly_not_silently(tmp_path):
    mono_audio = _sustaining_zone_audio(freq=220.0).astype(np.float32)
    sf.write(str(tmp_path / "mono.wav"), mono_audio, pp.SR_IN, subtype="PCM_16")

    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": len(mono_audio),
         "file": "mono.wav"},
    ])
    (tmp_path / "patch.json").write_text(json.dumps(patch))

    meta = pp.process_patch(tmp_path)  # must not raise
    z = meta["zones"][0]

    assert z["kind"] == "error"
    assert z["loop"] == {"enabled": False}
    # No mono FLAC was silently written in place of the stereo requirement.
    assert not (tmp_path / "mono.flac").exists()
    assert (tmp_path / "mono.wav").exists()
    assert z["file"] == "mono.wav"


def test_main_survives_bad_patch_and_continues(tmp_path, monkeypatch, capsys):
    bad_dir = tmp_path / "01_bad"
    bad_dir.mkdir()
    (bad_dir / "patch.json").write_text("{not valid json")

    good_dir = tmp_path / "02_good"
    good_dir.mkdir()
    _make_patch_dir(good_dir, [(60, 100, 1, _sustaining_zone_audio(freq=220.0))])

    monkeypatch.setattr(sys, "argv", ["postprocess.py", str(tmp_path)])

    # A batch with any failure must exit non-zero (requirement #2) so a
    # systemic bug can't hide behind an always-0 exit code -- but the
    # *other*, good patch must still have been fully processed first.
    with pytest.raises(SystemExit) as exc_info:
        pp.main()
    assert exc_info.value.code != 0

    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "01_bad" in captured.err

    on_disk = json.loads((good_dir / "patch.json").read_text())
    assert on_disk["sample_rate"] == pp.SR_OUT
    assert on_disk["zones"][0]["loop"]["enabled"] is True

    # Machine-readable failure list so a later pass can retry just the
    # patches/zones that failed.
    fail_path = tmp_path / "postprocess_failures.json"
    assert fail_path.exists()
    failures = json.loads(fail_path.read_text())
    assert any(f["scope"] == "patch" and f["patch"] == "01_bad" for f in failures)


def test_main_exits_cleanly_with_no_failures(tmp_path, monkeypatch):
    good_dir = tmp_path / "01_good"
    good_dir.mkdir()
    _make_patch_dir(good_dir, [(60, 100, 1, _sustaining_zone_audio(freq=220.0))])

    monkeypatch.setattr(sys, "argv", ["postprocess.py", str(tmp_path)])
    pp.main()  # must NOT raise SystemExit when nothing failed

    fail_path = tmp_path / "postprocess_failures.json"
    assert fail_path.exists()
    assert json.loads(fail_path.read_text()) == []


def test_main_summary_line_reports_zone_failures(tmp_path, monkeypatch, capsys):
    good_audio = _sustaining_zone_audio(freq=220.0)
    good_stereo = np.stack([good_audio, good_audio], axis=1).astype(np.float32)
    patch_dir = tmp_path / "01_mixed"
    patch_dir.mkdir()
    sf.write(str(patch_dir / "good.wav"), good_stereo, pp.SR_IN, subtype="PCM_16")

    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": len(good_audio),
         "file": "good.wav"},
        {"key": 61, "velocity": 100, "layer": 1, "frames": 0,
         "file": "missing.wav"},
    ])
    (patch_dir / "patch.json").write_text(json.dumps(patch))

    monkeypatch.setattr(sys, "argv", ["postprocess.py", str(tmp_path)])
    with pytest.raises(SystemExit):
        pp.main()

    captured = capsys.readouterr()
    # The per-patch summary line itself must surface the failed count, not
    # just the aggregate end-of-run total.
    assert "1/2 FAILED" in captured.out


def test_unexpected_filename_gets_marked_unprocessable(tmp_path):
    # A zone whose "file" was never a .wav to begin with (not an
    # already-processed .flac either) -- simulates a schema/data problem
    # upstream. Must not be silently skipped with no kind/loop/release.
    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": 0, "file": "weird.aiff"},
    ])
    (tmp_path / "patch.json").write_text(json.dumps(patch))

    meta = pp.process_patch(tmp_path)  # must not raise
    z = meta["zones"][0]

    assert z["kind"] == "error"
    assert z["loop"] == {"enabled": False}
    assert "release" in z


def test_flac_name_that_does_not_exist_is_not_treated_as_processed(tmp_path):
    # A zone whose "file" already says .flac but nothing was ever written
    # there (corrupt patch.json, hand-edited, whatever) must not be
    # silently skipped as "already processed" just because of the suffix.
    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": 0, "file": "ghost.flac"},
    ])
    (tmp_path / "patch.json").write_text(json.dumps(patch))

    meta = pp.process_patch(tmp_path)  # must not raise
    z = meta["zones"][0]

    assert z["kind"] == "error"
    assert z["loop"] == {"enabled": False}


def test_zone_failure_after_read_stays_retryable(tmp_path, monkeypatch):
    """Requirement #5: previously, src.unlink() ran before z["file"] was
    reassigned and before measure_release, so a failure in measure_release
    still left z["file"] pointing at the written .flac -- the next run's
    suffix check would then skip it forever, permanently stranding a zone
    with valid audio but stale "error" metadata. Verify the reordered
    "point of no return" (compute everything, THEN write/commit/unlink
    last) keeps a mid-pipeline failure retryable.
    """
    audio = _sustaining_zone_audio(freq=220.0)
    stereo = np.stack([audio, audio], axis=1).astype(np.float32)
    sf.write(str(tmp_path / "zone0.wav"), stereo, pp.SR_IN, subtype="PCM_16")

    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": len(audio),
         "file": "zone0.wav"},
    ])
    (tmp_path / "patch.json").write_text(json.dumps(patch))

    def boom(*_a, **_k):
        raise RuntimeError("synthetic failure late in the pipeline")

    monkeypatch.setattr(pp, "measure_release", boom)

    meta = pp.process_patch(tmp_path)  # must not raise
    z = meta["zones"][0]

    assert z["kind"] == "error"
    # The critical assertion: z["file"] still points at the ORIGINAL .wav
    # (not a .flac that got written before the failure), and that .wav
    # still exists, so the very next process_patch() call retries this
    # zone from scratch instead of being permanently skipped.
    assert z["file"] == "zone0.wav"
    assert (tmp_path / "zone0.wav").exists()
    assert not (tmp_path / "zone0.flac").exists()

    # Prove it's actually retryable: remove the monkeypatch and rerun.
    monkeypatch.undo()
    meta2 = pp.process_patch(tmp_path)
    z2 = meta2["zones"][0]
    assert z2["kind"] == "sustaining"
    assert z2["file"] == "zone0.flac"


def test_find_loop_performance():
    """Regression guard, now exercised through find_loop directly.

    estimate_period (the target of the previous perf fix, O(n^2)
    np.correlate -> O(n log n) fftconvolve) no longer exists: the
    length-based rewrite made it genuinely dead weight -- gating candidate
    lengths off a single estimated pitch period is exactly the bug this
    rewrite fixes, so there was nothing left for it to usefully do, and
    the new full-region FFT cross-correlation already produces sample-
    exact endpoint alignment without it (see find_loop's docstring).

    This asserts find_loop itself stays fast on a production-sized
    ~117,600-frame steady-state region (the standard 3.5s-hold synthetic
    test signal below). The old np.correlate-based estimate_period alone
    took ~7.4s at this size; find_loop's FFT cross-correlation over the
    *entire* region (a strictly bigger computation) should still complete
    in well under a second -- measured around 5-10ms.
    """
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    x = make_stereo(mono)

    start = time.perf_counter()
    loop = pp.find_loop(x, SR, HOLD)
    elapsed = time.perf_counter() - start

    assert loop is not None
    assert elapsed < 1.0, (
        f"find_loop took {elapsed:.2f}s on a {HOLD}-frame-hold region -- "
        "the old np.correlate-based period estimate alone took ~8s at a "
        "comparable size, so this looks like a performance regression"
    )


# ---------------------------------------------------------------------------
# --reloop mode: re-run classification/loop detection on already-encoded
# FLACs in place, without re-encoding audio or touching sample files.
# process_patch deletes the source .wav after writing the .flac, so a
# find_loop improvement (like this one) can only reach already-processed
# output through this path -- otherwise it needs a full re-render.
# ---------------------------------------------------------------------------


def test_reloop_leaves_file_frames_and_audio_bytes_untouched(tmp_path):
    zones_audio = [
        (60, 100, 1, _sustaining_zone_audio(freq=220.0)),
        (60, 64, 1, _decaying_zone_audio(freq=440.0)),
    ]
    _make_patch_dir(tmp_path, zones_audio)

    first = pp.process_patch(tmp_path)
    # Keyed by (key, velocity), not just key -- these two zones share the
    # same MIDI key (60) at different velocities, same as real multisample
    # layers do.
    before = {
        (z["key"], z["velocity"]): {
            "file": z["file"],
            "frames": z["frames"],
            "bytes": (tmp_path / z["file"]).read_bytes(),
        }
        for z in first["zones"]
    }

    second = pp.reloop_patch(tmp_path)

    assert len(second["zones"]) == len(first["zones"])
    for z in second["zones"]:
        b = before[(z["key"], z["velocity"])]
        assert z["file"] == b["file"]
        assert z["frames"] == b["frames"]
        assert (tmp_path / z["file"]).read_bytes() == b["bytes"], (
            "reloop must never rewrite the audio file"
        )
        # Schema contract still holds after a reloop pass.
        assert "loop" in z and "kind" in z and "release" in z

    on_disk = json.loads((tmp_path / "patch.json").read_text())
    assert on_disk == second


def test_reloop_can_improve_loop_detection_without_reencoding(tmp_path, monkeypatch):
    """The actual reason --reloop exists: pick up a find_loop improvement
    on already-processed output. Simulates having processed this patch
    with a worse loop detector (the pilot render used the old, narrowly
    period-constrained find_loop and now needs re-looping without a 37GB
    re-render) by monkeypatching find_loop to find nothing on the first
    pass, then showing reloop_patch (with the real, fixed find_loop)
    updates the loop -- with the audio file never rewritten.
    """
    zones_audio = [(60, 100, 1, _sustaining_zone_audio(freq=1200.0))]
    _make_patch_dir(tmp_path, zones_audio)

    monkeypatch.setattr(pp, "find_loop", lambda *a, **k: None)
    first = pp.process_patch(tmp_path)
    z0 = first["zones"][0]
    assert z0["kind"] == "sustaining"
    assert z0["loop"]["enabled"] is False  # old/worse detector found nothing

    flac_path = tmp_path / z0["file"]
    original_bytes = flac_path.read_bytes()
    original_file = z0["file"]
    original_frames = z0["frames"]

    monkeypatch.undo()  # restore the real (fixed) find_loop

    second = pp.reloop_patch(tmp_path)
    z1 = second["zones"][0]

    assert z1["loop"]["enabled"] is True, "the fixed detector should find this loop"
    assert z1["file"] == original_file
    assert z1["frames"] == original_frames
    assert flac_path.read_bytes() == original_bytes, "audio must never be rewritten by reloop"


def test_reloop_skips_missing_and_error_zones(tmp_path):
    """A zone with no usable audio (kind "missing"/"error") must be left
    exactly as-is by reloop -- there's no .flac to read, and reloop must
    never invent or touch sample files."""
    patch = _minimal_patch_dict([
        {"key": 60, "velocity": 100, "layer": 1, "frames": 0,
         "file": "does_not_exist.wav"},
    ])
    (tmp_path / "patch.json").write_text(json.dumps(patch))

    first = pp.process_patch(tmp_path)
    assert first["zones"][0]["kind"] == "missing"

    second = pp.reloop_patch(tmp_path)
    assert second["zones"][0]["kind"] == "missing"
    assert second["zones"][0]["loop"] == {"enabled": False}
    assert second["zones"][0]["file"] == "does_not_exist.wav"


def test_main_reloop_flag_invokes_reloop_not_process(tmp_path, monkeypatch):
    zones_audio = [(60, 100, 1, _sustaining_zone_audio(freq=220.0))]
    _make_patch_dir(tmp_path, zones_audio)
    pp.process_patch(tmp_path)  # get to an already-encoded state first

    flac_name = json.loads((tmp_path / "patch.json").read_text())["zones"][0]["file"]
    original_bytes = (tmp_path / flac_name).read_bytes()

    monkeypatch.setattr(sys, "argv", ["postprocess.py", "--reloop", str(tmp_path)])
    pp.main()  # must not raise, must not try to re-encode from a .wav
    # that no longer exists

    assert (tmp_path / flac_name).read_bytes() == original_bytes
    on_disk = json.loads((tmp_path / "patch.json").read_text())
    assert on_disk["zones"][0]["file"] == flac_name
