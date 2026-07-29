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
    # Values below are for the length-based FFT search (see find_loop):
    # a pure 220 Hz tone is periodic at every lag, so the "prefer longer"
    # tiebreak walks all the way out near the end of the reachable region
    # (region_hi - REF_WIN), not just to a small multiple of the pitch
    # period.
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    x = make_stereo(mono)
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    assert loop["start"] == 48000
    assert loop["end"] == 163200
    assert loop["crossfade"] == 2000
    # This specific case exercises the MAX_XFADE branch of the min(): both
    # start//4 (12000) and length//4 (28800) exceed MAX_XFADE (2000), so
    # MAX_XFADE is the binding constraint here.
    assert loop["crossfade"] == pp.MAX_XFADE
    # Independent sanity check against the documented bound, using the
    # loop's own reported values (not a copy of the implementation).
    length = loop["end"] - loop["start"]
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
    inharmonic bell-like material) with an exact copy spliced in at
    L_target=1.9s. A single pure tone can't demonstrate this bug (every
    period multiple of a pure tone is trivially a perfect match, which is
    exactly why the bug passed unnoticed on this file's own 220 Hz test
    signal) -- the point is that old-style short-lag candidates must
    genuinely NOT correlate, only the engineered long lag should.
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
    overlap = 4096  # > REF_WIN, so the engineered repeat is unambiguous
    mono[start_lo + l_target: start_lo + l_target + overlap] = \
        mono[start_lo: start_lo + overlap]
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


def test_stereo_divergent_endpoint_rejected_even_when_mono_cancels():
    """Regression test for a second, separate bug found in pilot
    validation: find_loop used to score correlation and minimize endpoint
    discontinuity on x.mean(axis=1) -- the MONO mix. On stereo-divergent
    material (stereo-detuned/panned tone pairs, common in JV patches) the
    L and R endpoint errors can be near-exact opposites that CANCEL in the
    mono sum, producing a "perfect" 0.0% mono match at a lag whose real
    per-channel discontinuity is far above the validator's 5%-of-peak
    threshold -- an audible click the mono-based search couldn't see by
    construction. Real pilot examples: F#2_v3 (mono 0.0%, per-channel
    8.0%), C1_v2 (mono 0.1%, per-channel 6.9%). 2,400+ looped zones in the
    447-patch pilot failed this way.

    This builds a clean stereo-identical sine (lots of genuinely good,
    per-channel-identical candidates exist naturally), finds where the
    (already-fixed) length search would otherwise land, then poisons
    exactly that one endpoint sample with opposite-sign L/R deltas (+8%,
    -8% of peak) so the mono average at that point is untouched (the
    errors cancel) while the per-channel discontinuity there is a clear
    8%. A reconstruction of the mono-based scoring (the actual pre-fix
    code) is shown to pick exactly that poisoned lag; the real, per-
    channel-aware find_loop must not.
    """
    t = np.arange(int(6.0 * SR)) / SR
    base = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    peak = 0.5
    start_lo = int(1.0 * SR)

    # Establish where the (correct) search naturally lands on the clean,
    # stereo-identical signal, so we know exactly which sample to poison.
    baseline = pp.find_loop(make_stereo(base), SR, HOLD)
    assert baseline is not None
    l_bad = baseline["end"] - baseline["start"]
    idx = start_lo + l_bad

    delta = 0.08 * peak  # 8% of peak -- comfortably above a 5% threshold
    left = base.copy()
    right = base.copy()
    left[idx] += delta
    right[idx] -= delta
    x = np.stack([left, right], axis=1)

    # Confirm the engineered signal actually reproduces the cancellation:
    # mono difference at (start_lo, idx) is ~0, per-channel is ~8%.
    mono = x.mean(axis=1)
    mono_dv = abs(mono[start_lo] - mono[idx])
    per_channel_dv = np.abs(x[start_lo] - x[idx]).max()
    assert mono_dv < 0.0005 * peak, (
        f"engineered mono discontinuity {mono_dv} should be ~0 (cancelled)"
    )
    assert per_channel_dv > 0.05 * peak, (
        f"engineered per-channel discontinuity {per_channel_dv} should "
        "clearly exceed the 5%-of-peak validator threshold"
    )

    # Self-contained proof the OLD mono-based scoring would pick this
    # exact bad lag: reconstruct it (mono mix, single-channel dv) and
    # confirm it selects l_bad.
    def find_loop_mono_reconstruction(stereo, sr, hold_frames):
        m = stereo.mean(axis=1)
        s_lo = int(1.0 * sr)
        r_hi = min(hold_frames, len(m)) - int(0.05 * sr)
        region = m[s_lo:r_hi].astype(np.float64)
        win = min(pp.REF_WIN, len(region) // 2)
        ref = region[:win]
        ref_norm = float(np.linalg.norm(ref))
        numerator = pp.fftconvolve(region, ref[::-1], mode="valid")
        csq = np.cumsum(np.concatenate(([0.0], region ** 2)))
        win_energy = csq[win:] - csq[:-win]
        win_norm = np.sqrt(np.maximum(win_energy, 0.0))
        score = numerator / (ref_norm * win_norm + 1e-12)
        l_min = int(pp.MIN_LOOP_SECONDS * sr)
        l_max = len(region) - win
        lags = np.arange(len(score))
        mask = (lags >= l_min) & (lags <= l_max) & np.isfinite(score) & (score >= 0.90)
        cand_lags, cand_scores = lags[mask], score[mask]
        dv = np.abs(region[0] - region[cand_lags])
        best_dv = float(np.min(dv))
        tol = max(best_dv * 2.0, 0.0005 * float(np.max(np.abs(m))), 1e-9)
        near = cand_lags[dv <= tol]
        chosen = int(near[np.argmax(near)])
        return s_lo, s_lo + chosen

    mono_start, mono_end = find_loop_mono_reconstruction(x, SR, HOLD)
    assert mono_end - mono_start == l_bad, (
        "expected the mono-mix reconstruction to pick the poisoned lag "
        "(that's the bug being reproduced) -- if it picked something "
        "else, this test isn't exercising the cancellation case"
    )

    # The actual fix: per-channel scoring/discontinuity must NOT pick
    # l_bad, and whatever it does pick must have a genuinely small
    # per-channel discontinuity.
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    chosen_length = loop["end"] - loop["start"]
    assert chosen_length != l_bad, (
        "find_loop picked the same poisoned lag the mono-based scoring "
        "did -- per-channel matching isn't actually being applied"
    )
    chosen_dv = np.abs(x[loop["start"]] - x[loop["end"]]).max()
    assert chosen_dv < 0.01 * peak, (
        f"find_loop's chosen endpoint has a {100*chosen_dv/peak:.2f}%-of-"
        "peak per-channel discontinuity -- should have found one of the "
        "many genuinely clean candidates on this otherwise-clean signal"
    )


def test_find_loop_declines_when_best_candidate_is_still_bad():
    """Found in pilot validation while measuring the stereo-divergence fix
    above: the 0.90 correlation gate is a WINDOWED, aggregate similarity
    check -- it does not guarantee the boundary sample itself lines up.
    On some real material (seen on patches like Mighty Pad and Pipe
    Organ 1), literally every candidate that cleared 0.90 still had a
    60-70%-of-peak per-channel endpoint jump, because the material is
    never truly periodic and some lag just happens to clear the
    aggregate-shape bar anyway. Without an explicit ceiling, find_loop
    returned that "best of a bad lot" candidate as a loop -- looping
    fraction went up, but 75/550 looped zones in one measurement still
    failed the validator's own 5%-of-peak threshold.

    A linear chirp (steadily rising frequency) is periodic nowhere, so no
    lag is ever a genuinely clean repeat, but a nearby-in-time window can
    still look structurally similar enough to just clear 0.90.
    """
    t = np.arange(int(6.0 * SR)) / SR
    f0, f1 = 2000.0, 2600.0
    k = (f1 - f0) / t[-1]
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    mono = 0.4 * np.sin(phase)
    x = make_stereo(mono)

    kind, _ = pp.classify(x, HOLD)
    assert kind == "sustaining"

    # Confirm this signal actually exercises the ceiling and not just the
    # 0.90 correlation gate: with the ceiling effectively disabled, some
    # candidate DOES clear 0.90 but with a large endpoint discontinuity.
    original_ceiling = pp.MAX_ENDPOINT_DV_FRACTION
    try:
        pp.MAX_ENDPOINT_DV_FRACTION = 1.0
        uncapped = pp.find_loop(x, SR, HOLD)
    finally:
        pp.MAX_ENDPOINT_DV_FRACTION = original_ceiling

    assert uncapped is not None, (
        "expected some candidate to clear the 0.90 correlation gate even "
        "though none of them are a genuinely clean repeat -- if nothing "
        "clears the gate at all, this test isn't exercising the ceiling"
    )
    peak = float(np.abs(x).max())
    uncapped_dv = np.abs(x[uncapped["start"]] - x[uncapped["end"]]).max()
    assert uncapped_dv > pp.MAX_ENDPOINT_DV_FRACTION * peak, (
        "the uncapped candidate's discontinuity should clearly exceed "
        "the ceiling -- otherwise this isn't the 'best of a bad lot' case"
    )

    # With the real ceiling active, find_loop must decline rather than
    # hand back that bad candidate.
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is None, (
        "find_loop returned a loop whose only available candidates all "
        "have a bad endpoint match -- it should decline instead, per the "
        "design principle that no loop is better than a bad one"
    )


def test_prefer_longer_tolerance_never_exceeds_endpoint_ceiling():
    """Found while measuring the endpoint-ceiling fix above on real pilot
    data: the "prefer longer among comparable candidates" tiebreak uses a
    tolerance of `best_dv * 2.0` -- so a best_dv of, say, 3% of peak
    (which clears MAX_ENDPOINT_DV_FRACTION on its own) gives a tolerance
    of 6%, letting the tiebreak pick a LONGER candidate at up to 6%,
    silently exceeding the very ceiling it was just checked against. This
    is what let 28/503 looped zones in one real-data measurement still
    exceed the validator's 5% threshold even after the ceiling was added.

    Uses broadband bandpass-noise (near-zero autocorrelation almost
    everywhere, like the high-frequency regression test above) so only
    two deliberately spliced-in candidates pass the 0.90 gate at all,
    giving full control: a SHORTER one at 3% endpoint discontinuity (just
    under the ceiling) and a LONGER one at 6% (just over it). Without
    clamping the tolerance window to the ceiling, "prefer longer" picks
    the 6% one; with the clamp, it must pick the 3% one instead.
    """
    from scipy.signal import butter, sosfiltfilt

    n_total = int(6.0 * SR)
    rng = np.random.default_rng(7)
    sos = butter(4, [900, 6000], btype="bandpass", fs=SR, output="sos")
    base = sosfiltfilt(sos, rng.standard_normal(n_total))
    base /= np.max(np.abs(base)) + 1e-9
    peak = 0.5
    left = base.copy() * peak
    right = base.copy() * peak

    start_lo = int(1.0 * SR)
    overlap = 4096  # > REF_WIN
    l_near, l_far = int(1.0 * SR), int(1.8 * SR)
    for l_target, delta_frac in [(l_near, 0.03), (l_far, 0.06)]:
        idx = start_lo + l_target
        left[idx:idx + overlap] = base[start_lo:start_lo + overlap] * peak
        right[idx:idx + overlap] = base[start_lo:start_lo + overlap] * peak
        delta = delta_frac * peak
        left[idx] += delta
        right[idx] -= delta

    x = np.stack([left, right], axis=1)

    kind, _ = pp.classify(x, HOLD)
    assert kind == "sustaining"

    # Confirm the setup: without the ceiling (and therefore without the
    # tolerance clamp derived from it), the old "prefer longer" logic
    # picks the farther, worse (6%) candidate.
    original_ceiling = pp.MAX_ENDPOINT_DV_FRACTION
    try:
        pp.MAX_ENDPOINT_DV_FRACTION = 1.0
        unclamped = pp.find_loop(x, SR, HOLD)
    finally:
        pp.MAX_ENDPOINT_DV_FRACTION = original_ceiling
    assert unclamped is not None
    assert unclamped["end"] - unclamped["start"] == l_far, (
        "expected the unclamped tiebreak to pick the longer, worse "
        "candidate -- if it picked something else, this test isn't "
        "exercising the tolerance-window bug"
    )

    # With the real (clamped) logic, it must pick the shorter, genuinely
    # acceptable candidate instead.
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    assert loop["end"] - loop["start"] == l_near, (
        "expected the clamped tiebreak to pick the shorter candidate "
        "within the ceiling, not the longer one that exceeds it"
    )
    chosen_dv = np.abs(x[loop["start"]] - x[loop["end"]]).max()
    assert chosen_dv <= pp.MAX_ENDPOINT_DV_FRACTION * peak


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
