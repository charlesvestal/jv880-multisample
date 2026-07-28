import json
import sys
from pathlib import Path

import numpy as np
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
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    x = make_stereo(mono)
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    length = loop["end"] - loop["start"]
    assert loop["crossfade"] <= min(pp.MAX_XFADE, loop["start"] // 4, length // 4)


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
    pp.main()  # must not raise even though one patch is malformed

    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "01_bad" in captured.err

    on_disk = json.loads((good_dir / "patch.json").read_text())
    assert on_disk["sample_rate"] == pp.SR_OUT
    assert on_disk["zones"][0]["loop"]["enabled"] is True
