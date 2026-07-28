"""Acceptance tests for tools/emit_presets.py (Task 6: preset emitters).

Uses synthetic patch.json-shaped metadata throughout -- no real renders are
needed to exercise the emitter logic.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import emit_presets as ep  # noqa: E402

CALIB_PATH = Path(__file__).resolve().parents[1] / "calib" / "calibration.json"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def make_zone(key, velocity, layer, kind="sustaining", release=0.5,
              loop_enabled=True, crossfade=500, file=None):
    z = {
        "key": key, "velocity": velocity, "layer": layer,
        "frames": 200000,
        "file": file or f"n{key}_v{layer}.flac",
        "kind": kind,
        "sustain_ratio": 0.4 if kind == "sustaining" else 0.0,
        "release": release,
    }
    if kind in ep.FAILURE_KINDS:
        z["loop"] = {"enabled": False}
    elif loop_enabled:
        z["loop"] = {"enabled": True, "start": 48000, "end": 158000,
                      "crossfade": crossfade, "score": 0.97}
    else:
        z["loop"] = {"enabled": False}
    return z


def make_meta(reverb_type="Hall1", lfo_form="SIN", lfo1_stripped=True, sync=1,
              chorus_output="Mix", zone_overrides=None, extra_zones=None,
              chorus_rate=50):
    """25 keys (C1..C7, every 3 semitones) x 3 velocity layers, all
    'sustaining' by default -- matches the design doc's sampling grid.
    `zone_overrides` maps (key, layer) -> dict of overrides applied to that
    zone (e.g. {"kind": "missing"}), for exercising the failure-skip path.
    """
    zone_overrides = zone_overrides or {}
    zones = []
    for key in range(24, 97, 3):
        for layer, vel in ((1, 32), (2, 72), (3, 110)):
            z = make_zone(key, vel, layer)
            override = zone_overrides.get((key, layer))
            if override:
                z.update(override)
                if override.get("kind") in ep.FAILURE_KINDS:
                    z["loop"] = {"enabled": False}
            zones.append(z)
    if extra_zones:
        zones.extend(extra_zones)

    return {
        "name": "Test Patch", "bank": "A", "index": 0, "sample_rate": 48000,
        "effects": {
            "reverb": {"type": reverb_type, "level": 80, "time": 64, "feedback": 20},
            "chorus": {"type": "Chorus1", "level": 40, "depth": 30,
                       "rate": chorus_rate, "feedback": 0, "output": chorus_output},
            "reverb_send": [100, 100, 0, 0], "chorus_send": [80, 80, 0, 0],
            "tone_level": [100, 100, 0, 0], "bend_up": 2, "bend_down": 2,
        },
        "lfo1": {"stripped": lfo1_stripped, "reason": "strippable", "form": lfo_form,
                 "rate": 60, "delay": 25, "sync": sync,
                 "pitch": 0, "tvf": 0, "tva": 20},
        "lfo2": {"stripped": False, "reason": "no LFO depth", "form": "TRI",
                 "rate": 0, "delay": 0, "sync": 0, "pitch": 0, "tvf": 0, "tva": 0},
        "zones": zones,
    }


CAL = {
    "chorus_rate_hz": {"0": 0.1, "16": 0.5, "32": 0.9, "96": 2.2734,
                        "104": 2.9416, "127": 6.0},
    "chorus_depth_norm": {"0": 0.0, "127": 1.0},
    "chorus_mix": {"0": 0.0, "127": 0.8},
    "reverb_wet": {"0": 0.0, "127": 0.7},
    "reverb_rt60": {
        "0": {"0": 0.7, "64": 1.1, "127": 1.9},
        "1": {"0": 0.6, "64": 0.9, "127": 2.1},
        "2": {"0": 0.8, "64": 1.4, "127": 2.8},
        "3": {"0": 2.0, "64": 3.0, "127": 9.9},
        "4": {"0": 2.2, "16": 2.0773, "32": 1.8122, "64": 2.4, "127": 8.0},
        "5": {"0": 1.8, "64": 3.0, "127": 12.5},
    },
}

REAL_CAL = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else None


def parse(meta, cal=CAL, prefix="Samples/x"):
    return ET.fromstring(ep.build_dspreset(meta, cal, prefix))


# ---------------------------------------------------------------------------
# AC 1: well-formed XML, one <sample> per zone
# ---------------------------------------------------------------------------

def test_dspreset_is_valid_xml_with_one_sample_per_zone():
    root = parse(make_meta())
    samples = root.findall(".//sample")
    assert len(samples) == 25 * 3


def test_dspreset_has_groups_containing_samples():
    root = parse(make_meta())
    groups = root.find("groups")
    assert groups is not None
    assert groups.find(".//sample") is not None


# ---------------------------------------------------------------------------
# AC 2: key ranges tile 0..127, no gaps, contiguous
# ---------------------------------------------------------------------------

def test_key_ranges_tile_0_to_127_no_gaps():
    keys = [24, 27, 33, 96]
    spans = ep.key_ranges(keys)
    spans_sorted = sorted(spans, key=lambda s: s[0])
    assert spans_sorted[0][1] == 0, "lowest span must start at 0"
    assert spans_sorted[-1][2] == 127, "highest span must end at 127"
    for (_, _, hi), (_, lo2, _) in zip(spans_sorted, spans_sorted[1:]):
        assert lo2 == hi + 1, f"gap or overlap between {hi} and {lo2}"


def test_key_ranges_from_full_dspreset_tile_without_gaps():
    root = parse(make_meta())
    spans = sorted({(int(s.get("loNote")), int(s.get("hiNote")))
                    for s in root.findall(".//sample")})
    assert spans[0][0] == 0
    assert spans[-1][1] == 127
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert lo == hi + 1


def test_key_ranges_single_key_covers_full_range():
    spans = ep.key_ranges([60])
    assert spans == [(60, 0, 127)]


# ---------------------------------------------------------------------------
# AC 3: velocity ranges tile 1..127, no gaps/overlaps
# ---------------------------------------------------------------------------

def test_vel_ranges_tile_1_to_127_for_three_layers():
    vr = ep.vel_ranges(3)
    assert vr[0][0] == 1
    assert vr[-1][1] == 127
    for (_, hi), (lo, _) in zip(vr, vr[1:]):
        assert lo == hi + 1
    # matches the design doc's documented 3-layer split exactly
    assert vr == [(1, 42), (43, 85), (86, 127)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_vel_ranges_tile_without_gaps_for_various_layer_counts(n):
    vr = ep.vel_ranges(n)
    assert len(vr) == n
    assert vr[0][0] == 1
    assert vr[-1][1] == 127
    for (_, hi), (lo, _) in zip(vr, vr[1:]):
        assert lo == hi + 1


def test_velocity_ranges_from_full_dspreset_tile_1_to_127():
    root = parse(make_meta())
    vr = sorted({(int(s.get("loVel")), int(s.get("hiVel")))
                 for s in root.findall(".//sample")})
    assert vr[0][0] == 1 and vr[-1][1] == 127
    for (_, hi), (lo, _) in zip(vr, vr[1:]):
        assert lo == hi + 1


# ---------------------------------------------------------------------------
# AC 4: reverb types 0-5 -> reverb; 6-7 -> delay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rtype", ["Room1", "Room2", "Stage1", "Stage2", "Hall1", "Hall2"])
def test_hall_and_room_types_emit_reverb_never_delay(rtype):
    root = parse(make_meta(reverb_type=rtype))
    types = [e.get("type") for e in root.findall(".//effect")]
    assert "reverb" in types
    assert "delay" not in types


def test_delay_type_emits_delay_never_reverb():
    root = parse(make_meta(reverb_type="Delay"))
    types = [e.get("type") for e in root.findall(".//effect")]
    assert "delay" in types
    assert "reverb" not in types


def test_pan_dly_emits_delay_with_nonzero_stereo_offset():
    root = parse(make_meta(reverb_type="Pan-Dly"))
    delay_el = [e for e in root.findall(".//effect") if e.get("type") == "delay"][0]
    assert float(delay_el.get("stereoOffset")) != 0.0


def test_plain_delay_type_has_zero_stereo_offset():
    root = parse(make_meta(reverb_type="Delay"))
    delay_el = [e for e in root.findall(".//effect") if e.get("type") == "delay"][0]
    assert float(delay_el.get("stereoOffset")) == 0.0


# ---------------------------------------------------------------------------
# AC 5: chorus modRate from calibration table, including the raw=24 gap
# ---------------------------------------------------------------------------

def test_chorus_effect_present_with_calibrated_mod_rate():
    root = parse(make_meta(chorus_rate=50))
    ch = [e for e in root.findall(".//effect") if e.get("type") == "chorus"][0]
    # raw rate 50 sits between the 32 (0.9) and 96 (2.2734) calibration
    # points in CAL -- interpolated, so strictly between them.
    assert 0.9 < float(ch.get("modRate")) < 2.2734


def test_interp_table_matches_hand_computed_value():
    # Between two adjacent points: raw=100 sits between 96 (2.2734) and
    # 104 (2.9416) in CAL -- t = (100-96)/(104-96) = 0.5
    expected = 2.2734 + 0.5 * (2.9416 - 2.2734)
    got = ep.interp_table(CAL["chorus_rate_hz"], 100)
    assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.skipif(REAL_CAL is None, reason="calib/calibration.json not present")
def test_chorus_rate_24_gap_does_not_yield_zero_hz():
    """calib/calibration.json deliberately omits the "24" key (see its
    module docstring in analyze_calibration.py) because that raw setting's
    measurement was dominated by a non-chorus artifact. It is an
    INTERPOLATION GAP, not a zero -- table.get("24", 0) would silently
    produce a 0 Hz chorus, which this must not do."""
    assert "24" not in REAL_CAL["chorus_rate_hz"]
    rate = ep.interp_table(REAL_CAL["chorus_rate_hz"], 24)
    assert rate > 0.0
    # Sanity: should land between the real neighbouring measured points
    # (16 -> 0.7782, 32 -> 0.9009).
    assert 0.7782 < rate < 0.9009


def test_chorus_rate_24_gap_end_to_end_through_dspreset():
    """Same trap, exercised through the full pipeline: a patch whose raw
    chorus rate is exactly 24 must not emit modRate="0" (or anything close
    to it) in the generated XML."""
    root = parse(make_meta(chorus_rate=24), cal=CAL)
    ch = [e for e in root.findall(".//effect") if e.get("type") == "chorus"][0]
    mod_rate = float(ch.get("modRate"))
    assert mod_rate > 0.05, f"chorus rate=24 gap produced near-zero modRate: {mod_rate}"


@pytest.mark.skipif(REAL_CAL is None, reason="calib/calibration.json not present")
def test_reverb_rt60_hall1_dip_is_preserved_not_smoothed():
    """reverb_rt60 type 4 (Hall1) has a genuine measured dip (~0.27s)
    between raw 16 (2.0773) and raw 32 (1.8122). Interpolation must
    reproduce that dip, not fit a monotonic curve that erases it."""
    table = REAL_CAL["reverb_rt60"]["4"]
    v16 = table["16"]
    v32 = table["32"]
    assert v32 < v16, "fixture assumption: the measured dip must exist in the data"
    mid = ep.interp_table(table, 24)
    assert v32 <= mid <= v16, "interpolation should sit between the two real points"
    assert mid < v16, "the dip must show up in interpolated values, not be smoothed away"


# ---------------------------------------------------------------------------
# AC 6 & 7: LFO modulator scope/target/absence
# ---------------------------------------------------------------------------

def test_stripped_synced_lfo_emits_voice_scope():
    root = parse(make_meta(lfo1_stripped=True, sync=1))
    lfo = root.find(".//lfo")
    assert lfo is not None
    assert lfo.get("scope") == "voice"


def test_stripped_free_running_lfo_emits_global_scope():
    root = parse(make_meta(lfo1_stripped=True, sync=0))
    lfo = root.find(".//lfo")
    assert lfo is not None
    assert lfo.get("scope") == "global"


def test_stripped_lfo_binds_to_tva_target():
    # make_meta's lfo1 has tva=20, pitch=0, tvf=0
    root = parse(make_meta(lfo1_stripped=True))
    lfo = root.find(".//lfo")
    bindings = lfo.findall("binding")
    assert len(bindings) == 1
    assert bindings[0].get("parameter") == "AMP_VOLUME"


def test_stripped_lfo_shape_tri_maps_to_sine():
    root = parse(make_meta(lfo1_stripped=True, lfo_form="TRI"))
    assert root.find(".//lfo").get("shape") == "sine"


@pytest.mark.parametrize("form", ["SIN", "SAW", "SQU"])
def test_stripped_lfo_shape_maps_correctly(form):
    root = parse(make_meta(lfo1_stripped=True, lfo_form=form))
    expected = {"SIN": "sine", "SAW": "saw", "SQU": "square"}[form]
    assert root.find(".//lfo").get("shape") == expected


@pytest.mark.parametrize("form", ["RND1", "RND2"])
def test_random_waveform_emits_no_lfo(form):
    root = parse(make_meta(lfo1_stripped=True, lfo_form=form))
    assert root.find(".//lfo") is None


def test_unstripped_lfo_emits_no_modulator():
    root = parse(make_meta(lfo1_stripped=False))
    assert root.find(".//lfo") is None


def test_stripped_lfo_with_zero_depth_emits_no_modulator():
    meta = make_meta(lfo1_stripped=True)
    meta["lfo1"]["tva"] = 0
    root = parse(meta)
    assert root.find(".//lfo") is None


def test_build_lfo_modulator_directly_multiple_targets():
    lfo = {"stripped": True, "form": "SIN", "rate": 60, "delay": 0, "sync": 1,
           "pitch": 10, "tvf": 0, "tva": -30}
    m = ep.build_lfo_modulator(lfo)
    assert m is not None
    params = {b["parameter"] for b in m["bindings"]}
    assert params == {"AMP_VOLUME", "GROUP_TUNING"}


# ---------------------------------------------------------------------------
# loopCrossfade passthrough
# ---------------------------------------------------------------------------

def test_loop_crossfade_passed_through_unchanged():
    overrides = {(24, 1): {"loop_crossfade_test": True}}
    meta = make_meta()
    # directly set a distinctive crossfade value on one zone
    for z in meta["zones"]:
        if z["key"] == 24 and z["layer"] == 1:
            z["loop"]["crossfade"] = 137
    root = parse(meta)
    sample = [s for s in root.findall(".//sample") if s.get("rootNote") == "24"
              and int(s.get("loVel")) == 1][0]
    assert sample.get("loopCrossfade") == "137"


def test_loop_attributes_present_for_looped_zone():
    root = parse(make_meta())
    s = root.find(".//sample")
    assert s.get("loopEnabled") == "1"
    assert int(s.get("loopStart")) == 48000
    assert int(s.get("loopEnd")) == 158000
    assert int(s.get("loopCrossfade")) <= 2000


def test_non_looped_zone_has_no_loop_attributes():
    meta = make_meta()
    for z in meta["zones"]:
        if z["key"] == 24 and z["layer"] == 1:
            z["kind"] = "decaying"
            z["loop"] = {"enabled": False}
    root = parse(meta)
    sample = [s for s in root.findall(".//sample") if s.get("rootNote") == "24"
              and int(s.get("loVel")) == 1][0]
    assert sample.get("loopEnabled") is None
    assert sample.get("loopStart") is None


# ---------------------------------------------------------------------------
# missing/error zones must be skipped entirely
# ---------------------------------------------------------------------------

def test_missing_zone_produces_no_sample():
    meta = make_meta(zone_overrides={(24, 2): {"kind": "missing", "file": "does_not_exist.wav"}})
    root = parse(meta)
    samples = root.findall(".//sample")
    assert len(samples) == 25 * 3 - 1
    paths = [s.get("path") for s in samples]
    assert not any("does_not_exist" in p for p in paths)


def test_error_zone_produces_no_sample():
    meta = make_meta(zone_overrides={(60, 1): {"kind": "error", "file": "corrupt.wav"}})
    root = parse(meta)
    paths = [s.get("path") for s in root.findall(".//sample")]
    assert not any("corrupt" in p for p in paths)


def test_missing_zone_produces_no_sfz_region():
    meta = make_meta(zone_overrides={(24, 2): {"kind": "missing", "file": "does_not_exist.wav"}})
    sfz = ep.build_sfz(meta, "Samples/x")
    assert "does_not_exist" not in sfz
    assert sfz.count("<region>") == 25 * 3 - 1


def test_key_with_all_layers_missing_leaves_no_gap_in_neighbours():
    """If an entire key's zones all fail, the key must vanish from the
    grid entirely -- and its neighbours must still tile 0..127 with no
    gap where it used to be."""
    overrides = {(24, 1): {"kind": "missing", "file": "x.wav"},
                 (24, 2): {"kind": "missing", "file": "x.wav"},
                 (24, 3): {"kind": "missing", "file": "x.wav"}}
    meta = make_meta(zone_overrides=overrides)
    root = parse(meta)
    root_notes = {int(s.get("rootNote")) for s in root.findall(".//sample")}
    assert 24 not in root_notes
    spans = sorted({(int(s.get("loNote")), int(s.get("hiNote")))
                    for s in root.findall(".//sample")})
    assert spans[0][0] == 0
    assert spans[-1][1] == 127
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert lo == hi + 1


def test_partial_key_failure_still_tiles_velocity():
    """A key that loses one of its three velocity layers must still tile
    1..127 across its surviving layers with no gap."""
    meta = make_meta(zone_overrides={(24, 2): {"kind": "missing", "file": "x.wav"}})
    root = parse(meta)
    samples = [s for s in root.findall(".//sample") if s.get("rootNote") == "24"]
    assert len(samples) == 2
    vr = sorted((int(s.get("loVel")), int(s.get("hiVel"))) for s in samples)
    assert vr[0][0] == 1 and vr[-1][1] == 127
    assert vr[0][1] + 1 == vr[1][0]


# ---------------------------------------------------------------------------
# effective_send
# ---------------------------------------------------------------------------

def test_effective_send_averages_active_tones_only():
    meta = make_meta()
    # tone_level = [100,100,0,0], reverb_send = [100,100,0,0]
    assert ep.effective_send(meta, "reverb") == pytest.approx(100.0)


def test_effective_send_zero_when_no_active_tones():
    meta = make_meta()
    meta["effects"]["tone_level"] = [0, 0, 0, 0]
    assert ep.effective_send(meta, "reverb") == 0.0


def test_effective_send_ignores_inactive_tone_contribution():
    meta = make_meta()
    meta["effects"]["tone_level"] = [100, 0, 0, 0]
    meta["effects"]["reverb_send"] = [40, 127, 0, 0]
    assert ep.effective_send(meta, "reverb") == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# chorus routing order
# ---------------------------------------------------------------------------

def test_chorus_feeds_reverb_when_output_is_reverb():
    root = parse(make_meta(chorus_output="Reverb"))
    types = [e.get("type") for e in root.find("effects")]
    assert types.index("chorus") < types.index("reverb")


def test_chorus_parallel_after_reverb_when_output_is_mix():
    # Design doc's mapping table: chorusoutput=Reverb -> chorus before
    # reverb (chorus feeds it); chorusoutput=Mix -> "chorus parallel /
    # after" -- i.e. reverb is NOT fed by chorus here, so reverb comes
    # first and chorus is appended after it.
    root = parse(make_meta(chorus_output="Mix"))
    types = [e.get("type") for e in root.find("effects")]
    assert types.index("chorus") > types.index("reverb")


# ---------------------------------------------------------------------------
# AC 8: SFZ region count, key/vel ranges, loop opcodes, ampeg_release
# ---------------------------------------------------------------------------

def test_sfz_has_one_region_per_zone():
    sfz = ep.build_sfz(make_meta(), "Samples/x")
    assert sfz.count("<region>") == 25 * 3


def test_sfz_region_has_matching_key_vel_ranges_and_release():
    sfz = ep.build_sfz(make_meta(), "Samples/x")
    assert "lokey=0" in sfz  # lowest key's span starts at 0
    assert "hikey=127" in sfz  # highest key's span ends at 127
    assert "lovel=1" in sfz
    assert "hivel=127" in sfz
    assert "ampeg_release=" in sfz


def test_sfz_looped_zone_has_loop_opcodes():
    sfz = ep.build_sfz(make_meta(), "Samples/x")
    assert "loop_mode=loop_continuous" in sfz
    assert "loop_start=48000 loop_end=158000" in sfz


def test_sfz_non_looped_zone_has_no_loop_points_only_no_loop_mode():
    meta = make_meta()
    for z in meta["zones"]:
        z["kind"] = "decaying"
        z["loop"] = {"enabled": False}
    sfz = ep.build_sfz(meta, "Samples/x")
    assert "loop_mode=no_loop" in sfz
    assert "loop_mode=loop_continuous" not in sfz
    assert "loop_start=" not in sfz


# ---------------------------------------------------------------------------
# AC 9 (structural half): sample paths reference the zone's actual file
# ---------------------------------------------------------------------------

def test_sample_paths_use_sample_prefix_and_zone_file():
    root = parse(make_meta(), prefix="Samples/000_TestPatch")
    sample = root.find(".//sample")
    assert sample.get("path").startswith("Samples/000_TestPatch/")
    assert sample.get("path").endswith(".flac")


def test_end_to_end_emitted_sample_paths_exist_on_disk(tmp_path):
    """Integration-level check of AC 9: write real FLAC-named (empty)
    files at the location `sample_prefix` implies (mirroring what main()
    assembles: presets sit in `library_dir`, samples in
    `library_dir/Samples/<patchdir>/`), run build_dspreset, and verify
    every emitted <sample path> resolves relative to the preset's own
    directory -- while a deliberately-missing zone must never appear."""
    library_dir = tmp_path  # where the .dspreset itself would be written
    patch_name = "000_TestPatch"
    samples_dir = library_dir / "Samples" / patch_name
    samples_dir.mkdir(parents=True)

    meta = make_meta(zone_overrides={(24, 2): {"kind": "missing", "file": "ghost.flac"}})
    for z in meta["zones"]:
        if z.get("kind") not in ep.FAILURE_KINDS:
            (samples_dir / z["file"]).write_bytes(b"")  # placeholder, existence is what matters

    sample_prefix = f"Samples/{patch_name}"
    xml_text = ep.build_dspreset(meta, CAL, sample_prefix)
    root = ET.fromstring(xml_text)

    missing = [s.get("path") for s in root.findall(".//sample")
               if not (library_dir / s.get("path")).exists()]
    assert missing == []
    assert not any("ghost" in (s.get("path") or "") for s in root.findall(".//sample"))
