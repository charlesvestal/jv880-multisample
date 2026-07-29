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

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi):
    """Mirror src/jv_sampler.cpp's note_name() exactly (NOTE_NAMES[midi%12]
    + str(midi/12 - 1), e.g. 24 -> "C1", 27 -> "D#1", 96 -> "C7"), so
    synthetic fixtures use the SAME filename shape the real renderer
    produces -- including the '#' in sharp note names -- rather than a
    simplified n<key> form that would never exercise whatever character
    handling the real pipeline needs."""
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def make_zone(key, velocity, layer, kind="sustaining", release=0.5,
              loop_enabled=True, crossfade=500, file=None):
    z = {
        "key": key, "velocity": velocity, "layer": layer,
        "frames": 200000,
        "file": file or f"{note_name(key)}_v{layer}.flac",
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


def parse(meta, cal=CAL, prefix="x"):
    return ET.fromstring(ep.build_dspreset(meta, cal, prefix))


def iattr(el, name):
    """int() an XML attribute without tripping `str | None` type checks:
    Element.get() is typed Optional[str], and these tests only ever call
    this on an attribute they've already established is present, so a
    plain "0" default is never actually exercised at runtime -- it only
    exists to give int() a str, not None, to satisfy Pyright."""
    return int(el.get(name, "0"))


def fattr(el, name):
    """float() counterpart to iattr -- see its docstring."""
    return float(el.get(name, "0"))


def sattr(el, name):
    """str-typed .get() with a "" default, for callers that immediately
    call a str method (.startswith/.endswith) or use `in` -- Element.get()
    alone is typed Optional[str], which neither supports."""
    return el.get(name, "")


def find1(parent, path):
    """ET's .find() is typed to return Optional[Element]; these tests only
    ever call it where the element is known (by test construction) to
    exist, so assert that instead of leaving a `| None` for every
    subsequent .get()/.findall() call on the result to trip over."""
    found = parent.find(path)
    assert found is not None, f"expected to find {path!r}"
    return found


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
    spans = sorted({(iattr(s, "loNote"), iattr(s, "hiNote"))
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
    vr = sorted({(iattr(s, "loVel"), iattr(s, "hiVel"))
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
    assert fattr(delay_el, "stereoOffset") != 0.0


def test_plain_delay_type_has_zero_stereo_offset():
    root = parse(make_meta(reverb_type="Delay"))
    delay_el = [e for e in root.findall(".//effect") if e.get("type") == "delay"][0]
    assert fattr(delay_el, "stereoOffset") == 0.0


# ---------------------------------------------------------------------------
# Gap 3: measured pan_dly_stereo_offset_s
# ---------------------------------------------------------------------------

def test_pan_dly_stereo_offset_uses_measured_calibration_value():
    """With a measured pan_dly_stereo_offset_s in calibration.json, that
    value (not the old PAN_DLY_STEREO_OFFSET_FRAC formula) drives
    stereoOffset -- verified directly via _pan_dly_stereo_offset rather than
    through a full preset build, so the assertion is exact, not just
    "nonzero"."""
    cal_with_measurement = {"pan_dly_stereo_offset_s": 0.035}
    got = ep._pan_dly_stereo_offset(cal_with_measurement, delay_time_s=0.5)
    assert got == pytest.approx(0.035)


def test_pan_dly_stereo_offset_falls_back_without_measurement():
    """No measured value (e.g. the synthetic CAL fixture, which predates
    this measurement) -- falls back to the old reasoned proportion, exactly
    as before this task."""
    cal_without = {}
    got = ep._pan_dly_stereo_offset(cal_without, delay_time_s=0.2)
    expected = min(ep.PAN_DLY_STEREO_OFFSET_MAX_S, ep.PAN_DLY_STEREO_OFFSET_FRAC * 0.2)
    assert got == pytest.approx(expected)


def test_pan_dly_stereo_offset_never_approaches_delay_time():
    """The original bug this whole mapping exists to avoid repeating: a
    stereoOffset at or beyond delayTime inverts which channel leads. Even a
    generously large measured value must be capped well inside a SHORT
    delayTime (regression guard for PAN_DLY_STEREO_OFFSET_MAX_FRAC_OF_DELAY)."""
    cal_large = {"pan_dly_stereo_offset_s": 0.5}   # implausibly large on purpose
    short_delay = 0.03
    got = ep._pan_dly_stereo_offset(cal_large, delay_time_s=short_delay)
    assert got <= 0.5 * short_delay + 1e-9
    assert got < short_delay, "stereoOffset must never reach, let alone exceed, delayTime"


# ---------------------------------------------------------------------------
# AC 5: chorus modRate from calibration table, including the raw=24 gap
# ---------------------------------------------------------------------------

def test_chorus_effect_present_with_calibrated_mod_rate():
    root = parse(make_meta(chorus_rate=50))
    ch = [e for e in root.findall(".//effect") if e.get("type") == "chorus"][0]
    # raw rate 50 sits between the 32 (0.9) and 96 (2.2734) calibration
    # points in CAL -- interpolated, so strictly between them.
    assert 0.9 < fattr(ch, "modRate") < 2.2734


def test_interp_table_matches_hand_computed_value():
    # Between two adjacent points: raw=100 sits between 96 (2.2734) and
    # 104 (2.9416) in CAL -- t = (100-96)/(104-96) = 0.5
    expected = 2.2734 + 0.5 * (2.9416 - 2.2734)
    got = ep.interp_table(CAL["chorus_rate_hz"], 100)
    assert got == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Gap 2: chorus depth (chorus_depth_norm) -- the table this task exists to
# not repeat the "shipped inverted, no test caught it" mistake with.
# ---------------------------------------------------------------------------

def test_chorus_mod_depth_uses_calibrated_curve_when_present():
    """A deliberately NON-linear chorus_depth_norm (unlike CAL's own
    0.0/1.0-only fixture, whose 2-point interpolation happens to be
    indistinguishable from the proportional fallback) must actually change
    the result relative to the flat proportional mapping -- proving the
    measured curve, not just amount(), is what's driving modDepth."""
    cal = dict(CAL)
    cal["chorus_depth_norm"] = {"0": 0.0, "64": 0.9, "127": 1.0}
    meta = make_meta()
    meta["effects"]["chorus"]["depth"] = 64
    root = parse(meta, cal=cal)
    ch = [e for e in root.findall(".//effect") if e.get("type") == "chorus"][0]
    got = fattr(ch, "modDepth")
    expected_calibrated = 0.9 * ep.CHORUS_MAX_MOD_DEPTH
    expected_proportional = ep.amount(64, ep.CHORUS_MAX_MOD_DEPTH)
    assert got == pytest.approx(expected_calibrated, abs=1e-4)
    assert got != pytest.approx(expected_proportional, abs=1e-4)


def test_chorus_mod_depth_falls_back_to_proportional_without_table():
    """No chorus_depth_norm at all (e.g. a calibration.json predating this
    measurement) -- falls back to the old flat proportional mapping, not a
    crash or a fabricated curve."""
    cal_without = {k: v for k, v in CAL.items() if k != "chorus_depth_norm"}
    got = ep._chorus_mod_depth(cal_without, 64)
    assert got == pytest.approx(ep.amount(64, ep.CHORUS_MAX_MOD_DEPTH))


def test_chorus_mod_depth_zero_at_zero_raw_depth():
    """raw depth=0 must produce (near-)zero modDepth regardless of whether
    a measured curve is present -- the exact property the FIRST shipped
    table violated (raw=0 mapped to MAXIMUM depth)."""
    cal = {"chorus_depth_norm": {"0": 0.1335, "64": 0.519, "127": 1.0}}
    got = ep._chorus_mod_depth(cal, 0)
    assert got < 0.1, f"raw depth=0 should be near-zero modDepth, got {got}"


def test_chorus_mod_depth_monotonic_across_measured_table():
    """End-to-end sanity: modDepth must not decrease as raw depth rises,
    for the real measured curve (if calib/calibration.json exists)."""
    if REAL_CAL is None or "chorus_depth_norm" not in REAL_CAL:
        pytest.skip("no real chorus_depth_norm to check")
    values = [ep._chorus_mod_depth(REAL_CAL, raw) for raw in (0, 16, 32, 48, 64, 80, 96, 112, 127)]
    for a, b in zip(values, values[1:]):
        assert b >= a - 1e-9, f"modDepth decreased across the measured sweep: {values}"


@pytest.mark.skipif(REAL_CAL is None, reason="calib/calibration.json not present")
def test_chorus_rate_24_gap_does_not_yield_zero_hz():
    """calib/calibration.json deliberately omits the "24" key (see its
    module docstring in analyze_calibration.py) because that raw setting's
    measurement was dominated by a non-chorus artifact. It is an
    INTERPOLATION GAP, not a zero -- table.get("24", 0) would silently
    produce a 0 Hz chorus, which this must not do."""
    assert REAL_CAL is not None  # narrows for Pyright; guaranteed by skipif above
    assert "24" not in REAL_CAL["chorus_rate_hz"]
    rate = ep.interp_table(REAL_CAL["chorus_rate_hz"], 24)
    assert rate > 0.0
    # Should land between the real neighbouring measured points. Read those
    # from the table rather than hardcoding them: the calibration is a
    # regenerated measurement (it changed materially when the octave/time-
    # stretch renderer bug was fixed), so pinning literals here would make
    # this test fail on a legitimate recalibration rather than on a real
    # regression in the gap-interpolation behaviour it exists to guard.
    lo = REAL_CAL["chorus_rate_hz"]["16"]
    hi = REAL_CAL["chorus_rate_hz"]["32"]
    assert min(lo, hi) <= rate <= max(lo, hi), (
        f"interpolated {rate} Hz at the raw=24 gap is outside its measured "
        f"neighbours ({lo}, {hi})"
    )


def test_chorus_rate_24_gap_end_to_end_through_dspreset():
    """Same trap, exercised through the full pipeline: a patch whose raw
    chorus rate is exactly 24 must not emit modRate="0" (or anything close
    to it) in the generated XML."""
    root = parse(make_meta(chorus_rate=24), cal=CAL)
    ch = [e for e in root.findall(".//effect") if e.get("type") == "chorus"][0]
    mod_rate = fattr(ch, "modRate")
    assert mod_rate > 0.05, f"chorus rate=24 gap produced near-zero modRate: {mod_rate}"


@pytest.mark.skipif(REAL_CAL is None, reason="calib/calibration.json not present")
def test_reverb_rt60_hall1_dip_is_preserved_not_smoothed():
    """A genuine measured RT60 dip must survive interpolation, not be
    smoothed into a monotonic curve.

    Which reverb type exhibits a dip is a property of the measurement, not a
    fixed fact: the original table had one in Hall1 at raw 16->32, but that
    dip vanished when the calibration was regenerated after the octave/time-
    stretch renderer fix. So locate a real dip in the current data rather
    than assuming a specific one, and skip if the corrected measurements are
    genuinely monotonic everywhere (a legitimate outcome, not a failure)."""
    assert REAL_CAL is not None  # narrows for Pyright; guaranteed by skipif above

    found = None
    for rtype, table in sorted(REAL_CAL["reverb_rt60"].items()):
        pts = sorted((int(k), v) for k, v in table.items() if v is not None)
        for (k0, v0), (k1, v1) in zip(pts, pts[1:]):
            if v1 < v0:
                found = (rtype, k0, v0, k1, v1)
                break
        if found:
            break

    if found is None:
        pytest.skip("no measured RT60 dip in the current calibration -- "
                    "nothing for interpolation to smooth away")

    rtype, k0, v0, k1, v1 = found
    mid = ep.interp_table(REAL_CAL["reverb_rt60"][rtype], (k0 + k1) // 2)
    assert v1 <= mid <= v0, (
        f"type {rtype}: interpolation at the midpoint of the measured dip "
        f"({k0}->{k1}, {v0}->{v1}s) gave {mid}s, outside the two real points"
    )
    assert mid < v0, "the dip must survive interpolation, not be smoothed away"


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
    lfo = find1(root, ".//lfo")
    bindings = lfo.findall("binding")
    assert len(bindings) == 1
    b = bindings[0]
    assert b.get("parameter") == "AMP_VOLUME"
    # Verified against the official DecentSampler developer guide's
    # Appendix B: Group Volume is type="amp" level="group", requires a
    # groupIndex identifying which <group> (there's only ever one here,
    # index 0). "voice" is never a valid `level` value at all.
    assert b.get("type") == "amp"
    assert b.get("level") == "group"
    assert b.get("groupIndex") == "0"
    assert b.get("modBehavior") == "modulate"
    # translationOutputMin/Max carry the actual per-target depth (see
    # _lfo_binding_range's docstring) -- <binding> itself has no modAmount
    # attribute at all.
    assert b.get("modAmount") is None
    assert fattr(b, "translationOutputMin") < 0 < fattr(b, "translationOutputMax")


def test_modulators_is_top_level_not_nested_in_group():
    """The official developer guide states <modulators> "lives below the
    top-level <DecentSampler> element" -- it is a SIBLING of <groups> and
    <effects>, not something nested inside an individual <group>. Getting
    this wrong means real DecentSampler doesn't recognize the block as a
    modulator container at all and the LFO is silently dropped."""
    root = parse(make_meta(lfo1_stripped=True))
    modulators = root.find("modulators")
    assert modulators is not None, "<modulators> must exist as a top-level element"
    assert modulators in list(root), "<modulators> must be a direct child of <DecentSampler>"
    group = find1(root, ".//group")
    assert group.find("modulators") is None, "<modulators> must NOT be nested inside <group>"


def test_tvf_only_depth_emits_no_lfo_no_valid_target():
    """TVF has no valid DecentSampler binding target in this pipeline's
    output: FX_FILTER_FREQUENCY is real but requires an actual filter
    effect in the chain, and build_effects() never emits one (the JV's
    TVF filter is already baked into the render). A patch whose ONLY
    nonzero LFO depth is tvf must therefore emit no <lfo> at all, rather
    than a binding that looks plausible but targets the wrong effect."""
    meta = make_meta(lfo1_stripped=True)
    meta["lfo1"]["tva"] = 0
    meta["lfo1"]["pitch"] = 0
    meta["lfo1"]["tvf"] = 40
    root = parse(meta)
    assert root.find(".//lfo") is None


def test_build_lfo_modulator_tvf_alone_returns_none():
    lfo = {"stripped": True, "form": "SIN", "rate": 60, "delay": 0, "sync": 1,
           "pitch": 0, "tvf": 63, "tva": 0}
    assert ep.build_lfo_modulator(lfo) is None


def test_stripped_lfo_shape_tri_maps_to_sine():
    root = parse(make_meta(lfo1_stripped=True, lfo_form="TRI"))
    assert find1(root, ".//lfo").get("shape") == "sine"


@pytest.mark.parametrize("form", ["SIN", "SAW", "SQU"])
def test_stripped_lfo_shape_maps_correctly(form):
    root = parse(make_meta(lfo1_stripped=True, lfo_form=form))
    expected = {"SIN": "sine", "SAW": "saw", "SQU": "square"}[form]
    assert find1(root, ".//lfo").get("shape") == expected


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
    # GROUP_TUNING is real (verified against Appendix B) but is type="amp"
    # like AMP_VOLUME, NOT type="generic" -- an earlier version of this
    # code got the parameter name right but the type wrong.
    for b in m["bindings"]:
        assert b["type"] == "amp"
        assert b["level"] == "group"
        assert b["groupIndex"] == ep.GROUP_INDEX


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
              and iattr(s, "loVel") == 1][0]
    assert sample.get("loopCrossfade") == "137"


def test_loop_attributes_present_for_looped_zone():
    root = parse(make_meta())
    s = find1(root, ".//sample")
    assert s.get("loopEnabled") == "1"
    assert iattr(s, "loopStart") == 48000
    assert iattr(s, "loopEnd") == 158000
    assert iattr(s, "loopCrossfade") <= 2000


def test_sample_release_attribute_matches_measured_zone_release():
    """DecentSampler's per-sample envelope release attribute is `release`
    (alongside attack/decay/sustain), confirmed against the official
    developer guide -- NOT `ampegRelease`, which isn't a real attribute at
    all and would be silently ignored, discarding every measured release
    time from postprocess.py's measure_release without any error."""
    meta = make_meta()
    for z in meta["zones"]:
        if z["key"] == 24 and z["layer"] == 1:
            z["release"] = 1.234
    root = parse(meta)
    sample = [s for s in root.findall(".//sample") if s.get("rootNote") == "24"
              and iattr(s, "loVel") == 1][0]
    assert sample.get("release") == "1.234"
    assert sample.get("ampegRelease") is None


def test_non_looped_zone_has_no_loop_attributes():
    meta = make_meta()
    for z in meta["zones"]:
        if z["key"] == 24 and z["layer"] == 1:
            z["kind"] = "decaying"
            z["loop"] = {"enabled": False}
    root = parse(meta)
    sample = [s for s in root.findall(".//sample") if s.get("rootNote") == "24"
              and iattr(s, "loVel") == 1][0]
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
    paths = [sattr(s, "path") for s in samples]
    assert not any("does_not_exist" in p for p in paths)


def test_error_zone_produces_no_sample():
    meta = make_meta(zone_overrides={(60, 1): {"kind": "error", "file": "corrupt.wav"}})
    root = parse(meta)
    paths = [sattr(s, "path") for s in root.findall(".//sample")]
    assert not any("corrupt" in p for p in paths)


def test_missing_zone_produces_no_sfz_region():
    meta = make_meta(zone_overrides={(24, 2): {"kind": "missing", "file": "does_not_exist.wav"}})
    sfz = ep.build_sfz(meta, "x")
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
    root_notes = {iattr(s, "rootNote") for s in root.findall(".//sample")}
    assert 24 not in root_notes
    spans = sorted({(iattr(s, "loNote"), iattr(s, "hiNote"))
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
    vr = sorted((iattr(s, "loVel"), iattr(s, "hiVel")) for s in samples)
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
    types = [e.get("type") for e in find1(root, "effects")]
    assert types.index("chorus") < types.index("reverb")


def test_chorus_parallel_after_reverb_when_output_is_mix():
    # Design doc's mapping table: chorusoutput=Reverb -> chorus before
    # reverb (chorus feeds it); chorusoutput=Mix -> "chorus parallel /
    # after" -- i.e. reverb is NOT fed by chorus here, so reverb comes
    # first and chorus is appended after it.
    root = parse(make_meta(chorus_output="Mix"))
    types = [e.get("type") for e in find1(root, "effects")]
    assert types.index("chorus") > types.index("reverb")


# ---------------------------------------------------------------------------
# AC 8: SFZ region count, key/vel ranges, loop opcodes, ampeg_release
# ---------------------------------------------------------------------------

def test_sfz_has_one_region_per_zone():
    sfz = ep.build_sfz(make_meta(), "x")
    assert sfz.count("<region>") == 25 * 3


def test_sfz_region_has_matching_key_vel_ranges_and_release():
    sfz = ep.build_sfz(make_meta(), "x")
    assert "lokey=0" in sfz  # lowest key's span starts at 0
    assert "hikey=127" in sfz  # highest key's span ends at 127
    assert "lovel=1" in sfz
    assert "hivel=127" in sfz
    assert "ampeg_release=" in sfz


def test_sfz_looped_zone_has_loop_opcodes():
    sfz = ep.build_sfz(make_meta(), "x")
    assert "loop_mode=loop_continuous" in sfz
    assert "loop_start=48000 loop_end=158000" in sfz


def test_sfz_non_looped_zone_has_no_loop_points_only_no_loop_mode():
    meta = make_meta()
    for z in meta["zones"]:
        z["kind"] = "decaying"
        z["loop"] = {"enabled": False}
    sfz = ep.build_sfz(meta, "x")
    assert "loop_mode=no_loop" in sfz
    assert "loop_mode=loop_continuous" not in sfz
    assert "loop_start=" not in sfz


# ---------------------------------------------------------------------------
# AC 9 (structural half): sample paths reference the zone's actual file
# ---------------------------------------------------------------------------

def test_sample_paths_use_sample_prefix_and_zone_file():
    # Real layout is flat: jv_sampler + postprocess.py write FLACs
    # directly into <out_dir>/<patchdir>/, with no "Samples/" wrapper (see
    # main()'s sample_prefix, which is just pdir.name).
    root = parse(make_meta(), prefix="000_TestPatch")
    sample = find1(root, ".//sample")
    assert sattr(sample, "path").startswith("000_TestPatch/")
    assert sattr(sample, "path").endswith(".flac")


def test_realistic_note_names_including_sharp_survive_unmangled():
    """Real rendered filenames are note-name based (A1_v1.flac,
    D#4_v2.flac, C1_v3.flac -- see src/jv_sampler.cpp's note_name()), not
    a simplified n<key> form. In particular, sharp keys produce a literal
    '#' in the filename -- confirm it survives XML attribute
    serialization/parsing and SFZ text output unmangled (not escaped,
    not URL-encoded, not dropped)."""
    meta = make_meta()
    # key=27 -> D#1 (MIDI 27 % 12 == 3 -> "D#", 27 // 12 - 1 == 1)
    sharp_zone = next(z for z in meta["zones"] if z["key"] == 27 and z["layer"] == 1)
    assert sharp_zone["file"] == "D#1_v1.flac"  # fixture sanity check

    root = parse(meta, prefix="000_TestPatch")
    sample = [s for s in root.findall(".//sample") if s.get("rootNote") == "27"
              and iattr(s, "loVel") == 1][0]
    assert sample.get("path") == "000_TestPatch/D#1_v1.flac"

    sfz = ep.build_sfz(meta, "000_TestPatch")
    assert "sample=D#1_v1.flac" in sfz


def test_end_to_end_emitted_sample_paths_exist_on_disk(tmp_path):
    """Integration-level check of AC 9: write real FLAC-named (empty)
    files at the location `sample_prefix` implies, mirroring what main()
    ACTUALLY assembles -- a flat layout with no "Samples/" wrapper: the
    .dspreset lives directly in `library_dir` and its samples live in
    `library_dir/<patchdir>/` (patch.json's own directory), not under any
    "Samples/" subfolder. An earlier version of this test (and of
    main()'s sample_prefix) assumed a "Samples/<patchdir>/" wrapper that
    doesn't exist anywhere on disk -- this fixture would have failed to
    catch that regression because it built its OWN matching-but-wrong
    "Samples/" directory instead of the real layout. Verify every emitted
    <sample path> resolves relative to the preset's own directory, using
    realistic note-name filenames (including a sharp), while a
    deliberately-missing zone must never appear."""
    library_dir = tmp_path  # where the .dspreset itself would be written
    patch_name = "000_TestPatch"
    patch_dir = library_dir / patch_name  # NOT library_dir / "Samples" / patch_name
    patch_dir.mkdir(parents=True)

    meta = make_meta(zone_overrides={(24, 2): {"kind": "missing", "file": "ghost.flac"}})
    for z in meta["zones"]:
        if z.get("kind") not in ep.FAILURE_KINDS:
            (patch_dir / z["file"]).write_bytes(b"")  # placeholder, existence is what matters

    sample_prefix = patch_name  # matches main()'s pdir.name, no "Samples/" prefix
    xml_text = ep.build_dspreset(meta, CAL, sample_prefix)
    root = ET.fromstring(xml_text)

    missing = [s.get("path") for s in root.findall(".//sample")
               if not (library_dir / s.get("path")).exists()]
    assert missing == []
    assert not any("ghost" in (s.get("path") or "") for s in root.findall(".//sample"))
    # A sharp-named sample must resolve too, not just plain ones.
    sharp_paths = [s.get("path") for s in root.findall(".//sample") if "#" in (s.get("path") or "")]
    assert sharp_paths, "fixture sanity check: expected at least one sharp-named sample"
    assert all((library_dir / p).exists() for p in sharp_paths)


def test_main_uses_flat_sample_prefix_matching_real_layout(tmp_path, monkeypatch, capsys):
    """End-to-end main(): build a patch directory in the REAL flat layout
    jv_sampler/postprocess.py actually produce (FLACs alongside
    patch.json, no "Samples/" wrapper), run main() against it, and verify
    the emitted .dspreset's <sample path> attributes resolve relative to
    the library directory main() wrote the preset into."""
    library_dir = tmp_path / "JV-880 Internal"
    patch_dir = library_dir / "000_Test Patch"
    patch_dir.mkdir(parents=True)

    meta = make_meta()
    meta["name"] = "Test Patch"
    for z in meta["zones"]:
        (patch_dir / z["file"]).write_bytes(b"")
    (patch_dir / "patch.json").write_text(json.dumps(meta))

    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps(CAL))

    monkeypatch.setattr(sys, "argv", ["emit_presets.py", str(library_dir), str(cal_path)])
    ep.main()

    dspreset_files = list(library_dir.glob("*.dspreset"))
    assert len(dspreset_files) == 1
    root = ET.parse(dspreset_files[0]).getroot()
    samples = root.findall(".//sample")
    assert samples
    missing = [s.get("path") for s in samples if not (library_dir / s.get("path")).exists()]
    assert missing == []


# ---------------------------------------------------------------------------
# Convolution IR path (reverb types 0-5)
# ---------------------------------------------------------------------------

def _cal_with_ir():
    cal = dict(CAL)
    cal["reverb_ir"] = {
        str(t): {str(step): f"ir/reverb_t{t}_time_{step:03d}.wav"
                 for step in (0, 16, 32, 48, 64, 80, 96, 112, 127)}
        for t in range(6)
    }
    return cal


def test_reverb_types_emit_convolution_when_ir_bank_present():
    """With an IR bank, true reverbs use the captured JV algorithm."""
    root = parse(make_meta(reverb_type="Hall1"), _cal_with_ir())
    effects = {e.get("type") for e in root.findall(".//effect")}
    assert "convolution" in effects
    assert "reverb" not in effects, "should not emit both a modelled and a captured reverb"
    conv = [e for e in root.findall(".//effect") if e.get("type") == "convolution"][0]
    assert conv.get("irFile", "").endswith(".wav")
    assert 0.0 <= float(conv.get("mix")) <= 1.0


def test_delay_types_still_emit_delay_even_with_ir_bank():
    """Delay/Pan-Dly are genuinely time+feedback+pan; an IR is the wrong model."""
    root = parse(make_meta(reverb_type="Pan-Dly"), _cal_with_ir())
    effects = {e.get("type") for e in root.findall(".//effect")}
    assert "delay" in effects
    assert "convolution" not in effects


def test_ir_snaps_to_nearest_captured_time_step():
    cal = _cal_with_ir()
    assert ep._nearest_ir(cal, 4, 90)[0].endswith("_096.wav")   # 90 -> 96
    assert ep._nearest_ir(cal, 4, 50)[0].endswith("_048.wav")   # 50 -> 48
    assert ep._nearest_ir(cal, 4, 0)[0].endswith("_000.wav")
    assert all(ok for ok in (ep._nearest_ir(cal, 4, r)[1] for r in (0, 50, 90)))


def test_silent_ir_step_zeroes_wet_instead_of_eating_the_dry_signal():
    """convolution `mix` is a BLEND, so mix>0 against an all-zero IR quietly
    attenuates the dry signal while adding nothing. The JV really does emit no
    wet signal at reverbtime 0, so the correct emission is mix=0 -- and the
    irFile should point at an AUDIBLE step so the knob still works."""
    cal = _cal_with_ir()
    cal["reverb_ir_silent"] = {"4": [0]}

    ir, wet_ok = ep._nearest_ir(cal, 4, 0)
    assert wet_ok is False
    assert not ir.endswith("_000.wav"), "should redirect away from the silent capture"

    meta = make_meta(reverb_type="Hall1")
    meta["effects"]["reverb"].update(time=0, level=127)
    conv = [e for e in parse(meta, cal).findall(".//effect")
            if e.get("type") == "convolution"][0]
    assert float(conv.get("mix") or 0) == 0.0
    assert not (conv.get("irFile") or "").endswith("_000.wav")


def test_audible_step_is_unaffected_by_the_silent_manifest():
    cal = _cal_with_ir()
    cal["reverb_ir_silent"] = {"4": [0]}
    meta = make_meta(reverb_type="Hall1")
    meta["effects"]["reverb"].update(time=96, level=127)
    conv = [e for e in parse(meta, cal).findall(".//effect")
            if e.get("type") == "convolution"][0]
    assert float(conv.get("mix") or 0) > 0.0
    assert (conv.get("irFile") or "").endswith("_096.wav")


def test_emitted_library_stages_every_ir_it_references(tmp_path):
    """A preset's irFile is relative to the preset, so the WAV must physically
    land in the library -- otherwise DecentSampler renders no reverb at all and
    it reads as 'the reverb is too quiet'. Shipped broken twice already."""
    calib_root = tmp_path / "calib"
    (calib_root / "ir_synth").mkdir(parents=True)
    (calib_root / "ir_synth" / "reverb_t4_time_096.wav").write_bytes(b"RIFFfake")

    out_root = tmp_path / "lib"
    out_root.mkdir()
    (out_root / "p.dspreset").write_text(
        '<DecentSampler><effects>'
        '<effect type="convolution" irFile="ir_synth/reverb_t4_time_096.wav" mix="0.3"/>'
        '</effects></DecentSampler>')

    assert ep._stage_referenced_irs(out_root, calib_root) == 1
    assert (out_root / "ir_synth" / "reverb_t4_time_096.wav").exists()


def test_staging_fails_loudly_when_a_referenced_ir_is_absent(tmp_path):
    out_root = tmp_path / "lib"
    out_root.mkdir()
    (out_root / "p.dspreset").write_text(
        '<DecentSampler><effects>'
        '<effect type="convolution" irFile="ir_synth/nope.wav" mix="0.3"/>'
        '</effects></DecentSampler>')
    with pytest.raises(SystemExit):
        ep._stage_referenced_irs(out_root, tmp_path / "calib")


def test_missing_ir_bank_falls_back_to_parametric_reverb():
    """A missing IR must degrade to an audible reverb, never to silence."""
    root = parse(make_meta(reverb_type="Hall1"), CAL)   # CAL has no reverb_ir
    effects = {e.get("type") for e in root.findall(".//effect")}
    assert "reverb" in effects and "convolution" not in effects
