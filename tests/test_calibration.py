"""Acceptance tests for calib/calibration.json (Task 4: effect calibration).

Skips (does not fail) if calibration.json is absent: it's a generated
artifact (~99 WAV renders + analysis), not something committed for every
checkout. Regenerate with:

    cmake --build build --target calibrate
    ./build/calibrate --roms <dir> --out calib
    python3 tools/analyze_calibration.py calib

Tolerances below are deliberately documented, not tuned to force a pass:
this file must never be adjusted to paper over a genuinely bad measurement
(see tools/analyze_calibration.py's module docstring for the investigation
that shaped the chorus-rate measurement approach and its one intentionally
omitted, unreliable raw setting).
"""
import json
from pathlib import Path

import pytest

CALIB_PATH = Path(__file__).resolve().parents[1] / "calib" / "calibration.json"

# "Small tolerance" per the task spec, interpreted as: a later raw setting's
# measured value may sit up to this many Hz (chorus) / seconds (RT60) BELOW
# the value at the previous raw setting before it counts as a real monotonic
# violation, rather than requiring bit-for-bit non-decreasing values (real
# acoustic measurements have some settle/estimation noise even when the
# underlying physical trend is genuinely monotonic).
CHORUS_RATE_TOL_HZ = 0.15
RT60_TOL_SEC = 0.25


def _load():
    if not CALIB_PATH.exists():
        pytest.skip(f"{CALIB_PATH} not found -- run tools/calibrate.cpp + "
                     f"tools/analyze_calibration.py to generate it")
    return json.loads(CALIB_PATH.read_text())


def _sorted_present(table):
    """(raw:int, value) pairs sorted by raw, dropping any null/None entries
    -- an omitted or null entry means that raw setting's measurement was
    judged unreliable and intentionally not reported (see
    analyze_calibration.py); it must not be treated as a 0 or otherwise
    fabricated when checking trends across the rest of the sweep."""
    items = [(int(k), v) for k, v in table.items() if v is not None]
    items.sort(key=lambda kv: kv[0])
    return items


def test_all_five_tables_present_and_nonempty():
    data = _load()
    for key in ("chorus_rate_hz", "chorus_depth_norm", "chorus_mix",
                "reverb_rt60", "reverb_wet"):
        assert key in data, f"missing table: {key}"
        assert len(data[key]) > 0, f"table is empty: {key}"


def test_chorus_rate_monotonic_non_decreasing():
    data = _load()
    items = _sorted_present(data["chorus_rate_hz"])
    assert len(items) >= 2, "need at least 2 measured chorus-rate points"
    for i in range(1, len(items)):
        (raw_prev, v_prev), (raw_cur, v_cur) = items[i - 1], items[i]
        assert v_cur >= v_prev - CHORUS_RATE_TOL_HZ, (
            f"chorus rate dropped from {v_prev} Hz (raw={raw_prev}) to "
            f"{v_cur} Hz (raw={raw_cur}), beyond the {CHORUS_RATE_TOL_HZ} Hz tolerance"
        )


def test_chorus_rate_127_at_least_2x_rate_0():
    data = _load()
    items = _sorted_present(data["chorus_rate_hz"])
    lo_raw, lo_val = items[0]
    hi_raw, hi_val = items[-1]
    assert lo_val > 0, "chorus rate at the lowest measured raw setting must be > 0 Hz"
    assert hi_val >= 2.0 * lo_val, (
        f"chorus rate at raw={hi_raw} ({hi_val} Hz) is not at least 2x "
        f"the rate at raw={lo_raw} ({lo_val} Hz)"
    )


def test_reverb_rt60_monotonic_per_type():
    data = _load()
    rt60 = data["reverb_rt60"]
    for rtype in range(6):   # types 0-5 only: 6/7 are delays, excluded by design
        key = str(rtype)
        assert key in rt60, f"missing reverb_rt60 entry for type {rtype}"
        items = _sorted_present(rt60[key])
        assert len(items) >= 2, f"need at least 2 measured RT60 points for type {rtype}"
        for i in range(1, len(items)):
            (raw_prev, v_prev), (raw_cur, v_cur) = items[i - 1], items[i]
            assert v_cur >= v_prev - RT60_TOL_SEC, (
                f"type {rtype}: RT60 dropped from {v_prev}s (raw={raw_prev}) to "
                f"{v_cur}s (raw={raw_cur}), beyond the {RT60_TOL_SEC}s tolerance"
            )


def test_reverb_wet_max_greater_than_min():
    data = _load()
    items = _sorted_present(data["reverb_wet"])
    assert len(items) >= 2, "need at least 2 measured reverb_wet points"
    lo_val = min(v for _, v in items)
    hi_val = max(v for _, v in items)
    assert hi_val > lo_val, (
        f"reverb_wet does not increase across the sweep (min={lo_val}, max={hi_val})"
    )
