#!/usr/bin/env python3
"""Measure the JV's wet/dry reverb ratio PER REVERB TYPE, and fit it.

emit_presets currently derives convolution `mix` from a single global fit of
wet/dry ratio against (reverb level x average active send), across all six
convolution types at once. That fit sits at r = 0.77, and tools/audit_fidelity
shows the cost: the wet/dry ratio lands outside +-6 dB on about a third of
patches, systematically too dry, and it does so with the chorus excluded from
both sides -- so it is the reverb path, not the chorus approximation.

A single fit cannot be right in principle. Room1 and Hall2 are different
algorithms with different output gains, so the same level-and-send setting
means a different amount of reverb in each. This measures each type
separately, over as many patches as the internal bank actually provides,
and writes per-type coefficients to calibration.json.

Usage:
    python3 tools/calibrate_reverb_ratio.py --roms ROMS --library LIB \
        [--per-type N] [--calibration calib/calibration.json] [--dry-run]
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 64000
BANK_BASE = {"A": 0, "B": 64, "Internal": 128}
REVERB_NAMES = ["Room1", "Room2", "Stage1", "Stage2", "Hall1", "Hall2"]

# A patch whose reverb-only residual is this far under its dry signal carries
# no usable ratio -- the measurement would be fitting noise.
MIN_USABLE_RATIO = 0.02


def measure(wave_inject, roms, idx, tmp):
    out = Path(tmp) / str(idx)
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(wave_inject), "groundtruth", "--roms", str(roms),
                           "--patch-index", str(idx), "--out-dir", str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if not line:
        return None
    eff = json.loads(line[-1])
    if eff["reverb_type"] > 5:
        return None            # Delay / Pan-Dly are parametric, not convolution
    wet, sr = sf.read(str(out / "groundtruth_wet.wav"), always_2d=True)
    dry, _ = sf.read(str(out / "groundtruth_dry.wav"), always_2d=True)
    n = min(len(wet), len(dry))
    w, d = wet[:n].mean(axis=1), dry[:n].mean(axis=1)
    if np.sqrt(np.mean(d ** 2)) < 1e-5:
        return None
    # Remove the dry by least squares rather than plain subtraction: the two
    # renders' dry levels differ slightly, and a raw difference folds that
    # error straight into the ratio being fitted.
    a = float(np.dot(w, d) / max(np.dot(d, d), 1e-12))
    resid = w - a * d
    R = float(np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(d ** 2)))

    active = [s for s, lv in zip(eff["reverb_send"], eff["tone_level"]) if lv > 0]
    send = (sum(active) / len(active)) if active else 0.0
    pred = (eff["reverb_level"] / 127.0) * (send / 127.0)
    return {"index": idx, "name": eff["name"], "type": eff["reverb_type"],
            "level": eff["reverb_level"], "send": round(send), "pred": pred, "R": R}


def fit(rows):
    """Least-squares R = slope*pred + intercept, with the correlation."""
    p = np.array([r["pred"] for r in rows])
    R = np.array([r["R"] for r in rows])
    if len(rows) < 3 or p.std() < 1e-6:
        return None
    slope, intercept = np.polyfit(p, R, 1)
    corr = float(np.corrcoef(p, R)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept),
            "r": corr, "n": len(rows),
            "r_min": float(R.min()), "r_max": float(R.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--wave-inject", default="build/wave_inject")
    ap.add_argument("--tmp", default="/tmp/reverb_ratio")
    ap.add_argument("--per-type", type=int, default=14)
    ap.add_argument("--calibration", default="calib/calibration.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    library = Path(args.library)
    candidates = []
    for f in sorted(library.glob("*/patch.json")):
        m = json.loads(f.read_text())
        if str(m["bank"]) not in BANK_BASE:
            continue
        rv = m["effects"]["reverb"]
        if rv["type"] in ("Delay", "Pan-Dly"):
            continue
        candidates.append((BANK_BASE[str(m["bank"])] + int(m["index"]),
                           REVERB_NAMES.index(rv["type"])))

    # Spread the budget evenly over types, and within a type over the whole
    # level/send range rather than whatever happens to come first in the bank.
    by_type = defaultdict(list)
    for idx, t in candidates:
        by_type[t].append(idx)
    selected = []
    for t, idxs in sorted(by_type.items()):
        if len(idxs) > args.per_type:
            step = len(idxs) / args.per_type
            idxs = [idxs[int(i * step)] for i in range(args.per_type)]
        selected += [(i, t) for i in idxs]
    print(f"measuring {len(selected)} patches across {len(by_type)} types\n")

    rows = []
    for n, (idx, _) in enumerate(selected, 1):
        r = measure(Path(args.wave_inject), args.roms, idx, args.tmp)
        if r is None:
            continue
        if r["R"] < MIN_USABLE_RATIO:
            continue
        rows.append(r)
        if n % 10 == 0 or n == len(selected):
            print(f"  [{n}/{len(selected)}] {len(rows)} usable")

    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)

    print(f"\n{'type':<9}{'n':>4}{'slope':>9}{'intercept':>11}{'r':>8}{'R range':>16}")
    fits = {}
    for t in sorted(by):
        f = fit(by[t])
        if f is None:
            print(f"{REVERB_NAMES[t]:<9}{len(by[t]):>4}   (too few points to fit)")
            continue
        fits[str(t)] = {k: f[k] for k in ("slope", "intercept")}
        print(f"{REVERB_NAMES[t]:<9}{f['n']:>4}{f['slope']:>9.3f}{f['intercept']:>11.3f}"
              f"{f['r']:>8.3f}   {f['r_min']:.2f}-{f['r_max']:.2f}")

    g = fit(rows)
    print(f"\n{'GLOBAL':<9}{g['n']:>4}{g['slope']:>9.3f}{g['intercept']:>11.3f}{g['r']:>8.3f}")

    # Does per-type actually beat global on this data? If it does not, saying
    # so is the useful result -- shipping a more complicated model that fits
    # no better would just be a more confident way to be wrong.
    err_g, err_t = [], []
    for r in rows:
        pg = g["slope"] * r["pred"] + g["intercept"]
        f = fits.get(str(r["type"]))
        pt = (f["slope"] * r["pred"] + f["intercept"]) if f else pg
        for pred, acc in ((pg, err_g), (pt, err_t)):
            acc.append(20 * np.log10(max(pred, 1e-3) / max(r["R"], 1e-3)))
    err_g, err_t = np.abs(np.array(err_g)), np.abs(np.array(err_t))
    print(f"\nmean |error| vs measured ratio:  global {err_g.mean():.2f} dB   "
          f"per-type {err_t.mean():.2f} dB")
    print(f"within 6 dB:                     global {(err_g<6).mean()*100:.0f}%       "
          f"per-type {(err_t<6).mean()*100:.0f}%")

    if args.dry_run:
        print("\n--dry-run: calibration.json not written")
        return
    if err_t.mean() >= err_g.mean():
        print("\nper-type does NOT improve on global; leaving calibration.json alone")
        return

    path = Path(args.calibration)
    cal = json.loads(path.read_text())
    cal["reverb_ratio_fit"] = fits
    cal["reverb_ratio_fit_global"] = {"slope": g["slope"], "intercept": g["intercept"]}
    path.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"\nwrote per-type fits to {path}")
    Path("/tmp/reverb_ratio_rows.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
