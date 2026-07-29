#!/usr/bin/env python3
"""Automated fidelity audit: the shipped presets against the plugin itself.

Every fidelity defect in this project so far was found by a human listening,
not by a test, and each one slipped through for the same structural reason --
the metric in place was blind in exactly the dimension that broke:

  * the IR bank shipped as pure digital silence, because validation measured
    in-memory arrays instead of the written files;
  * reverb came out 18 dB quiet, because envelope correlation is
    level-insensitive and could not see it;
  * the IRs carried up to 17 ms of injection latency, because nothing
    measured onset;
  * the dry signal came out several dB down, because nothing measured the
    dry path separately from the wet one.

So this audit deliberately measures SEPARATE, ORTHOGONAL quantities rather
than one aggregate score, and fails loudly per-metric. An aggregate is what
let a preset with correct decay shape and inaudible level pass as 0.95 "good".

For each patch it renders the plugin's own dry and full-effect output, builds
the reconstruction from the SHIPPED .dspreset exactly as ab_compare does, and
reports:

    level_db    total output level error, reconstruction vs plugin
    wetdry_db   wet/dry ratio error (how much reverb, independent of level)
    predelay_ms reverb onset error (is the reverb late?)
    decay_corr  decay envelope correlation (does it decay the same way?)
    tilt_db     spectral tilt error, HF/LF balance (is it dull or bright?)

Usage:
    python3 tools/audit_fidelity.py --roms ROMS --library LIB [--limit N]
"""
import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_compare as ab   # noqa: E402  (reuse the reconstruction, don't fork it)

SR = 64000
BANK_BASE = {"A": 0, "B": 64, "Internal": 128}

# Thresholds. Chosen from what has actually shipped broken, not from taste:
# the level defect was 18 dB, the pre-delay defect 15 ms.
TOL = {
    "level_db": 3.0,
    "wetdry_db": 6.0,
    "predelay_ms": 5.0,
    "decay_corr": 0.80,
    "tilt_db": 6.0,
}


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def db(a, b):
    return 20.0 * np.log10(max(a, 1e-12) / max(b, 1e-12))


def reverb_residual(wet, dry):
    """Reverb-only contribution, with the dry removed by least squares.

    A plain `wet - dry` difference is contaminated whenever the two renders'
    dry levels differ even slightly, which shows up as reverb arriving
    instantly and hid the pre-delay defect on the first attempt.
    """
    n = min(len(wet), len(dry))
    w, d = wet[:n].mean(axis=1), dry[:n].mean(axis=1)
    a = float(np.dot(w, d) / max(np.dot(d, d), 1e-12))
    return w - a * d, d


def onset_ms(x, sr, floor_db=-45.0, smooth=32):
    env = np.convolve(np.abs(x), np.ones(smooth) / smooth, mode="same")
    if not env.any():
        return None
    return float(np.argmax(env > env.max() * 10 ** (floor_db / 20)) / sr * 1000.0)


def decay_correlation(a, b, sr, hold_seconds=3.5):
    """Correlation of the two decay envelopes, measured AFTER note-off.

    Restricted to the release tail on purpose. The renderer holds every note
    for 3.5 s, so on a sustained patch the analysis window is almost entirely
    steady state: both envelopes are flat, the correlation is computed on
    noise, and the result is meaningless -- which is what it looked like, with
    sustained patches (Overdrive -0.244, Pan Pipe 0.116, E.Organ 2 0.173)
    scoring far below percussive ones that genuinely decay in-window. A reverb
    decay is only observable once the source stops driving it.
    """
    start = int(hold_seconds * sr)
    n = min(len(a), len(b))
    if n - start > sr * 0.2:
        a, b = a[start:n], b[start:n]
    n = min(len(a), len(b))
    k = np.ones(int(sr * 0.02)) / int(sr * 0.02)
    ea = 20 * np.log10(np.maximum(np.convolve(np.abs(a[:n]), k, mode="same"), 1e-9))
    eb = 20 * np.log10(np.maximum(np.convolve(np.abs(b[:n]), k, mode="same"), 1e-9))
    keep = (ea > ea.max() - 60) | (eb > eb.max() - 60)
    if keep.sum() < sr * 0.1:
        return float("nan")
    return float(np.corrcoef(ea[keep], eb[keep])[0, 1])


def spectral_tilt_db(x, sr):
    """Energy above 2 kHz relative to below, in dB -- a blunt brightness proxy."""
    mono = x.mean(axis=1) if x.ndim > 1 else x
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    lo = spec[(freqs > 100) & (freqs <= 2000)].sum()
    hi = spec[(freqs > 2000) & (freqs < 16000)].sum()
    return 10 * np.log10(max(hi, 1e-20) / max(lo, 1e-20))


def run_groundtruth(wave_inject, roms, idx, out_dir):
    """Reverb-only ground truth: the patch with reverb restored, chorus still
    zeroed. Lets the audit separate the reverb path -- where convolution is an
    exact model of what DecentSampler does -- from the chorus path, which is
    an approximation of undocumented internals. Without that split, an error
    in the approximate part is indistinguishable from an error in the exact
    part, and the whole reverb result becomes unfalsifiable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(wave_inject), "groundtruth", "--roms", str(roms),
                           "--patch-index", str(idx), "--out-dir", str(out_dir)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"groundtruth failed:\n{proc.stderr}")
    meta = json.loads([l for l in proc.stdout.splitlines() if l.startswith("{")][-1])
    wet, sr = sf.read(str(out_dir / "groundtruth_wet.wav"), always_2d=True)
    dry, _ = sf.read(str(out_dir / "groundtruth_dry.wav"), always_2d=True)
    assert sr == SR
    return wet, dry, meta


def audit_patch(wave_inject, roms, tmp, library, idx, stem, key, reverb_only=False):
    if reverb_only:
        full, dry, meta = run_groundtruth(wave_inject, roms, idx, Path(tmp) / f"rev_{stem}")
        meta.setdefault("reverb_type", -1)
    else:
        full, dry, meta = ab.run_abcompare(wave_inject, roms, idx, key, Path(tmp) / stem)
    fx = ab.preset_effects(library / f"{stem}.dspreset")
    recon, _ = ab.build_reconstruction(dry, fx, library, use_chorus=not reverb_only)

    # Group <volume> is part of what ships and directly sets the dry level, so
    # the audit must apply it -- omitting it would measure a preset nobody has.
    root = ET.parse(library / f"{stem}.dspreset").getroot()
    grp = root.find(".//group")
    if grp is not None and grp.get("volume"):
        vol = float(grp.get("volume"))
        if reverb_only:
            # The shipped volume compensates the chorus blend as well, so applying
            # it whole against a chorus-free reconstruction would double-count.
            cm = float(fx.get("chorus", {}).get("mix", 0.0))
            vol *= (1.0 - cm) ** 0.5
        recon = recon * vol

    n = min(len(full), len(recon))
    full, recon, dryn = full[:n], recon[:n], dry[:n]

    # A patch whose dry render is essentially silent at this key (out of its
    # sampled range, or a layer that does not speak) carries no information:
    # every ratio below divides by it and produces garbage rather than a
    # finding. The first run of this audit reported +240 dB on four such
    # patches, which is a measurement artefact, not a defect.
    if rms(dryn) < 1e-5:
        raise ValueError("dry render is silent at this key; nothing to compare")

    res_p, dry_mono = reverb_residual(full, dryn)
    res_r, _ = reverb_residual(recon, dryn)

    # Output level, as total RMS. An earlier version recovered each rendering's
    # direct component by least-squares projection onto the dry render, which
    # is a cleaner idea and a much worse measurement: on a heavily modulated
    # patch the full render decorrelates from the dry one, the projection
    # collapses toward zero, and the ratio of two near-zero numbers reported
    # +240 dB on four patches. Total RMS needs no correlation to hold, and is
    # what "sounds quieter than the plugin" actually refers to.
    a_p = rms(full.mean(axis=1))
    a_r = rms(recon.mean(axis=1))

    R_p, R_r = rms(res_p) / max(rms(dry_mono), 1e-12), rms(res_r) / max(rms(dry_mono), 1e-12)
    on_p, on_r = onset_ms(res_p, SR), onset_ms(res_r, SR)

    return {
        "name": meta["name"], "type": meta["reverb_type"],
        # Retained so the makeup gain can be fitted from measurement rather
        # than derived: the theoretical 1/(1-mix) overshoots badly, because a
        # chorus or early-reflection wet path is strongly CORRELATED with the
        # dry it was derived from, so a blend at mix removes far less direct
        # sound than (1-mix) implies.
        "a_plugin": a_p, "a_recon": a_r, "R_plugin": R_p, "R_recon": R_r,
        "conv_mix": float(fx.get("convolution", {}).get("mix", 0.0)),
        "chorus_mix": float(fx.get("chorus", {}).get("mix", 0.0)),
        "group_volume": float(grp.get("volume")) if (grp is not None and grp.get("volume")) else 1.0,
        "level_db": db(a_r, a_p),
        "wetdry_db": db(R_r, R_p) if R_p > 1e-6 else float("nan"),
        "predelay_ms": (on_r - on_p) if (on_p is not None and on_r is not None) else float("nan"),
        "decay_corr": decay_correlation(res_p, res_r, SR),
        "tilt_db": spectral_tilt_db(recon, SR) - spectral_tilt_db(full, SR),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--wave-inject", default="build/wave_inject")
    ap.add_argument("--tmp", default="/tmp/audit_fidelity")
    ap.add_argument("--key", type=int, default=60)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--reverb-only", action="store_true",
                    help="compare against reverb-only ground truth, excluding the "
                         "approximate chorus model from both sides")
    args = ap.parse_args()

    library = Path(args.library)
    patches = []
    for f in sorted(library.glob("*/patch.json")):
        m = json.loads(f.read_text())
        bank = str(m["bank"])
        if bank not in BANK_BASE:
            continue        # expansion boards are not in the internal enumeration
        stem = [p.stem for p in library.glob("*.dspreset") if p.stem.endswith(m["name"])]
        if stem:
            patches.append((BANK_BASE[bank] + int(m["index"]), stem[0]))
    patches.sort()
    if args.limit and len(patches) > args.limit:
        step = len(patches) / args.limit
        patches = [patches[int(i * step)] for i in range(args.limit)]

    rows = []
    for i, (idx, stem) in enumerate(patches, 1):
        try:
            r = audit_patch(Path(args.wave_inject), args.roms, args.tmp, library, idx, stem,
                            args.key, reverb_only=args.reverb_only)
        except Exception as exc:            # one bad patch must not lose the run
            print(f"  [{i}/{len(patches)}] {stem}: FAILED ({exc!r})", file=sys.stderr)
            continue
        rows.append(r)
        print(f"  [{i}/{len(patches)}] {r['name'][:20]:<21} "
              f"level {r['level_db']:+6.1f} dB  wet/dry {r['wetdry_db']:+6.1f} dB  "
              f"predelay {r['predelay_ms']:+6.1f} ms   decay {r['decay_corr']:.3f}   "
              f"tilt {r['tilt_db']:+5.1f} dB")

    if not rows:
        sys.exit("no patches audited")

    print(f"\n{'metric':<14}{'median':>9}{'mean|err|':>11}{'worst':>9}{'tol':>7}{'outliers':>10}")
    failed = []
    for metric, tol in TOL.items():
        vals = np.array([r[metric] for r in rows if r[metric] == r[metric]])
        if not len(vals):
            continue
        if metric == "decay_corr":
            bad = vals < tol
            print(f"{metric:<14}{np.median(vals):>9.3f}{vals.mean():>11.3f}"
                  f"{vals.min():>9.3f}{tol:>7.2f}{int(bad.sum()):>7}/{len(vals)}")
        else:
            bad = np.abs(vals) > tol
            worst = vals[np.argmax(np.abs(vals))]
            print(f"{metric:<14}{np.median(vals):>+9.2f}{np.abs(vals).mean():>11.2f}"
                  f"{worst:>+9.2f}{tol:>7.1f}{int(bad.sum()):>7}/{len(vals)}")
        if bad.mean() > 0.25:
            failed.append(f"{metric}: {int(bad.sum())}/{len(vals)} outside tolerance")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")

    print()
    for r in sorted(rows, key=lambda r: -abs(r["level_db"] if r["level_db"] == r["level_db"] else 0))[:5]:
        print(f"  worst level error: {r['name'][:20]:<21} {r['level_db']:+.1f} dB")

    if failed:
        print("\nAUDIT FAILED")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    print("\nAUDIT PASSED")


if __name__ == "__main__":
    main()
