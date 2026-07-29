#!/usr/bin/env python3
"""Correct the captured IRs' spectral tilt against the plugin's own reverb.

Reported on listening as everything sounding muddier, and confirmed by
measurement: comparing the reverb-only decay of the reconstruction against the
plugin's, level-normalised so this is purely tonal, the shipped IRs carry
about +4 dB too much energy near 250 Hz and 5 dB too little at 8 kHz -- a
+3.5 dB low/high imbalance.

Part of that is structural. The emulator's output path is not flat (it posts
two output frames per DSP tick, and has a comb null at 16 kHz), measuring
-1.7 dB over 8-16 kHz. Our dry SAMPLES already carry that rolloff, and the
captured IR carries it a second time, so convolving them applies it twice
where the JV applies it once -- its reverb runs before the DAC. That accounts
for roughly 3.4 dB of the 5.2 dB deficit at 8 kHz. The remainder is most
likely HF lost to quantisation: the raw IR captures peak near -39 dBFS, and a
reverb's high frequencies decay fastest, so they sit closest to the noise
floor.

Rather than model those separately, this measures the total mismatch and
inverts it. The correction is fitted on one set of patches and validated on a
DISJOINT set, because a curve with this many degrees of freedom will always
improve the patches it was fitted to.

Usage:
    python3 tools/correct_ir_spectrum.py --roms ROMS --library LIB \
        [--train N] [--test N] [--apply]
"""
import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_compare as ab   # noqa: E402

SR = 64000
IR_SR = 48000
BANK_BASE = {"A": 0, "B": 64, "Internal": 128}
HOLD_S = 3.5

# Octave-ish bands. Deliberately coarse: the goal is to remove a broad tonal
# tilt, not to chase per-patch spectral detail that no single curve can fix.
BANDS = [(60, 125), (125, 250), (250, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
# Cap on how much correction is applied in any band -- a sanity bound against
# a bad fit, not a tuning knob. Capping at 4 dB was tried on the theory that it
# would "keep half the fix" while protecting decay accuracy. It does not: the
# measured low/high imbalance on held-out patches goes 6.87 dB uncorrected ->
# 6.53 dB at a 4 dB cap -> 1.34 dB uncapped. Almost the entire benefit lives in
# the last few dB, because the imbalance is driven by the band extremes. The
# correction magnitude is not a proxy for the correction's effect.
MAX_CORRECTION_DB = 8.0


def residual(wet, dry):
    n = min(len(wet), len(dry))
    w, d = wet[:n].mean(axis=1), dry[:n].mean(axis=1)
    a = float(np.dot(w, d) / max(np.dot(d, d), 1e-12))
    return w - a * d


def band_power(x, sr):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return np.array([X[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS])


def measure_patch(wave_inject, roms, library, idx, stem, tmp):
    out = Path(tmp) / str(idx)
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(wave_inject), "groundtruth", "--roms", str(roms),
                        "--patch-index", str(idx), "--out-dir", str(out)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None
    wet, sr = sf.read(str(out / "groundtruth_wet.wav"), always_2d=True)
    dry, _ = sf.read(str(out / "groundtruth_dry.wav"), always_2d=True)
    fx = ab.preset_effects(library / f"{stem}.dspreset")
    if "convolution" not in fx or float(fx["convolution"].get("mix", 0)) <= 0:
        return None
    recon, _ = ab.build_reconstruction(dry, fx, library, use_chorus=False)

    grp = ET.parse(library / f"{stem}.dspreset").getroot().find(".//group")
    if grp is not None and grp.get("volume"):
        cm = float(fx.get("chorus", {}).get("mix", 0.0))
        recon = recon * float(grp.get("volume")) * (1 - cm) ** 0.5

    st = int(HOLD_S * sr)
    rp, rr = residual(wet, dry)[st:], residual(recon, dry)[st:]
    if len(rp) < sr * 0.2 or len(rr) < sr * 0.2:
        return None
    bp, br = band_power(rp, sr), band_power(rr, sr)
    if bp.min() <= 0 or br.min() <= 0:
        return None
    d = 10 * np.log10(br / bp)
    return d - d.mean()          # tonal only; overall level is tracked elsewhere


def collect(library):
    out = []
    for f in sorted(library.glob("*/patch.json")):
        m = json.loads(f.read_text())
        if str(m["bank"]) not in BANK_BASE:
            continue
        if m["effects"]["reverb"]["type"] in ("Delay", "Pan-Dly"):
            continue
        stems = [p.stem for p in library.glob("*.dspreset") if p.stem.endswith(m["name"])]
        if stems:
            out.append((BANK_BASE[str(m["bank"])] + int(m["index"]), stems[0]))
    return sorted(out)


def apply_correction(ir_dir, curve_db):
    """Apply the inverse tilt to every IR in the bank, in the frequency domain."""
    centres = np.array([np.sqrt(lo * hi) for lo, hi in BANDS])
    n_done = 0
    for f in sorted(Path(ir_dir).glob("*.wav")):
        x, sr = sf.read(str(f), always_2d=True)
        if not np.abs(x).any():
            continue
        N = len(x)
        freqs = np.fft.rfftfreq(N, 1 / sr)
        # Interpolate the band corrections into a smooth curve, held flat
        # outside the measured range rather than extrapolated -- extrapolating
        # a fitted tilt past where it was measured is how you get a bright,
        # thin IR instead of a corrected one.
        gain_db = np.interp(np.log(np.maximum(freqs, 1.0)), np.log(centres), curve_db,
                            left=curve_db[0], right=curve_db[-1])
        gain = 10 ** (gain_db / 20.0)
        y = np.zeros_like(x)
        for c in range(x.shape[1]):
            y[:, c] = np.fft.irfft(np.fft.rfft(x[:, c]) * gain, N)
        peak = np.abs(y).max()
        if peak >= 1.0:
            y = y / peak * 0.99
        sf.write(str(f), y, sr, subtype="PCM_24")
        n_done += 1
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--wave-inject", default="build/wave_inject")
    ap.add_argument("--ir-dir", default="calib/ir_synth")
    ap.add_argument("--tmp", default="/tmp/ir_spectrum")
    ap.add_argument("--train", type=int, default=18)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    library = Path(args.library)
    patches = collect(library)
    # Interleave so train and test both span the whole bank rather than
    # train landing on pianos and test on synths.
    train = patches[0::2][:args.train]
    test = patches[1::2][:args.test]
    print(f"train {len(train)} patches, test {len(test)} patches (disjoint)\n")

    def gather(group, label):
        rows = []
        for idx, stem in group:
            d = measure_patch(Path(args.wave_inject), args.roms, library, idx, stem, args.tmp)
            if d is not None:
                rows.append(d)
        print(f"  {label}: {len(rows)} usable")
        return np.array(rows)

    tr = gather(train, "train")
    te_before = gather(test, "test (before)")
    if not len(tr) or not len(te_before):
        sys.exit("not enough usable patches")

    err = np.median(tr, axis=0)
    curve = np.clip(-err, -MAX_CORRECTION_DB, MAX_CORRECTION_DB)
    hdr = "".join(f"{lo}".rjust(8) for lo, _ in BANDS)
    print(f"\n{'band Hz':<12}{hdr}")
    print(f"{'train err':<12}" + "".join(f"{v:+8.1f}" for v in err))
    print(f"{'correction':<12}" + "".join(f"{v:+8.1f}" for v in curve))

    def imbalance(a):
        return a[:3].mean() - a[5:].mean()
    print(f"\nBEFORE  test: mean|err| {np.abs(te_before).mean():.2f} dB, "
          f"low/high imbalance {imbalance(np.median(te_before, axis=0)):+.2f} dB")

    if not args.apply:
        print("\n(dry run -- pass --apply to write the corrected IR bank)")
        return

    n = apply_correction(args.ir_dir, curve)
    print(f"\napplied to {n} IRs; re-emit presets, then re-measuring the TEST set")
    for d in sorted(Path(args.library).parent.glob("*/")):
        subprocess.run([sys.executable, "tools/emit_presets.py", str(d),
                        "calib/calibration.json"], capture_output=True)
    te_after = gather(test, "test (after)")
    print(f"\nAFTER   test: mean|err| {np.abs(te_after).mean():.2f} dB, "
          f"low/high imbalance {imbalance(np.median(te_after, axis=0)):+.2f} dB")
    print(f"{'after err':<12}" + "".join(f"{v:+8.1f}" for v in np.median(te_after, axis=0)))

    cal = Path("calib/calibration.json")
    c = json.loads(cal.read_text())
    c["ir_spectral_correction_db"] = {str(lo): float(v) for (lo, _), v in zip(BANDS, curve)}
    cal.write_text(json.dumps(c, indent=2) + "\n")
    print(f"\nrecorded the correction in {cal}")


if __name__ == "__main__":
    main()
