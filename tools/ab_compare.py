#!/usr/bin/env python3
"""Render listening A/Bs: the JV-880 itself vs. our DecentSampler reconstruction.

For each requested patch this writes a pair of WAVs at the same note:

    <patch> A plugin.wav   -- the emulator with its full native effect chain
    <patch> B decentsampler.wav -- our dry sample plus the effect chain the
                                   SHIPPED .dspreset actually asks for

The B side is built by reading effect parameters out of the emitted .dspreset
rather than recomputing them, so what gets auditioned is the artifact that
ships, not a parallel implementation of it that could agree with the emitter
while both are wrong.

IMPORTANT -- what the B side is and isn't:

  * Convolution reverb is EXACT. DecentSampler's convolution `mix` is a plain
    blend, so `dry*(1-mix) + convolve(dry, ir)*mix` is not an approximation of
    what DS does, it is what DS does.
  * The parametric delay is a faithful model: a feedback tap line with DS's
    `stereoOffset` applied as a channel time offset.
  * CHORUS IS APPROXIMATE. DecentSampler does not document its chorus
    internals, so the model here is a conventional modulated delay line and
    `modDepth` is dimensionless in DS. A chorus mismatch in these files is
    therefore evidence about THIS MODEL as much as about the shipped preset.
    Use --no-chorus to A/B the parts that are exactly reproducible.

There is no DecentSampler CLI to render through, so this is a simulation of
DS, not a capture of it. It is honest about the reverb and delay paths and
explicitly not authoritative about chorus.
"""
import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

SR = 64000          # emulator native rate
OUT_SR = 48000      # delivered rate


def run_abcompare(wave_inject, roms, patch_index, key, out_dir):
    """Render the plugin (full effects) and dry sides for one patch/key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(wave_inject), "abcompare", "--roms", str(roms),
         "--patch-index", str(patch_index), "--key", str(key),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"wave_inject abcompare failed:\n{proc.stderr}")
    meta = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            meta = json.loads(line)
    if meta is None:
        raise SystemExit("abcompare produced no effects JSON")
    full, sr1 = sf.read(str(out_dir / f"full_k{key}.wav"), always_2d=True)
    dry, sr2 = sf.read(str(out_dir / f"dry_k{key}.wav"), always_2d=True)
    assert sr1 == SR and sr2 == SR, f"expected {SR} Hz, got {sr1}/{sr2}"
    return full, dry, meta


def preset_effects(preset_path):
    """Effect parameters exactly as the shipped .dspreset declares them."""
    root = ET.parse(preset_path).getroot()
    out = {}
    for e in root.findall(".//effect"):
        attrs = {k: v for k, v in e.attrib.items() if k != "type"}
        out[e.get("type")] = attrs
    return out


def apply_convolution(dry, ir, mix):
    """DecentSampler's convolution: a straight blend of dry and convolved."""
    if mix <= 0:
        return dry.copy()
    # Deliberately NOT normalised. An earlier version scaled the wet path to
    # the dry peak before blending, which made the A/B sound right while the
    # shipped preset was 21.8 dB quiet in DecentSampler -- the simulation was
    # hiding the exact defect it existed to find. DS convolves with the IR as
    # written, so the IR's own level is part of what is under test, and the
    # level is now fixed where it belongs: in the IR (see ir_capture's
    # normalize_ir), not here.
    wet = np.stack([fftconvolve(dry[:, c], ir[:, c % ir.shape[1]])[:len(dry)]
                    for c in range(dry.shape[1])], axis=1)
    return dry * (1.0 - mix) + wet * mix


def apply_delay(dry, delay_time_s, feedback, stereo_offset_s, wet_level, sr=SR):
    """Feedback delay line with DS's stereoOffset as a per-channel time offset."""
    if wet_level <= 0 or delay_time_s <= 0:
        return dry.copy()
    n = len(dry)
    out = dry.copy()
    for c in range(dry.shape[1]):
        offset = stereo_offset_s if c == 1 else 0.0
        d = int(round((delay_time_s + offset) * sr))
        if d <= 0:
            continue
        tap = np.zeros(n)
        gain, pos = 1.0, d
        # Stop once a repeat is inaudible (-60 dB) rather than at a fixed count.
        while pos < n and gain > 1e-3:
            tap[pos:] += dry[:n - pos, c] * gain
            gain *= feedback
            pos += d
            if feedback <= 0:
                break
        out[:, c] += tap * wet_level
    return out


def apply_chorus(dry, mix, mod_depth, mod_rate_hz, sr=SR):
    """APPROXIMATE model of DecentSampler's chorus (see module docstring).

    A conventional modulated delay line: `modDepth` is dimensionless in DS, so
    it is interpreted here as a fraction of a musically typical 5 ms sweep over
    a 15 ms base delay, with the two channels in quadrature for width.
    """
    if mix <= 0:
        return dry.copy()
    n = len(dry)
    t = np.arange(n) / sr
    base_s, sweep_s = 0.015, 0.005
    out = dry.copy()
    for c in range(dry.shape[1]):
        phase = 0.0 if c == 0 else np.pi / 2
        delay_s = base_s + mod_depth * sweep_s * np.sin(2 * np.pi * mod_rate_hz * t + phase)
        idx = np.arange(n) - delay_s * sr
        wet = np.interp(idx, np.arange(n), dry[:, c], left=0.0, right=0.0)
        out[:, c] = dry[:, c] * (1.0 - mix) + wet * mix
    return out


def build_reconstruction(dry, fx, library_dir, use_chorus=True):
    """Apply the shipped preset's effect chain to the dry render."""
    sig = dry.copy()
    notes = []

    if use_chorus and "chorus" in fx:
        mix = float(fx["chorus"].get("mix", 0))
        if mix > 0:
            sig = apply_chorus(sig, mix,
                               float(fx["chorus"].get("modDepth", 0)),
                               float(fx["chorus"].get("modRate", 0)))
            notes.append(f"chorus(mix={mix:.3f}, depth={fx['chorus'].get('modDepth')}, "
                         f"rate={fx['chorus'].get('modRate')}Hz) [approximate]")

    if "convolution" in fx:
        mix = float(fx["convolution"].get("mix", 0))
        ir_rel = fx["convolution"].get("irFile")
        if mix > 0 and ir_rel:
            ir, ir_sr = sf.read(str(Path(library_dir) / ir_rel), always_2d=True)
            if ir_sr != SR:
                ir = resample_poly(ir, up=SR, down=ir_sr, axis=0)
            sig = apply_convolution(sig, ir, mix)
            notes.append(f"convolution({Path(ir_rel).name}, mix={mix:.3f}) [exact]")
        else:
            notes.append(f"convolution(mix={mix:.3f}) -> no wet contribution")

    if "delay" in fx:
        wet = float(fx["delay"].get("wetLevel", 0))
        if wet > 0:
            sig = apply_delay(sig,
                              float(fx["delay"].get("delayTime", 0)),
                              float(fx["delay"].get("feedback", 0)),
                              float(fx["delay"].get("stereoOffset", 0)),
                              wet)
            notes.append(f"delay(t={fx['delay'].get('delayTime')}s, "
                         f"fb={fx['delay'].get('feedback')}, "
                         f"off={fx['delay'].get('stereoOffset')}s, wet={wet:.3f})")
    return sig, notes


def envelope_correlation(a, b, sr=SR):
    """Correlation of the two signals' log envelopes.

    Compares how the sound EVOLVES rather than sample-by-sample agreement:
    a reverb tail reconstructed from an IR will never be sample-identical to
    the hardware's, but it should decay along the same curve.
    """
    n = min(len(a), len(b))
    am, bm = np.abs(a[:n]).mean(axis=1), np.abs(b[:n]).mean(axis=1)
    w = int(sr * 0.02)
    k = np.ones(w) / w
    ae = 20 * np.log10(np.maximum(np.convolve(am, k, mode="same"), 1e-9))
    be = 20 * np.log10(np.maximum(np.convolve(bm, k, mode="same"), 1e-9))
    keep = (ae > ae.max() - 60) | (be > be.max() - 60)
    if keep.sum() < sr * 0.1:
        return float("nan")
    return float(np.corrcoef(ae[keep], be[keep])[0, 1])


def write48(path, x):
    y = resample_poly(x, up=3, down=4, axis=0)
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak
    sf.write(str(path), y, OUT_SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True)
    ap.add_argument("--library", required=True,
                    help="emitted library dir (holds .dspreset files and ir_synth/)")
    ap.add_argument("--out", required=True, help="directory to write A/B pairs into")
    ap.add_argument("--wave-inject", default="build/wave_inject")
    ap.add_argument("--tmp", default="/tmp/ab_compare")
    ap.add_argument("--key", type=int, default=60)
    ap.add_argument("--no-chorus", action="store_true",
                    help="skip the approximate chorus model on the B side")
    ap.add_argument("patches", nargs="+",
                    help="'<patch index>:<preset stem>', e.g. '0:A00 A.Piano 1'")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    library = Path(args.library)
    rows = []

    for spec in args.patches:
        idx_str, _, stem = spec.partition(":")
        idx = int(idx_str)
        preset = library / f"{stem}.dspreset"
        if not preset.exists():
            raise SystemExit(f"no such preset: {preset}")

        full, dry, meta = run_abcompare(Path(args.wave_inject), args.roms, idx,
                                        args.key, Path(args.tmp) / stem)
        fx = preset_effects(preset)
        recon, notes = build_reconstruction(dry, fx, library,
                                            use_chorus=not args.no_chorus)

        corr = envelope_correlation(full, recon)
        rows.append((meta["name"], corr, notes))

        write48(out_dir / f"{stem} -- A plugin.wav", full)
        write48(out_dir / f"{stem} -- B decentsampler.wav", recon)

        print(f"\n{meta['name']}  (patch {idx}, key {args.key})")
        print(f"  plugin: reverb type {meta['reverb_type']} "
              f"level {meta['reverb_level']} time {meta['reverb_time']}, "
              f"chorus level {meta['chorus_level']} depth {meta['chorus_depth']}")
        for n in notes:
            print(f"  B side: {n}")
        print(f"  envelope correlation: {corr:.4f}")

    print(f"\n{'patch':<20}{'corr':>8}")
    for name, corr, _ in rows:
        print(f"{name[:19]:<20}{corr:>8.4f}")
    good = [c for _, c, _ in rows if c == c]
    if good:
        print(f"{'mean':<20}{sum(good)/len(good):>8.4f}")
    print(f"\nwrote {2*len(rows)} files to {out_dir}")


if __name__ == "__main__":
    main()
