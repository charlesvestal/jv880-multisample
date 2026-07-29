#!/usr/bin/env python3
"""Build a small .dslibrary for auditioning over a slow link.

A full preset is 75 samples / ~40 MB, so even a three-preset taster is far too
big to send directly. This keeps only a few root keys and velocity layers per
preset and re-tiles the remaining zones across the keyboard, producing
something a few MB in size that still loads and plays in DecentSampler.

What it is for: judging TONE -- reverb character, level, brightness, decay.
The effect chain, the IRs and the group volume are exactly what the full
library ships. What it is NOT for: judging the multisampling itself. With
fewer root keys, notes are pitch-shifted further from their source sample, so
timbre drifts across the keyboard in a way the full library does not.

Keep the root keys dense. Several JV patches pan by key position -- A.Piano 2
sweeps from 19 dB left at A2 to 19 dB right at D#5 -- which is a smooth
gradient over the full library's 25 root keys but becomes a STAIRCASE here,
each sample holding one pan position across its whole span. At 5 keys the
largest jump between neighbours is 12.9 dB and reads as the reverb lurching
into one ear; at 9 keys it is 6.9 dB, matching the full library's own 6.7 dB.
That artifact belongs to this trimming, not to the library being auditioned.

Usage:
    python3 tools/make_mini_dslibrary.py <out.dslibrary> --keys 36,48,60,72 \\
        --layers 1,3 "<lib>:<preset stem>" [...]
"""
import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def trim_preset(preset_path, keys, layers):
    """Keep only the chosen root keys / velocity layers, re-tiled to cover
    the whole keyboard and full velocity range."""
    tree = ET.parse(preset_path)
    root = tree.getroot()
    groups = root.find("groups")
    if groups is None:
        raise SystemExit(f"{preset_path.name}: no <groups>")
    group = groups.find("group")

    samples = group.findall("sample")
    kept = [s for s in samples if int(s.get("rootNote", -1)) in keys]
    if not kept:
        raise SystemExit(f"{preset_path.name}: none of {sorted(keys)} are root keys")

    # Velocity layers are identified by their order within a root key, since
    # the preset stores explicit loVel/hiVel rather than a layer index.
    by_key = {}
    for s in kept:
        by_key.setdefault(int(s.get("rootNote")), []).append(s)
    chosen = []
    for key, group_samples in by_key.items():
        group_samples.sort(key=lambda s: int(s.get("loVel", 0)))
        picks = [group_samples[i - 1] for i in layers if 0 < i <= len(group_samples)]
        # Re-tile velocity so the surviving layers still cover 1..127; without
        # this, dropping the top layer leaves hard notes silent.
        edges = [1 + round(i * 127 / len(picks)) for i in range(len(picks))] + [128]
        for i, s in enumerate(picks):
            s.set("loVel", str(edges[i]))
            s.set("hiVel", str(edges[i + 1] - 1))
        chosen += picks

    # Re-tile key ranges across 0..127 around the surviving root notes.
    chosen.sort(key=lambda s: int(s.get("rootNote")))
    roots = sorted({int(s.get("rootNote")) for s in chosen})
    bounds = {}
    for i, r in enumerate(roots):
        lo = 0 if i == 0 else (roots[i - 1] + r) // 2 + 1
        hi = 127 if i == len(roots) - 1 else (r + roots[i + 1]) // 2
        bounds[r] = (lo, hi)
    for s in chosen:
        lo, hi = bounds[int(s.get("rootNote"))]
        s.set("loNote", str(lo))
        s.set("hiNote", str(hi))

    for s in samples:
        if s not in chosen:
            group.remove(s)
    return tree, [s.get("path") for s in chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--keys", default="36,48,60,72")
    ap.add_argument("--layers", default="1,3")
    ap.add_argument("presets", nargs="+")
    args = ap.parse_args()

    keys = {int(k) for k in args.keys.split(",")}
    layers = [int(v) for v in args.layers.split(",")]
    out = Path(args.out)
    name = out.stem

    staged = []
    for spec in args.presets:
        lib_str, _, stem = spec.rpartition(":")
        lib = Path(lib_str)
        preset = lib / f"{stem}.dspreset"
        if not preset.exists():
            raise SystemExit(f"preset not found: {preset}")
        tree, sample_paths = trim_preset(preset, keys, layers)

        refs = list(sample_paths)
        refs += [e.get("irFile") for e in tree.getroot()
                 .findall('.//effect[@type="convolution"]')]
        files = []
        for rel in dict.fromkeys(r for r in refs if r):
            src = (lib / rel).resolve()
            if not src.exists():
                raise SystemExit(f"{preset.name}: missing {rel}")
            files.append((src, rel))
        staged.append((stem, tree, files))
        size = sum(p.stat().st_size for p, _ in files)
        print(f"{stem}: {len(sample_paths)} samples, {size / 1e6:.1f} MB")

    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for stem, tree, files in staged:
            z.writestr(f"{name}/{stem}.dspreset",
                       ET.tostring(tree.getroot(), encoding="unicode"))
            for src, rel in files:
                z.write(src, f"{name}/{rel}")
    tmp.replace(out)
    print(f"\n-> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
