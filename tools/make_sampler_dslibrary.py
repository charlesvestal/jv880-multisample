#!/usr/bin/env python3
"""Build a small .dslibrary containing only selected presets.

Same container shape as tools/make_dslibrary.py (stored ZIP, one top-level
folder), but includes just the named presets and ONLY the sample files those
presets actually reference -- so a 7 GB library becomes a ~100 MB taster that
is practical to download over a slow link.

Sample paths are read from each preset's own XML rather than guessed from the
directory layout, so whatever relative paths the emitter wrote are exactly
what gets packaged, and a missing file is reported rather than silently
producing a broken library.

Usage:
    python3 tools/make_sampler_dslibrary.py <out.dslibrary> <lib>:<preset stem> [...]

Example:
    python3 tools/make_sampler_dslibrary.py /tmp/taster.dslibrary \\
        "/path/JV-880 Internal:A00 A.Piano 1" \\
        "/path/SR-JV80-04 Vintage Synth:00 Prologue"
"""
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def collect(lib: Path, stem: str):
    """Return (preset path, [(sample abs path, relative path)]) for one preset."""
    preset = lib / f"{stem}.dspreset"
    if not preset.exists():
        sys.exit(f"preset not found: {preset}")
    root = ET.parse(preset).getroot()
    seen, out = set(), []
    for s in root.findall(".//sample"):
        rel = s.get("path")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        src = (lib / rel).resolve()
        if not src.exists():
            sys.exit(f"{preset.name}: references missing sample {rel}")
        out.append((src, rel))
    if not out:
        sys.exit(f"{preset.name}: no samples referenced")
    return preset, out


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    name = out.stem

    entries = []
    for spec in sys.argv[2:]:
        lib_str, _, stem = spec.rpartition(":")
        lib = Path(lib_str)
        preset, samples = collect(lib, stem)
        entries.append((preset, samples))
        size = sum(p.stat().st_size for p, _ in samples)
        print(f"{stem}: {len(samples)} samples, {size / 1e6:.0f} MB")

    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for preset, samples in entries:
            # Flatten every preset into one library folder. Sample paths keep
            # their per-patch subfolder, which is already unique per preset,
            # so presets from different source libraries can't collide.
            z.write(preset, f"{name}/{preset.name}")
            for src, rel in samples:
                z.write(src, f"{name}/{rel}")
    tmp.replace(out)
    print(f"\n-> {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
