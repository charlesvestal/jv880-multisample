#!/usr/bin/env python3
"""Split one rendered board into several smaller .dslibrary files.

A whole board is 3-12 GB, which is awkward to move around and heavy for
DecentSampler on iOS to open. This packages the same presets in fixed-size
groups instead, each a self-contained library with only the samples and
impulse responses its own presets reference.

    python3 tools/make_chunked_dslibrary.py "<board dir>" <out dir> [--size 32]
"""
import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def preset_refs(preset, lib):
    """Every file a preset needs, as (absolute source, path inside the zip)."""
    root = ET.parse(preset).getroot()
    rels = [s.get("path") for s in root.findall(".//sample")]
    rels += [e.get("irFile") for e in root.findall('.//effect[@type="convolution"]')]
    out = []
    for rel in dict.fromkeys(r for r in rels if r):
        src = (lib / rel).resolve()
        if not src.exists():
            raise SystemExit(f"{preset.name}: references missing file {rel}")
        out.append((src, rel))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=32, help="presets per library")
    args = ap.parse_args()

    lib = Path(args.board).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    presets = sorted(lib.glob("*.dspreset"))
    if not presets:
        raise SystemExit(f"no .dspreset files in {lib}")

    chunks = [presets[i:i + args.size] for i in range(0, len(presets), args.size)]
    width = len(str(len(chunks)))
    for n, chunk in enumerate(chunks, 1):
        # No square brackets in the filename. Some SMB clients and iOS hide
        # or fail on them, which made a set of these invisible over a share.
        name = f"{lib.name} part{n:0{width}d}of{len(chunks)}"
        dest = out / f"{name}.dslibrary"
        tmp = dest.with_suffix(dest.suffix + ".partial")
        total = 0
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for preset in chunk:
                z.write(preset, f"{name}/{preset.name}")
                for src, rel in preset_refs(preset, lib):
                    z.write(src, f"{name}/{rel}")
                    total += src.stat().st_size
        tmp.replace(dest)
        print(f"  {name}.dslibrary  {len(chunk)} presets, {dest.stat().st_size/2**30:.2f} GB")
    print(f"\n{len(chunks)} libraries -> {out}")


if __name__ == "__main__":
    main()
