#!/usr/bin/env python3
"""Combine several rendered boards into ONE .dslibrary, sharing identical samples.

Built for drum kits, where sharing is the point. The kits on a board reuse each
other's sounds heavily -- Roland built them that way -- so packaging each kit
with its own copies stores the same audio many times over. Across the 52 JV-880
kits, 12,688 sample files hold only 6,608 distinct recordings: 46.8% of the
bytes are duplicates.

Deduplication is by CONTENT HASH, not by filename or by wave number. Two zones
can name the same wave and still render differently (a retuned tom), and two
differently-named zones can be bit-identical. Only the audio decides.

A deduplicated sample keeps the relative path of its FIRST occurrence, so every
file in the archive still says where it came from; later presets simply point
at that copy instead of shipping their own.

    python3 tools/make_combined_dslibrary.py <out.dslibrary> <board dir> [...]
"""
import argparse
import hashlib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="path of the .dslibrary to write")
    ap.add_argument("boards", nargs="+", help="rendered board directories")
    ap.add_argument("--name", default=None,
                    help="folder name inside the archive (default: the out stem)")
    args = ap.parse_args()

    out = Path(args.out)
    libname = args.name or out.stem

    canonical = {}          # content hash -> path inside the archive
    staged = {}             # path inside the archive -> source file
    presets = []            # (preset name inside archive, XML tree)
    dup_bytes = 0
    total_refs = 0

    for board_arg in args.boards:
        board = Path(board_arg)
        found = sorted(board.glob("*.dspreset"))
        if not found:
            raise SystemExit(f"{board}: no .dspreset files -- emit presets first")
        for preset in found:
            tree = ET.parse(preset)
            root = tree.getroot()

            for el in root.findall(".//sample") + root.findall('.//effect[@type="convolution"]'):
                attr = "path" if el.tag == "sample" else "irFile"
                rel = el.get(attr)
                if not rel:
                    continue
                src = (board / rel).resolve()
                if not src.exists():
                    raise SystemExit(f"{preset.name}: references missing file {rel}")
                total_refs += 1
                h = file_hash(src)
                if h in canonical:
                    dup_bytes += src.stat().st_size
                else:
                    # First sighting wins, and keeps its own board/kit path so
                    # the archive still records where the audio came from.
                    canonical[h] = f"{board.name}/{rel}"
                    staged[canonical[h]] = src
                el.set(attr, canonical[h])

            # Board name in the preset title: 52 kits in one library are
            # otherwise four indistinguishable "Rhythm 1"s per board.
            presets.append((f"{board.name} {preset.stem}", tree))

    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, tree in presets:
            z.writestr(f"{libname}/{name}.dspreset",
                       ET.tostring(tree.getroot(), encoding="unicode"))
        for arc, src in staged.items():
            z.write(src, f"{libname}/{arc}")
    tmp.replace(out)

    size = sum(s.stat().st_size for s in staged.values())
    print(f"presets        {len(presets)}")
    print(f"sample refs    {total_refs}")
    print(f"unique files   {len(staged)}   ({size / 1e6:.1f} MB)")
    print(f"deduplicated   {total_refs - len(staged)} refs, saving {dup_bytes / 1e6:.1f} MB")
    print(f"-> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
