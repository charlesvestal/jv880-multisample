#!/usr/bin/env python3
"""Repair `effects.reverb.type` in already-rendered patch.json files.

Background: `reverbtype` was decoded as a 4-bit field when it is actually 3
bits (see src/jv_patch.cpp).  The extra bit is an unrelated flag, so 40 of the
192 internal patches decoded to invalid types 8-15 and serialized as
"Unknown", which then broke preset emission.

Only that one field is wrong, and it is derived purely from ROM data -- the
rendered audio is unaffected.  So rather than re-render tens of GB, re-read
each patch's ROM bytes and rewrite just the reverb type in place.

Usage:  python3 tools/fix_reverb_type.py "<library root>" [<roms dir>]
"""
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np

PATCH_SIZE = 0x16A
REVERB_NAMES = ["Room1", "Room2", "Stage1", "Stage2",
                "Hall1", "Hall2", "Delay", "Pan-Dly"]
INTERNAL_BANKS = {"A": 0x010CE0, "B": 0x018CE0, "Internal": 0x008CE0}
AA = [2, 0, 3, 4, 1, 9, 13, 10, 18, 17, 6, 15, 11, 16, 8, 5, 12, 7, 14, 19]
DD = [2, 0, 4, 5, 7, 6, 3, 1]


def unscramble(src: np.ndarray) -> np.ndarray:
    n = len(src)
    i = np.arange(n, dtype=np.int64)
    addr = (i & ~0xFFFFF).astype(np.int64)
    for j in range(20):
        addr |= ((i >> j) & 1) << AA[j]
    s = src[addr]
    out = np.zeros(n, dtype=np.uint8)
    for j in range(8):
        out |= (((s >> DD[j]) & 1) << j).astype(np.uint8)
    return out


def build_patch_index(roms: Path) -> dict:
    """Map (bank, index) -> 362-byte patch record for every known patch."""
    index = {}
    rom2 = np.fromfile(roms / "jv880_rom2.bin", dtype=np.uint8)
    for bank, off in INTERNAL_BANKS.items():
        for i in range(64):
            index[(bank, i)] = bytes(rom2[off + i * PATCH_SIZE:
                                          off + (i + 1) * PATCH_SIZE])

    for f in sorted(glob(str(roms / "expansions" / "*.[bB][iI][nN]"))):
        u = unscramble(np.fromfile(f, dtype=np.uint8))
        count = int(u[0x67]) | (int(u[0x66]) << 8)
        offset = (int(u[0x8F]) | (int(u[0x8E]) << 8)
                  | (int(u[0x8D]) << 16) | (int(u[0x8C]) << 24))
        if not (0 < count <= 256 and offset + count * PATCH_SIZE <= len(u)):
            continue
        # Board name must match what jv_sampler wrote as the zone's "bank".
        base = Path(f).name
        cut = base.find(" - CS ")
        if cut == -1:
            hexpos = base.find(" 0x")
            cut = hexpos if hexpos != -1 else base.rfind(".")
        name = base[:cut].rstrip().replace("_", " ")
        for i in range(count):
            index[(name, i)] = bytes(u[offset + i * PATCH_SIZE:
                                       offset + (i + 1) * PATCH_SIZE])
    return index


def main() -> None:
    root = Path(sys.argv[1])
    roms = Path(sys.argv[2] if len(sys.argv) > 2
                else "/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880")

    index = build_patch_index(roms)
    fixed = unchanged = missing = 0

    for jf in sorted(root.glob("*/patch.json")):
        meta = json.loads(jf.read_text())
        key = (meta["bank"], meta["index"])
        patch = index.get(key)
        if patch is None:
            print(f"  WARN no ROM patch for {key} ({jf.parent.name})")
            missing += 1
            continue

        correct = REVERB_NAMES[patch[12] & 0x07]      # 3 bits, not 4
        current = meta["effects"]["reverb"]["type"]
        if current == correct:
            unchanged += 1
            continue
        meta["effects"]["reverb"]["type"] = correct
        jf.write_text(json.dumps(meta, indent=2))
        fixed += 1

    print(f"{root.name}: {fixed} fixed, {unchanged} already correct, "
          f"{missing} unmatched")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
