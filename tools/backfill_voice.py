#!/usr/bin/env python3
"""Merge each patch's voice mode into an already-rendered patch.json.

The voice fields (key_assign, portamento, ...) were added to jv_sampler after
the library was rendered. Re-running the sampler would regenerate patch.json
from scratch and destroy what postprocess wrote into the zones -- kind, loop
points, release -- so this merges the new block in instead, leaving everything
else byte-identical.

No audio is touched. Monophony and glide are playback-time properties, and
every note was sampled in isolation regardless.

    python3 tools/backfill_voice.py --roms ROMS --library LIB
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--sampler", default="build/jv_sampler")
    args = ap.parse_args()

    lib = Path(args.library)
    total = merged = 0
    for board_dir in sorted(p for p in lib.iterdir() if p.is_dir()):
        proc = subprocess.run([args.sampler, "--roms", args.roms,
                               "--board", board_dir.name, "--dump-voice"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  {board_dir.name}: dump failed, skipped")
            continue
        # Key on (bank, index): patch NAMES repeat within a board.
        voices = {}
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                voices[(d["bank"], d["index"])] = d["voice"]

        n = 0
        for pj in sorted(board_dir.glob("*/patch.json")):
            meta = json.loads(pj.read_text())
            v = voices.get((str(meta["bank"]), int(meta["index"])))
            if v is None:
                continue
            if meta.get("voice") == v:
                continue
            meta["voice"] = v
            pj.write_text(json.dumps(meta, indent=2))
            n += 1
        total += len(list(board_dir.glob("*/patch.json")))
        merged += n
        print(f"  {board_dir.name}: {n} updated")
    print(f"\n{merged} of {total} patch.json files updated")


if __name__ == "__main__":
    main()
