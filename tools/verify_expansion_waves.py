#!/usr/bin/env python3
"""Prove each expansion board actually rendered with ITS OWN waves.

The bug this exists for: an expansion patch addresses its waves through PCM
banks 3-6 (waverom_exp), a region startSC55() does not populate. Miss that
and the wave numbers still resolve -- against the INTERNAL wave ROM -- so
every patch on every expansion board renders a confident, distinct,
completely wrong instrument. An Experience-board string patch came out as an
acoustic piano.

Nothing structural catches it. Zones are present, ranges tile, references
resolve, and each patch differs from its neighbours with a plausible
spectrum. Even the fidelity audit passes, because it compares a preset to
the emulator and both sides are wrong in the same way.

So this compares the two renders directly: the same patch WITH expansion
waves loaded and WITHOUT. If a board's waves are being used, the two must
differ. If they match, that board fell back to internal waves.

    python3 tools/verify_expansion_waves.py --roms ROMS [--patch 0]
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
# Above this, the two renders are effectively the same audio and the board is
# not using its own waves. Genuine expansion renders sit far below it: the
# reference case measured +0.026.
SAME_SOUND_CORRELATION = 0.90


def render(sampler, roms, board, patch, out, no_expansion=False):
    cmd = [str(sampler), "--roms", str(roms), "--board", board,
           "--out", str(out), "--patch", str(patch)]
    if no_expansion:
        cmd.append("--no-expansion-waves")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    wavs = sorted(Path(out).glob("*/C4_v3.wav")) or sorted(Path(out).glob("*/*.wav"))
    if not wavs:
        return None
    x, _ = sf.read(str(wavs[0]), always_2d=True)
    return x.mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True)
    ap.add_argument("--sampler", default=str(REPO / "build" / "jv_sampler"))
    ap.add_argument("--patch", type=int, default=0)
    args = ap.parse_args()

    listing = subprocess.run([args.sampler, "--roms", args.roms, "--list"],
                             capture_output=True, text=True)
    boards = [ln.split("\t")[0] for ln in listing.stdout.splitlines()
              if "\t" in ln and ln.split("\t")[0].startswith("SR-JV80")]
    print(f"checking {len(boards)} expansion boards\n")

    failures = []
    for board in boards:
        tmp = Path(tempfile.mkdtemp())
        try:
            a = render(args.sampler, args.roms, board, args.patch, tmp / "with")
            b = render(args.sampler, args.roms, board, args.patch, tmp / "without",
                       no_expansion=True)
            if a is None or b is None:
                print(f"  {board:<42} SKIPPED (render failed)")
                continue
            n = min(len(a), len(b))
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n else 1.0
            ok = corr < SAME_SOUND_CORRELATION
            print(f"  {board:<42} correlation {corr:+.3f}  {'ok' if ok else 'USING INTERNAL WAVES'}")
            if not ok:
                failures.append(board)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} board(s) did NOT use their own waves:")
        for b in failures:
            print(f"   {b}")
        sys.exit(1)
    print("\nall boards render with their own expansion waves")


if __name__ == "__main__":
    main()
