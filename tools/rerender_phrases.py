#!/usr/bin/env python3
"""Re-render tempo-locked phrase patches with a hold long enough to contain them.

Roland names these patches with their tempo -- "125:BtMenu 1", "83:Kick It" --
so they can be identified exactly rather than guessed at from audio. Three
earlier attempts to detect them from envelope shape all failed, finding
tremolo and chorus instead.

Why they need re-rendering: the standard grid holds each note 3.5 s, and a
two-bar phrase at 61 BPM runs nearly 8 s. Everything below about 137 BPM is
therefore cut mid-bar, and a loop point chosen inside a truncated phrase has
no reason to land on a musical boundary.

They are NOT excluded. Many are hybrids whose tones behave differently from
one another (125:ElevatMe reads [0,0,0,6] across its four tones), so dropping
the patch would discard playable material along with the loop. And the JV is
itself a PCM sampler -- a higher key plays the wave faster -- so multisampling
reproduces the hardware rather than fighting it.

    python3 tools/rerender_phrases.py --roms ROMS --library LIB [--dry-run]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BPM_NAME = re.compile(r"^\s*(\d{2,3})\s*[:.]")
MIN_BPM, MAX_BPM = 60, 200
BARS = 2          # capture two bars of 4/4
HOLD_CEILING_S = 16.0
HOLD_FLOOR_S = 4.0


def hold_for(bpm):
    """Seconds needed for BARS bars of 4/4 at this tempo, plus a little slack
    so the final beat is not sitting exactly on the boundary."""
    beats = BARS * 4
    return max(HOLD_FLOOR_S, min(HOLD_CEILING_S, beats * 60.0 / bpm * 1.15))


def rendered_seconds(pdir):
    """Longest rendered zone in seconds, from patch.json, or 0 if unknown.

    Works both before and after postprocess: `frames` is written by the
    sampler and preserved through FLAC encoding.
    """
    pj = pdir / "patch.json"
    if not pj.exists():
        return 0.0
    try:
        meta = json.loads(pj.read_text())
    except Exception:
        return 0.0
    rate = float(meta.get("sample_rate") or 48000)
    frames = [z.get("frames") or 0 for z in meta.get("zones", [])]
    return (max(frames) / rate) if frames else 0.0


def rom_patch_names(board, roms, sampler):
    """{index: true patch name} straight from the ROM, colon intact."""
    if not roms:
        return {}
    proc = subprocess.run([sampler, "--roms", roms, "--board", board, "--dump-voice"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return {}
    out = {}
    for i, line in enumerate(l for l in proc.stdout.splitlines() if l.startswith("{")):
        out[i] = json.loads(line)["name"]
    return out


# A directory left by an interrupted render: "090_85BumpnNite". The index
# prefix and the patch name are both in the directory name, with the colon
# stripped by the sampler's filename sanitiser.
DIR_NAME = re.compile(r"^(\d{3})_\s*(\d{2,3})\D")


def find_targets(library, roms=None, sampler="build/jv_sampler"):
    """Phrase patches, discovered WITHOUT depending on patch.json alone.

    A render that was interrupted leaves a directory holding partial .wav
    files and NO patch.json -- exactly the state that most needs repairing.
    Keying discovery off patch.json made those patches invisible to the tool
    meant to fix them, which is how four of them survived two repair passes.
    Directory names carry the index and tempo, so they are the reliable key.
    """
    out = []
    seen = set()
    for pj in sorted(Path(library).glob("*/*/patch.json")):
        meta = json.loads(pj.read_text())
        m = BPM_NAME.match(meta["name"])
        if not m:
            continue
        bpm = int(m.group(1))
        if not (MIN_BPM <= bpm <= MAX_BPM):
            continue
        seen.add(pj.parent)
        out.append((pj.parent, meta["name"], bpm, pj.parent.parent.name))

    # Orphans: a directory with no patch.json. Its NAME cannot be trusted to
    # identify a phrase patch, because the sampler strips the colon when
    # sanitising -- so "60s E.Piano" and "78 RPM" look exactly like a tempo
    # label. The real name comes from the ROM, keyed by the index prefix, and
    # only then is the strict BPM test applied.
    for board in sorted(Path(library).iterdir()):
        if not board.is_dir():
            continue
        orphans = [d for d in sorted(board.iterdir())
                   if d.is_dir() and d not in seen and DIR_NAME.match(d.name)]
        if not orphans:
            continue
        names = rom_patch_names(board.name, roms, sampler)
        if not names:
            continue
        for d in orphans:
            idx = int(DIR_NAME.match(d.name).group(1))
            real = names.get(idx)
            if not real:
                continue
            m = BPM_NAME.match(real)
            if not m:
                continue
            bpm = int(m.group(1))
            if MIN_BPM <= bpm <= MAX_BPM:
                out.append((d, real, bpm, board.name))
    return out


def render_one(job):
    pdir, name, bpm, board, roms, sampler, index = job
    hold = hold_for(bpm)
    # Delete immediately before rendering THIS patch, never in a batch up
    # front. An earlier version removed every target directory first and then
    # rendered; interrupting it destroyed 87 patch directories that had not
    # been reached yet. Now an interruption can cost at most the one patch
    # currently in flight.
    if pdir.exists():
        shutil.rmtree(pdir)
    proc = subprocess.run(
        [sampler, "--roms", roms, "--board", board, "--out", str(pdir.parent),
         "--patch", str(index), "--hold", f"{hold:.2f}"],
        capture_output=True, text=True)
    return name, bpm, hold, proc.returncode, proc.stderr[-200:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--sampler", default="build/jv_sampler")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = find_targets(args.library, args.roms, args.sampler)
    print(f"tempo-labelled phrase patches: {len(targets)}")
    by_board = {}
    for pdir, name, bpm, board in targets:
        by_board.setdefault(board, []).append((pdir, name, bpm))
    for b, v in sorted(by_board.items()):
        span = f"{min(x[2] for x in v)}-{max(x[2] for x in v)} BPM"
        print(f"   {b:<28}{len(v):>4}   {span}, holds "
              f"{hold_for(max(x[2] for x in v)):.1f}-{hold_for(min(x[2] for x in v)):.1f}s")
    if args.dry_run:
        print("\n--dry-run: nothing re-rendered")
        return

    jobs = []
    skipped = 0
    for pdir, name, bpm, board in targets:
        # Skip only if the patch is ALREADY rendered long enough for its own
        # tempo. Checking for the mere presence of .wav files was wrong: after
        # an interrupted run was repaired with the standard batch renderer,
        # every target had .wav again -- at the default 3.5 s hold -- so the
        # whole job skipped itself and re-rendered nothing.
        #
        # Note a fast phrase legitimately needs barely more than the default
        # (136 BPM wants 4.06 s), so this compares against the hold THIS
        # patch requires, not against a fixed multiple of the default.
        if rendered_seconds(pdir) >= hold_for(bpm) - 0.15:
            skipped += 1
            continue
        # Index comes from the directory prefix, not patch.json: an orphaned
        # directory (interrupted render) has no patch.json, and those are
        # precisely the ones needing repair. The prefix is the sampler's own
        # enumeration index, which is what --patch expects.
        m = DIR_NAME.match(pdir.name)
        if m:
            index = int(m.group(1))
        else:
            index = int(json.loads((pdir / "patch.json").read_text())["index"])
        jobs.append((pdir, name, bpm, board, args.roms, args.sampler, index))
    if skipped:
        print(f"skipping {skipped} already re-rendered")

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name, bpm, hold, rc, err in ex.map(render_one, jobs):
            if rc == 0:
                ok += 1
            else:
                fail += 1
                print(f"   FAILED {name} ({bpm} BPM): {err}", file=sys.stderr)
            if (ok + fail) % 25 == 0:
                print(f"   {ok + fail}/{len(jobs)} rendered")
    print(f"\nre-rendered {ok}, failed {fail}")


if __name__ == "__main__":
    main()
