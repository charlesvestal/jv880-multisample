#!/usr/bin/env python3
"""Render JV-880 boards in parallel, resumably.

Drives the built `jv_sampler` binary (Task 3) across every patch on one or
more boards, distributing work across CPU cores via ProcessPoolExecutor.
Safe to interrupt and re-run: any patch directory whose patch.json is
complete (non-empty zones list, every zone's file present on disk) is
skipped rather than re-rendered.

Usage:
    tools/run_batch.py --list
    tools/run_batch.py --board "SR-JV80-01 Pop"
    tools/run_batch.py --board all --jobs 8
    tools/run_batch.py --board "SR-JV80-01 Pop" --limit 2 --out /tmp/probe

Must be run from the repository root (the sampler binary path below is
relative, matching how the rest of the pipeline invokes it).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROMS = Path(os.environ.get("JV880_ROMS", ""))
OUT_ROOT = Path(os.environ.get("JV880_OUT", ""))
SAMPLER = Path("build/jv_sampler")

# Conservative raw-WAV footprint per patch before post-processing: 75 zones
# at 64 kHz/16-bit/stereo, several seconds each. This is a fail-fast sanity
# estimate for preflight, not an accounting of actual usage.
BYTES_PER_PATCH_ESTIMATE = 75 * 1_000_000  # 75 MB

STDERR_TAIL_LINES = 20
PROGRESS_INTERVAL_S = 30

# concurrent.futures.as_completed()'s default indefinite wait
# (Event.wait(None)) is not reliably interrupted by SIGINT -- it can swallow
# Ctrl-C for the remainder of a board's run (observed empirically: a 20s
# render ran to completion despite SIGINT at t=4s). Polling with a bounded
# timeout instead guarantees control returns to Python bytecode at least
# this often, which is where the pending KeyboardInterrupt actually gets
# raised.
WAIT_POLL_S = 1.0


# --------------------------------------------------------------------------
# Board discovery
# --------------------------------------------------------------------------

def list_boards() -> list[tuple[str, int]]:
    """Run `jv_sampler --list` and parse stdout into [(name, count), ...].

    Only stdout is parsed. jv_sampler writes diagnostics about unusable
    boards (e.g. the 97/98 Experience II/III slots) to stderr; those must
    never be mistaken for board lines, so stderr is ignored entirely here.
    Each stdout line is "<name>\t<count>" -- names can contain almost
    anything except a tab, so split from the right per the reviewed format.
    """
    proc = subprocess.run(
        [str(SAMPLER), "--roms", str(ROMS), "--list"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"jv_sampler --list failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}"
        )
    boards = []
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        name, count = line.rsplit("\t", 1)
        boards.append((name, int(count)))
    return boards


# --------------------------------------------------------------------------
# Resumability
# --------------------------------------------------------------------------

def patch_done(pdir: Path) -> bool:
    """A patch directory counts as complete iff patch.json parses, has a
    non-empty "zones" list, and every zone's referenced file exists.

    Zones may reference either the raw .wav render (this tool's output) or
    a post-processed .flac (tools/postprocess.py rewrites zone["file"] in
    place, as a separate later stage) -- both count as complete. Treating
    only .wav as done would re-render the entire library the moment
    post-processing starts converting files.
    """
    meta_path = pdir / "patch.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    zones = meta.get("zones")
    if not zones:
        return False
    for z in zones:
        fn = z.get("file") if isinstance(z, dict) else None
        if not fn or not (pdir / fn).is_file():
            return False
    return True


def find_patch_dir(board_out: Path, index: int) -> Path | None:
    """Locate the `<index:03d>_<sanitized name>` dir for a patch index.

    Avoids reimplementing jv_sampler's C++ sanitize() in Python: the
    zero-padded index prefix is unique per board, so a glob on it is exact.
    """
    prefix = f"{index:03d}_"
    matches = sorted(p for p in board_out.glob(prefix + "*") if p.is_dir())
    return matches[0] if matches else None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _ignore_sigint() -> None:
    """ProcessPoolExecutor initializer: workers ignore SIGINT so a Ctrl-C
    at the terminal is handled once, by the main process, instead of
    raising KeyboardInterrupt independently in every worker and each
    printing its own traceback.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def render_one(board: str, index: int, board_out: Path) -> tuple[str, int, int, str]:
    """Render a single patch via a `jv_sampler --patch N` subprocess.

    Returns (board, index, returncode, stderr_tail). Never raises: even a
    failure to launch the subprocess is reported through the return value
    so one bad patch can never take down the worker pool.
    """
    try:
        proc = subprocess.run(
            [str(SAMPLER), "--roms", str(ROMS), "--board", board,
             "--out", str(board_out), "--patch", str(index)],
            capture_output=True, text=True, check=False,
        )
        rc = proc.returncode
        stderr = proc.stderr or ""
    except OSError as e:
        rc = -1
        stderr = str(e)
    tail = "\n".join(stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
    return board, index, rc, tail


class Interrupted(Exception):
    """Raised out of render_board() on Ctrl-C, carrying whatever partial
    results had already landed so main() doesn't lose them.
    """

    def __init__(self, failures: list[dict], skipped: int, attempted: int):
        super().__init__("interrupted")
        self.failures = failures
        self.skipped = skipped
        self.attempted = attempted


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def render_board(
    executor: ProcessPoolExecutor,
    board: str,
    total_patches: int,
    out_root: Path,
    limit: int | None,
) -> tuple[list[dict], int, int]:
    """Render every not-yet-complete patch on one board.

    Returns (failures, skipped_count, attempted_count). Raises Interrupted
    on Ctrl-C after cancelling not-yet-started work.
    """
    board_out = out_root / board
    board_out.mkdir(parents=True, exist_ok=True)

    n = total_patches if limit is None else min(limit, total_patches)
    todo = []
    skipped = 0
    for i in range(n):
        existing = find_patch_dir(board_out, i)
        if existing is not None and patch_done(existing):
            skipped += 1
            continue
        todo.append(i)

    print(f"[{board}] {n} patches total, {skipped} already complete, "
          f"{len(todo)} to render", flush=True)
    if not todo:
        return [], skipped, 0

    failures: list[dict] = []
    done = 0
    start = time.monotonic()
    last_report = start

    futures = {executor.submit(render_one, board, i, board_out): i for i in todo}
    pending = set(futures)
    try:
        while pending:
            # Bounded timeout, not as_completed()'s default indefinite wait
            # -- see WAIT_POLL_S comment. This is what makes Ctrl-C actually
            # work instead of blocking silently through it.
            finished, pending = wait(pending, timeout=WAIT_POLL_S,
                                      return_when=FIRST_COMPLETED)
            for fut in finished:
                _, index, rc, stderr_tail = fut.result()
                done += 1
                if rc != 0:
                    failures.append({
                        "board": board, "index": index,
                        "returncode": rc, "stderr_tail": stderr_tail,
                    })
                    last_line = stderr_tail.splitlines()[-1] if stderr_tail else ""
                    print(f"[{board}] FAILED patch {index:03d} (exit {rc}): {last_line}",
                          file=sys.stderr, flush=True)

                now = time.monotonic()
                if now - last_report >= PROGRESS_INTERVAL_S or done == len(todo):
                    elapsed = now - start
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = len(todo) - done
                    eta = remaining / rate if rate > 0 else None
                    print(f"[{board}] {done}/{len(todo)} rendered "
                          f"({len(failures)} failed) -- elapsed {fmt_duration(elapsed)}, "
                          f"ETA {fmt_duration(eta)}", flush=True)
                    last_report = now
    except KeyboardInterrupt:
        print(f"\n[{board}] interrupt received -- cancelling queued patches "
              f"and waiting for in-flight ones to stop...", file=sys.stderr, flush=True)
        for f in pending:
            f.cancel()
        raise Interrupted(failures, skipped, done)

    ok = len(todo) - len(failures)
    print(f"[{board}] finished: {ok} rendered, {len(failures)} failed, "
          f"{skipped} already complete", flush=True)
    return failures, skipped, len(todo)


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

def preflight_binary_and_roms() -> list[str]:
    problems = []
    if not SAMPLER.is_file():
        problems.append(f"sampler binary not found: {SAMPLER.resolve()} "
                         f"(build it first)")
    elif not os.access(SAMPLER, os.X_OK):
        problems.append(f"sampler binary is not executable: {SAMPLER.resolve()}")
    if not ROMS.is_dir():
        problems.append(f"ROMs directory not found: {ROMS}")
    return problems


def existing_ancestor(p: Path) -> Path:
    p = p.resolve()
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent
    return p


def check_disk_space(out_root: Path, n_patches: int) -> tuple[bool, str | None]:
    needed = n_patches * BYTES_PER_PATCH_ESTIMATE
    probe = existing_ancestor(out_root)
    usage = shutil.disk_usage(str(probe))
    if usage.free < needed:
        return False, (
            f"not enough free space at {probe} to render {n_patches} patches: "
            f"need ~{needed / 1e9:.1f} GB, have {usage.free / 1e9:.1f} GB free"
        )
    return True, None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    global ROMS
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", help='board name (exact, as printed by --list), or "all"')
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                     help="parallel worker processes (default: os.cpu_count())")
    ap.add_argument("--list", action="store_true",
                     help="print board names and patch counts, then exit")
    ap.add_argument("--limit", type=int, default=None,
                     help="render at most N patches per board (testing)")
    ap.add_argument("--roms", type=Path, default=ROMS,
                     help="directory holding the JV-880 ROM images "
                          "(default: $JV880_ROMS)")
    ap.add_argument("--out", type=Path, default=OUT_ROOT,
                     help=f"output root (default: {OUT_ROOT}). "
                          f"NEVER point this at the internal drive -- a full "
                          f"run is 250-315 GB.")
    args = ap.parse_args()
    # --roms overrides the module-level default. Setting the environment too,
    # not just the global: workers run in separate processes that re-import
    # this module and would otherwise read the ORIGINAL env value, which is
    # how a run with --roms failed with "cannot open ./jv880_rom1.bin".
    ROMS = args.roms
    os.environ["JV880_ROMS"] = str(args.roms)
    os.environ["JV880_OUT"] = str(args.out)

    if args.list:
        try:
            boards = list_boards()
        except (RuntimeError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for name, count in boards:
            print(f"{name}\t{count}")
        total = sum(c for _, c in boards)
        print(f"\n{len(boards)} libraries totalling {total:,} patches")
        return 0

    if not args.board:
        print("error: --board <name|all> is required (use --list to see names)",
              file=sys.stderr)
        return 1
    if args.jobs < 1:
        print("error: --jobs must be >= 1", file=sys.stderr)
        return 1

    problems = preflight_binary_and_roms()
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1

    out_root: Path = args.out
    free_gb = shutil.disk_usage(out_root.parent if out_root.exists() else Path.cwd()).free / 2**30
    if free_gb < 200:
        print(f"warning: only {free_gb:.0f} GB free at {out_root}. A full "
              f"21-board render needs roughly 250-315 GB.", file=sys.stderr)

    try:
        boards = list_boards()
    except (RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    board_map = dict(boards)

    if args.board == "all":
        targets = boards
    else:
        if args.board not in board_map:
            print(f"error: unknown board {args.board!r}. "
                  f"Use --list to see valid names.", file=sys.stderr)
            return 1
        targets = [(args.board, board_map[args.board])]

    total_needed_patches = sum(
        (count if args.limit is None else min(args.limit, count))
        for _, count in targets
    )
    ok, msg = check_disk_space(out_root, total_needed_patches)
    if not ok:
        print(f"error: {msg}", file=sys.stderr)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)

    all_failures: list[dict] = []
    total_skipped = 0
    total_attempted = 0
    interrupted = False
    run_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_ignore_sigint) as ex:
        for board, count in targets:
            try:
                failures, skipped, attempted = render_board(ex, board, count, out_root, args.limit)
            except Interrupted as intr:
                all_failures.extend(intr.failures)
                total_skipped += intr.skipped
                total_attempted += intr.attempted
                interrupted = True
                print("waiting for the worker pool to drain...", file=sys.stderr, flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break
            all_failures.extend(failures)
            total_skipped += skipped
            total_attempted += attempted

    elapsed = time.monotonic() - run_start

    failures_path = out_root / "render_failures.json"
    if all_failures:
        failures_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "board_arg": args.board,
            "failures": all_failures,
        }, indent=2))
    elif failures_path.exists():
        failures_path.unlink()

    ok_count = total_attempted - len(all_failures)
    print("\n" + "=" * 60)
    print(f"run summary ({len(targets)} board(s) targeted):")
    print(f"  already complete (skipped): {total_skipped}")
    print(f"  rendered ok:                {ok_count}")
    print(f"  failed:                     {len(all_failures)}")
    if all_failures:
        print(f"  failure details:            {failures_path}")
    print(f"  elapsed:                    {fmt_duration(elapsed)}")
    if interrupted:
        print("  ** interrupted by user (Ctrl-C) -- re-run to resume **")
    print("=" * 60)

    if interrupted:
        return 130
    return 1 if all_failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
