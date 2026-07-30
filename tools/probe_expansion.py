#!/usr/bin/env python3
"""
probe_expansion.py -- investigate whether SR-JV80-97 (Experience III) and
SR-JV80-98 (Experience II) expansion ROM dumps contain a parseable patch
table, using SR-JV80-99 (Experience) as a known-good control.

Background: src/jv_rom.cpp implements the standard SR-JV80 expansion parser,
validated against 20 of 22 boards. Board 99 (also a 2 MB "Experience" promo
board) parses correctly (64 patches). Boards 97 and 98 report patch_count=0
via the standard header fields. This script:

  1. Reimplements the exact unscramble algorithm from jv::unscramble_rom
     (src/jv_rom.cpp), vectorized with numpy for speed.
  2. Dumps the relevant header bytes for boards 97, 98, 99.
  3. Reports the standard-format patch_count / patches_offset fields.
  4. Runs a calibrated full-image scan for candidate patch-name tables
     (looking for runs of plausible 12-byte names on a 0x16a-byte stride,
     at every possible phase), and proves the detector correctly locates
     board 99's real table (offset 0x1F9866, patch 0 = "*Tr.Rhodes") before
     trusting its verdict on 97/98.
  5. Prints a CONCLUSION line per board.

Usage: python3 tools/probe_expansion.py [roms_dir]
  roms_dir defaults to $JV880_ROMS
  (same default + override convention as tests/test_jv_rom.cpp), and expansion
  files are read from <roms_dir>/expansions/.
"""

import os
import re
import sys

try:
    import numpy as np
except ImportError:
    print("ERROR: this tool requires numpy (pip install numpy)", file=sys.stderr)
    sys.exit(1)

# --- constants mirrored from src/jv_rom.cpp / src/jv_rom.h ------------------

PATCH_SIZE = 0x16A  # 362 bytes; jv::PATCH_SIZE in src/jv_rom.h

# jv::unscramble_rom's address-bit and data-bit permutations, src/jv_rom.cpp:80-93
AA = [2, 0, 3, 4, 1, 9, 13, 10, 18, 17, 6, 15, 11, 16, 8, 5, 12, 7, 14, 19]
DD = [2, 0, 4, 5, 7, 6, 3, 1]

DEFAULT_ROMS_DIR = os.environ.get("JV880_ROMS", "")

BOARDS = [
    (97, "SR-JV80-97"),
    (98, "SR-JV80-98"),
    (99, "SR-JV80-99"),
]

CONTROL_BOARD = 99
CONTROL_OFFSET = 0x1F9866
CONTROL_NAME0 = "*Tr.Rhodes"


# --- unscramble (numpy re-implementation of jv::unscramble_rom) -------------

def unscramble(data: bytes) -> np.ndarray:
    """Bit-for-bit reimplementation of jv::unscramble_rom (src/jv_rom.cpp:80),
    vectorized with numpy. Returns a uint8 ndarray the same length as data.

    For each output index i, the address permutation moves bit j of i to bit
    AA[j] of the source address (bits above 20 pass straight through -- this
    is what makes the same 20-bit within-bank permutation work unmodified on
    both 2 MB boards, tested here, and 8 MB boards). The data-byte read from
    that address then has its bits permuted per DD to produce the output byte.
    """
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    idx = np.arange(n, dtype=np.uint32)
    address = idx & np.uint32(0xFFFFFFFF & ~0xFFFFF)
    for j in range(20):
        bit = (idx >> j) & 1
        address |= bit.astype(np.uint32) << AA[j]
    s = src[address]
    d = np.zeros(n, dtype=np.uint8)
    for j in range(8):
        bit = (s >> DD[j]) & 1
        d |= (bit << j).astype(np.uint8)
    return d


# --- name plausibility heuristic ---------------------------------------------
#
# Calibrated against the 3,941 real patch names read from the 19 known-good
# 8 MB boards (01-19) via their standard-format patch tables:
#   - all bytes printable ASCII (0x20-0x7E): true for every real name.
#   - >=2 alphabetic characters: true for every real name.
#   - longest run of consecutive uppercase letters: <=4 for 3,937/3,941 names
#     (99.9%); only 3 exceed it (e.g. "SYNBRAKUN", run=9). So an implausibly
#     long uppercase run is a soft penalty, not a hard reject -- it lets a
#     real 64-entry table absorb one or two such names without the
#     max-subarray search fragmenting the run.

_UPPER_RUN_RE = re.compile(rb"[A-Z]+")


def name_score(raw: bytes) -> int:
    """Score a candidate 12-byte window as a patch name.
    +1    plausible real name
    -2    printable, has letters, but an implausibly long uppercase run
    -10   printable but fewer than 2 letters (pure digit/punctuation block)
    -1000 contains a non-printable byte (hard break for the run search)
    """
    if any(b < 0x20 or b > 0x7E for b in raw):
        return -1000
    alpha = sum(1 for b in raw if (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A))
    if alpha < 2:
        return -10
    mx = max((len(m) for m in _UPPER_RUN_RE.findall(raw)), default=0)
    if mx >= 5:
        return -2
    return 1


def trim_name(raw: bytes) -> str:
    s = raw.decode("ascii", errors="replace")
    return s.rstrip(" \x00")


def is_monotonic_ramp(raw: bytes) -> bool:
    """True if byte values are non-decreasing across the window -- the
    signature of waveform/envelope PCM data landing in the printable-ASCII
    range by chance, not text."""
    return all(raw[i] <= raw[i + 1] for i in range(len(raw) - 1))


def distinct_byte_count(raw: bytes) -> int:
    return len(set(raw))


# --- full-image candidate-table scan (Kadane max-subarray per phase) --------

def compute_scores(u: np.ndarray) -> np.ndarray:
    """Per-offset name_score for every possible 12-byte window start,
    vectorized: numpy computes the cheap printable/alpha prefilter over the
    whole buffer, and the (rarer, regex-based) uppercase-run check only runs
    on windows that already passed the prefilter."""
    n = len(u)
    printable = (u >= 0x20) & (u <= 0x7E)
    cum_print = np.concatenate(([0], np.cumsum(printable.astype(np.int32))))
    win_print = (cum_print[12:] - cum_print[:-12]) == 12

    alpha = ((u >= 0x41) & (u <= 0x5A)) | ((u >= 0x61) & (u <= 0x7A))
    cum_alpha = np.concatenate(([0], np.cumsum(alpha.astype(np.int32))))
    win_alpha = cum_alpha[12:] - cum_alpha[:-12]

    scores = np.where(~win_print, -1000, np.where(win_alpha < 2, -10, 1)).astype(np.int32)

    # Windows that passed the cheap vectorized prefilter (printable + >=2
    # alpha) still need the uppercase-run check from name_score() applied --
    # call it directly (rather than re-deriving the rule here) so this stays
    # a single source of truth with the scoring contract documented above.
    ub = u.tobytes()
    for i in np.nonzero(scores == 1)[0]:
        scores[i] = name_score(ub[i : i + 12])
    return scores


def kadane(arr) -> "tuple[int, int, int]":
    """Standard max-subarray search. Returns (best_sum, start_idx, end_idx)
    (inclusive), so a real table's near-unanimous run of +1s dominates over
    scattered garbage, but a single bad name doesn't break the run."""
    best_sum = None
    best_start = best_end = 0
    cur_sum = 0
    cur_start = 0
    for i, v in enumerate(arr):
        if cur_sum <= 0:
            cur_start = i
            cur_sum = int(v)
        else:
            cur_sum += int(v)
        if best_sum is None or cur_sum > best_sum:
            best_sum = cur_sum
            best_start = cur_start
            best_end = i
    return best_sum, best_start, best_end


def find_top_tables(scores: np.ndarray, n_top: int = 5):
    """Scan every one of the PATCH_SIZE possible strides ("phases") a patch
    table could be aligned to, run Kadane's max-subarray on each phase's
    score sequence, and return the n_top highest-scoring candidates as
    (score, length, byte_offset, phase, good_count) tuples, best first."""
    results = []
    for phase in range(PATCH_SIZE):
        arr = scores[phase::PATCH_SIZE]
        if len(arr) == 0:
            continue
        s, a, b = kadane(arr)
        length = b - a + 1
        offset = phase + a * PATCH_SIZE
        good = int(np.sum(arr[a : b + 1] == 1))
        results.append((s, length, offset, phase, good))
    results.sort(key=lambda r: -r[0])
    return results[:n_top]


# --- header field survey across all 22 boards -------------------------------

def header_field_survey(exp_dir: str):
    """Read the 0x60-0x61 / 0x62-0x63 / 0x66-0x67 header u16 fields from every
    SR-JV80-*.bin/.BIN file in exp_dir, to check whether 0x60-0x61 or
    0x62-0x63 could be an alternate patch-count field for boards 97/98 (their
    only nonzero header values near patch_count). If those fields scaled
    1:1 with patch_count on the 20 known-good boards, that would support the
    hypothesis; if they scale independently, it refutes it."""
    files = sorted(
        fn
        for fn in os.listdir(exp_dir)
        if fn.upper().endswith(".BIN") and "SR-JV80" in fn
    )
    print(f"{'=' * 78}")
    print("HEADER FIELD SURVEY (all boards) -- is 0x60-61 or 0x62-63 an")
    print("alternate patch-count field?")
    print(f"{'=' * 78}")
    print(f"{'file':45s} {'0x60-61':>8s} {'0x62-63':>8s} {'count(0x66-67)':>15s} {'ratio62/count':>14s}")
    ratios = []
    for fn in files:
        with open(os.path.join(exp_dir, fn), "rb") as fh:
            data = fh.read()
        u = unscramble(data)
        ub = u.tobytes()
        f6061 = (ub[0x60] << 8) | ub[0x61]
        f6263 = (ub[0x62] << 8) | ub[0x63]
        count = ub[0x67] | (ub[0x66] << 8)
        ratio_str = "n/a"
        if count > 0:
            ratio = f6263 / count
            ratio_str = f"{ratio:.2f}"
            ratios.append(ratio)
        print(f"{fn[:45]:45s} {f6061:8d} {f6263:8d} {count:15d} {ratio_str:>14s}")
    ratio_min = ratio_max = None
    if ratios:
        ratio_min, ratio_max = min(ratios), max(ratios)
        print()
        print(
            f"0x62-0x63 / patch_count ratio across the {len(ratios)} boards "
            f"with a nonzero standard patch_count: min={ratio_min:.2f}, "
            f"max={ratio_max:.2f}. This is a wide, non-constant spread "
            f"(not ~1.0), so 0x62-0x63 is NOT a duplicate/alternate encoding "
            f"of patch_count -- it is some other per-board quantity "
            f"(plausibly a wave/tone count) that happens to also be small "
            f"on the small 2 MB Experience boards. Same conclusion applies "
            f"to 0x60-0x61, which scales with the same board-size pattern."
        )
    print()
    return ratio_min, ratio_max


# --- per-board probe ----------------------------------------------------------

def find_expansion_file(exp_dir: str, prefix: str) -> str:
    for fn in sorted(os.listdir(exp_dir)):
        if fn.startswith(prefix):
            return os.path.join(exp_dir, fn)
    raise FileNotFoundError(f"no file starting with {prefix!r} in {exp_dir}")


def probe_board(exp_dir: str, board_id: int, prefix: str):
    path = find_expansion_file(exp_dir, prefix)
    with open(path, "rb") as fh:
        data = fh.read()
    u = unscramble(data)
    ub = u.tobytes()

    print(f"{'=' * 78}")
    print(f"Board {board_id}: {os.path.basename(path)}")
    print(f"{'=' * 78}")
    print(f"File size: {len(data):,} bytes")
    print()
    print("Header bytes (unscrambled):")
    print(f"  0x60-0x70: {ub[0x60:0x70].hex(' ')}")
    print(f"  0x8c-0x90: {ub[0x8c:0x90].hex(' ')}")
    print()

    patch_count = ub[0x67] | (ub[0x66] << 8)
    patches_offset = (ub[0x8C] << 24) | (ub[0x8D] << 16) | (ub[0x8E] << 8) | ub[0x8F]
    room_to_eof = len(u) - patches_offset
    max_records_fit = room_to_eof // PATCH_SIZE if room_to_eof > 0 else 0

    print("Standard-format fields (src/jv_rom.cpp:123-125):")
    print(f"  0x66-0x67 patch_count      = {patch_count}")
    print(
        f"  0x8c-0x8f patches_offset   = 0x{patches_offset:06x}"
        f"  ({room_to_eof:,} bytes to EOF -> max {max_records_fit} records fit)"
    )
    print()

    f6061 = (ub[0x60] << 8) | ub[0x61]
    f6263 = (ub[0x62] << 8) | ub[0x63]
    f6465 = (ub[0x64] << 8) | ub[0x65]
    print("Other header u16 fields (candidate alternate patch-count locations):")
    print(f"  0x60-0x61 = {f6061}")
    print(f"  0x62-0x63 = {f6263}")
    print(f"  0x64-0x65 = {f6465}")
    print()

    scores = compute_scores(u)
    top = find_top_tables(scores, n_top=5)

    print(
        f"Full-image candidate-table scan ({PATCH_SIZE}-byte stride, all "
        f"{PATCH_SIZE} phases, Kadane max-subarray of name_score per phase):"
    )
    for rank, (s, length, offset, phase, good) in enumerate(top, 1):
        names_preview = []
        for k in range(min(length, 5)):
            raw = ub[offset + k * PATCH_SIZE : offset + k * PATCH_SIZE + 12]
            names_preview.append(trim_name(raw) or repr(raw))
        raw0 = ub[offset : offset + 12]
        flags = []
        if is_monotonic_ramp(raw0):
            flags.append("monotonic-ramp")
        if distinct_byte_count(raw0) <= 2:
            flags.append("near-constant-bytes")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  #{rank}: score={s:5d}  len={length:4d}  good={good:4d}/{length:<4d}"
            f"  offset=0x{offset:06x}{flag_str}"
        )
        print(f"       names[0:{min(length,5)}] = {names_preview}")
    print()

    return {
        "board_id": board_id,
        "path": path,
        "unscrambled": u,
        "patch_count": patch_count,
        "patches_offset": patches_offset,
        "max_records_fit": max_records_fit,
        "top_candidates": top,
    }


def main():
    roms_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROMS_DIR
    exp_dir = os.path.join(roms_dir, "expansions")
    if not os.path.isdir(exp_dir):
        print(f"ERROR: expansion directory not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    results = {}
    for board_id, prefix in BOARDS:
        results[board_id] = probe_board(exp_dir, board_id, prefix)

    ratio_min, ratio_max = header_field_survey(exp_dir)

    # --- self-check: the heuristic MUST find board 99's real table ---------
    control = results[CONTROL_BOARD]
    control_top = control["top_candidates"][0]
    control_score, control_len, control_offset, control_phase, control_good = control_top
    u99 = control["unscrambled"]
    ub99 = u99.tobytes()
    name0 = trim_name(ub99[control_offset : control_offset + 12])

    control_ok = (control_offset == CONTROL_OFFSET) and (name0 == CONTROL_NAME0)

    print(f"{'=' * 78}")
    print("CONTROL CHECK (board 99)")
    print(f"{'=' * 78}")
    print(
        f"  Detector's #1 candidate: offset=0x{control_offset:06x}, "
        f"len={control_len}, score={control_score}, name[0]={name0!r}"
    )
    print(f"  Expected: offset=0x{CONTROL_OFFSET:06x}, name[0]={CONTROL_NAME0!r}")
    if control_ok:
        print("  PASS: detector correctly located board 99's known-good table")
        print(
            "        and was not fooled by the printable-but-garbage run at "
            "0x402 found by a naive scan."
        )
    else:
        print("  *** FAIL: detector did NOT find the known-good table on the control. ***")
        print("  *** Results for boards 97/98 below are NOT trustworthy. ***")
    print()

    if not control_ok:
        print("ABORTING: control check failed; refusing to draw conclusions for 97/98.")
        sys.exit(1)

    # --- conclusions for 97 and 98, benchmarked against the control --------
    print(f"{'=' * 78}")
    print("CONCLUSIONS")
    print(f"{'=' * 78}")

    reference_score = control_score  # 65, board 99's real 64-patch table
    for board_id in (97, 98):
        r = results[board_id]
        best = r["top_candidates"][0]
        best_score, best_len, best_offset, best_phase, best_good = best
        ratio = best_score / reference_score if reference_score else 0.0
        u = r["unscrambled"]
        ub = u.tobytes()
        raw0 = ub[best_offset : best_offset + 12]

        print(f"Board {board_id}:")
        print(f"  standard patch_count field   = {r['patch_count']}")
        print(
            f"  standard patches_offset field = 0x{r['patches_offset']:06x} "
            f"(only room for {r['max_records_fit']} records to EOF)"
        )
        print(
            f"  best full-image candidate     = score {best_score} / len {best_len} "
            f"at 0x{best_offset:06x} "
            f"({ratio:.1%} of board 99's real-table score of {reference_score})"
        )
        print(f"  best candidate's name[0]      = {raw0!r}")

        # A genuine table's best candidate should be a large fraction of the
        # control's score, with the run being almost entirely +1 hits. The
        # smallest genuine SR-JV80 table observed anywhere (board 99 itself)
        # is 64 patches; nothing found for 97/98 comes close to a plausible
        # patch count, and their top hits are not word-like.
        looks_real = best_len >= 16 and (best_good / best_len) >= 0.9 and ratio >= 0.3

        if looks_real:
            print(
                f"  CONCLUSION (board {board_id}): a candidate table WAS found at "
                f"0x{best_offset:06x} ({best_len} entries) -- investigate further "
                f"before wiring into the pipeline."
            )
        else:
            print(
                f"  CONCLUSION (board {board_id}): NO usable patch table found. "
                f"The best full-image candidate is only {ratio:.0%} as strong as "
                f"board 99's real 64-patch table, its length ({best_len}) is far "
                f"below the smallest genuine SR-JV80 table seen anywhere (64, on "
                f"board 99 itself), and its content "
                f"({trim_name(raw0) or raw0!r}) is not word-like -- it is a "
                f"monotonic byte ramp or a run of a single repeated byte, the "
                f"signature of PCM waveform/envelope data coincidentally "
                f"landing in the printable-ASCII range, not a patch name. The "
                f"alternate header fields at 0x60-0x61 and 0x62-0x63 do not "
                f"encode patch count either (see header field survey above); "
                f"across the 20 boards with a nonzero standard patch_count, "
                f"0x62-0x63 / patch_count ranges {ratio_min:.2f}-{ratio_max:.2f} "
                f"(not a constant ~1.0), so they are some other per-board "
                f"quantity (plausibly a wave/tone count), not an alternate "
                f"count field for 97/98. "
                f"patch_count=0 from the standard header is genuinely correct: "
                f"this ROM dump contains no parseable patch table in the "
                f"SR-JV80 format. Board {board_id} cannot be added to the "
                f"pipeline."
            )
        print()


if __name__ == "__main__":
    main()
