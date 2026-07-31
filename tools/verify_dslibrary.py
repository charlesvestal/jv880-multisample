#!/usr/bin/env python3
"""Check that every packaged .dslibrary is self-contained.

A preset references two kinds of file by relative path: its samples, and the
convolution impulse response for its reverb. Both must be inside the same
archive, or DecentSampler fails in a way that reads as something else -- a
missing IR renders no reverb at all, which was once reported as "the reverb is
too quiet" and cost a day.

Splitting a board into chunks makes this sharper. Each chunk carries only the
files its OWN presets use, and different presets need different IRs (one per
reverb type and time step), so a packager that collected samples but forgot
IRs, or that staged one chunk's IRs into another, would produce archives that
look complete and play dry.

    python3 tools/verify_dslibrary.py <dir of .dslibrary files>
"""
import argparse
import posixpath
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def check(path):
    """Return (presets, samples, irs, [problems])."""
    problems = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        presets = sorted(n for n in names if n.endswith(".dspreset"))
        if not presets:
            return 0, 0, 0, ["contains no .dspreset"]

        samples, irs = set(), set()
        for p in presets:
            base = posixpath.dirname(p)
            try:
                root = ET.fromstring(z.read(p))
            except ET.ParseError as exc:
                problems.append(f"{posixpath.basename(p)}: XML parse error: {exc}")
                continue

            for s in root.findall(".//sample"):
                rel = s.get("path")
                if not rel:
                    problems.append(f"{posixpath.basename(p)}: <sample> with no path")
                    continue
                target = posixpath.normpath(posixpath.join(base, rel))
                samples.add(target)
                if target not in names:
                    problems.append(f"{posixpath.basename(p)}: missing sample {rel}")

            for e in root.findall('.//effect[@type="convolution"]'):
                rel = e.get("irFile")
                if not rel:
                    problems.append(f"{posixpath.basename(p)}: convolution with no irFile")
                    continue
                target = posixpath.normpath(posixpath.join(base, rel))
                irs.add(target)
                if target not in names:
                    problems.append(f"{posixpath.basename(p)}: missing IR {rel}")

            # A group routed to a bus needs a bus to arrive at, or the send
            # goes nowhere and the reverb is silently absent.
            group = root.find(".//group")
            if group is not None and (group.get("output2Target") or "").startswith("BUS_"):
                if root.find("./buses/bus") is None:
                    problems.append(f"{posixpath.basename(p)}: routes to a bus it does not declare")

        # Files carried but referenced by nothing: harmless, but a sign the
        # chunker is copying more than the chunk needs.
        carried = {n for n in names if n.endswith((".flac", ".wav"))}
        orphans = carried - samples - irs
        if orphans:
            problems.append(f"{len(orphans)} audio file(s) referenced by no preset")

    return len(presets), len(samples), len(irs), problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    args = ap.parse_args()

    libs = sorted(Path(args.directory).glob("*.dslibrary"))
    if not libs:
        sys.exit(f"no .dslibrary files in {args.directory}")

    failed = 0
    print(f"{'library':<46}{'presets':>8}{'samples':>9}{'IRs':>5}  status")
    for lib in libs:
        n, s, i, problems = check(lib)
        status = "ok" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"  {lib.name[:44]:<44}{n:>8}{s:>9}{i:>5}  {status}")
        for p in problems[:5]:
            print(f"       {p}")
        if problems:
            failed += 1

    print(f"\n{len(libs)} libraries checked, {failed} with problems")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
