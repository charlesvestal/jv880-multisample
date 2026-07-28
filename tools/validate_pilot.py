#!/usr/bin/env python3
"""Validate a rendered JV-880 library against the pilot acceptance criteria.

Checks structure, audio format, loop quality, and preset validity across every
library under the given root. Exits non-zero if any check fails, so this can
gate the full render.

Usage:  python3 tools/validate_pilot.py "/Volumes/ExtFS/charlesvestal/JV-880 Multisamples"
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

EXPECTED_ZONES = 75
MIN_LOOP_FRACTION = 0.25
MAX_LOOP_DISCONTINUITY = 0.05   # fraction of that sample's peak

errors: list[str] = []
warnings: list[str] = []


def attr(el: ET.Element, name: str, where: str) -> str | None:
    """Fetch a required XML attribute, recording an error if it's absent.

    A malformed preset must produce a clear validation failure, not a
    TypeError from passing None into int() or Path.__truediv__.
    """
    v = el.get(name)
    if v is None:
        errors.append(f"{where}: <{el.tag}> missing required attribute '{name}'")
    return v


def iattr(el: ET.Element, name: str, where: str) -> int | None:
    v = attr(el, name, where)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        errors.append(f"{where}: <{el.tag}> attribute '{name}' is not an integer: {v!r}")
        return None


def check_audio(pdir: Path, meta: dict) -> tuple[int, int, int]:
    """Return (looped, total, skipped) zone counts for one patch."""
    looped = total = skipped = 0
    for z in meta.get("zones", []):
        # Zones that never rendered or failed post-processing are excluded from
        # the presets by design; don't hold their missing audio against us.
        if z.get("kind") in ("missing", "error"):
            skipped += 1
            continue
        total += 1
        f = pdir / z["file"]
        if not f.exists():
            errors.append(f"{pdir.name}/{z['file']}: missing")
            continue

        info = sf.info(str(f))
        if info.samplerate != 48000:
            errors.append(f"{f.name}: {info.samplerate} Hz, expected 48000")
        if info.channels != 2:
            errors.append(f"{f.name}: {info.channels} channels, expected 2")
        if "24" not in info.subtype:
            errors.append(f"{f.name}: subtype {info.subtype}, expected 24-bit")

        loop = z.get("loop", {})
        if not loop.get("enabled"):
            continue
        looped += 1
        x, _ = sf.read(str(f), always_2d=True)
        s, e = loop["start"], loop["end"]
        if s >= e or e >= len(x):
            errors.append(f"{f.name}: loop {s}-{e} out of range (len {len(x)})")
            continue
        peak = float(np.abs(x).max()) or 1.0
        disc = float(np.abs(x[s] - x[e]).max())
        if disc > MAX_LOOP_DISCONTINUITY * peak:
            errors.append(f"{f.name}: loop discontinuity {disc / peak:.1%}")
    return looped, total, skipped


def check_presets(lib: Path) -> tuple[bool, bool]:
    """Validate every .dspreset in a library. Returns (saw_reverb, saw_delay)."""
    presets = sorted(lib.glob("*.dspreset"))
    if not presets:
        errors.append(f"{lib.name}: no .dspreset files emitted")
        return False, False

    saw_reverb = saw_delay = False
    for f in presets:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError as ex:
            errors.append(f"{f.name}: XML parse error: {ex}")
            continue

        types = {e.get("type") for e in root.findall(".//effect")}
        saw_reverb |= "reverb" in types
        saw_delay |= "delay" in types

        samples = root.findall(".//sample")
        if not samples:
            errors.append(f"{f.name}: no <sample> elements")
            continue

        # Key ranges must tile 0..127 with no gaps.
        pairs = [(iattr(s, "loNote", f.name), iattr(s, "hiNote", f.name))
                 for s in samples]
        spans = sorted({(lo, hi) for lo, hi in pairs
                        if lo is not None and hi is not None})
        if not spans:
            errors.append(f"{f.name}: no usable key ranges")
        else:
            if spans[0][0] != 0 or spans[-1][1] != 127:
                errors.append(
                    f"{f.name}: key range {spans[0][0]}-{spans[-1][1]}, expected 0-127")
            for (_, hi), (lo, _) in zip(spans, spans[1:]):
                if lo != hi + 1:
                    errors.append(f"{f.name}: key gap between {hi} and {lo}")
                    break

        # Every referenced sample must resolve on disk.
        for s in samples:
            path = attr(s, "path", f.name)
            if path is None:
                break
            if not (f.parent / path).exists():
                errors.append(f"{f.name}: unresolved sample path {path}")
                break

    if not sorted(lib.glob("*.sfz")):
        errors.append(f"{lib.name}: no .sfz files emitted")
    return saw_reverb, saw_delay


def main() -> None:
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    libs = sorted(p for p in root.iterdir() if p.is_dir())
    if not libs:
        sys.exit(f"no libraries found under {root}")

    looped = total = skipped = patches = 0
    any_reverb = any_delay = False

    for lib in libs:
        pdirs = sorted(p for p in lib.iterdir() if (p / "patch.json").exists())
        if not pdirs:
            errors.append(f"{lib.name}: no patch directories")
            continue
        print(f"{lib.name}: {len(pdirs)} patches")

        for pdir in pdirs:
            patches += 1
            meta = json.loads((pdir / "patch.json").read_text())
            zones = meta.get("zones", [])
            if len(zones) != EXPECTED_ZONES:
                errors.append(
                    f"{pdir.name}: {len(zones)} zones, expected {EXPECTED_ZONES}")
            l, t, s = check_audio(pdir, meta)
            looped += l
            total += t
            skipped += s

        r, d = check_presets(lib)
        any_reverb |= r
        any_delay |= d

    frac = looped / total if total else 0.0
    print(f"\npatches: {patches}")
    print(f"zones:   {total} playable, {skipped} skipped (missing/error)")
    print(f"looped:  {looped} ({frac:.1%})")

    if total and frac < MIN_LOOP_FRACTION:
        errors.append(
            f"only {frac:.1%} of zones looped, expected >= {MIN_LOOP_FRACTION:.0%}")
    if not any_reverb:
        warnings.append("no preset emitted a reverb effect")
    if not any_delay:
        warnings.append("no preset emitted a delay effect "
                        "(Delay/Pan-Dly reverb types untested by this pilot)")

    for w in warnings:
        print(f"WARN: {w}")

    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for e in errors[:40]:
            print(f"  {e}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more")
        sys.exit(1)

    print("\nPILOT VALIDATION PASSED")


if __name__ == "__main__":
    main()
