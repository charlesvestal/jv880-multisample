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

# Zones = 25 key steps x N velocity layers. N is no longer fixed at 3: layers
# are derived per patch from its real tone velocity-switch points, so a patch
# with 4 distinct velocity regions legitimately has 100 zones. Validate the
# STRUCTURE (a whole number of layers, within the renderer's own 3-5 bounds)
# rather than a hardcoded count.
EXPECTED_KEYS = 25
MIN_LAYERS, MAX_LAYERS = 3, 5
# Fraction of SUSTAINING zones that must carry a loop. Not a fraction of all
# zones: whether a zone loops is decided by whether it sustains, and that is
# pure content. Measured across the finished 21-board library the two track
# each other at correlation 1.000 -- the Piano board is 1.7% sustaining
# (piano notes decay, and correctly do not loop) while Vocal is 49.5%. A
# threshold on ALL zones therefore just measures how percussive a board is:
# the full library sits at 22.1% and would fail a 25% bar while being
# entirely correct. The real defect is a zone that sustains and does NOT
# loop, which is what the user heard as a whistle or choir cutting off.
MIN_SUSTAINING_LOOP_FRACTION = 0.90
# Minimum normalized cross-correlation between the two windows DecentSampler
# blends at the loop seam. Replaces a raw endpoint-sample threshold, which a
# listening test showed does not predict audibility at all (40.9% raw sounded
# clean; 42.0% pulsed). Set slightly above postprocess.py's own accept floor
# (0.35) so validation catches anything the search should have declined.
MIN_CROSSFADE_CORRELATION = 0.30

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
    """Return (looped, total, skipped, sustaining) zone counts for one patch."""
    looped = total = skipped = sustaining = 0
    for z in meta.get("zones", []):
        # Zones that never rendered or failed post-processing are excluded from
        # the presets by design; don't hold their missing audio against us.
        if z.get("kind") in ("missing", "error"):
            skipped += 1
            continue
        total += 1
        if z.get("kind") == "sustaining":
            sustaining += 1
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
        # Score the CROSSFADE REGION, not the single endpoint sample.
        #
        # DecentSampler crossfades loops, so a mismatch between the lone
        # samples at loop_start and loop_end says almost nothing about what a
        # listener hears. Confirmed by listening test: JP-8 Strings at 40.9%
        # raw endpoint discontinuity sounds clean, while ChuChu Vox at 42.0%
        # and Whistle at 64.4% audibly pulse -- near-identical raw values,
        # opposite verdicts. What actually matters is whether the two windows
        # DS blends together correlate.
        xf = int(loop.get("crossfade", 0))
        if xf <= 0:
            continue
        xf = min(xf, s, e - s)
        if xf < 32:
            continue
        a = x[e - xf:e]
        b = x[s - xf:s]
        # Per channel: a poor blend in either channel is audible even if the
        # mono sum happens to cancel it out.
        worst = 1.0
        for ch in range(x.shape[1]):
            na = float(np.linalg.norm(a[:, ch]))
            nb = float(np.linalg.norm(b[:, ch]))
            if na < 1e-9 or nb < 1e-9:
                continue      # near-silent channel: neutral, not a failure
            worst = min(worst, float(np.dot(a[:, ch], b[:, ch]) / (na * nb)))
        if worst < MIN_CROSSFADE_CORRELATION:
            errors.append(
                f"{f.name}: crossfade region correlates {worst:.2f} "
                f"(min {MIN_CROSSFADE_CORRELATION})")
    return looped, total, skipped, sustaining


def check_release_consistency(pdir: Path, meta: dict) -> None:
    """Every zone in a patch must share one release value.

    The JV's TVA release is a PATCH parameter -- it does not vary note to
    note. Measuring it per zone made it do exactly that, and where a note
    decays before note-off the measurement falls off a cliff: Tr.Rhodes had
    C6 at 3.711 s beside neighbours at 0.11 s, heard as the top of the
    keyboard releasing far more slowly. 34 of 192 internal patches were
    affected, one spanning 0.050-5.040 s.

    Nothing caught it because each value was individually plausible -- 0.05 s
    is a fine release and so is 3.6 s. Only comparing them WITHIN a patch
    exposes it, which is what this does.
    """
    values = {round(z["release"], 4) for z in meta.get("zones", [])
              if z.get("kind") not in ("missing", "error")
              and z.get("release") is not None}
    if len(values) > 1:
        lo, hi = min(values), max(values)
        errors.append(
            f"{pdir.name}: {len(values)} different release values "
            f"({lo:.3f}-{hi:.3f}s) -- release is a patch parameter and must "
            f"be uniform across the keyboard")


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
        # Reverb lives on the BUS now, not the instrument chain, and is a
        # convolution rather than DecentSampler's parametric `reverb`. This
        # check looked only for type="reverb" among instrument effects, so
        # after the bus change it reported "no preset emitted a reverb
        # effect" on a library where every one of them had it -- a validator
        # that cannot see the reverb also cannot notice it going missing,
        # which is exactly how the silent-IR bug survived once already.
        saw_reverb |= bool(root.findall(
            "./buses/bus/effects/effect[@type='convolution']")) or "reverb" in types
        saw_delay |= "delay" in types

        # A group routed to a bus must actually have a bus to arrive at, or
        # the send goes nowhere and the reverb is silently absent.
        group = root.find(".//group")
        if group is not None and group.get("output2Target"):
            target = group.get("output2Target")
            if target.startswith("BUS_") and root.find("./buses/bus") is None:
                errors.append(f"{f.name}: routes to {target} but declares no <buses>")

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

    looped = total = skipped = sustaining = patches = 0
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
            n = len(zones)
            if n % EXPECTED_KEYS != 0:
                errors.append(
                    f"{pdir.name}: {n} zones is not a whole number of "
                    f"{EXPECTED_KEYS}-key layers")
            elif not (MIN_LAYERS <= n // EXPECTED_KEYS <= MAX_LAYERS):
                errors.append(
                    f"{pdir.name}: {n // EXPECTED_KEYS} velocity layers, "
                    f"expected {MIN_LAYERS}-{MAX_LAYERS}")
            check_release_consistency(pdir, meta)
            l, t, s, sus = check_audio(pdir, meta)
            looped += l
            total += t
            skipped += s
            sustaining += sus

        r, d = check_presets(lib)
        any_reverb |= r
        any_delay |= d

    frac = looped / total if total else 0.0
    sus_frac = looped / sustaining if sustaining else 1.0
    print(f"\npatches: {patches}")
    print(f"zones:   {total} playable, {skipped} skipped (missing/error)")
    print(f"looped:  {looped} ({frac:.1%} of all zones, "
          f"{sus_frac:.1%} of the {sustaining} sustaining ones)")

    if sustaining and sus_frac < MIN_SUSTAINING_LOOP_FRACTION:
        errors.append(
            f"only {sus_frac:.1%} of SUSTAINING zones looped, expected >= "
            f"{MIN_SUSTAINING_LOOP_FRACTION:.0%} -- a sustaining zone without a "
            f"loop cuts off abruptly when held")
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
