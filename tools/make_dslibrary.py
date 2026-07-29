#!/usr/bin/env python3
"""Package a rendered library folder into a DecentSampler `.dslibrary`.

A .dslibrary is a plain ZIP containing ONE top-level folder named after the
library, holding the .dspreset files and their sample folders -- verified
against a real-world library (ASIMOV v1.0.dslibrary), which uses exactly this
shape with compression method "store".

Stored, not deflated, on purpose: the samples are already FLAC, so deflate
costs significant time for essentially no size reduction. ZIP64 is enabled
because these libraries run well past the 4 GB classic-ZIP ceiling.

Only .dspreset files are included. The .sfz presets describe the same zones
but are a different format with no role inside a DecentSampler library; they
remain available in the source folder.

Usage:
    python3 tools/make_dslibrary.py "<library folder>" [<output dir>]
"""
import sys
import zipfile
from pathlib import Path


def iter_library_files(lib: Path):
    """Yield (absolute path, path relative to the library folder)."""
    for p in sorted(lib.rglob("*")):
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue          # .DS_Store and friends
        if p.suffix == ".sfz":
            continue          # not part of a DecentSampler library
        if p.name == "patch.json":
            continue          # build metadata, not needed at play time
        yield p, p.relative_to(lib)


def build(lib: Path, out_dir: Path) -> Path:
    out = out_dir / f"{lib.name}.dslibrary"
    files = list(iter_library_files(lib))
    total = sum(p.stat().st_size for p, _ in files)
    print(f"{lib.name}: {len(files)} files, {total / 1e9:.2f} GB")

    tmp = out.with_suffix(".dslibrary.partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, (src, rel) in enumerate(files, 1):
            # Everything lives under a single top-level folder named after the
            # library, matching how DecentSampler libraries are distributed.
            z.write(src, f"{lib.name}/{rel.as_posix()}")
            if i % 2000 == 0:
                print(f"  {i}/{len(files)}")
    tmp.replace(out)          # atomic: never leave a half-written .dslibrary
    print(f"  -> {out}  ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def main() -> None:
    lib = Path(sys.argv[1])
    if not lib.is_dir():
        sys.exit(f"not a directory: {lib}")
    if not sorted(lib.glob("*.dspreset")):
        sys.exit(f"no .dspreset files in {lib} -- nothing to package")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else lib.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    build(lib, out_dir)


if __name__ == "__main__":
    main()
