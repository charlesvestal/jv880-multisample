#!/usr/bin/env python3
"""One command: JV-880 ROMs in, DecentSampler libraries out.

    python3 tools/build_library.py --roms <rom dir> --out <output dir>

Runs the whole pipeline in order, skipping stages whose output already exists,
so an interrupted build resumes rather than restarting:

    1. calibrate     measure the JV's effects (reverb IRs, chorus, delay,
                     portamento) by rendering test tones on the emulator
    2. render        every patch on every board, 25 keys x up to 5 velocity
                     layers, effects stripped so the samples are dry
    3. phrases       re-render tempo-labelled loop patches at a hold long
                     enough to contain a full two bars
    4. postprocess   resample to 48 kHz, find loop points, measure release,
                     encode FLAC
    5. emit          write .dspreset and .sfz presets
    6. validate      check every zone, key range and effect reference
    7. package       zip each board into a .dslibrary

No Roland PCM ships in this repository -- no samples, no patch data. The
effect calibration DOES ship, impulse responses included: those are the JV
reverb's response to a synthetic impulse this project injects into a copy of
the wave ROM, so no Roland waveform is in the signal. Stage 1 is therefore
skipped by default; pass --stop-after calibrate to re-measure from your ROMs.

You need:
  * JV-880 ROM images (rom1, rom2, waverom1, waverom2, nvram) and, optionally,
    SR-JV80 expansion ROMs in an `expansions/` subdirectory
  * the schwung-jv880 emulator sources (a submodule: git submodule update --init)
  * cmake, a C++17 compiler, Python 3 with numpy/scipy/soundfile
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd, desc, cwd=REPO):
    print(f"\n=== {desc} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        sys.exit(f"FAILED: {desc}")
    print(f"    ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roms", required=True, help="directory holding the JV-880 ROM images")
    ap.add_argument("--out", required=True, help="where the rendered library is written")
    ap.add_argument("--libraries", default=None,
                    help="where .dslibrary files are written (default: <out>/../JV-880 Libraries)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--board", default="all", help="restrict to one board (see --list-boards)")
    ap.add_argument("--list-boards", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="re-measure the effects from your ROMs instead of using "
                         "the calibration that ships with the repo")
    ap.add_argument("--stop-after", choices=["calibrate", "render", "phrases",
                                             "postprocess", "emit", "validate"],
                    help="stop once this stage completes")
    args = ap.parse_args()

    roms, out = Path(args.roms).resolve(), Path(args.out).resolve()
    libs = Path(args.libraries).resolve() if args.libraries else out.parent / "JV-880 Libraries"
    if not roms.is_dir():
        sys.exit(f"--roms is not a directory: {roms}")

    py = [sys.executable]
    sampler = str(REPO / "build" / "jv_sampler")

    run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], "configure")
    run(["cmake", "--build", "build", "-j", str(args.jobs)], "build")

    if args.list_boards:
        subprocess.run(py + [str(REPO / "tools" / "run_batch.py"), "--list",
                             "--roms", str(roms)])
        return

    if args.calibrate:
        run(py + [str(REPO / "tools" / "ir_capture.py"), "--roms", str(roms)],
            "1/7 calibrate effects (impulse responses, chorus, delay)")
    if args.stop_after == "calibrate":
        return

    run(py + [str(REPO / "tools" / "run_batch.py"), "--board", args.board,
              "--jobs", str(args.jobs), "--roms", str(roms), "--out", str(out)],
        "2/7 render patches")
    if args.stop_after == "render":
        return

    run(py + [str(REPO / "tools" / "rerender_phrases.py"), "--roms", str(roms),
              "--library", str(out), "--jobs", str(args.jobs), "--sampler", sampler],
        "3/7 re-render tempo-locked phrase patches")
    if args.stop_after == "phrases":
        return

    for board in sorted(p for p in out.iterdir() if p.is_dir()):
        if any(board.glob("*/*.wav")):
            run(py + [str(REPO / "tools" / "postprocess.py"), str(board)],
                f"4/7 postprocess {board.name}")
    if args.stop_after == "postprocess":
        return

    for board in sorted(p for p in out.iterdir() if p.is_dir()):
        run(py + [str(REPO / "tools" / "emit_presets.py"), str(board),
                  str(REPO / "calib" / "calibration.json")], f"5/7 emit {board.name}")
    if args.stop_after == "emit":
        return

    run(py + [str(REPO / "tools" / "validate_pilot.py"), str(out)], "6/7 validate")
    if args.stop_after == "validate":
        return

    libs.mkdir(parents=True, exist_ok=True)
    for board in sorted(p for p in out.iterdir() if p.is_dir()):
        run(py + [str(REPO / "tools" / "make_dslibrary.py"), str(board), str(libs)],
            f"7/7 package {board.name}")

    print(f"\nDone. .dslibrary files are in {libs}")


if __name__ == "__main__":
    main()
