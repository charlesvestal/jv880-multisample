#!/usr/bin/env python3
"""Capture a DecentSampler plugin state that has a preset loaded.

Why this needs a human, once: DecentSampler exposes only three automatable
parameters (main volume, tuning, bypass) and no way to load a preset
programmatically. Its saved state is an opaque binary blob -- not zlib, and
containing no readable path -- so a state referencing a given preset cannot be
synthesised. But the plugin hosts and renders offline perfectly well through
pedalboard, so one captured state unlocks everything after it.

Run this, load the preset in the window that opens, then close the window.
The state is written to disk and tools/render_ds.py reuses it forever.

    python3 tools/capture_ds_state.py --preset "<path to .dspreset>"

If DecentSampler reads the preset from disk each time the state is restored --
which is the usual design, since samples are external anyway -- then editing
the .dspreset and re-applying this same state is enough to render any
variation of it. render_ds.py verifies that assumption rather than trusting it.
"""
import argparse
from pathlib import Path

VST3 = "/Library/Audio/Plug-Ins/VST3/DecentSampler.vst3"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True,
                    help="the .dspreset to load in the editor (shown for convenience)")
    ap.add_argument("--out", default="calib/ds_state.bin")
    ap.add_argument("--plugin", default=VST3)
    args = ap.parse_args()

    from pedalboard import load_plugin
    plugin = load_plugin(args.plugin)

    print(f"\nA DecentSampler window will open.\n"
          f"  1. Load this preset in it:\n       {args.preset}\n"
          f"  2. Play a note to confirm you hear it\n"
          f"  3. Close the window\n")
    input("Press Enter to open the editor... ")
    plugin.show_editor()

    state = bytes(plugin.raw_state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(state)
    print(f"\ncaptured {len(state)} bytes -> {out}")
    print("If this is the same size as an empty state (~1044 bytes), the preset "
          "probably did not load -- rerun and check step 2.")


if __name__ == "__main__":
    main()
