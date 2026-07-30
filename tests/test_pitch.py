"""Pitch regression test (critical bug fix, Task 3 follow-up).

Every render before this fix was exactly one octave low and time-stretched
2x. Root cause: MCU::updateSC55(nSamples) in the third-party emulator does
NOT produce nSamples stereo frames -- mcu.h's MCU_PostSample() bumps
sample_write_ptr once for the L value and once for the R value, so it
counts int16 VALUES posted, not frames (see src/jv_render.cpp's
run_frames() for the full, empirically-verified mechanism). The old code
passed the desired frame count directly as nSamples, so the emulator wrote
only half the requested frames while drain() still copied a full frame's
worth -- reading unwritten/stale buffer content past the real audio for
the second half of every chunk. That silently doubled every rendered
note's duration and halved its pitch.

All 90 prior automated tests passed because every one of them was relative
or structural (frame counts, JSON shape, determinism, dryness ratios) --
none checked an absolute pitch against real-world Hz. This test closes
that gap: it renders a real patch through the actual jv_sampler CLI and
FFT-verifies the fundamental frequency at three widely-spaced notes (MIDI
36, 60, 84), so a uniform rate error, an octave error, or a per-key
transposition bug (which a single test note could miss) all get caught.
A one-semitone transposition error alone is a ~6% frequency ratio, well
outside this test's 3% tolerance.

This runs as part of the normal pytest suite -- it is NOT a manual
script. It skips (does not fail) only when the ROMs or the built
jv_sampler binary aren't available in this environment, matching
tests/test_calibration.py's convention for artifacts that need personal,
copyrighted ROM files this repo can't ship; anywhere the ROMs and a build
are present (this dev environment included), it actually renders and
measures.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
JV_SAMPLER = REPO_ROOT / "build" / "jv_sampler"

# ROMs are supplied by the user and are not in this repository, so every
# test that needs them SKIPS when JV880_ROMS is unset or does not exist --
# a contributor without ROMs should get a clean run, not a wall of errors.
# Test the RAW env var, not the Path: Path("") resolves to "." which exists
# and is a directory, so an unset JV880_ROMS silently looks valid.
_ROMS_ENV = os.environ.get("JV880_ROMS", "")
ROMS_DIR = Path(_ROMS_ENV) if _ROMS_ENV else None
requires_roms = pytest.mark.skipif(
    not (_ROMS_ENV and Path(_ROMS_ENV).is_dir()),
    reason="set JV880_ROMS to a directory of JV-880 ROM images to run this")

# "Square Lead" (internal patch 117): a clean, single-oscillator-ish tone
# whose fundamental is unambiguously the strongest spectral component
# across low/mid/high registers -- chosen empirically. "Pipe Organ 1" (the
# patch used to hand-verify this fix during development) has legitimate
# strong 16'/32' drawbar content that outweighs its own fundamental in a
# plain FFT peak search, which would make a naive single-peak test pass or
# fail for the wrong reason regardless of this bug; several other lead/organ
# patches were checked too (see the fix's PR discussion). Square Lead was
# verified directly against a pre-fix build: it measures within ~1% of
# expected at all three notes below on the fixed renderer, and is off by
# ~45-50% (never anywhere near TOLERANCE) on the pre-fix renderer.
PATCH_INDEX = 117
PATCH_DIR_NAME = "117_Square Lead"

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_name(midi: int) -> str:
    """Mirror src/jv_sampler.cpp's note_name() exactly."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


# A few percent, per the task spec: generous relative to the ~1% natural
# FFT-bin/measurement noise observed in practice, but nowhere close to the
# ~50% deviation an octave/rate bug produces or the ~6% a single semitone
# of mistranspositon produces, so there is no ambiguity between "pass" and
# "this is broken."
TOLERANCE = 0.03

TEST_NOTES = [36, 60, 84]   # widely spaced: low, middle, high registers


def expected_freq(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def _dominant_frequency(path: Path, sr_expected: int = 64000) -> float:
    """FFT-based fundamental estimate: Hann-windowed spectrum of up to 1s of
    the sustained portion (skipping the attack), strongest bin above 20 Hz."""
    x, sr = sf.read(str(path))
    assert sr == sr_expected, f"{path}: sample rate {sr} Hz, expected {sr_expected} Hz"
    mono = x.mean(axis=1) if x.ndim > 1 else x

    start = int(0.3 * sr)
    end = min(len(mono), start + int(1.0 * sr))
    if end - start < sr // 8:
        start, end = 0, len(mono)   # fall back to the whole clip if too short

    seg = (mono[start:end] * np.hanning(end - start)).astype(np.float64)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    mask = freqs > 20   # ignore DC/subsonic content
    peak_idx = np.argmax(spec[mask])
    return float(freqs[mask][peak_idx])


@pytest.fixture(scope="module")
def rendered_patch_dir(tmp_path_factory):
    # Guard the EMPTY case explicitly: Path("") resolves to "." which exists,
    # so an unset JV880_ROMS would sail past a plain .exists() check and fail
    # in the renderer instead of skipping.
    if ROMS_DIR is None or not ROMS_DIR.is_dir():
        pytest.skip("set JV880_ROMS to a directory of JV-880 ROM images")
    if not JV_SAMPLER.exists():
        pytest.skip(f"{JV_SAMPLER} not built -- run: cmake --build build --target jv_sampler")

    out_dir = tmp_path_factory.mktemp("pitch_test")
    result = subprocess.run(
        [str(JV_SAMPLER), "--roms", str(ROMS_DIR), "--board", "JV-880 Internal",
         "--patch", str(PATCH_INDEX), "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"jv_sampler failed (exit {result.returncode}):\n{result.stderr}"
    )
    pdir = out_dir / PATCH_DIR_NAME
    assert pdir.is_dir(), f"expected output directory missing: {pdir}"
    return pdir


@requires_roms
@pytest.mark.parametrize("midi", TEST_NOTES)
def test_fundamental_matches_equal_temperament(rendered_patch_dir, midi):
    fn = rendered_patch_dir / f"{_note_name(midi)}_v2.wav"
    assert fn.exists(), f"expected zone file missing: {fn}"

    expected = expected_freq(midi)
    measured = _dominant_frequency(fn)
    ratio = measured / expected

    assert abs(ratio - 1.0) <= TOLERANCE, (
        f"MIDI {midi} ({_note_name(midi)}): expected {expected:.2f} Hz, "
        f"measured {measured:.2f} Hz (ratio {ratio:.4f}, tolerance "
        f"+-{TOLERANCE:.0%}). A ratio near 0.5 is exactly the signature of "
        f"the updateSC55 frame-count bug this test guards against "
        f"(octave down / 2x time-stretch)."
    )
