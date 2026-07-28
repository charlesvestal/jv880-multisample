#!/usr/bin/env python3
"""Resample renders to 48 kHz, detect loops, measure release, encode FLAC.

Turns the raw 64 kHz WAV renders produced by the C++ renderer (Task 3) into
48 kHz / 24-bit FLAC files, annotating each zone in ``patch.json`` with loop
points (for sustaining zones only) and a measured release time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR_IN = 64000
SR_OUT = 48000
MAX_XFADE = 2000


def resample_to_48k(x):
    # 64000 -> 48000 is exactly 3/4, so an exact rational resample suffices
    # (no fractional-ratio filter design needed).
    return resample_poly(x, up=3, down=4, axis=0)


def classify(x, hold_frames):
    """Sustaining if energy just before note-off is still substantial."""
    if len(x) == 0:
        return "silent", 0.0
    peak = np.abs(x).max()
    if peak <= 0:
        return "silent", 0.0
    lo = max(0, hold_frames - int(0.25 * SR_OUT))
    hi = min(len(x), hold_frames)
    if hi <= lo:
        return "decaying", 0.0
    sustain = float(np.sqrt(np.mean(x[lo:hi] ** 2)))
    ratio = sustain / peak
    return ("sustaining" if ratio > 0.18 else "decaying"), ratio


def estimate_period(mono, sr):
    """Fundamental period in frames via autocorrelation, refined to
    sub-sample precision by parabolic interpolation of the peak.

    Sub-sample precision matters: multiplying a merely-integer period by up
    to 60 periods (see find_loop) would amplify a fraction-of-a-sample
    truncation error into many samples of accumulated phase drift.
    """
    seg = mono - mono.mean()
    if len(seg) < 4096:
        return 0.0
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(sr / 1200), int(sr / 40)     # 40 Hz .. 1200 Hz
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return 0.0
    k = int(lo + np.argmax(ac[lo:hi]))
    if k <= 0 or k >= len(ac) - 1:
        return float(k)
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    offset = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(k) + offset


def find_loop(x, sr, hold_frames):
    """Correlation-matched loop points inside the steady-state region."""
    if len(x) == 0:
        return None
    mono = x.mean(axis=1) if x.ndim > 1 else x
    start_lo = int(1.0 * sr)
    region_hi = min(hold_frames, len(mono)) - int(0.05 * sr)
    if region_hi - start_lo < int(0.3 * sr):
        return None

    period = estimate_period(mono[start_lo:region_hi], sr)
    if period <= 0:
        return None

    win = max(int(round(period * 2)), 256)
    loop_start = start_lo

    # Try loop lengths from 8 to 60 periods; longer loops sound less static.
    # NOTE (not fixed here): this choice has no awareness of the patch's
    # own LFO1/LFO2 rate. A slow vibrato/tremolo whose cycle is longer than
    # the chosen loop length gets flattened once looped, since the loop
    # only ever replays one phase of the LFO. Inherited from the original
    # design; needs revisiting once we can audition real renders with
    # LFO-modulated sustains (Task 3 isn't built yet).
    # `period` is a non-integer estimate, so `period * n_per` rounds to a
    # slightly different sub-sample residual for each n_per. Rather than
    # trust a single windowed-correlation "best" (which tolerates small
    # phase error and can still leave a visible endpoint discontinuity),
    # score every candidate that passes the periodicity gate by its actual
    # sample-level endpoint match and keep the tightest one -- across ~50
    # candidates at least one lands very close to true phase alignment.
    best = None  # (dv, end, score)
    for n_per in range(8, 61):
        end = int(round(loop_start + period * n_per))
        if end + win >= region_hi:
            break
        a = mono[loop_start:loop_start + win]
        b = mono[end:end + win]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        score = float(np.dot(a, b) / denom)
        if score < 0.90:
            continue
        dv = abs(float(mono[loop_start]) - float(mono[end]))
        if best is None or dv < best[0]:
            best = (dv, end, score)

    if best is None:
        return None

    _, end, score = best
    length = end - loop_start
    if length <= 0:
        return None
    # Enforce the DecentSampler crossfade bound in code (not just documented):
    # loopCrossfade silently breaks looping when it's large relative to
    # loopStart or the loop length, so cap it hard on every path here.
    xfade = int(min(MAX_XFADE, loop_start // 4, length // 4))
    return {"enabled": True, "start": int(loop_start), "end": int(end),
            "crossfade": int(max(0, xfade)), "score": round(score, 4)}


def measure_release(x, sr, hold_frames):
    """Seconds for the post-note-off tail to fall 60 dB."""
    if len(x) == 0:
        return 0.1
    tail = x[hold_frames:]
    if len(tail) < sr // 20:
        return 0.1
    mono = np.abs(tail.mean(axis=1) if tail.ndim > 1 else tail)
    hop = 64
    n = (len(mono) // hop) * hop
    if n == 0:
        return 0.1
    env = mono[:n].reshape(-1, hop).max(axis=1) + 1e-12
    peak = env.max()
    if peak <= 1e-12:
        return 0.1
    db = 20 * np.log10(env / peak)
    below = np.where(db < -60)[0]
    if len(below):
        return float(max(0.05, below[0] * hop / sr))
    return float(len(mono) / sr)


def _mark_zone_unprocessable(z, kind):
    """Populate the schema-required per-zone fields with safe defaults.

    Task 6 reads zone["loop"] / zone["release"] unconditionally on every
    zone, so every zone must gain these keys regardless of whether it could
    actually be converted -- a missing render or a per-zone processing
    failure must never leave a zone short of the schema contract (that
    would surface as a KeyError deep into an hours-long batch render,
    rather than here where it's cheap to handle).

    kind: "missing" (the renderer never produced this file) or "error"
    (the file exists but couldn't be processed, e.g. zero-length audio or
    an unexpected channel count).
    """
    z["kind"] = kind
    z["sustain_ratio"] = 0.0
    z["loop"] = {"enabled": False}
    z["release"] = 0.0


def process_patch(pdir: Path, hold_seconds=3.5):
    meta = json.loads((pdir / "patch.json").read_text())
    hold_out = int(hold_seconds * SR_OUT)

    for z in meta["zones"]:
        src = pdir / z["file"]
        # A zone is already processed once z["file"] itself has been
        # rewritten to point at the .flac (see below), so re-running on the
        # same directory must recognise that name -- not just "file
        # missing" -- and skip it instead of trying to re-read a 48 kHz
        # FLAC as a 64 kHz WAV.
        if src.suffix.lower() != ".wav":
            continue

        if not src.exists():
            # Renderer never produced this file. Still fill in the schema
            # so a downstream KeyError doesn't surface hours into a batch.
            _mark_zone_unprocessable(z, "missing")
            continue

        try:
            x, sr = sf.read(str(src), always_2d=True)
            assert sr == SR_IN, f"unexpected rate {sr} in {src}"
            if x.shape[1] != 2:
                # Fail loudly rather than silently emitting a mono FLAC:
                # Task 3 is expected to always render stereo, so this is a
                # signal something upstream is wrong, not something to
                # quietly paper over by upmixing.
                raise ValueError(
                    f"expected stereo (2ch) audio, got {x.shape[1]}ch in {src}"
                )
            if len(x) == 0:
                # A zero-frame WAV resamples to a zero-frame array, and
                # soundfile happily "writes" a zero-frame FLAC that it then
                # can't reopen (verified: libsndfile reports "Format not
                # recognised"). Writing that file and deleting the source
                # would silently corrupt this zone, so refuse explicitly
                # before ever getting to sf.write/src.unlink below.
                raise ValueError(f"zero-length audio in {src}")

            y = resample_to_48k(x)

            kind, ratio = classify(y, hold_out)
            loop = find_loop(y, SR_OUT, hold_out) if kind == "sustaining" else None

            dst = src.with_suffix(".flac")
            sf.write(str(dst), y, SR_OUT, subtype="PCM_24")
            src.unlink()

            z["file"] = dst.name
            z["frames"] = int(len(y))
            z["kind"] = kind
            z["sustain_ratio"] = round(ratio, 4)
            z["loop"] = loop or {"enabled": False}
            z["release"] = round(measure_release(y, SR_OUT, hold_out), 4)
        except Exception as exc:
            # One degenerate zone (zero-length render, wrong channel count,
            # corrupt audio, ...) must not lose the work already done on
            # this patch's other zones, and must not abort the batch. The
            # source file (if any) is left untouched -- we only unlink it
            # on the success path above -- so nothing is lost on retry.
            print(
                f"  zone {z.get('file')!r} (key={z.get('key')}, "
                f"velocity={z.get('velocity')}): FAILED ({exc!r}) -- "
                "marking as error, continuing with next zone",
                file=sys.stderr,
            )
            _mark_zone_unprocessable(z, "error")

    meta["sample_rate"] = SR_OUT
    (pdir / "patch.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    root = Path(sys.argv[1])
    dirs = sorted(p for p in root.iterdir() if (p / "patch.json").exists())
    for i, d in enumerate(dirs, 1):
        # A batch runs over thousands of patches; one bad patch (malformed
        # patch.json, unexpected schema, disk error, ...) must not abort
        # the rest of the run. Per-zone failures are already isolated
        # inside process_patch -- this is the outer safety net for
        # failures at the whole-patch level.
        try:
            m = process_patch(d)
        except Exception as exc:
            print(
                f"[{i}/{len(dirs)}] {d.name}: PATCH FAILED ({exc!r}) -- "
                "skipping, continuing batch",
                file=sys.stderr,
            )
            continue
        looped = sum(1 for z in m["zones"] if z["loop"]["enabled"])
        print(f"[{i}/{len(dirs)}] {m['name']}: {looped}/{len(m['zones'])} looped")


if __name__ == "__main__":
    main()
