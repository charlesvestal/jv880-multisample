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
from scipy.signal import fftconvolve, resample_poly

SR_IN = 64000
SR_OUT = 48000
MAX_XFADE = 2000
# A zone whose kind is one of these has no usable audio; see
# _mark_zone_unprocessable's docstring for the Task 6 contract this implies.
FAILURE_KINDS = frozenset({"missing", "error"})


def resample_to_48k(x):
    # 64000 -> 48000 is exactly 3/4, so an exact rational resample suffices
    # (no fractional-ratio filter design needed).
    return resample_poly(x, up=3, down=4, axis=0)


def classify(x, hold_frames):
    """Sustaining if energy just before note-off is still substantial."""
    if len(x) == 0:
        return "silent", 0.0
    # Cast immediately: np.float64 happens to subclass Python's float (so
    # it round-trips through json.dumps for free), but np.float32 does
    # NOT -- and sf.read(..., dtype="float32") means `x` is float32 here,
    # so leaving `peak` as a bare numpy scalar would make `ratio` below a
    # non-JSON-serializable np.float32 too.
    peak = float(np.abs(x).max())
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
    # fftconvolve is O(n log n) vs np.correlate's direct O(n^2); on the
    # ~100-300k-frame steady-state regions this pipeline actually sees,
    # np.correlate cost ~8s per sustaining zone (measured: 117,600 frames
    # -> 7.4s) which would add hours to a multi-thousand-patch batch.
    # fftconvolve(seg, seg[::-1], "full") is the standard identity for
    # correlate(seg, seg, "full") and is ~1370x faster at that size
    # (verified numerically identical, max diff ~1e-11 -- floating-point
    # noise, not a behavior change). See test_estimate_period_performance.
    ac = fftconvolve(seg, seg[::-1], mode="full")[len(seg) - 1:]
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
    if not np.isfinite(period) or period <= 0:
        return None

    win = max(int(round(period * 2)), 256)
    loop_start = start_lo

    # Try loop lengths from 8 to 60 periods; longer loops sound less static.
    # NOTE (not fixed here): this choice has no awareness of the patch's
    # own LFO1/LFO2 rate. A slow vibrato/tremolo whose cycle is longer than
    # the chosen loop length gets flattened once looped, since the loop
    # only ever replays one phase of the LFO. Inherited from the original
    # design; real LFO-modulated renders exist now (Task 3 is built), but
    # this hasn't been evaluated against them yet -- needs auditioning
    # real pad/string patches before deciding how to size loops relative
    # to LFO period.
    # `period` is a non-integer estimate, so `period * n_per` rounds to a
    # slightly different sub-sample residual for each n_per. Rather than
    # trust a single windowed-correlation "best" (which tolerates small
    # phase error and can still leave a visible endpoint discontinuity),
    # score every candidate that passes the periodicity gate by its actual
    # sample-level endpoint match -- across ~50 candidates at least one
    # lands very close to true phase alignment.
    candidates = []  # (n_per, end, dv, score)
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
        candidates.append((n_per, end, dv, score))

    if not candidates:
        return None

    # Among the candidates, prefer a LONGER loop (fewer perceptible
    # repeats -- a real quality concern on the pads/strings this pipeline
    # is full of), but not at the cost of a meaningfully worse endpoint
    # match. Take every candidate within a small tolerance of the best
    # achievable discontinuity, then pick the longest of those.
    best_dv = min(c[2] for c in candidates)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    tol = max(best_dv * 2.0, 0.0005 * peak, 1e-9)
    near_best = [c for c in candidates if c[2] <= tol]
    n_per, end, dv, score = max(near_best, key=lambda c: c[0])

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

    TASK 6 CONTRACT: zone["file"] is deliberately left untouched here (it
    keeps whatever name it already had -- the never-rendered or
    unprocessable .wav). zone["kind"] is the flag to check: if
    zone["kind"] is "missing" or "error" (see FAILURE_KINDS), that file
    may not exist or may not be valid audio at all, and the zone MUST be
    skipped rather than referenced. Do not treat zone["file"] as a
    guaranteed-to-exist path without first checking zone["kind"].
    """
    z["kind"] = kind
    z["sustain_ratio"] = 0.0
    z["loop"] = {"enabled": False}
    # Same safe floor used everywhere else a release can't be measured
    # (see measure_release's own 0.1 fallbacks) -- 0.0 would be an instant
    # cutoff/click if this ever reached ampeg_release/DecentSampler release.
    z["release"] = 0.1


def process_patch(pdir: Path, hold_seconds=3.5):
    meta = json.loads((pdir / "patch.json").read_text())
    hold_out = int(hold_seconds * SR_OUT)

    for z in meta["zones"]:
        src = pdir / z["file"]
        suffix = src.suffix.lower()

        # A zone is already processed once z["file"] has been rewritten to
        # point at a .flac that actually exists on disk (see the success
        # path below) -- re-running on the same directory must recognise
        # that and skip it rather than trying to re-read a 48 kHz FLAC as
        # a 64 kHz WAV. Checking existence too (not just the suffix)
        # matters: a ".flac" name that does NOT exist is not "already
        # processed", it's a schema/data problem that still needs a
        # kind/loop/release set, same as any other unprocessable zone.
        if suffix == ".flac" and src.exists():
            continue

        if suffix != ".wav":
            # Neither a pending render nor a verified already-processed
            # output -- an unexpected filename. This used to be silently
            # skipped with no kind/loop/release ever set (the exact
            # KeyError-deep-in-a-batch failure _mark_zone_unprocessable
            # exists to prevent), so mark it explicitly instead.
            print(
                f"  zone {z.get('file')!r} (key={z.get('key')}, "
                f"velocity={z.get('velocity')}): unexpected file name "
                f"(not .wav, not a verified .flac) -- marking as error",
                file=sys.stderr,
            )
            _mark_zone_unprocessable(z, "error")
            continue

        if not src.exists():
            # Renderer never produced this file. Still fill in the schema
            # so a downstream KeyError doesn't surface hours into a batch.
            _mark_zone_unprocessable(z, "missing")
            continue

        try:
            x, sr = sf.read(str(src), always_2d=True, dtype="float32")
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

            # Everything above and below this point (up to "point of no
            # return") is pure computation on `y` -- nothing here mutates
            # `z` or touches the filesystem, so if any of it raises, the
            # zone is untouched and the next run retries cleanly from the
            # original .wav. Previously src.unlink() ran BEFORE
            # measure_release, so a failure in measure_release still left
            # z["file"] pointing at a written .flac with a ".flac" suffix
            # -- the next run's suffix check would then skip it forever,
            # stranding a zone with valid audio but "error" metadata that
            # never got corrected. See requirement #5 in the review.
            kind, ratio = classify(y, hold_out)
            loop = find_loop(y, SR_OUT, hold_out) if kind == "sustaining" else None
            release_val = round(measure_release(y, SR_OUT, hold_out), 4)

            # Point of no return: write the deliverable, then commit the
            # zone's metadata, and only then remove the original. Any
            # failure before this point leaves the .wav (and z) untouched;
            # the delete is deliberately the very last thing that happens.
            dst = src.with_suffix(".flac")
            sf.write(str(dst), y, SR_OUT, subtype="PCM_24")

            z["file"] = dst.name
            z["frames"] = int(len(y))
            z["kind"] = kind
            z["sustain_ratio"] = round(ratio, 4)
            z["loop"] = loop or {"enabled": False}
            z["release"] = release_val

            try:
                src.unlink()
            except OSError as exc:
                # The .flac + metadata above are already valid and
                # committed; failing to clean up the old .wav is untidy,
                # not a reason to discard a successful conversion.
                print(
                    f"  warning: converted {src.name} but could not remove "
                    f"the original ({exc!r})",
                    file=sys.stderr,
                )
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

    # A systemic bug that quietly errors out a chunk of zones must not
    # look identical to a clean run over 4,000+ patches: track every
    # failure so it's counted, summarized, exits non-zero, and is written
    # out for a later pass to retry -- mirroring what the render
    # orchestrator (Task 7) does for render failures.
    failures = []
    patch_failures = 0
    zone_failures = 0

    for i, d in enumerate(dirs, 1):
        # A batch runs over thousands of patches; one bad patch (malformed
        # patch.json, unexpected schema, disk error, ...) must not abort
        # the rest of the run. Per-zone failures are already isolated
        # inside process_patch -- this is the outer safety net for
        # failures at the whole-patch level.
        try:
            m = process_patch(d)
        except Exception as exc:
            patch_failures += 1
            failures.append({"scope": "patch", "patch": d.name, "reason": repr(exc)})
            print(
                f"[{i}/{len(dirs)}] {d.name}: PATCH FAILED ({exc!r}) -- "
                "skipping, continuing batch",
                file=sys.stderr,
            )
            continue

        bad_zones = [z for z in m["zones"] if z.get("kind") in FAILURE_KINDS]
        zone_failures += len(bad_zones)
        for z in bad_zones:
            failures.append({
                "scope": "zone",
                "patch": d.name,
                "key": z.get("key"),
                "velocity": z.get("velocity"),
                "file": z.get("file"),
                "kind": z.get("kind"),
            })

        looped = sum(1 for z in m["zones"] if z["loop"]["enabled"])
        status = f"{looped}/{len(m['zones'])} looped"
        if bad_zones:
            status += f", {len(bad_zones)}/{len(m['zones'])} FAILED"
        print(f"[{i}/{len(dirs)}] {m['name']}: {status}")

    total = len(dirs)
    print(
        f"\n{total - patch_failures}/{total} patches processed "
        f"({patch_failures} patch failures, {zone_failures} zone failures "
        f"across {total - patch_failures} attempted patches)."
    )

    fail_path = root / "postprocess_failures.json"
    fail_path.write_text(json.dumps(failures, indent=2))

    if failures:
        print(f"{len(failures)} failures written to {fail_path} -- retry these.",
              file=sys.stderr)
        sys.exit(1)
    print("No failures.")


if __name__ == "__main__":
    main()
