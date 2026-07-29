#!/usr/bin/env python3
"""Resample renders to 48 kHz, detect loops, measure release, encode FLAC.

Turns the raw 64 kHz WAV renders produced by the C++ renderer (Task 3) into
48 kHz / 24-bit FLAC files, annotating each zone in ``patch.json`` with loop
points (for sustaining zones only) and a measured release time.
"""
import argparse
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


MIN_LOOP_SECONDS = 0.05
REF_WIN = 2048  # reference window (frames) for the cross-correlation search
# Matches the validator's own threshold: a candidate whose best achievable
# per-channel endpoint discontinuity still exceeds this fraction of peak
# is not looped at all, rather than looped at the least-bad candidate.
MAX_ENDPOINT_DV_FRACTION = 0.05


def find_loop(x, sr, hold_frames):
    """Correlation-matched loop points inside the steady-state region.

    Searches loop LENGTH directly via FFT cross-correlation over the whole
    steady-state region, rather than constraining candidates to multiples
    of an estimated pitch period. The period-multiple approach (the
    original design) put a hard ceiling on how far the search could reach:
    at 60 periods, a ~1245 Hz note (period ~39 frames) only reaches ~48ms,
    while its real repeat lived at 1.9s -- so nothing above the bass
    register could ever find its actual loop point, and it silently looked
    like "this material isn't loopable" (pilot render: 7,112 sustaining
    zones, only 1,284 looped -- 3.8%). Low notes happened to work because
    their long periods pushed 60 multiples into a useful range, which is
    why this passed unnoticed on a 220 Hz synthetic test signal.

    Everything below operates PER CHANNEL, not on the mono mix. Scoring
    and minimizing discontinuity on x.mean(axis=1) was a separate, clean
    bug: a real per-channel endpoint mismatch can be near-exactly opposite
    in sign between L and R (common on stereo-detuned/panned JV patches),
    which cancels in the mono average to a "perfect" 0.0% match while the
    actual per-channel discontinuity a sampler plays back is 5-50%+ of
    peak -- an audible click that the mono-based search couldn't see by
    construction, on 2,400+ zones in the pilot. Endpoint discontinuity is
    now `max(|L diff|, |R diff|)`, matching how the validator (and a real
    player) judges it; correlation score is the worse of the two
    per-channel scores at each lag, so a candidate only qualifies if BOTH
    channels genuinely match, not just their sum.

    For every lag L from MIN_LOOP_SECONDS up to as long as the region
    allows, this scores normalized cross-correlation between a REF_WIN-
    frame window at loop_start and one at loop_start+L, per channel.
    Cross-correlation at every lag is computed in one FFT call per channel
    (`fftconvolve`, the same O(n log n) trick used for the earlier
    autocorrelation fix) rather than a Python loop over candidate
    lengths -- this runs per sustaining zone, ~7,000+ times a batch, and
    needs to stay in the low tens of milliseconds even doubled for stereo.
    Because every lag is scored at full sample resolution already (not a
    coarse grid), there's no separate "fine" refinement pass afterward:
    the endpoint discontinuity (dv) used for the final pick is exact for
    every candidate, on every channel.
    """
    if len(x) == 0:
        return None
    # Treat mono input as a single "channel" so the rest of this function
    # doesn't need two code paths.
    chans = x if x.ndim > 1 else x[:, None]
    n_channels = chans.shape[1]

    start_lo = int(1.0 * sr)
    region_hi = min(hold_frames, len(chans)) - int(0.05 * sr)
    if region_hi - start_lo < int(0.3 * sr):
        return None

    loop_start = start_lo
    # float64 throughout: the cumulative-sum trick below for rolling
    # window energy is numerically sensitive to precision loss over
    # ~100k+ samples, and the source may already be float32 (see
    # sf.read(..., dtype="float32") in process_patch).
    region = chans[loop_start:region_hi].astype(np.float64)  # (n_region, C)
    n_region = len(region)

    win = min(REF_WIN, n_region // 2)
    if win < 64:
        return None

    l_min = int(MIN_LOOP_SECONDS * sr)
    l_max = n_region - win
    if l_max < l_min:
        return None

    # If EVERY channel's reference window is near-silent, there's no real
    # audio to loop at all (a fully silent zone shouldn't get a "perfect"
    # loop just because 0-0=0 everywhere) -- match the old mono
    # behaviour's `ref_norm <= 1e-12: return None` for that case exactly.
    ref_norms = [float(np.linalg.norm(region[:win, c])) for c in range(n_channels)]
    if max(ref_norms) <= 1e-12:
        return None

    # Per-channel normalized cross-correlation at every lag, combined by
    # taking the WORSE (minimum) of the per-channel scores -- a candidate
    # only counts as a good match if every channel independently clears
    # the bar, not just their average.
    combined_score = None
    for c in range(n_channels):
        chan = region[:, c]
        ref = chan[:win]
        ref_norm = ref_norms[c]
        if ref_norm <= 1e-12:
            # A near-silent reference on THIS channel while at least one
            # OTHER channel has real signal (e.g. a hard-panned tone)
            # can't meaningfully judge any lag itself -- numerator/denom
            # would be near 0/0 noise, not a real signal of mismatch.
            # Treat it as neutral (score 1.0 everywhere) rather than
            # letting numerical noise veto every candidate; the endpoint
            # discontinuity check below still applies to this channel
            # with correct arithmetic regardless (0 - 0 stays 0).
            chan_score = np.ones(n_region - win + 1)
        else:
            # numerator[L] = sum_j chan[L+j] * ref[j] for every lag L at
            # once -- the standard correlation-as-convolution identity
            # (mode="valid" since we only want in-range lags).
            numerator = fftconvolve(chan, ref[::-1], mode="valid")
            csq = np.cumsum(np.concatenate(([0.0], chan ** 2)))
            win_energy = csq[win:] - csq[:-win]
            win_norm = np.sqrt(np.maximum(win_energy, 0.0))
            denom = ref_norm * win_norm + 1e-12
            chan_score = numerator / denom
        combined_score = (
            chan_score if combined_score is None
            else np.minimum(combined_score, chan_score)
        )

    lags = np.arange(len(combined_score))
    # Correlation gate unchanged from the mono version (0.90); the bug
    # being fixed is what the score/discontinuity are computed ON, not
    # this threshold.
    mask = (lags >= l_min) & (lags <= l_max) & np.isfinite(combined_score) & (combined_score >= 0.90)
    if not np.any(mask):
        return None

    cand_lags = lags[mask]
    cand_scores = combined_score[mask]

    # Endpoint discontinuity: worst case across channels -- max(|L diff|,
    # |R diff|) -- matching exactly what the validator checks
    # (np.abs(x[s] - x[e]).max()) and what a sampler actually plays back.
    # A mono-mixed difference can be near zero even when both channels are
    # individually well outside tolerance, if their errors are opposite in
    # sign; that cancellation was the entire bug.
    start_vals = region[0]              # (C,)
    end_vals = region[cand_lags]        # (len(cand_lags), C)
    dv_vals = np.max(np.abs(start_vals - end_vals), axis=1)

    # Among the candidates, prefer a LONGER loop (fewer perceptible
    # repeats -- a real quality concern on the pads/strings this pipeline
    # is full of), but not at the cost of a meaningfully worse endpoint
    # match. Take every candidate within a small tolerance of the best
    # achievable discontinuity, then pick the longest of those.
    #
    # NOTE (not fixed here): this still has no explicit awareness of the
    # patch's own LFO1/LFO2 rate. Loops now generally land much longer
    # than the old ~60-period ceiling (often close to the full region),
    # which incidentally makes it less likely a loop is shorter than one
    # LFO cycle -- but nothing here specifically checks that. Needs
    # auditioning real LFO-modulated pad/string renders to know whether
    # this needs a dedicated fix.
    best_dv = float(np.min(dv_vals))
    peak = float(np.max(np.abs(chans))) if chans.size else 0.0

    # The 0.90 correlation gate is a WINDOWED, aggregate similarity check
    # (2048 samples) -- it does not guarantee the single boundary sample
    # actually lines up. On some material (seen in pilot validation on
    # patches like Mighty Pad/Pipe Organ) literally every candidate that
    # clears 0.90 still has a large per-channel endpoint jump (60-70% of
    # peak), and without this check the "best of a bad lot" would still
    # be returned as a loop. Match the design's own stated principle (see
    # design doc, "no loop is better than a bad one") and the validator's
    # own 5%-of-peak threshold: if even the closest achievable per-channel
    # match is still clearly bad, decline to loop at all rather than hand
    # back a candidate the validator would reject anyway.
    if peak > 0 and best_dv > MAX_ENDPOINT_DV_FRACTION * peak:
        return None

    # The "prefer longer" tolerance window is relative to best_dv (up to
    # 2x it), which can itself approach the ceiling above without
    # exceeding it -- e.g. best_dv at 3% of peak passes the check above,
    # but a tol of best_dv*2 = 6% would then let the tiebreak pick a
    # LONGER candidate at up to 6%, over the ceiling it was just checked
    # against. Clamp the tolerance to the same ceiling so the tiebreak can
    # never hand back a candidate the check above was meant to rule out.
    tol = max(best_dv * 2.0, 0.0005 * peak, 1e-9)
    if peak > 0:
        tol = min(tol, MAX_ENDPOINT_DV_FRACTION * peak)
    near_mask = dv_vals <= tol
    near_lags = cand_lags[near_mask]
    near_scores = cand_scores[near_mask]

    best_idx = int(np.argmax(near_lags))
    end = loop_start + int(near_lags[best_idx])
    chosen_score = float(near_scores[best_idx])

    length = end - loop_start
    if length <= 0:
        return None
    # Enforce the DecentSampler crossfade bound in code (not just documented):
    # loopCrossfade silently breaks looping when it's large relative to
    # loopStart or the loop length, so cap it hard on every path here.
    xfade = int(min(MAX_XFADE, loop_start // 4, length // 4))
    return {"enabled": True, "start": int(loop_start), "end": int(end),
            "crossfade": int(max(0, xfade)), "score": round(chosen_score, 4)}


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


def reloop_patch(pdir: Path, hold_seconds=3.5):
    """Re-run classification and loop detection on the already-encoded
    48 kHz FLACs in place, WITHOUT re-encoding audio or touching sample
    files (zone["file"], zone["frames"], and the audio bytes are never
    modified) -- only zone["kind"], zone["sustain_ratio"], zone["loop"],
    and zone["release"] are refreshed.

    process_patch deletes the source .wav after writing the .flac, so it
    cannot be re-run to pick up a loop-detection improvement; that would
    need a full re-render (37GB for the pilot library alone). This is the
    path for iterating on loop quality against already-processed output --
    which, per the find_loop rewrite above, we know we'll need to do again.
    The FLACs are lossless 48kHz/24-bit, so reading them back for
    classification loses nothing relative to the .wav they came from.
    """
    meta = json.loads((pdir / "patch.json").read_text())
    hold_out = int(hold_seconds * SR_OUT)

    for z in meta["zones"]:
        fname = z.get("file")
        # Only a zone that actually has encoded audio can be re-looped; a
        # "missing"/"error" zone (see FAILURE_KINDS) has no .flac to read,
        # and re-loop must never invent or touch sample files -- leave
        # those zones exactly as process_patch (or a previous reloop) left
        # them.
        if not fname or Path(fname).suffix.lower() != ".flac":
            continue
        src = pdir / fname
        if not src.exists():
            continue

        try:
            y, sr = sf.read(str(src), always_2d=True, dtype="float32")
            assert sr == SR_OUT, (
                f"unexpected rate {sr} in {src} (expected {SR_OUT}; this "
                "should already be an encoded 48kHz FLAC)"
            )

            kind, ratio = classify(y, hold_out)
            loop = find_loop(y, SR_OUT, hold_out) if kind == "sustaining" else None
            release_val = round(measure_release(y, SR_OUT, hold_out), 4)

            z["kind"] = kind
            z["sustain_ratio"] = round(ratio, 4)
            z["loop"] = loop or {"enabled": False}
            z["release"] = release_val
        except Exception as exc:
            # Unlike process_patch, we deliberately do NOT downgrade the
            # zone to _mark_zone_unprocessable here: the .flac is valid
            # audio that already had working (if possibly imperfect)
            # metadata from a prior run, so a transient re-loop failure
            # shouldn't discard that in favour of an "error" zone with no
            # usable loop info at all. Leave the previous metadata as-is.
            print(
                f"  reloop zone {fname!r} (key={z.get('key')}, "
                f"velocity={z.get('velocity')}): FAILED ({exc!r}) -- "
                "leaving previous metadata, continuing with next zone",
                file=sys.stderr,
            )

    meta["sample_rate"] = SR_OUT
    (pdir / "patch.json").write_text(json.dumps(meta, indent=2))
    return meta


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root",
        help="Directory containing per-patch subdirectories, each with a patch.json.",
    )
    p.add_argument(
        "--reloop",
        action="store_true",
        help=(
            "Re-run classification and loop detection on the existing "
            "48kHz FLACs in place, without re-encoding audio or touching "
            "sample files. Use this to pick up a find_loop improvement "
            "without a full re-render."
        ),
    )
    return p


def main():
    args = _build_arg_parser().parse_args()
    root = Path(args.root)
    process_fn = reloop_patch if args.reloop else process_patch
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
            m = process_fn(d)
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
