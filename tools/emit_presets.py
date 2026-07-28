#!/usr/bin/env python3
"""Emit DecentSampler `.dspreset` and SFZ `.sfz` presets from a patch.json
(produced by jv_sampler / postprocess.py) plus the measured effect
calibration table (calib/calibration.json).

Scope note: this file and tests/test_emit_presets.py are the only files
this task touches. It reads patch.json / calibration.json but does not
modify the pipeline that produces them.

Schema this module consumes (verified 2026-07-28 against the CURRENT
src/jv_sampler.cpp + tools/postprocess.py, not just the design doc -- the
two disagree in minor ways and the code wins):

    {"name": str, "bank": str, "index": int, "sample_rate": int,
     "effects": {
        "reverb": {"type": <one of REVERB_NAMES>, "level": 0-127,
                   "time": 0-127, "feedback": 0-127},
        "chorus": {"type": "Chorus1|Chorus2|Chorus3", "level": 0-127,
                   "depth": 0-127, "rate": 0-127, "feedback": 0-127,
                   "output": "Mix|Reverb"},
        "reverb_send": [int,int,int,int],   # per tone, 0-127
        "chorus_send": [int,int,int,int],   # per tone, 0-127
        "tone_level": [int,int,int,int],    # per tone, 0-127; 0 = inactive
        "bend_up": int, "bend_down": int},
     "lfo1": {"stripped": bool, "reason": str,
              "form": "TRI|SIN|SAW|SQU|RND1|RND2",
              "rate": 0-127, "delay": 0-127, "sync": 0|1,
              "pitch": -63..63, "tvf": -63..63, "tva": -63..63},
     "lfo2": {...same shape as lfo1...},
     "zones": [{"key": int, "velocity": int, "layer": int, "frames": int,
                "file": str,
                # the following are added by postprocess.py:
                "kind": "sustaining|decaying|silent|missing|error",
                "sustain_ratio": float,
                "loop": {"enabled": bool, "start"?: int, "end"?: int,
                         "crossfade"?: int, "score"?: float},
                "release": float}]}

CRITICAL CONTRACT (see postprocess.py's `_mark_zone_unprocessable`
docstring): a zone with kind "missing" (never rendered) or "error"
(rendered but unprocessable) may have a `file` that does not exist or is
not valid audio. Every zone with any OTHER kind is guaranteed, by
postprocess.py's own "point of no return" write-then-commit ordering, to
have `file` pointing at a real FLAC on disk. This module never emits a
<sample>/<region> for a zone in FAILURE_KINDS.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Kept as a literal copy of postprocess.FAILURE_KINDS rather than an
# import: postprocess.py is being actively edited by another task in this
# same branch, and this module's only contract with it is the *value* of
# this constant (see module docstring). Importing would couple this file's
# behavior to whatever postprocess.py happens to look like mid-edit.
FAILURE_KINDS = frozenset({"missing", "error"})

REVERB_NAMES = ["Room1", "Room2", "Stage1", "Stage2",
                "Hall1", "Hall2", "Delay", "Pan-Dly"]
DELAY_TYPE_INDICES = {6, 7}  # Delay, Pan-Dly

# Damping chosen per reverb type: rooms/stages read as smaller, more
# damped/duller spaces; halls read as bigger, brighter, longer-ringing
# spaces. This ordering (not the exact numbers, which are undocumented on
# real DecentSampler reverb) is the part that matters -- see the module's
# self-review note on musical plausibility.
REVERB_DAMPING = {0: 0.55, 1: 0.50, 2: 0.45, 3: 0.40, 4: 0.30, 5: 0.20}

# roomSize normalization: a 6-second RT60 is already a very large hall, so
# it's used as the "roomSize saturates to 1.0" reference point. Anything
# measured at or above that reads as a full-size (1.0) room; below that it
# scales linearly, floored so no reverb type ever reports a literal 0
# (DecentSampler's own default is 0.7; a hard 0 is untested territory).
ROOMSIZE_RT60_REF = 6.0
ROOMSIZE_FLOOR = 0.05

# Delay/Pan-Dly (reverb types 6-7) were never swept by calibrate.cpp (it
# only sweeps type in 0..5 -- see tools/calibrate.cpp's `with_reverb` loop
# and tools/analyze_calibration.py's TYPE_RE, which never sees "6"/"7").
# There is therefore no measured delay-time table; this is an explicit,
# documented, uncalibrated best-effort mapping of the same raw "time" byte
# onto a musically sane slapback/delay range, not a measured curve.
DELAY_TIME_MIN_S = 0.02
DELAY_TIME_MAX_S = 1.0
PAN_DLY_STEREO_OFFSET = 3.0

# DecentSampler only supports these three LFO shapes; TRI has no exact
# match and is approximated by sine (see design doc's capability table).
# RND1/RND2 have no equivalent at all and are deliberately absent here, so
# lfo_shape.get(form) returning None is exactly the "no LFO" signal.
LFO_SHAPE = {"TRI": "sine", "SIN": "sine", "SAW": "saw", "SQU": "square"}

# No calibration table exists for the general-purpose LFO1/LFO2 rate --
# calib/calibration.json only measures the *chorus* LFO's rate. This is a
# documented, monotonic exponential curve spanning a musically sane
# 0.05-10 Hz (consistent with the measured chorus range of 0.69-9.51 Hz),
# not a measured result.
LFO_RATE_HZ_MIN = 0.05
LFO_RATE_HZ_MAX = 10.0
LFO_DELAY_MAX_S = 2.0


def interp_table(table, raw):
    """Linear interpolation over a ``{str(raw): value}`` calibration table.

    Handles missing interior keys transparently: the bracket search below
    only ever looks at keys that are actually PRESENT in `table`, so a gap
    like chorus_rate_hz's missing "24" is interpolated straight across
    using its nearest present neighbours (16 and 32) rather than being
    treated as a present-but-zero entry. Values outside the measured range
    are clamped to the nearest measured endpoint.
    """
    if not table:
        raise ValueError("empty calibration table")
    pts = sorted((int(k), float(v)) for k, v in table.items())
    keys = [k for k, _ in pts]
    vals = [v for _, v in pts]

    if raw <= keys[0]:
        return vals[0]
    if raw >= keys[-1]:
        return vals[-1]
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k0 <= raw <= k1:
            if k1 == k0:
                return vals[i]
            t = (raw - k0) / (k1 - k0)
            return vals[i] + t * (vals[i + 1] - vals[i])
    return vals[-1]  # unreachable given the bounds checks above


def key_ranges(zone_keys):
    """Contiguous key spans covering 0..127, one per distinct sampled key.

    Boundaries fall at the midpoint between adjacent sampled keys (integer
    floor division, so an odd gap's extra unit goes to the LOWER key --
    an arbitrary but consistent tie-break). The lowest span always starts
    at 0 and the highest always ends at 127, so the full keyboard sounds
    even though only every-3rd-semitone was actually sampled.

    Returns a list of (key, loNote, hiNote) tuples, ascending by key.
    """
    keys = sorted(set(zone_keys))
    spans = []
    for i, k in enumerate(keys):
        lo = 0 if i == 0 else (keys[i - 1] + k) // 2 + 1
        hi = 127 if i == len(keys) - 1 else (k + keys[i + 1]) // 2
        spans.append((k, lo, hi))
    return spans


def vel_ranges(n_layers):
    """Tile 1..127 across `n_layers` contiguous, non-overlapping bands.

    Boundaries are placed by even division (round(i * 127 / n_layers)),
    which reproduces the design doc's documented 3-layer split exactly
    (1-42, 43-85, 86-127) while still degrading gracefully to fewer bands
    when some velocity layers of a key were skipped (see FAILURE_KINDS).
    """
    if n_layers <= 0:
        return []
    bounds = [1 + round(i * 127 / n_layers) for i in range(n_layers + 1)]
    bounds[-1] = 128  # force the last band to end at 127, not a rounded value
    return [(bounds[i], bounds[i + 1] - 1) for i in range(n_layers)]


def effective_send(meta, which):
    """Average `{which}_send` across active tones only (tone_level > 0).

    Per the design doc: DS effects are global per-preset, but the JV-880
    sends are per-tone, so the per-tone send amounts are averaged across
    whichever tones are actually contributing sound. A patch with no
    active tones (degenerate/empty) sends nothing, by definition.
    """
    fx = meta["effects"]
    sends = fx[f"{which}_send"]
    levels = fx["tone_level"]
    active = [s for s, lv in zip(sends, levels) if lv > 0]
    if not active:
        return 0.0
    return sum(active) / len(active)


def _valid_zones(meta):
    """Zones safe to reference on disk -- see module docstring's CRITICAL
    CONTRACT. Order is preserved from meta["zones"]."""
    return [z for z in meta.get("zones", []) if z.get("kind") not in FAILURE_KINDS]


def _clamp01(x):
    return max(0.0, min(1.0, x))


def build_effects(meta, cal):
    """Return [(type, {attr: value}), ...] in signal-chain order.

    Reverb types 0-5 -> `reverb` (roomSize/damping/wetLevel); types 6-7
    (Delay, Pan-Dly) -> `delay` (delayTime/feedback/stereoOffset/wetLevel).
    Chorus is always emitted as `chorus` (mix/modDepth/modRate). Both
    effects are ALWAYS present (even at level 0 / silent) so they stay
    adjustable/bypassable via the UI knobs in build_dspreset, matching the
    design doc's "stay adjustable and removable" intent -- an effect is
    never silently omitted just because its wet contribution measures to
    ~0 for this particular patch.

    Chain order: chorus first when `chorus.output == "Reverb"` (it feeds
    the reverb bus), otherwise reverb/delay last.
    """
    fx = meta["effects"]
    rv, ch = fx["reverb"], fx["chorus"]

    if rv["type"] not in REVERB_NAMES:
        raise ValueError(f"unknown reverb type {rv['type']!r}")
    rtype = REVERB_NAMES.index(rv["type"])

    reverb_send_norm = effective_send(meta, "reverb") / 127.0
    chorus_send_norm = effective_send(meta, "chorus") / 127.0

    if rtype in DELAY_TYPE_INDICES:
        wet = _clamp01(interp_table(cal["reverb_wet"], rv["level"]) * reverb_send_norm)
        delay_time = DELAY_TIME_MIN_S + (rv["time"] / 127.0) * (DELAY_TIME_MAX_S - DELAY_TIME_MIN_S)
        reverb_effect = ("delay", {
            "delayTime": round(delay_time, 4),
            "feedback": round(rv["feedback"] / 127.0, 4),
            "stereoOffset": PAN_DLY_STEREO_OFFSET if rtype == 7 else 0.0,
            "wetLevel": round(wet, 4),
        })
    else:
        rt60 = interp_table(cal["reverb_rt60"][str(rtype)], rv["time"])
        room_size = round(min(1.0, max(ROOMSIZE_FLOOR, rt60 / ROOMSIZE_RT60_REF)), 4)
        wet = _clamp01(interp_table(cal["reverb_wet"], rv["level"]) * reverb_send_norm)
        reverb_effect = ("reverb", {
            "roomSize": room_size,
            "damping": REVERB_DAMPING[rtype],
            "wetLevel": round(wet, 4),
        })

    mod_rate = min(LFO_RATE_HZ_MAX, max(0.0, interp_table(cal["chorus_rate_hz"], ch["rate"])))
    mod_depth = _clamp01(interp_table(cal["chorus_depth_norm"], ch["depth"]))
    mix = _clamp01(interp_table(cal["chorus_mix"], ch["level"]) * chorus_send_norm)
    chorus_effect = ("chorus", {
        "mix": round(mix, 4),
        "modDepth": round(mod_depth, 4),
        "modRate": round(mod_rate, 4),
    })

    if ch["output"] == "Reverb":
        return [chorus_effect, reverb_effect]
    return [reverb_effect, chorus_effect]


def _lfo_rate_hz(raw):
    raw = max(0, min(127, raw))
    span = LFO_RATE_HZ_MAX / LFO_RATE_HZ_MIN
    return round(LFO_RATE_HZ_MIN * (span ** (raw / 127.0)), 4)


def _lfo_delay_seconds(raw):
    return round((max(0, min(127, raw)) / 127.0) * LFO_DELAY_MAX_S, 4)


def build_lfo_modulator(lfo_meta):
    """Build an in-memory modulator description for one stripped LFO, or
    None if this LFO shouldn't emit a <lfo> element at all (not stripped,
    RND1/RND2 waveform, or stripped-but-zero-depth-everywhere).

    Returns {"shape", "frequency", "scope", "delayTime", "bindings": [...]}
    where each binding is {"type", "level", "parameter", "modAmount"}.
    A stripped LFO can bind MULTIPLE targets at once (pitch/TVF/TVA are
    independent depths on the same LFO), each with its own modAmount
    scaled from the JV's -63..63 depth range.
    """
    if not lfo_meta or not lfo_meta.get("stripped"):
        return None
    shape = LFO_SHAPE.get(lfo_meta.get("form"))
    if shape is None:
        return None  # RND1/RND2 (or an unrecognized form) -- not reproducible

    bindings = []
    tva = lfo_meta.get("tva", 0)
    tvf = lfo_meta.get("tvf", 0)
    pitch = lfo_meta.get("pitch", 0)
    if tva:
        bindings.append({"type": "amp", "level": "voice", "parameter": "AMP_VOLUME",
                          "modAmount": round(_clamp01(abs(tva) / 63.0), 4)})
    if tvf:
        bindings.append({"type": "effect", "level": "instrument",
                          "parameter": "FX_FILTER_FREQUENCY",
                          "modAmount": round(_clamp01(abs(tvf) / 63.0), 4)})
    if pitch:
        bindings.append({"type": "generic", "level": "group", "parameter": "GROUP_TUNING",
                          "modAmount": round(_clamp01(abs(pitch) / 63.0), 4)})
    if not bindings:
        return None  # stripped, valid waveform, but nothing left to bind

    return {
        "shape": shape,
        "frequency": _lfo_rate_hz(lfo_meta.get("rate", 0)),
        "scope": "voice" if lfo_meta.get("sync") else "global",
        "delayTime": _lfo_delay_seconds(lfo_meta.get("delay", 0)),
        "bindings": bindings,
    }


def _group_zones_by_key(zones):
    by_key = {}
    for z in zones:
        by_key.setdefault(z["key"], []).append(z)
    return by_key


def build_dspreset(meta, cal, sample_prefix):
    """Build a well-formed DecentSampler .dspreset XML document (as a
    string) for `meta` (one patch.json's parsed content), using `cal`
    (parsed calibration.json) and `sample_prefix` as the directory the
    <sample path=...> values are relative to.

    Zones whose kind is in FAILURE_KINDS are skipped entirely (see module
    docstring's CRITICAL CONTRACT) -- they never produce a <sample>.
    """
    zones = _valid_zones(meta)

    root = ET.Element("DecentSampler", {"pluginVersion": "1"})

    groups_el = ET.SubElement(root, "groups")
    group_el = ET.SubElement(groups_el, "group", {
        # A simple hold-then-release shape: the renderer already holds the
        # note for the full sample duration, so attack/decay are
        # effectively instantaneous and sustain is full; the measured,
        # per-zone release (see postprocess.py's measure_release) is the
        # only envelope stage that varies and is set per-<sample> below.
        "attack": "0.001", "decay": "0.001", "sustain": "1.0",
    })

    lfo_mods = [m for m in (build_lfo_modulator(meta.get("lfo1", {})),
                            build_lfo_modulator(meta.get("lfo2", {}))) if m]
    if lfo_mods:
        modulators_el = ET.SubElement(group_el, "modulators")
        for lm in lfo_mods:
            lfo_el = ET.SubElement(modulators_el, "lfo", {
                "shape": lm["shape"],
                "frequency": str(lm["frequency"]),
                "scope": lm["scope"],
                "delayTime": str(lm["delayTime"]),
            })
            for b in lm["bindings"]:
                ET.SubElement(lfo_el, "binding", {
                    "type": b["type"], "level": b["level"],
                    "parameter": b["parameter"], "modAmount": str(b["modAmount"]),
                })

    by_key = _group_zones_by_key(zones)
    for key, lo, hi in key_ranges(by_key.keys()):
        layer_zones = sorted(by_key[key], key=lambda z: z["velocity"])
        for z, (vlo, vhi) in zip(layer_zones, vel_ranges(len(layer_zones))):
            attrs = {
                "path": f"{sample_prefix}/{z['file']}",
                "rootNote": str(key),
                "loNote": str(lo), "hiNote": str(hi),
                "loVel": str(vlo), "hiVel": str(vhi),
                # NOTE: ampegRelease is NOT in the pre-verified <sample>
                # attribute list this task was given (path/rootNote/loNote/
                # hiNote/loVel/hiVel/start/end/tuning/volume/pan/loopStart/
                # loopEnd/loopEnabled/loopCrossfade). It is added anyway
                # because the design doc requires .dspreset "full fidelity:
                # reverb, chorus, LFOs, loops, release" and ampegRelease is
                # a standard, widely-used DecentSampler per-sample envelope
                # attribute -- but since it wasn't in the supplied verified
                # list, flag it for a quick doc/spot-check before shipping.
                "ampegRelease": str(z.get("release", 0.1)),
            }
            loop = z.get("loop") or {}
            if loop.get("enabled"):
                attrs["loopEnabled"] = "1"
                attrs["loopStart"] = str(loop["start"])
                attrs["loopEnd"] = str(loop["end"])
                # Passed through UNCHANGED: postprocess.py already caps this
                # to DecentSampler's silent-failure-prone bound (see its
                # find_loop docstring); inflating it here would reintroduce
                # exactly the bug that capping was meant to prevent.
                attrs["loopCrossfade"] = str(loop.get("crossfade", 0))
            ET.SubElement(group_el, "sample", attrs)

    effects_el = ET.SubElement(root, "effects")
    for etype, eattrs in build_effects(meta, cal):
        ET.SubElement(effects_el, "effect",
                      {"type": etype, **{k: str(v) for k, v in eattrs.items()}})

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def build_sfz(meta, sample_prefix):
    """Build an SFZ text document for `meta`. SFZ has no standard
    reverb/chorus, so this is dry by design; the captured effect settings
    are written into header comments for reference (per the design doc's
    documented format-parity limitation, not a shortcut).
    """
    fx = meta.get("effects", {})
    lines = [
        f"// {meta.get('name', '')} -- JV-880 {meta.get('bank', '')}"
        f"#{meta.get('index', '')}",
        "// SFZ has no standard reverb/chorus; captured effect settings "
        "(for reference only):",
        f"//   reverb: {fx.get('reverb', {})}",
        f"//   chorus: {fx.get('chorus', {})}",
        f"//   lfo1:   {meta.get('lfo1', {})}",
        f"//   lfo2:   {meta.get('lfo2', {})}",
        "",
        "<control>",
        f"default_path={sample_prefix}/",
        "",
    ]

    zones = _valid_zones(meta)
    by_key = _group_zones_by_key(zones)
    for key, lo, hi in key_ranges(by_key.keys()):
        layer_zones = sorted(by_key[key], key=lambda z: z["velocity"])
        for z, (vlo, vhi) in zip(layer_zones, vel_ranges(len(layer_zones))):
            lines.append("<region>")
            lines.append(f"sample={z['file']}")
            lines.append(f"lokey={lo} hikey={hi} pitch_keycenter={key}")
            lines.append(f"lovel={vlo} hivel={vhi}")
            lines.append(f"ampeg_release={z.get('release', 0.1)}")
            loop = z.get("loop") or {}
            if loop.get("enabled"):
                lines.append("loop_mode=loop_continuous")
                lines.append(f"loop_start={loop['start']} loop_end={loop['end']}")
            else:
                lines.append("loop_mode=no_loop")
            lines.append("")

    return "\n".join(lines) + "\n"


def _sanitize_filename(s):
    """Mirror src/jv_sampler.cpp's `sanitize()` exactly, so preset
    filenames stay consistent with the patch-directory names already on
    disk (alnum, space, '-', '_', '.' only; no trailing spaces)."""
    out = "".join(c for c in s if c.isalnum() or c in " -_.")
    return out.rstrip(" ") or "patch"


def _check_zone_files_exist(meta, patch_dir):
    """Belt-and-suspenders integration check: postprocess.py's contract
    guarantees kind not in FAILURE_KINDS implies z['file'] exists on disk
    (see module docstring), but this is the point where a violated
    contract would otherwise silently ship a preset referencing a missing
    file (acceptance criterion 9). Mutates an in-memory copy only -- never
    writes back to patch.json."""
    for z in meta.get("zones", []):
        if z.get("kind") in FAILURE_KINDS:
            continue
        if not (patch_dir / z.get("file", "")).exists():
            print(
                f"  warning: {patch_dir.name} zone key={z.get('key')} "
                f"velocity={z.get('velocity')}: kind={z.get('kind')!r} but "
                f"file {z.get('file')!r} does not exist on disk -- "
                "excluding from emitted presets",
                file=sys.stderr,
            )
            z["kind"] = "error"
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library_dir",
                     help="directory containing one subdirectory per rendered "
                          "patch, each with a patch.json")
    ap.add_argument("calibration", nargs="?",
                     default=str(Path(__file__).resolve().parent.parent
                                 / "calib" / "calibration.json"),
                     help="path to calibration.json (default: calib/calibration.json)")
    ap.add_argument("--out", default=None,
                     help="output directory for .dspreset/.sfz (default: library_dir)")
    args = ap.parse_args()

    root = Path(args.library_dir)
    out_root = Path(args.out) if args.out else root
    cal = json.loads(Path(args.calibration).read_text())

    patch_dirs = sorted(p for p in root.iterdir() if (p / "patch.json").exists())
    if not patch_dirs:
        print(f"no patch.json found under {root}", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdir in patch_dirs:
        meta = json.loads((pdir / "patch.json").read_text())
        meta = _check_zone_files_exist(meta, pdir)
        sample_prefix = f"Samples/{pdir.name}"

        stem = f"{meta['bank']}{meta['index']:02d} {_sanitize_filename(meta['name'])}"
        (out_root / f"{stem}.dspreset").write_text(build_dspreset(meta, cal, sample_prefix))
        (out_root / f"{stem}.sfz").write_text(build_sfz(meta, sample_prefix))
        n += 1
        print(f"[{n}/{len(patch_dirs)}] emitted {stem}")

    print(f"emitted {n} preset pairs to {out_root}")


if __name__ == "__main__":
    main()
