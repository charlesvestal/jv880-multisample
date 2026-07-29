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

# Delay/Pan-Dly (reverb types 6-7) time/feedback are now measured --
# calibrate.cpp sweeps both types' reverbtime (on a percussive Marimba base
# patch, dry signal present) and analyze_calibration.py locates the first
# echo via cross-correlation against the same patch's own dry-attack shape,
# writing calib/calibration.json's "delay_time_s" and "delay_feedback"
# tables (see that file's DELAY_HOP block for the measurement rationale).
# Real numbers: Delay (type 6) reaches ~0.49s at raw=127, Pan-Dly (type 7)
# ~0.24s -- both comfortably in "slapback/short delay" territory, not the
# ~1s ping-pong echo the previous invented DELAY_TIME_MAX_S=1.0 produced on
# every max-time patch (reported by listening: "pads all have an odd
# ping-pong delay").
#
# DELAY_TIME_MIN_S/MAX_S remain ONLY as an explicit, documented fallback
# for when a real measured table isn't available (e.g.
# tests/test_emit_presets.py's synthetic CAL fixture, which predates
# delay_time_s) -- see _delay_time_s() below. They are never used when a
# real calibration.json is loaded.
DELAY_TIME_MIN_S = 0.02
DELAY_TIME_MAX_S = 1.0

# Pan-Dly (type 7) measurably alternates its successive echoes between
# channels (each repeat lands close to hard-panned, alternating sides --
# confirmed directly by comparing L/R peak amplitude at each repeat's
# measured arrival time); plain Delay (type 6) measured as genuinely
# centred/mono, every repeat landing at ~50/50 L/R to within noise. That
# measurement is WHETHER stereoOffset should be nonzero at all, and is
# recorded per-type in calibration.json's "delay_pan_alternation".
#
# The stereoOffset MAGNITUDE below is not, and cannot be, derived the same
# way: DecentSampler's stereoOffset is a TIME offset in seconds between the
# L and R channels' own delay taps ("half of this amount is subtracted
# from the left channel's delay time and half is added to the right" --
# official developer guide's appendix on the delay effect), not an
# amplitude pan -- there is no DS parameter that reproduces a hard
# alternating L/R gain swing at all. PAN_DLY_STEREO_OFFSET_FRAC is
# therefore a documented, REASONED proportion of the genuinely measured
# delayTime (not a measurement), in the same spirit as the developer
# guide's own worked example (stereoOffset=0.01-0.02s against a 0.5s
# delayTime, a 2-4% stereo widening) -- picked somewhat more generous here
# since the real hardware's channel alternation is much stronger than a
# subtle Haas widening, while PAN_DLY_STEREO_OFFSET_MAX_S caps it well
# below any plausible delayTime so it can never invert which channel
# leads, which is what made the old fixed stereoOffset=3.0 nonsensical
# against a ~1s (or, now, ~0.25-0.5s) delayTime.
PAN_DLY_STEREO_OFFSET_FRAC = 0.15
PAN_DLY_STEREO_OFFSET_MAX_S = 0.05

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

# --- LFO binding targets: verified 2026-07-28 against the official
# DecentSampler developer guide (Appendix B: The <binding> element,
# https://decentsampler-developers-guide.readthedocs.io/en/latest/
# appendix-b-the-binding-element.html), NOT assumed from the design doc's
# capability table -- see this file's module docstring for what changed
# after verification.
#
# Group Volume:  type="amp" level="group" parameter="AMP_VOLUME"
#                range 0.0-16.0, Modulatable: Yes, requires groupIndex.
# Group Tuning:  type="amp" level="group" parameter="GROUP_TUNING"
#                range -36.0-36.0 (semitones), Modulatable: Yes, requires
#                groupIndex.
# Both confirmed correct (my original parameter *names* were right); what
# was wrong was `type="generic"` for tuning (real type is "amp", same as
# volume) and `level="voice"` for volume ("voice" is not a valid `level`
# value at all -- valid values are ui/instrument/group/tag/midi -- and
# groupIndex is a required companion attribute I had omitted).
#
# TVF -> a filter cutoff target (FX_FILTER_FREQUENCY) is real
# (type="effect" level="instrument", requires effectIndex naming an
# actual filter effect in the <effects> chain) -- but this pipeline's
# build_effects() NEVER emits a filter/lowpass effect: the JV's own TVF
# filter is already baked into the rendered audio, not reconstructed as a
# DecentSampler effect (see design doc's "What stays baked" section).
# There is therefore no real effect index this preset could honestly
# point at -- binding to effectIndex=0 would silently modulate whatever
# happens to be first in OUR chain (reverb/delay or chorus), which is
# actively wrong, not just absent. Per instruction: a silently-ignored
# (or silently-WRONG) binding is worse than no binding, so TVF depth is
# deliberately dropped -- there is no valid DecentSampler equivalent in
# this pipeline's output, full stop.
GROUP_INDEX = "0"  # this module always emits exactly one <group>

# modBehavior="modulate": the LFO adds a zero-centered delta around the
# target's current/base value (the correct semantic for tremolo/vibrato).
# The alternative, "set" (the DecentSampler default), would force the
# parameter to literally EQUAL the LFO's instantaneous value each cycle,
# discarding whatever base value the group would otherwise have.
LFO_MOD_BEHAVIOR = "modulate"

# translationOutputMin/Max define each binding's own +/- swing at full
# LFO excursion (see module docstring) -- these are the values that
# actually encode "how deep", since a single <lfo> element's own
# modAmount is shared across every one of its bindings and can't
# represent independent per-target depths on its own (see
# _lfo_binding_range's docstring).
#
# Neither range is calibrated (same caveat as LFO_RATE_HZ_*: JV's tva/
# pitch LFO depth was never swept by calibrate.cpp), so these are
# documented, musically-reasoned choices, not measurements:
TVA_MOD_RANGE = 0.5           # +/- linear gain delta at full tva depth
GROUP_TUNING_MOD_RANGE_ST = 0.5  # +/- semitones at full pitch depth


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
    # A key may be PRESENT with a null value: analyze_calibration.py records an
    # unmeasurable point as null rather than omitting it or inventing a number
    # (e.g. reverb_rt60 type 1 raw 0, whose tail never crossed -35 dB). Treat
    # null exactly like an absent key -- interpolate across it from the nearest
    # real measurements. Coercing null to 0.0 would fabricate an instant decay.
    pts = sorted((int(k), float(v)) for k, v in table.items() if v is not None)
    if not pts:
        raise ValueError("calibration table has no measured values")
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


# --- Effect AMOUNT mapping: proportional, not measured -----------------------
#
# chorus_depth_norm / chorus_mix / reverb_wet were originally read from the
# measured calibration. They are not trustworthy and are no longer used:
#
#   chorus_depth_norm  raw 0 -> 1.0, raw 127 -> 0.69   (inverted, ~flat at 0.7)
#   chorus_mix         0,0,0,0, 0.42, 0.38, 0.67, 0.51, 1.0   (non-monotonic)
#   reverb_wet         0, 0.52, 0.80, 1.0, 0.0, 0.0, ...      (mid -> NO reverb)
#
# Every preset therefore received ~0.73 chorus modDepth regardless of its real
# setting, which is heavy pitch modulation -- audible as a warble on material
# with no chorus at all (A.Piano 1 has pitch LFO depth 0 on all four tones).
#
# The measurement was misapplied rather than merely noisy. Rate (Hz) and decay
# time (seconds) are genuinely device-specific and DO need measuring -- those
# tables are validated by tests and remain in use. But an AMOUNT is just an
# amount: a 0-127 level maps proportionally onto a 0-1 level. Measuring it
# added noise to a quantity whose mapping is known by construction.
#
# CHORUS_MAX_MOD_DEPTH is a deliberate, uncalibrated ceiling: DecentSampler's
# chorus modDepth defaults to 0.2, and the JV's chorus is a gentle stereo
# thickener rather than a vibrato, so full JV depth maps to a moderate value.
#
# Re-investigated (2026-07-29) using PURE-WET isolation -- drylevel=0,
# chorussendlevel=127, all other tones muted, sweeping chorusdepth alone --
# specifically to rule out the ORIGINAL failure's cause (dry/multi-tone
# contamination). Four independent measurement attempts were tried on that
# isolated signal: (1) RMS envelope excursion (the original metric, now
# free of dry contamination), (2) autocorrelation-based pitch tracking,
# (3) Hilbert-transform instantaneous frequency, (4) (3) plus robust
# MAD-based outlier rejection to reject the comb-filter-null artifacts (3)
# is prone to. ALL FOUR produced the same non-monotonic shape (rising then
# falling across the sweep, peaking mid-range) or, for the pitch-tracking
# attempts specifically, additional gross octave-lock errors at higher
# depth settings -- not the monotonic "zero depth -> zero modulation" shape
# a real chorus-depth control should have. Isolating the signal ruled out
# the ORIGINAL contamination theory but did not produce a clean
# measurement; a plausible remaining cause is this base patch's own
# baked-in chorusfeedback=100/127 (see tools/calibrate.cpp's neighbouring
# note on the SAME patch's chorusfeedback -- zeroing it was already shown
# elsewhere to make an unrelated measurement categorically worse, so it
# isn't a free fix here either). Per the task's own rule ("if a quantity
# doesn't measure cleanly, say so and fall back to a documented
# proportional mapping rather than shipping a noisy table"), this stays
# proportional.
CHORUS_MAX_MOD_DEPTH = 0.35


def amount(raw, ceiling=1.0):
    """Map a JV 0-127 amount onto a 0-1 amount, proportionally."""
    return _clamp01(max(0.0, min(127, float(raw))) / 127.0 * ceiling)


def _delay_time_s(cal, rtype, raw_time):
    """Measured echo spacing for reverb type `rtype` (6=Delay, 7=Pan-Dly)
    at raw byte `raw_time`, from calibration.json's "delay_time_s" table
    (see this module's comment above PAN_DLY_STEREO_OFFSET_FRAC for how it
    was measured). Falls back to the documented-but-uncalibrated linear
    guess (DELAY_TIME_MIN_S..MAX_S) only when no real table is available at
    all -- e.g. tests/test_emit_presets.py's synthetic CAL fixture, which
    predates this table; a real calib/calibration.json always has it."""
    table = cal.get("delay_time_s", {}).get(str(rtype))
    if table:
        try:
            return interp_table(table, raw_time)
        except ValueError:
            pass
    return DELAY_TIME_MIN_S + (raw_time / 127.0) * (DELAY_TIME_MAX_S - DELAY_TIME_MIN_S)


def _delay_feedback(cal, rtype, raw_feedback):
    """Measured per-repeat gain for reverb type `rtype` at raw byte
    `raw_feedback`, from calibration.json's "delay_feedback" table. Falls
    back to a straight proportional mapping when unavailable -- DS's own
    `feedback` parameter is ALSO defined as a 0-1 linear per-repeat gain
    (see the official developer guide), so this fallback is a reasonable
    amount()-style default, not a wild guess, even when unmeasured."""
    table = cal.get("delay_feedback", {}).get(str(rtype))
    if table:
        try:
            return interp_table(table, raw_feedback)
        except ValueError:
            pass
    return amount(raw_feedback)


def _pan_dly_stereo_offset(delay_time_s):
    """DS stereoOffset for a Pan-Dly patch with the given (measured)
    delayTime -- see this module's PAN_DLY_STEREO_OFFSET_FRAC comment for
    why this is a documented, reasoned proportion, not a measurement."""
    return round(min(PAN_DLY_STEREO_OFFSET_MAX_S,
                      PAN_DLY_STEREO_OFFSET_FRAC * delay_time_s), 4)


def _nearest_ir(cal, rtype, raw_time):
    """Path of the captured IR whose time step is nearest `raw_time`.

    The IR bank samples `reverbtime` at step 16 (9 steps per type). RT60
    varies smoothly across the full 0-127 range -- adjacent raw values differ
    by well under 1% -- so snapping to the nearest captured step costs at most
    a few percent of decay time, far below audibility. Returns None if no IR
    was captured for this reverb type, so the caller can fall back rather than
    emit a preset referencing a file that does not exist.
    """
    bank = cal.get("reverb_ir", {}).get(str(rtype))
    if not bank:
        return None
    steps = sorted(int(k) for k in bank)
    nearest = min(steps, key=lambda k: abs(k - int(raw_time)))
    return bank[str(nearest)]


def zone_vel_range(zones):
    """Per-zone (lovel, hivel), preferring the renderer's own values.

    jv_sampler now derives velocity layers from each patch's real tone
    switch points and writes explicit `lovel`/`hivel` per zone, so a patch
    that switches at velocity 100 gets a layer boundary exactly there rather
    than at an arbitrary third. Fall back to even division only for
    patch.json files written before that change.
    """
    if all("lovel" in z and "hivel" in z for z in zones):
        return [(int(z["lovel"]), int(z["hivel"])) for z in zones]
    return vel_ranges(len(zones))


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
        wet = _clamp01(amount(rv["level"]) * reverb_send_norm)
        delay_time = _delay_time_s(cal, rtype, rv["time"])
        reverb_effect = ("delay", {
            "delayTime": round(delay_time, 4),
            "feedback": round(_delay_feedback(cal, rtype, rv["feedback"]), 4),
            "stereoOffset": _pan_dly_stereo_offset(delay_time) if rtype == 7 else 0.0,
            "wetLevel": round(wet, 4),
        })
    else:
        wet = _clamp01(amount(rv["level"]) * reverb_send_norm)
        ir = _nearest_ir(cal, rtype, rv["time"])
        if ir is not None:
            # Convolution with an IR captured from the emulator itself, rather
            # than an approximation via roomSize/damping. The IRs are rendered
            # pure-wet (drylevel=0, reverbsendlevel=127) so they are the JV's
            # own reverb algorithm, not a model of it -- which no roomSize
            # value can be. The user's verdict on the parametric version was
            # that they could barely hear it.
            reverb_effect = ("convolution", {
                "irFile": ir,
                "mix": round(wet, 4),
            })
        else:
            # No IR captured for this type (should not happen for 0-5, but a
            # missing file must degrade to something audible rather than to
            # silence).
            rt60 = interp_table(cal["reverb_rt60"][str(rtype)], rv["time"])
            room_size = round(min(1.0, max(ROOMSIZE_FLOOR, rt60 / ROOMSIZE_RT60_REF)), 4)
            reverb_effect = ("reverb", {
                "roomSize": room_size,
                "damping": REVERB_DAMPING[rtype],
                "wetLevel": round(wet, 4),
            })

    mod_rate = min(LFO_RATE_HZ_MAX, max(0.0, interp_table(cal["chorus_rate_hz"], ch["rate"])))
    mod_depth = amount(ch["depth"], CHORUS_MAX_MOD_DEPTH)
    mix = _clamp01(amount(ch["level"]) * chorus_send_norm)
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


def _lfo_binding_range(depth, full_scale_range):
    """+/- output swing for one binding, scaled from a JV -63..63 depth.

    A single <lfo> element's own `modAmount` attribute (0-1) is shared
    across ALL of its <binding> children -- it can't represent "this LFO
    modulates pitch a little but volume a lot" on its own. Real per-target
    depth therefore has to live in EACH binding's own translationOutputMin/
    Max (its individual +/- swing at modAmount=1.0, DecentSampler's
    default), which is what this returns. `<binding>` itself has no
    modAmount attribute at all (verified against Appendix B's attribute
    list -- confirming an earlier mistake where I'd put modAmount on the
    binding instead of, correctly, leaving the shared one on <lfo>).
    """
    depth_frac = _clamp01(abs(depth) / 63.0)
    return round(depth_frac * full_scale_range, 4)


def build_lfo_modulator(lfo_meta):
    """Build an in-memory modulator description for one stripped LFO, or
    None if this LFO shouldn't emit a <lfo> element at all (not stripped,
    RND1/RND2 waveform, or stripped-but-zero-depth-everywhere).

    Returns {"shape", "frequency", "scope", "delayTime", "bindings": [...]}
    where each binding is {"type", "level", "groupIndex"?, "parameter",
    "modBehavior", "translationOutputMin", "translationOutputMax"}.

    A stripped LFO can bind MULTIPLE targets at once (pitch/TVA are
    independent depths on the same LFO); TVF depth is deliberately never
    bound to anything -- see GROUP_INDEX's neighbouring comment block for
    why FX_FILTER_FREQUENCY has no honest target in this pipeline's output.
    """
    if not lfo_meta or not lfo_meta.get("stripped"):
        return None
    shape = LFO_SHAPE.get(lfo_meta.get("form"))
    if shape is None:
        return None  # RND1/RND2 (or an unrecognized form) -- not reproducible

    bindings = []
    tva = lfo_meta.get("tva", 0)
    pitch = lfo_meta.get("pitch", 0)
    # tvf is intentionally never read here -- see the GROUP_INDEX comment
    # block above for why TVF has no valid DecentSampler binding target in
    # this pipeline's output.
    if tva:
        delta = _lfo_binding_range(tva, TVA_MOD_RANGE)
        bindings.append({
            "type": "amp", "level": "group", "groupIndex": GROUP_INDEX,
            "parameter": "AMP_VOLUME", "modBehavior": LFO_MOD_BEHAVIOR,
            "translation": "linear",
            "translationOutputMin": -delta, "translationOutputMax": delta,
        })
    if pitch:
        delta = _lfo_binding_range(pitch, GROUP_TUNING_MOD_RANGE_ST)
        bindings.append({
            "type": "amp", "level": "group", "groupIndex": GROUP_INDEX,
            "parameter": "GROUP_TUNING", "modBehavior": LFO_MOD_BEHAVIOR,
            "translation": "linear",
            "translationOutputMin": -delta, "translationOutputMax": delta,
        })
    if not bindings:
        return None  # stripped, valid waveform, but nothing left to bind
        # (either all depths were zero, or TVF was the only nonzero depth
        # and it has no valid binding target -- see above)

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

    by_key = _group_zones_by_key(zones)
    for key, lo, hi in key_ranges(by_key.keys()):
        layer_zones = sorted(by_key[key], key=lambda z: z["velocity"])
        for z, (vlo, vhi) in zip(layer_zones, zone_vel_range(layer_zones)):
            attrs = {
                "path": f"{sample_prefix}/{z['file']}",
                "rootNote": str(key),
                "loNote": str(lo), "hiNote": str(hi),
                "loVel": str(vlo), "hiVel": str(vhi),
                # DecentSampler's per-sample envelope release attribute is
                # simply `release` (alongside attack/decay/sustain) --
                # confirmed against the official developer guide. An
                # earlier version of this code used `ampegRelease`, which
                # is not a real DecentSampler attribute at all: it would
                # have been silently ignored, discarding every per-zone
                # release time measured in postprocess.py's
                # measure_release without any error.
                "release": str(z.get("release", 0.1)),
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

    # <modulators> is a TOP-LEVEL element, a sibling of <groups> and
    # <effects> directly under <DecentSampler> -- confirmed against the
    # official developer guide's own words: "This section lives below the
    # top-level <DecentSampler> element". An earlier version of this code
    # nested it inside <group>, which real DecentSampler does not
    # recognize as a modulator container at all -- the LFO would have been
    # silently dropped, exactly the "looks like it works but doesn't"
    # failure mode this verification pass was meant to catch. Placement
    # here has no bearing on scope="voice" vs "global": that's controlled
    # purely by the <lfo>'s own `scope` attribute, not by where the
    # <modulators> block sits in the tree.
    lfo_mods = [m for m in (build_lfo_modulator(meta.get("lfo1", {})),
                            build_lfo_modulator(meta.get("lfo2", {}))) if m]
    if lfo_mods:
        modulators_el = ET.SubElement(root, "modulators")
        for lm in lfo_mods:
            lfo_el = ET.SubElement(modulators_el, "lfo", {
                "shape": lm["shape"],
                "frequency": str(lm["frequency"]),
                "scope": lm["scope"],
                "delayTime": str(lm["delayTime"]),
                # Left at the DecentSampler default (1.0): per-target depth
                # is expressed via each binding's own translationOutputMin/
                # Max instead (see _lfo_binding_range's docstring for why).
                "modAmount": "1.0",
            })
            for b in lm["bindings"]:
                battrs = {
                    "type": b["type"], "level": b["level"],
                    "parameter": b["parameter"],
                    "modBehavior": b["modBehavior"],
                    "translation": b["translation"],
                    "translationOutputMin": str(b["translationOutputMin"]),
                    "translationOutputMax": str(b["translationOutputMax"]),
                }
                if "groupIndex" in b:
                    battrs["groupIndex"] = b["groupIndex"]
                ET.SubElement(lfo_el, "binding", battrs)

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
        for z, (vlo, vhi) in zip(layer_zones, zone_vel_range(layer_zones)):
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
        # NOTE: no "Samples/" wrapper. The real on-disk layout produced by
        # jv_sampler + postprocess.py is flat -- FLACs live directly in
        # <out_dir>/<pdir.name>/, alongside patch.json, and presets are
        # written into <out_dir> itself (see main()'s `out_root`, which
        # defaults to `library_dir` == <out_dir>) -- so the correct
        # relative path from a preset to its samples is just the patch
        # directory's own name. An earlier version of this used
        # "Samples/<pdir.name>", which doesn't exist anywhere on disk and
        # would have made every emitted <sample>/<region> unresolvable.
        sample_prefix = pdir.name

        # Preset filenames are what a player's browser shows, so keep them
        # short and scannable. Internal banks ("A"/"B"/"Internal") are useful
        # prefixes and stay. An expansion's bank IS the board name, which is
        # already the enclosing library folder -- repeating it gives
        # "SR-JV80-04 Vintage Synth00 Prologue" for all 255 presets, where the
        # index also runs straight into the board name. Drop the redundant
        # prefix in that case and lead with the index.
        bank = str(meta["bank"])
        prefix = "" if bank == out_root.name else bank
        sep = "" if prefix else ""
        stem = f"{prefix}{sep}{meta['index']:02d} {_sanitize_filename(meta['name'])}"
        (out_root / f"{stem}.dspreset").write_text(build_dspreset(meta, cal, sample_prefix))
        (out_root / f"{stem}.sfz").write_text(build_sfz(meta, sample_prefix))
        n += 1
        print(f"[{n}/{len(patch_dirs)}] emitted {stem}")

    print(f"emitted {n} preset pairs to {out_root}")


if __name__ == "__main__":
    main()
