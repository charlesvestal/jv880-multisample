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
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Kept as a literal copy of postprocess.FAILURE_KINDS rather than an
# import: postprocess.py is being actively edited by another task in this
# same branch, and this module's only contract with it is the *value* of
# this constant (see module docstring). Importing would couple this file's
# behavior to whatever postprocess.py happens to look like mid-edit.
FAILURE_KINDS = frozenset({"missing", "error"})
# Kinds the emitters refuse to map. "silent" is not a FAILURE -- the file
# exists and post-processing succeeded -- but a zone of digital silence is
# still a key that answers with nothing, so it is excluded from presets and
# its velocity band handed to a neighbour (see _valid_zones).
SKIP_KINDS = FAILURE_KINDS | frozenset({"silent"})

REVERB_NAMES = ["Room1", "Room2", "Stage1", "Stage2",
                "Hall1", "Hall2", "Delay", "Pan-Dly"]
DELAY_TYPE_INDICES = {6, 7}  # Delay, Pan-Dly

# Damping chosen per reverb type: rooms/stages read as smaller, more
# damped/duller spaces; halls read as bigger, brighter, longer-ringing
# spaces. This ordering (not the exact numbers, which are undocumented on
# real DecentSampler reverb) is the part that matters -- see the module's
# self-review note on musical plausibility.
REVERB_DAMPING = {0: 0.55, 1: 0.50, 2: 0.45, 3: 0.40, 4: 0.30, 5: 0.20}

# --- Why not convolution IRs --------------------------------------------------
#
# Capturing the JV's reverb as an impulse response was tried and abandoned.
# Every sound the JV can produce is a pitched PCM wave, so there is no way to
# excite the reverb with a true impulse. Three attempts, each better than the
# last, none usable:
#
#   1. Marimba note as excitation -- the "IR" was the marimba's own tone
#      convolved with the reverb, smearing its pitch across every sample.
#      Reported on listening as "a bunch of resonance".
#   2. Frequency-domain deconvolution of wet against dry -- the note's
#      harmonic nulls make the division unstable; recovered IR peaked 244ms
#      in rather than at zero.
#   3. TVA envelope forced to a ~26ms click, used raw (no deconvolution) --
#      decay profile was smooth and monotonic, but still audibly resonant,
#      and spectral flatness only reached 0.134.
#
# DecentSampler's own reverb is a perfectly good algorithm. The reason it was
# inaudible was not the model but two bugs below it: roomSize was derived
# against a 6-second reference that made every JV hall tiny, and wetLevel was
# being multiplied by EFFECT_MAX_MIX -- correct for chorus/convolution `mix`,
# which are blends, but wrong for reverb `wetLevel`, which is an additive
# return level and does not displace the dry signal.
#
# ROOMSIZE_RT60_REF is lowered from 6.0 to 3.0 so a typical JV hall (measured
# RT60 1.5-3.8s) maps to a substantial room rather than a small one.

# roomSize normalization: a 6-second RT60 is already a very large hall, so
# it's used as the "roomSize saturates to 1.0" reference point. Anything
# measured at or above that reads as a full-size (1.0) room; below that it
# scales linearly, floored so no reverb type ever reports a literal 0
# (DecentSampler's own default is 0.7; a hard 0 is untested territory).
ROOMSIZE_RT60_REF = 3.0
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
# recorded per-type in calibration.json's "delay_pan_alternation" -- with a
# clean injected impulse (vs. the old note-based cross-correlation), this
# reads even more starkly than before: type 6 measures 0.0 (perfectly
# centred, up from 0.0225) and type 7 measures 1.0 (FULLY hard-alternating,
# up from 0.6582) -- confirming DS's stereoOffset, a mere TIME offset
# between channels, structurally cannot reproduce what Pan-Dly actually
# does (a hard amplitude pan swing per repeat, not a phase/time shift).
#
# The stereoOffset MAGNITUDE below used to be a reasoned proportion of
# delayTime (PAN_DLY_STEREO_OFFSET_FRAC), analogised from the official
# developer guide's own generic worked example -- never measured. It is now
# MEASURED: for several real Pan-Dly patches, tools/ir_capture.py's
# find_stereo_offset() renders the real hardware's own wet output, computes
# its stereo width (RMS(L-R)/RMS(L+R) over the decay region -- the same
# metric validated for chorus depth), then simulates DecentSampler's own
# delay effect (independent L/R taps at delayTime -/+ stereoOffset/2, per
# the developer guide's own formula) at a sweep of candidate stereoOffset
# values and picks whichever makes the SIMULATED width closest to the real
# one. Six Pan-Dly patches measured stereoOffset in the 0.015-0.1s range
# (median 0.035s, written to calibration.json's "pan_dly_stereo_offset_s")
# -- close to, but a bit more generous than, the old guessed 0.15x-of-
# delayTime/cap-0.05s formula happened to land for a mid-range delayTime.
# PAN_DLY_STEREO_OFFSET_FRAC/MAX_S remain as the fallback for when no
# measured value is available (e.g. the synthetic test fixture).
PAN_DLY_STEREO_OFFSET_FRAC = 0.15
PAN_DLY_STEREO_OFFSET_MAX_S = 0.05
# However large the measured (or fallback) value, it must never approach
# the patch's own delayTime: DS subtracts half of it from the LEFT tap's
# time, and a value at or beyond delayTime would zero out or invert which
# channel leads -- exactly what made the very first, un-measured
# stereoOffset=3.0 (three seconds) nonsensical. Capping at half of
# delayTime keeps the left tap's own delay always positive and sane
# regardless of how short a given patch's delayTime is.
PAN_DLY_STEREO_OFFSET_MAX_FRAC_OF_DELAY = 0.5

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


def rhythm_key_spans(zone_keys):
    """One span per key, covering only that key -- the drum-kit mapping.

    A rhythm set is not a multisample of one instrument, it is 61 unrelated
    instruments side by side. Widening a span the way key_ranges() does would
    make the kick sound for several semitones around C2, transposed, and would
    silence whichever neighbour it displaced. Setting loNote == hiNote ==
    rootNote also guarantees no pitch shift is ever applied: the played note
    always equals the sample's root.
    """
    return [(k, k, k) for k in sorted(set(zone_keys))]


def spans_for(meta, zone_keys):
    """Key spans appropriate to what this preset is: a kit or a patch."""
    if meta.get("kind") == "rhythm":
        return rhythm_key_spans(zone_keys)
    return key_ranges(zone_keys)


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
    CONTRACT. Order is preserved from meta["zones"].

    In a RHYTHM SET, zones that rendered to DIGITAL SILENCE are dropped too.
    They are real files with valid metadata, so nothing else rejects them, but
    mapping one means a key that answers a keystroke with nothing. 418 of the
    12,688 drum zones are silent this way -- almost all velocity layer 1, in
    17 of the 52 kits -- because a JV rhythm tone can sit below its own
    velocity threshold at velocity 16. Dropping them alone would leave
    velocity 1-32 unmapped, so fill_velocity_gaps() stretches the surviving
    layers over it.

    A PATCH keeps its silent zones, and this asymmetry is deliberate. A kit
    maps one sample to one key, so dropping a key simply removes it. A patch
    TILES spans between its sampled keys, so dropping the silent ones makes
    the survivors stretch across the gap. 319 patches contain silent zones,
    and in the worst cases nearly all of them are: SR-JV80-10's CR-78 Hi-Hat
    has 72 of 75 zones silent because it is a percussion patch that only
    sounds over a few keys. Dropping those would smear three samples across
    the whole keyboard, transposed -- turning faithful silence into a wrong
    note. Silence is the correct answer there.
    """
    skip = SKIP_KINDS if meta.get("kind") == "rhythm" else FAILURE_KINDS
    return [z for z in meta.get("zones", []) if z.get("kind") not in skip]


def fill_velocity_gaps(ranges):
    """Stretch (lovel, hivel) pairs so they tile 1..127 with no holes.

    Only closes gaps; it never moves a boundary between two surviving layers.
    That keeps a patch's real velocity-switch points intact while making sure
    a key always answers, however softly it is struck. Input must be sorted
    ascending by lovel.
    """
    if not ranges:
        return []
    out = [list(r) for r in ranges]
    out[0][0] = 1            # softest surviving layer reaches down to 1
    out[-1][1] = 127         # loudest reaches up to 127
    for i in range(len(out) - 1):
        # Close any hole left by a dropped layer between these two.
        if out[i][1] < out[i + 1][0] - 1:
            out[i][1] = out[i + 1][0] - 1
    return [tuple(r) for r in out]


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
# CHORUS_MAX_MOD_DEPTH is a deliberate, uncalibrated CEILING: DecentSampler's
# chorus modDepth defaults to 0.2, and the JV's chorus is a gentle stereo
# thickener rather than a vibrato, so full JV depth maps to a moderate value.
# This number itself still cannot be derived from a physical measurement --
# DecentSampler's modDepth is explicitly documented as dimensionless ("0 is
# no modulation, 1.0 is max modulation"), so there is no unit conversion from
# a measured JV excursion to it. What CAN be (and, per the second
# investigation below, now has been) measured is the CURVE's SHAPE --
# whether depth is monotonic and how its effect actually scales across
# 0-127 -- which now multiplies this ceiling via chorus_depth_norm below
# instead of a flat proportional (raw/127) assumption.
#
# --- History: two failed measurement attempts, then a working one --------
#
# Attempt 1 (original): a musical note excitation, RMS envelope excursion.
# Produced raw 0 -> MAXIMUM depth (inverted, ~flat at 0.7) -- shipped once
# and made every preset in the library audibly warble.
#
# Attempt 2 (2026-07-29, pure-wet isolation): drylevel=0, chorussendlevel=127,
# all other tones muted, sweeping chorusdepth alone on a MUSICAL NOTE carrier
# -- ruling out the original failure's dry/multi-tone contamination theory.
# Four independent methods on that isolated signal: (1) RMS envelope
# excursion, (2) autocorrelation-based pitch tracking, (3) Hilbert-transform
# instantaneous frequency, (4) (3) plus MAD-based outlier rejection. ALL FOUR
# produced the same non-monotonic shape (rising then falling, peaking
# mid-range) or gross octave-lock errors -- isolating the signal ruled out
# contamination but did not produce a clean measurement.
#
# Attempt 3 (2026-07-29, injected impulse tool): with tools/wave_inject.cpp's
# clean synthetic excitation available, a THIRD attempt tried the task's own
# suggested method -- cross-correlate short windows of a pure-wet chorus
# render against its own dry reference sine to track the delay line's
# instantaneous lag. This ALSO failed cleanly: the recovered lag curve is
# dominated by an unexplained slow drift (tens-hundreds of samples/second)
# that swamps any real periodic signal; fitting a sinusoid at the KNOWN
# chorus rate to the detrended curve gives a flat, near-zero "amplitude" at
# EVERY depth setting including 0 and 127 -- no signal, not just noisy.
#
# Attempt 4 (2026-07-29, injected sine, ENERGY-domain metric) -- the one that
# worked: this chorus is a STEREO effect (independently modulated L/R delay
# lines), so instead of tracking phase/lag directly, measure how much a
# pure-wet chorused sine's L and R channels DECORRELATE: RMS(L-R)/RMS(L+R).
# This has no periodicity-aliasing failure mode (it's a plain energy ratio,
# not a cross-correlation), and it is cleanly, strongly MONOTONIC in raw
# chorusdepth -- and reproduces almost identically (0.999 shape-correlation)
# across two unrelated carrier frequencies (50Hz, 200Hz), strong evidence
# it's measuring something real rather than one tone's own artifact. See
# tools/ir_capture.py's cmd_chorus_depth for the full method. The resulting
# calibration.json "chorus_depth_norm" table is what CHORUS_MAX_MOD_DEPTH now
# scales (see build_effects below) instead of a flat proportional guess.
# Lowered 0.35 -> 0.10 after measuring what the JV's chorus actually does to
# PITCH. At the chorus LFO rate -- the only component the chorus can be
# responsible for, since the patch's own multi-tone detuning is broadband --
# MIDI EPiano (chorus depth 116/127, about as deep as the bank goes) measures
# 6.14 cents on the dry render and 4.07 cents with the JV's chorus engaged.
# The JV's chorus does not add pitch modulation there at all; it slightly
# reduces it. What it produces is doubling and stereo width, which `mix`
# already carries.
#
# DecentSampler's chorus is a genuinely pitch-modulating one, so mapping the
# JV's depth byte onto modDepth manufactures a vibrato the JV never had --
# reported as the electric piano sounding cartoonish. An earlier measurement
# appeared to clear this at 0.31 by comparing total pitch deviation, plugin
# 27.5 cents vs ours 26.7; that number is meaningless, because the DRY render
# measures 35.2 cents on its own. The metric was reading the patch's beating,
# not the chorus.
#
# 0.10 was still audibly too much in DecentSampler on MIDI EPiano (which
# emitted 0.089), so this is now 0.03. The measurement supports going
# essentially to zero -- the JV adds NO pitch modulation on that patch -- and
# `mix` still carries the doubling and stereo width that its chorus really
# does produce. A trace is kept rather than a hard zero only because DS's
# chorus at modDepth 0 is an untested corner.
#
# This remains a judgement call, not a measurement, and is flagged as such:
# modDepth is dimensionless and undocumented, there is no DecentSampler CLI to
# render through, and our own chorus model adds only +0.14 cents at 0.31 --
# so the model cannot be used to calibrate DS's response either. This keeps
# audible movement while staying far away from vibrato; only listening in
# DecentSampler can confirm the value.
# DecentSampler's chorus is juce::dsp::Chorus -- confirmed from JUCE's own
# source, whose constants make the mapping exact rather than guessed:
#
#     delay(t) = centreDelay + maximumDelayModulation * oscVolumeMultiplier
#                              * depth * sin(2*pi*rate*t)
#     with maximumDelayModulation = 20 ms, oscVolumeMultiplier = 0.5
#
# so the delay swings +/- 10*modDepth ms, and differentiating gives the pitch
# swing a listener actually hears:
#
#     cents = 1200 * log2(1 + 0.0628 * modDepth * modRate)
#
# That is why the earlier values were cartoonish: modDepth 0.089 at 1.6 Hz is
# +/- 15.4 cents of vibrato, and 0.30 is +/- 51 cents. It also shows the old
# flat ceiling was the wrong shape entirely -- pitch swing scales with RATE as
# well as depth, so a fixed modDepth gave patches with fast chorus several
# times the vibrato of slow ones, for no reason related to the JV.
#
# Solving for a target pitch swing instead makes it rate-independent.
CHORUS_MAX_CENTS = 4.0
CHORUS_DELAY_MOD_MS = 10.0      # juce maximumDelayModulation * oscVolumeMultiplier


def amount(raw, ceiling=1.0):
    """Map a JV 0-127 amount onto a 0-1 amount, proportionally."""
    return _clamp01(max(0.0, min(127, float(raw))) / 127.0 * ceiling)


def _chorus_mod_depth(cal, raw_depth, mod_rate_hz):
    """DecentSampler modDepth giving a target PITCH swing, not a fixed depth.

    DS's chorus is juce::dsp::Chorus, whose delay swings +/- 10*modDepth ms at
    modRate, so the audible vibrato is

        cents = 1200 * log2(1 + 0.0628 * modDepth * modRate)

    (see the CHORUS_MAX_CENTS block). Inverting that for a target swing makes
    the result rate-independent: a fast-chorus patch and a slow one get the
    same amount of pitch movement, which a fixed modDepth did not -- it gave
    them vibrato in proportion to their rate.

    The target is scaled by the measured chorus_depth_norm curve so deeper JV
    settings still move more, and the ceiling is small because the JV's chorus
    barely modulates pitch at all: on MIDI EPiano, at chorus depth 116/127,
    pitch modulation AT the chorus LFO rate measures 6.14 cents on the dry
    render and 4.07 cents with the JV's chorus engaged -- it adds none. What
    the JV's chorus produces is doubling and stereo width, which `mix` carries.
    """
    depth_norm = 1.0
    table = cal.get("chorus_depth_norm")
    if table:
        try:
            depth_norm = _clamp01(interp_table(table, raw_depth))
        except ValueError:
            depth_norm = _clamp01(raw_depth / 127.0)
    else:
        depth_norm = _clamp01(raw_depth / 127.0)

    cents = CHORUS_MAX_CENTS * depth_norm
    rate = max(0.05, float(mod_rate_hz))
    # Invert cents = 1200*log2(1 + k*modDepth*rate), k = 2*pi*CHORUS_DELAY_MOD_MS/1000
    k = 2.0 * math.pi * CHORUS_DELAY_MOD_MS / 1000.0
    ratio = 2.0 ** (cents / 1200.0) - 1.0
    return _clamp01(ratio / (k * rate))


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


def _pan_dly_stereo_offset(cal, delay_time_s):
    """DS stereoOffset for a Pan-Dly patch with the given (measured)
    delayTime. Prefers calibration.json's "pan_dly_stereo_offset_s" -- a
    value MEASURED by tools/ir_capture.py's find_stereo_offset() (see this
    module's comment above PAN_DLY_STEREO_OFFSET_FRAC) -- falling back to
    the old reasoned proportion only when no measurement is available
    (e.g. tests/test_emit_presets.py's synthetic CAL fixture, which
    predates this measurement). Either way, capped at half of THIS patch's
    own delayTime so a short delayTime can never be inverted."""
    measured = cal.get("pan_dly_stereo_offset_s")
    if measured is not None:
        base = measured
    else:
        base = min(PAN_DLY_STEREO_OFFSET_MAX_S, PAN_DLY_STEREO_OFFSET_FRAC * delay_time_s)
    return round(min(base, PAN_DLY_STEREO_OFFSET_MAX_FRAC_OF_DELAY * delay_time_s), 4)


# --- Send-vs-mix ceiling ----------------------------------------------------
#
# On the JV, `choruslevel` and `reverblevel` are SEND amounts: the dry signal
# always continues at full level and the effect is added alongside it. In
# DecentSampler, `mix` (chorus, convolution) is a BLEND -- the docs are
# explicit that "1.0 is just chorus, 0 is just dry signal" -- so mapping a
# full send straight onto mix=1.0 DELETES the direct sound.
#
# Measured on the 192 internal patches, the naive mapping produced mix > 0.5
# on 102 of them and > 0.8 on 39, i.e. half the library had its direct sound
# mostly replaced by pure effect output. That is both washy and, for chorus,
# maximally modulated -- the effect return is 100% modulated signal.
#
# A send at maximum means the effect sits at full level ALONGSIDE full dry,
# which is an equal blend. So a full JV send maps to mix 0.5, not 1.0.
EFFECT_MAX_MIX = 0.5

# Reverb depth for the CONVOLUTION path is no longer that flat ceiling. The
# ceiling was calibrated against a simulation that normalised the wet path,
# so it silently assumed a gain-neutral IR; the shipped IRs were 9-14 dB down,
# and the two errors compounded into a wet signal 21.8 dB below the dry on
# A.Piano 1 -- reported as "I'm not sure I hear the reverb". The IRs are now
# gain-neutral (ir_capture.normalize_ir), and how much reverb a patch gets is
# set here, from measurement instead of reasoning.
#
# Ground truth: for 14 internal patches spanning all 6 convolution types, the
# JV's own reverb-only contribution was measured against its own dry render as
# a wet/dry RMS ratio R. Regressed on the patch's reverb level x average send
# over active tones, R = 1.428*p - 0.048 (r = 0.77 across R = 0.33..1.49).
#
# Correlation 0.77, not 0.99: R also depends on reverb type and on the
# patch's own spectrum, and 14 patches will not resolve that. This is a
# measured central tendency, not a per-patch prediction.
#
# A blend cannot reproduce the JV exactly in any case -- the JV ADDS reverb
# beside undiminished dry, whereas `mix` trades one for the other. Matching
# the RATIO is what preserves the character, so mix = R/(1+R); the small
# overall level drop that comes with it is a volume-knob problem, not a tonal
# one. The cap keeps some direct sound on the wettest patches.
REVERB_RATIO_SLOPE = 1.428
REVERB_RATIO_INTERCEPT = -0.048
REVERB_RATIO_MAX = 1.7          # highest ratio measured across 88 patches

# A `mix` of m nominally yields a wet/dry ratio of m/(1-m). Measured against
# the emitted presets it yields only about 0.72x that -- convolution reflects
# some of the dry signal back correlated with itself, so part of what the
# blend adds does not read as "wet" at all. Setting mix = R/(1+R) as if the
# blend were ideal therefore lands systematically dry, by 4-7 dB on the
# wettest patches, which is why Pop Piano 1 (plugin ratio 1.53) was reported
# as sounding like it had no reverb.
#
# Solving mix against the measured efficiency instead: mix = R/(R + EFF).
REVERB_MIX_EFFICIENCY = 0.72
# Cap high enough that a genuinely wet patch is not clipped by the cap
# instead of by its own measurement -- at the old 0.6 the wettest patches
# could not reach their measured ratio at all. Dry is restored afterwards by
# blend_makeup_gain, so a high mix costs level, not presence.
REVERB_MIX_MAX = 0.78

# Ceiling on the dry-restoring makeup gain (see blend_makeup_gain). At the
# emitted maxima -- reverb mix 0.6 with chorus mix 0.5 -- exact compensation
# would be 5x (+14 dB). That is more trust than a derived number deserves at
# the extreme, so cap at +10 dB and let the very wettest patches stay slightly
# under rather than risk a preset that arrives far hotter than the plugin.
MAX_MAKEUP_GAIN = 3.16


def blend_makeup_gain(effects):
    """Linear gain restoring the dry signal that DecentSampler's blends remove.

    `mix` on convolution and chorus is a BLEND: it does not add the effect, it
    crossfades to it, so the direct sound comes out attenuated by (1 - mix) for
    each blending effect in the chain. The JV does the opposite -- a send routes
    a COPY to the effect and the dry path stays at full level. So a preset that
    matches the JV's wet/dry ratio still arrives with its dry signal several dB
    down (A.Piano 1: -4.9 dB from reverb, another -1.3 dB from chorus), which
    is audible as the sampled version sounding weaker than the plugin.

    The compensation is 1/sqrt(prod(1 - mix)) -- an ENERGY correction, not an
    amplitude one. Compensating amplitude by the obvious 1/prod(1 - mix)
    overshoots by about 3 dB, because the wet path is not independent of the
    dry: a chorus voice, and a reverb's early reflections, are both derived
    from the dry signal and partly correlated with it, so a blend at `mix`
    removes far less audible direct sound than (1 - mix) implies. Where wet and
    dry are uncorrelated their POWERS add, which is the square-root law.
    Confirmed by measurement rather than assumed: fitting the exponent against
    the makeup gain that actually equalises output level, over 24 patches
    audited against the plugin, gives 0.534 (median error -0.05 dB) versus
    +3.12 dB median for the amplitude form. 0.5 is used as the principled
    value the fit is indistinguishable from.

    This cannot clip in practice because the samples ARE the plugin's own dry
    signal, so the result reproduces the plugin's level -- and the ground-truth
    renders peak around -12 dBFS.

    `delay` is excluded deliberately: it uses an additive `wetLevel`, not a
    blend, so it never takes anything away from the dry path.
    """
    attenuation = 1.0
    for kind, attrs in effects:
        if kind in ("convolution", "chorus"):
            attenuation *= max(1e-3, 1.0 - float(attrs.get("mix", 0.0)))
    return min(MAX_MAKEUP_GAIN, 1.0 / math.sqrt(attenuation))


def _convolution_mix(wet, cal=None, rtype=None):
    """DecentSampler convolution `mix` reproducing the JV's wet/dry ratio.

    `wet` is the patch's reverb level scaled by its average send over active
    tones, i.e. the predictor the ground-truth ratios were regressed on. See
    REVERB_RATIO_SLOPE for the measurement and for why a ratio match (rather
    than a level match) is the right target for a blend control.

    Uses calibration.json's per-type fit when present (see
    tools/calibrate_reverb_ratio.py, 88 patches), falling back to the global
    coefficients. Per-type is a small but real improvement -- mean |error|
    3.44 -> 3.31 dB, 82% -> 84% of patches within 6 dB.

    Do not expect much more from this predictor. Measured over 88 patches the
    global correlation is 0.63, not the 0.77 an earlier 14-patch sample
    suggested, and three separate attempts to explain the residual scatter all
    came back negative: keeping the IRs' native per-type output gain instead of
    normalising it was WORSE (3.66 dB), weighting each tone's send by its level
    changed nothing (3.41 dB), and per-type fitting bought 0.13 dB. What is
    left looks genuinely patch-dependent -- how much reverb a given patch
    returns depends on the spectrum being fed into the algorithm, which no
    function of level and send can capture.
    """
    slope, intercept = REVERB_RATIO_SLOPE, REVERB_RATIO_INTERCEPT
    if cal:
        f = (cal.get("reverb_ratio_fit") or {}).get(str(rtype)) \
            or cal.get("reverb_ratio_fit_global")
        if f:
            slope, intercept = float(f["slope"]), float(f["intercept"])
    ratio = slope * _clamp01(wet) + intercept
    ratio = max(0.0, min(REVERB_RATIO_MAX, ratio))
    if ratio <= 0.0:
        return 0.0
    return round(min(REVERB_MIX_MAX, ratio / (ratio + REVERB_MIX_EFFICIENCY)), 4)


def _nearest_ir(cal, rtype, raw_time):
    """Return `(irFile, wet_ok)` for the IR whose time step is nearest `raw_time`.

    The IR bank samples `reverbtime` at step 16 (9 steps per type). RT60
    varies smoothly across the full 0-127 range -- adjacent raw values differ
    by well under 1% -- so snapping to the nearest captured step costs at most
    a few percent of decay time, far below audibility. Returns (None, False)
    if no IR was captured for this reverb type, so the caller can fall back
    rather than emit a preset referencing a file that does not exist.

    `wet_ok` is False when the nearest step captured as digital silence, which
    happens at reverbtime 0 for every type. Note what that does and does NOT
    mean. The JV genuinely DOES produce a wet signal at reverbtime 0 -- an
    A/B of House Hunter (type 4, level 127, time 0) against its own dry render
    measures the reverb-only contribution at 0.9x the dry RMS, correlated 0.41
    with the dry at lag 0: a short, dense, filtered ambience rather than a
    tail. What it means is that this response UNDERFLOWS to zero under impulse
    excitation: the emulator's fixed-point reverb, given a single-sample
    impulse and a near-zero decay time, produces output below its own
    resolution. So no IR can be captured for these steps -- not with a louder
    impulse, since the excitation is already full-scale.

    Given that, mix must be 0. DecentSampler's convolution `mix` is a BLEND,
    so any mix above 0 against an all-zero IR attenuates the dry signal while
    adding nothing (House Hunter: mix 0.32 -> -3.4 dB of dry for zero
    benefit). Pointing these patches at the nearest AUDIBLE step instead would
    be worse than silence: at type 4 that step carries an 849 ms tail where
    the truth is a brief ambience. So irFile references the nearest audible
    step purely so the knob does something if a player raises it, while the
    shipped mix stays 0.

    KNOWN LIMITATION: the 6 of 309 pilot patches with reverbtime < 8 therefore
    ship without their ambience. Capturing it needs a sustained excitation and
    deconvolution, which this pipeline deliberately abandoned -- deconvolved
    IRs carried audible source resonance.
    """
    bank = cal.get("reverb_ir", {}).get(str(rtype))
    if not bank:
        return None, False
    steps = sorted(int(k) for k in bank if bank[k])
    if not steps:
        return None, False
    silent = {int(s) for s in (cal.get("reverb_ir_silent", {}).get(str(rtype)) or [])}
    nearest = min(steps, key=lambda k: abs(k - int(raw_time)))
    if nearest not in silent:
        return bank[str(nearest)], True
    audible = [k for k in steps if k not in silent]
    if not audible:
        return None, False
    return bank[str(min(audible, key=lambda k: abs(k - int(raw_time))))], False


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


# The JV's portamento time byte is 0-127; DecentSampler's glideTime is in
# SECONDS. Measured rather than assumed: tools/wave_inject's `portamento`
# subcommand renders a real two-note glide at each setting (the JV only
# glides BETWEEN notes, so a single-note render cannot show it), and the
# 5-95% duration is read off an autocorrelation pitch track. The curve is
# steeply exponential and lands in calibration.json's "portamento_time_s":
#
#     raw   32     48     64     80     96     112     127
#     s     0.04   0.16   0.44   1.12   2.50   6.12    13.68
#
# Settings up to ~24 are instant. The previous value here was an assumed
# 1.5 s maximum, which was too fast by about 9x -- a patch at raw 127 glides
# for over thirteen seconds, and no amount of reasoning would have guessed
# that.
#
# Measured across a FIFTH (keys 48->55). The interval matters: the JV has a
# portamento TYPE byte selecting Rate or Time behaviour, and under Rate the
# duration scales with how far the pitch travels. DecentSampler's glideTime
# is a fixed time, so Rate-type patches cannot be exact at every interval;
# a fifth is used as the representative one.
GLIDE_TIME_FALLBACK_MAX_S = 1.5
MONO_TAG = "jv_solo"


def voice_attrs(meta, cal=None):
    """Group attributes for the patch's voice mode: monophony and glide.

    Sampling renders every note in isolation, which is right regardless of
    key assign -- monophony only matters when notes overlap, and that is the
    player's job, not the sample's. So this is carried purely as preset
    metadata.

    Returns (attrs, needs_mono_tag).
    """
    voice = meta.get("voice") or {}
    attrs = {}
    solo = str(voice.get("key_assign", "Poly")) == "Solo"

    # glideTime only when the patch actually enables portamento: the JV
    # stores a portamento TIME regardless (93 is a common stored default on
    # patches with portamento switched off), so keying off the time alone
    # would put a glide on most of the library.
    if voice.get("portamento"):
        raw = max(0, min(127, int(voice.get("portamento_time", 0))))
        table = (cal or {}).get("portamento_time_s")
        if table:
            secs = interp_table(table, raw)
        else:
            secs = raw / 127.0 * GLIDE_TIME_FALLBACK_MAX_S
        attrs["glideTime"] = f"{secs:.4f}"
        # The JV's solo-legato plays the glide only when the previous note is
        # still held, which is exactly DecentSampler's "legato" mode.
        attrs["glideMode"] = "legato" if voice.get("solo_legato") else "always"

    if solo:
        attrs["tags"] = MONO_TAG
    return attrs, solo


def _bus_effects(meta, cal):
    """Effects that belong on the parallel reverb SEND bus.

    Only the convolution reverb. A delay's `wetLevel` is ADDITIVE -- it passes
    its input through alongside the echoes -- so on a send bus it would return
    an undelayed copy of the dry to the main output and comb-filter against
    the original. It stays an insert, where additive wetLevel is exactly right.
    Convolution's `mix` is a blend, so at mix=1.0 it returns pure wet, which is
    what a send bus wants.
    """
    return [(kind, dict(attrs, mix=1.0)) for kind, attrs in build_effects(meta, cal)
            if kind == "convolution" and float(attrs.get("mix", 0.0)) > 0.0]


def _insert_effects(meta, cal):
    """Effects that stay in the instrument's own chain: chorus and delay."""
    return [(kind, attrs) for kind, attrs in build_effects(meta, cal)
            if kind != "convolution"]


def _reverb_target_ratio(meta, cal):
    """The JV's own wet/dry ratio for this patch, from the measured fit.

    Same predictor and per-type coefficients the blend used (see
    REVERB_RATIO_SLOPE), but now it is the send level directly rather than
    something to be converted into a crossfade position.
    """
    rv = meta["effects"]["reverb"]
    if rv["type"] not in REVERB_NAMES:
        return 0.0
    rtype = REVERB_NAMES.index(rv["type"])
    if rtype in DELAY_TYPE_INDICES:
        return 0.0
    wet = _clamp01(amount(rv["level"]) * (effective_send(meta, "reverb") / 127.0))
    slope, intercept = REVERB_RATIO_SLOPE, REVERB_RATIO_INTERCEPT
    if cal:
        f = (cal.get("reverb_ratio_fit") or {}).get(str(rtype)) \
            or cal.get("reverb_ratio_fit_global")
        if f:
            slope, intercept = float(f["slope"]), float(f["intercept"])
    return max(0.0, min(REVERB_RATIO_MAX, slope * wet + intercept))


def _reverb_send_level(meta, cal):
    """(group send level, bus volume) reproducing the JV's wet/dry ratio.

    With a parallel send the arithmetic is finally direct: the IRs are
    gain-neutral, so wet/dry is just send * busVolume. No blend efficiency
    factor, no cap to stop `mix` erasing the dry, and no makeup gain -- all
    three existed only to fight the crossfade, and all three are gone.

    Sends above unity are carried by busVolume, since a send level is a
    fraction of the signal being sent.
    """
    convolutions = [attrs for kind, attrs in build_effects(meta, cal)
                    if kind == "convolution"]
    if not convolutions or float(convolutions[0].get("mix", 0.0)) <= 0.0:
        return 0.0, 1.0
    ratio = _reverb_target_ratio(meta, cal)
    send = min(1.0, ratio)
    return send, (ratio / send if send > 0 else 1.0)


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

    # A rhythm set has no effect block at all. Unlike a patch -- where an
    # effect is deliberately kept even at level 0 so the UI knob still works
    # -- a drum kit is rendered dry with nothing to expose, and it declares
    # that with the type "Off", which is not one of the JV's eight reverb
    # names. Emitting no effects here is the accurate answer, not a fallback.
    if rv["type"] == "Off" and ch["type"] == "Off":
        return []

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
            "stereoOffset": _pan_dly_stereo_offset(cal, delay_time) if rtype == 7 else 0.0,
            "wetLevel": round(wet, 4),
        })
    else:
        wet = _clamp01(amount(rv["level"]) * reverb_send_norm)
        # IRs captured by injecting a synthetic impulse directly into the
        # wave ROM (see tools/wave_inject.cpp): excitation flatness 0.949 vs
        # 0.552 for the best musical note, so no deconvolution is needed and
        # the IR carries no source resonance. Validated across all 6 reverb
        # types against reverb-only ground truth, 5-8 patches per type (a
        # single representative patch is misleading -- A.Piano 1 alone scores
        # 0.997 for Hall1 against a type mean of 0.85): mean decay-envelope
        # correlation 0.8664.
        ir, wet_ok = _nearest_ir(cal, rtype, rv["time"])
        if ir is not None:
            # Convolution with an IR captured from the emulator itself, rather
            # than an approximation via roomSize/damping. The IRs are rendered
            # pure-wet (drylevel=0, reverbsendlevel=127) so they are the JV's
            # own reverb algorithm, not a model of it -- which no roomSize
            # value can be. The user's verdict on the parametric version was
            # that they could barely hear it.
            # convolution `mix` is a BLEND (1.0 = no dry signal), unlike the
            # parametric reverb's additive `wetLevel`. A full JV send means the
            # effect sits at full level ALONGSIDE full dry, so cap at an equal
            # blend -- without this, high-reverb patches (A.Piano 2 has level
            # 127 with full sends) emit mix=1.0 and lose their direct sound.
            reverb_effect = ("convolution", {
                "irFile": ir,
                "mix": _convolution_mix(wet, cal, rtype) if wet_ok else 0.0,
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
    mod_depth = _chorus_mod_depth(cal, ch["depth"], mod_rate)
    mix = _clamp01(amount(ch["level"]) * chorus_send_norm * EFFECT_MAX_MIX)
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
        # Restores the dry level DecentSampler's blend controls take away.
        # Only the chorus insert blends now -- the reverb moved to a parallel
        # SEND bus, which does not touch the dry path at all. See
        # blend_makeup_gain and build_dspreset's bus block.
        "volume": f"{blend_makeup_gain(_insert_effects(meta, cal)):.4f}",
    })

    # A true send, matching the JV: the dry goes to the main output at full
    # level and a COPY goes to the reverb bus. DecentSampler duplicates rather
    # than steals the signal, so unlike `mix` this cannot thin out the direct
    # sound.
    #
    # Only routed when a bus actually exists. Delay/Pan-Dly patches have no
    # convolution and so no bus, and an earlier version still stamped
    # output2Target="BUS_1" on them -- 683 presets pointing a send at a bus
    # that was never declared. Harmless in practice, since the send level was
    # 0, but malformed.
    # Voice mode: monophony and glide, from the patch's own key-assign and
    # portamento bytes.
    vattrs, needs_mono_tag = voice_attrs(meta, cal)
    for k, v in vattrs.items():
        group_el.set(k, v)
    if needs_mono_tag:
        tags_el = ET.SubElement(root, "tags")
        # polyphony="1" is how DecentSampler expresses monophony; there is no
        # direct attribute for it on <group>.
        ET.SubElement(tags_el, "tag", {"name": MONO_TAG, "polyphony": "1"})

    if _bus_effects(meta, cal):
        group_el.set("output1Target", "MAIN_OUTPUT")
        group_el.set("output2Target", "BUS_1")
        group_el.set("output2Volume", f"{_reverb_send_level(meta, cal)[0]:.4f}")

    by_key = _group_zones_by_key(zones)
    for key, lo, hi in spans_for(meta, by_key.keys()):
        layer_zones = sorted(by_key[key], key=lambda z: z["velocity"])
        for z, (vlo, vhi) in zip(layer_zones,
                                 fill_velocity_gaps(zone_vel_range(layer_zones))):
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
    for etype, eattrs in _insert_effects(meta, cal):
        ET.SubElement(effects_el, "effect",
                      {"type": etype, **{k: str(v) for k, v in eattrs.items()}})

    # The reverb bus. The JV routes a per-tone SEND to a global reverb block
    # whose output is added back -- it does not crossfade the dry away, and
    # its stereo image is the reverb's own rather than the note's. Modelling
    # that with DecentSampler's `mix` got both wrong: the dry arrived several
    # dB down, and the wet inherited each note's pan. Measured on A.Piano 2,
    # whose dry pans 25.5 dB across the keyboard by key position, the JV's
    # reverb moves only 5.1 dB -- it is effectively a fixed, centred bus --
    # while a per-channel blend dragged the reverb the full 25.5 dB with it.
    # That affects 36% of the library (>6 dB pan span) and 20% severely
    # (>20 dB).
    for etype, eattrs in _bus_effects(meta, cal):
        buses_el = root.find("buses") or ET.SubElement(root, "buses")
        bus_el = buses_el.find("bus")
        if bus_el is None:
            _, bus_volume = _reverb_send_level(meta, cal)
            bus_el = ET.SubElement(buses_el, "bus", {
                "busVolume": f"{bus_volume:.4f}",
                "output1Target": "MAIN_OUTPUT",
            })
            bus_fx = ET.SubElement(bus_el, "effects")
            # Collapse the send to mono BEFORE convolving, so the reverb sits
            # where the JV's does instead of following the note across the
            # stereo field.
            ET.SubElement(bus_fx, "effect",
                          {"type": "stereo_simulator", "width": "0"})
        bus_fx = bus_el.find("effects")
        ET.SubElement(bus_fx, "effect",
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
    for key, lo, hi in spans_for(meta, by_key.keys()):
        layer_zones = sorted(by_key[key], key=lambda z: z["velocity"])
        for z, (vlo, vhi) in zip(layer_zones,
                                 fill_velocity_gaps(zone_vel_range(layer_zones))):
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


def _stage_referenced_irs(out_root, calib_root):
    """Copy every IR a just-emitted preset references into the library.

    `irFile` is written relative to the preset, so the WAVs have to physically
    live inside the library folder or DecentSampler silently renders no reverb
    at all -- indistinguishable from "the reverb is too quiet", and reported as
    exactly that once already. This used to be a manual copy step, which meant
    the library was correct only if someone remembered; re-pointing the bank
    (calib/ir -> calib/ir_synth) left every preset referencing a directory that
    did not exist. Staging here makes the emitted library self-contained by
    construction. Returns the number of files copied.
    """
    wanted = set()
    for preset in out_root.glob("*.dspreset"):
        for eff in ET.parse(preset).getroot().findall('.//effect[@type="convolution"]'):
            if eff.get("irFile"):
                wanted.add(eff.get("irFile"))

    staged = 0
    for rel in sorted(wanted):
        src, dst = calib_root / rel, out_root / rel
        if not src.exists():
            raise SystemExit(f"emitted preset references missing IR: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Copy unconditionally. This used to skip when the sizes matched, which
        # is silently wrong for exactly the change most likely to happen to an
        # IR: re-processing one in place -- a different filter, a different
        # normalisation -- leaves the length and format identical, so the size
        # is identical, so the stale file survived in every library. That cost
        # a full spectral-correction validation run, which reported the
        # corrected bank as byte-for-byte no improvement. There are only tens
        # of these files; there is nothing to optimise here.
        shutil.copy2(src, dst)
        staged += 1
    return staged


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
        #
        # Two digits, deliberately left alone. A three-digit index would make
        # filename order match patch order (a two-digit index sorts as 0..9,
        # 10, 100, 101, ..., 11, 110), which would give the CHUNKER tidier
        # boundaries. It does NOT fix how DecentSampler lists presets: its
        # File Browser shows them in an order of its own, ignoring both
        # filename order and archive order -- measured, see docs/DEFERRED.md.
        # Changing the padding now would also leave stale two-digit presets
        # beside new three-digit ones in any already-rendered tree, and the
        # chunker would package both.
        bank = str(meta["bank"])
        prefix = "" if bank == out_root.name else bank
        sep = "" if prefix else ""
        stem = f"{prefix}{sep}{meta['index']:02d} {_sanitize_filename(meta['name'])}"
        (out_root / f"{stem}.dspreset").write_text(build_dspreset(meta, cal, sample_prefix))
        (out_root / f"{stem}.sfz").write_text(build_sfz(meta, sample_prefix))
        n += 1
        print(f"[{n}/{len(patch_dirs)}] emitted {stem}")

    staged = _stage_referenced_irs(out_root, Path(args.calibration).resolve().parent)
    print(f"emitted {n} preset pairs to {out_root} ({staged} IR files staged)")


if __name__ == "__main__":
    main()
