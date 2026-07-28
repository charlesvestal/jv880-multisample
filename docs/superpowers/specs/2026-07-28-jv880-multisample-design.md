# JV-880 Multisample Library — Design

**Date:** 2026-07-28
**Status:** Approved for pilot

## Goal

Produce a high-quality multisampled instrument library from the Roland JV-880, covering
the 192 internal patches and all 22 SR-JV80 expansion boards (~2,800 patches), delivered
as DecentSampler `.dspreset` and SFZ `.sfz` presets organized one library per board.

The target is **desktop DecentSampler and general SFZ players**, not the Schwung
Multisampler on Move. Design priorities are fidelity and flexibility, not device
constraints.

## Why emulator rendering, not audio capture

The `schwung-jv880` repo contains the full JV-880 emulator core (`mcu.cpp`,
`mcu_opcodes.cpp`, `pcm.cpp`) plus a working headless harness at
`tools/render_test/render_test.cpp`. Verified on 2026-07-28:

- Builds clean in ~2s with `clang++ -std=c++17 -O2`.
- Renders 5.75s of audio in 0.28s user time — **~20x realtime, single core**.
- All 192 internal patch names read correctly from `rom2` at the documented offsets.

This gives deterministic, repeatable, faster-than-realtime rendering with byte-level
control over patch parameters. No audio interface, no plugin GUI automation, no
real-time constraint.

## Source material

ROMs: `/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880/`

| File | Size | Purpose |
|---|---|---|
| `jv880_rom1.bin` | 32 KB | CPU program |
| `jv880_rom2.bin` | 256 KB | CPU program + patch tables |
| `jv880_waverom1.bin` | 2 MB | PCM wave data |
| `jv880_waverom2.bin` | 2 MB | PCM wave data |
| `jv880_nvram.bin` | 32 KB | User patch/performance storage |
| `expansions/` | 22 files | SR-JV80 boards 01-19, 97, 98, 99 |

Patch banks in `rom2`, 0x16a bytes per patch, 12-byte ASCII name at offset 0:

| Bank | Offset | Count |
|---|---|---|
| Preset A | `0x010ce0` | 64 |
| Preset B | `0x018ce0` | 64 |
| Internal | `0x008ce0` | 64 |

## Sampling grid

- **Keys:** every 3 semitones, C1 (MIDI 24) through C7 (MIDI 96) = **25 zones**.
  Outermost zones extend their key range to 0 and 127 so the full keyboard sounds.
  Maximum pitch-shift on any note is 1.5 semitones.
- **Velocities:** 3 layers. Rendered at the layer midpoint, mapped to the full band:

  | Layer | Render velocity | Zone range |
  |---|---|---|
  | 1 | 32 | 1-42 |
  | 2 | 72 | 43-85 |
  | 3 | 110 | 86-127 |

- **Per patch:** 25 x 3 = **75 samples**.
- **Total:** 4,197 patches x 75 = **~315,000 samples**.

### Verified patch counts (measured 2026-07-28)

Expansion headers were parsed directly (unscramble + `patch_count` at `0x66..0x67`,
`patches_offset` at `0x8c..0x8f`). Actual totals are ~50% higher than first estimated:

| Source | Patches |
|---|---|
| Internal (Preset A + Preset B + Internal) | 192 |
| 20 working SR-JV80 boards | 4,005 |
| **Total** | **4,197** |

Seven boards carry the maximum 255 patches (02 Orchestral, 04 Vintage Synth, 05 World,
06 Dance, 08 Keyboards 60s/70s, 11 Techno, 12 HipHop).

**Boards 97 and 98 do not parse.** SR-JV80-97 (Experience III) and SR-JV80-98
(Experience II) both report `patch_count = 0`. Both are 2 MB promo boards; SR-JV80-99
(Experience), also 2 MB, parses correctly at 64 patches. These two are excluded from the
main pipeline and investigated separately — they must not block the other 20 boards.

#### Boards 97 and 98: investigation (2026-07-28, Task 10)

`tools/probe_expansion.py` investigated whether 97/98 can be parsed, using board 99 as
a known-good control for the detection method. Findings:

- **Control validated.** A full-image scan for candidate patch-name tables (Kadane
  max-subarray search for runs of plausible 12-byte names, calibrated against the
  3,941 real names read from boards 01-19: printable ASCII, ≥2 letters, longest
  uppercase-letter run ≤4 for 99.9% of real names) correctly locates board 99's real
  table at offset `0x1F9866`, a clean 65-entry run, patch 0 = `*Tr.Rhodes` — and is
  not fooled by the printable-but-garbage run a naive scan finds at `0x402`.
- **Nothing comparable exists on 97/98.** Run against the full 2 MB image, the same
  detector's best candidate for board 97 scores 12 (18% of board 99's real-table
  score of 65) as a 12-entry run; for board 98 it scores 10 (15%) as a 10-entry run.
  Both "candidates" are visibly not text — monotonic ascending byte ramps (e.g.
  `27<BIS[_bflo`) or runs of a single repeated byte (`wwwwwwwwwwww`) — the signature
  of PCM waveform/envelope data coincidentally landing in the printable-ASCII range,
  not patch names.
- **The hinted alternate header fields are not patch counts.** `0x60-0x61` (187/198)
  and `0x62-0x63` (34/27 for boards 97/98) looked like candidates since they're the
  only nonzero values near where `patch_count` normally lives. Surveyed across all 20
  boards with a real `patch_count`, `0x62-0x63 / patch_count` ranges 0.30-1.63 — not
  a constant ratio — so this is some other per-board quantity (plausibly a wave/tone
  count) that scales with board size, not a duplicate encoding of patch count. It's
  small on 97/98 because they're small 2 MB boards, not because it's secretly the
  real patch count.
- **Geometry rules out the standard `patches_offset` field too.** It points to
  within ~5.4 KB (board 97) / ~5.2 KB (board 98) of end-of-file — room for at most 14
  patch records — far short of even the smallest genuine table seen anywhere (64, on
  board 99 itself).

**Conclusion: boards 97 and 98 contain no parseable patch table in the SR-JV80
format.** `patch_count = 0` from the standard header is genuinely correct, not a
parsing bug or an alignment/offset error. They remain excluded from the pipeline; the
21-library output (Internal + 20 boards) stands as final.

Revised cost: ~250-315 GB and ~5-6 hours on 8 cores. Fits within the 1.1 TB free on
ExtFS.

### Per-sample timing

| Phase | Duration |
|---|---|
| Settle after patch load | 0.5 s (discarded) |
| Note held | 3.5 s |
| Release tail after note-off | 2.5 s |
| **Total captured** | **~6.0 s** |

Samples that decay to silence before note-off (pianos, plucks, percussion) are truncated
at the decay point, which materially reduces total disk use.

### Output format

48 kHz / 24-bit stereo FLAC. The emulator renders at its native 64 kHz and is resampled
to 48 kHz (a clean 4:3 ratio) with a high-quality SRC. The emulator's `sample_buffer` is
`int16_t`, so 16 bits is the true source depth; 24-bit output preserves resampler output
without adding requantization noise.

Estimated size: ~150-170 GB before decay truncation, less after. Destination is ExtFS
(1.1 TB free). The internal drive has only 22 GB free and must not be used.

## Effects: render dry, reconstruct in DecentSampler

**All samples are rendered fully dry** — reverb and chorus levels zeroed in the patch
bytes before rendering. Each preset's effect settings are read from ROM and reconstructed
as DecentSampler effects, so they stay adjustable and removable, and loops stay clean.

### Parameters captured per patch

Patch-common bytes (offsets from the plugin's verified parameter table):

| Parameter | Offset | Range | Values |
|---|---|---|---|
| `reverbtype` | 12 (bits 0-3) | 0-7 | Room1, Room2, Stage1, Stage2, Hall1, Hall2, Delay, Pan-Dly |
| `reverblevel` | 13 | 0-127 | |
| `reverbtime` | 14 | 0-127 | |
| `reverbfeedback` | 15 | 0-127 | |
| `chorustype` | 12 (bits 4-5) | 0-2 | Chorus1, Chorus2, Chorus3 |
| `choruslevel` | 16 | 0-127 | |
| `chorusoutput` | 16 (bit 7) | 0-1 | Mix, Reverb |
| `chorusdepth` | 17 | 0-127 | |
| `chorusrate` | 18 | 0-127 | |
| `chorusfeedback` | 19 | 0-127 | |

Per-tone sends, 4 tones: `reverbsendlevel` (offset 82), `chorussendlevel` (offset 83),
both 0-127. Since DS effects are global per-preset, these are combined into a single wet
level weighted by each tone's contribution, counting only active tones.

### Mapping to DecentSampler

| JV | DecentSampler |
|---|---|
| Reverb types 0-5 (Room/Stage/Hall) | `<effect type="reverb">` — `roomSize`, `damping`, `wetLevel` |
| Reverb types 6-7 (Delay, Pan-Dly) | `<effect type="delay">` — `delayTime`, `feedback`, `stereoOffset`, `wetLevel` |
| Chorus 1/2/3 | `<effect type="chorus">` — `mix`, `modDepth`, `modRate` |
| `chorusoutput` = Reverb | chorus placed before reverb in the chain |
| `chorusoutput` = Mix | chorus parallel / after |

Reverb types 6 and 7 are delay effects, not reverbs. Mapping them to `reverb` would be
incorrect; they get a `delay` effect instead.

### Calibration by measurement

Rather than inventing 0-127 to Hz/seconds formulas, the emulator is measured directly.
A one-time calibration pass renders controlled test tones and extracts real values:

| Sweep | Measurement | Produces |
|---|---|---|
| `chorusrate` 0-127 | LFO period via envelope/FFT analysis | rate to Hz table |
| `chorusdepth` 0-127 | peak pitch deviation | depth to `modDepth` |
| `choruslevel` 0-127 | wet/dry energy ratio | level to `mix` |
| `reverbtime` 0-127, per type | RT60 from decay slope | time to `roomSize` |
| `reverblevel` 0-127 | wet/dry energy ratio | level to `wetLevel` |

Roughly 30 throwaway renders, run once, producing `calibration.json` consumed by the
preset emitter. This is the main quality differentiator over a hand-tuned guess and is
only possible because the emulator is cycle-accurate.

## LFO handling: conditional strip

Each patch has up to 8 LFOs (LFO1 and LFO2 per tone, 4 tones), each with waveform, rate,
delay, fade time, offset, key sync, and independent depths to pitch, TVF, and TVA.

Baking LFOs into samples causes two problems: every note gets the same frozen LFO phase
(chords pulse in lockstep, unlike hardware where a free-running LFO is caught at a
different phase per note), and periodic modulation fights loop-point detection.

### Capability mapping

| JV | DecentSampler | Fidelity |
|---|---|---|
| LFO to TVA | `<lfo>` to `AMP_VOLUME` | good |
| LFO to pitch | `<lfo>` to group tuning | good |
| LFO to TVF | `<lfo>` to `FX_FILTER_FREQUENCY` | good, requires filter in chain |
| `synchro` On | `scope="voice"` | exact semantic match |
| `synchro` Off | `scope="global"` | exact semantic match |
| `lfo delay` | `delayTime` | exact |
| SIN / SAW / SQU | `sine` / `saw` / `square` | exact |
| TRI | `sine` approximation | minor loss |
| RND1 / RND2 | none | **not reproducible** |
| `fadetime`, `offset` | none | small gap |

### Decision rule, applied per patch

Because rendering captures the **sum** of up to 4 tones, per-tone LFO differences cannot
be separated after the fact.

> **Measured outcome (2026-07-28):** across the 192 internal patches, only **6 are
> LFO1-strippable and 1 is LFO2-strippable**. Breakdown for LFO1: 130 have no LFO depth
> at all, 13 use RND1/RND2, and 43 have active tones that genuinely disagree on
> waveform (15), pitch depth (11), rate (10), or TVF depth (7). The conservative rule
> below is working as specified — it is simply stricter in practice than it looked on
> paper, so most LFO-bearing patches keep their modulation baked in.
>
> **Decision: ship these defaults unchanged and evaluate the pilot by ear.** Rendering
> costs only compute time, and whether baked frozen-phase modulation is actually
> objectionable is a listening question, not an analytical one. Revisit the tolerances
> only if the pilot reveals a real problem.

**Strip and recreate** when all hold:
- at least one active tone has non-zero LFO depth to pitch, TVF, or TVA;
- all active tones agree on LFO settings within these tolerances:
  - waveform: identical;
  - `rate`: within +/- 4 (of 0-127);
  - each depth (pitch / TVF / TVA): within +/- 6 (of -63..+63) **and** matching sign;
  - `synchro`: identical;
- waveform is TRI, SIN, SAW, or SQU.

Where tones agree within tolerance but are not identical, the mean of the active tones'
values is used. Tones with zero output level do not count as active.

**Bake** otherwise — divergent per-tone settings, or RND1/RND2 waveforms.

In both cases the actual LFO values are recorded in the patch metadata, so a baked patch
is documented rather than silently different.

Stripping is implemented by zeroing `lfo1/2pitchdepth`, `lfo1/2tvfdepth`, and
`lfo1/2tvadepth` on each tone before rendering.

## What stays baked

These are the instrument, not effects on it:

- TVF / TVA / pitch envelopes (release handled separately, see below)
- Filter cutoff and resonance
- Waveform selection and tone structure
- Velocity switching between tones (captured by the 3 velocity layers)
- Tone delay (`tonedelaymode` / `tonedelaytime`) — per-tone delayed onset is structural
  and cannot be reproduced without splitting tones into separate DS groups
- `analogfeel` — its per-note random detune adds welcome variation across the
  multisample rather than harming it

## Correctness fix: portamento

Several patches have portamento enabled. Rendering notes back-to-back within one emulator
instance would produce pitch glides *between* sampled notes, silently corrupting them.

**Portamento is forced off during rendering**, unconditionally. This is a correctness
requirement, not a fidelity preference. `bendrangeup` / `bendrangedown` are carried into
metadata instead.

## Loops and release

Each rendered sample is analyzed:

1. **Classify.** Compare RMS in the final 250 ms before note-off against peak RMS. Above
   threshold, the patch sustains; below, it decays.
2. **Decayers** (pianos, plucks, percussion) — no loop, truncate at silence.
3. **Sustainers** — find a loop in the steady-state region (roughly 1.0 s to 3.5 s):
   - estimate fundamental period by autocorrelation;
   - generate candidate loop points at zero crossings on period boundaries;
   - score candidates by cross-correlation between loop start and end neighbourhoods;
   - accept the best if it clears a match threshold, otherwise leave unlooped.
4. **Crossfade.** `loopCrossfade` is kept small — the parity notes in
   `schwung-sfz/tools/parity/presets/20_loop_crossfade.dspreset` record that
   DecentSampler *silently fails* when `loopCrossfade` is large relative to `loopStart`
   or sample length. 500 frames (~11 ms) is the documented-safe reference. Crossfade is
   capped at both a fixed ceiling and a fraction of `loopStart`.
5. **Release.** Measure the decay time of the recorded tail; emit as `release` in DS and
   `ampeg_release` in SFZ.

Rendering fully dry helps here: with no chorus LFO in the signal, there is no modulation
phase to match at the loop point.

## Output structure

One library folder per board, **21 total** (Internal + 20 working boards):

```
JV-880 Multisamples/
  JV-880 Internal/
    A01 A.Piano 1.dspreset
    A01 A.Piano 1.sfz
    ...
    Samples/
      A01_APiano1/
        C1_v1.flac  C1_v2.flac  C1_v3.flac
        D#1_v1.flac ...
  SR-JV80-01 Pop/
  SR-JV80-02 Orchestral/
  ...
```

Samples live under the board folder and are shared by both preset formats.

### Format parity

- **`.dspreset`** — full fidelity: reverb, chorus, LFOs, loops, release. Effect
  parameters are additionally exposed as labeled UI knobs so they can be adjusted or
  bypassed.
- **`.sfz`** — SFZ has no standard reverb or chorus. SFZ presets are dry, with correct
  key/velocity mapping, loops, and envelopes; the captured effect settings are written
  into header comments for reference.

This limitation is inherent to SFZ, not a shortcut.

## Components

| Component | Language | Responsibility |
|---|---|---|
| `jv_sampler` | C++ | Batch renderer on the emulator core: ROM/expansion loading, patch enumeration, byte preprocessing (dry, LFO strip, portamento off), grid rendering, decay truncation |
| `calibrate` | C++ + Python | Effect parameter sweeps, produces `calibration.json` |
| `postprocess` | Python | Resample to 48 kHz, loop detection, release measurement, FLAC encode, per-patch metadata JSON |
| `emit_presets` | Python | Generate `.dspreset` and `.sfz` from metadata + calibration |
| orchestrator | shell/Python | Parallel driver across 8 cores |

Each stage communicates through files on disk (raw WAV, then metadata JSON), so stages
can be re-run independently — re-emitting presets after a mapping fix must not require
re-rendering audio.

### Per-patch metadata

```json
{
  "board": "JV-880 Internal",
  "bank": "A", "index": 0, "name": "A.Piano 1",
  "effects": {
    "reverb": {"type": "Hall1", "level": 80, "time": 64, "feedback": 0},
    "chorus": {"type": "Chorus1", "level": 40, "depth": 20, "rate": 30,
               "feedback": 0, "output": "Mix"}
  },
  "lfo": {"stripped": true, "shape": "TRI", "rate_raw": 52, "synchro": false,
          "depths": {"pitch": 0, "tvf": 0, "tva": 18}},
  "zones": [
    {"key": 24, "lokey": 0, "hikey": 25, "lovel": 1, "hivel": 42,
     "file": "Samples/A01_APiano1/C1_v1.flac",
     "loop": {"enabled": true, "start": 48000, "end": 158000, "crossfade": 500},
     "release": 0.8}
  ]
}
```

## Risks

1. ~~**Expansion ROM parsing**~~ — **resolved 2026-07-28.** The unscramble algorithm from
   `jv880_plugin.cpp:496` was reimplemented and validated against all 22 boards: 20 parse
   cleanly with correct patch names and counts. Remaining sub-risk is limited to boards 97
   and 98 (see above), which are excluded from the main pipeline.
2. **Loop quality on evolving patches** — some pads modulate continuously and have no
   truly steady state. Mitigated by the accept-threshold: no loop is better than a bad one.
3. **LFO compatibility heuristic** — the "tones agree within tolerance" rule needs a
   sensible tolerance. Spot-check against known tremolo/vibrato patches
   (e.g. Preset B "TremoloStrng", "Velo Strings").
4. **Calibration accuracy** — measured rates and RT60s depend on clean test signals.
   Validate against a handful of patches by ear before trusting the table.

## Plan: pilot first

Full render is several hours and ~150 GB. Before committing to that:

1. Build the pipeline end to end.
2. Render the **192 internal patches plus one expansion board** (SR-JV80-04 Vintage
   Synth — synth-heavy, exercises LFOs and sustained loops hard).
3. Validate: A/B a sample of presets against the real plugin, check loop quality on pads,
   confirm effect reconstruction sounds close, verify both formats load.
4. Only then run the remaining 21 boards.

This surfaces systematic problems at ~10 GB instead of after 150 GB.

## Verification

- `jv_sampler` renders a known patch bit-identically across runs (determinism).
- Rendered dry patches show measurably no reverb/chorus energy vs. the wet reference.
- Calibration table round-trips: a rendered chorus at rate N measures back to the
  predicted Hz within tolerance.
- Generated `.dspreset` and `.sfz` both load without error in DecentSampler and sfizz.
- Looped sustainers show no discontinuity at the loop point (sample-level check).
- Spot A/B against the JV-880 plugin on a fixed set of reference patches.
