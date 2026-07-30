# Deferred work

Agreed with the user, not yet done. Recorded here so it survives a session
ending.

## 1. Monophonic patches -- DONE (2026-07-30)

Implemented. `src/jv_patch.cpp` reads the six patch-common voice bits,
`jv_sampler` writes a "voice" block into patch.json, and `emit_presets`
turns it into a `polyphony="1"` tag plus glideTime/glideMode.

267 of 4,197 presets (6.4%) are monophonic and 336 (8.0%) glide -- mostly
basses and solo leads, which is what the JV's Solo mode is for.

Two things worth remembering:
  - `tools/backfill_voice.py` merges the voice block into ALREADY-rendered
    patch.json files via `jv_sampler --dump-voice`. Re-running the sampler
    normally would regenerate patch.json and destroy what postprocess wrote
    into the zones (kind, loop points, release).
  - glideTime is emitted only when the portamento SWITCH is on. The JV stores
    a portamento time regardless -- 93 is a common stored default on patches
    with portamento off -- so keying off the time alone would have put a
    glide on most of the library.

Portamento time is now MEASURED, not assumed. `wave_inject portamento` renders
a real two-note glide at each setting -- the JV only glides BETWEEN notes, so
a single-note render cannot show it, which is why `Renderer::render_glide` was
added -- and the 5-95% duration comes off an autocorrelation pitch track:

    raw   32     48     64     80     96     112     127
    s     0.04   0.16   0.44   1.12   2.50   6.12    13.68

Settings up to ~24 are instant. The previous assumed 1.5 s maximum was about
9x too fast. Stored in calibration.json as "portamento_time_s".

Measured across a FIFTH. Still open: the portamento TYPE byte (25 bit 7)
selects Rate vs Time behaviour, and under Rate the duration scales with the
interval. DecentSampler's glideTime is a fixed time, so Rate-type patches
cannot be exact at every interval -- a fifth is used as representative.

## 2. Loop / phrase patches -- DONE (2026-07-30)

Handled, not excluded. Roland names these patches with their tempo
("125:BtMenu 1", "83:Kick It"), so all 190 are identified exactly rather than
guessed from audio -- three separate envelope-based detectors were tried
first and all found tremolo and chorus instead.

The real defect was truncation, not transposition. The standard grid holds
each note 3.5 s while a two-bar phrase at 61 BPM runs nearly 8 s, so anything
under ~137 BPM was cut mid-bar. `jv_sampler --hold` plus
`tools/rerender_phrases.py` re-render each one at a hold computed from its own
labelled tempo (4.0-11.6 s).

NOT excluded, for two reasons:
  - many are hybrids whose tones carry different key-follow values
    (125:ElevatMe reads [0,0,0,6]), so dropping the patch would discard
    playable material along with the loop;
  - the JV is itself a PCM sampler, so a higher key plays the wave faster.
    Multisampling reproduces the hardware rather than fighting it. An earlier
    claim in this file that "the JV never does this" was wrong.

Still open: loop POINTS are chosen by the generic sustain-loop finder and have
no reason to land on a bar boundary. Snapping them to a multiple of the
measured phrase period would make held notes repeat musically.

## 3. Drum kits / Rhythm Sets -- SKIPPED for now
Explicitly deferred by the user. Note these are NOT missing from the render by
accident: on the JV, drum kits live in **Rhythm Sets**, a separate ROM area
from Patches, and `src/jv_rom.cpp` enumerates Patches only. So no drum kit is
present in the 4,197-patch count at all.

If picked up later it is genuinely new scope: new ROM parsing, plus a
different sampling strategy (one sample per key, no pitch-tiling, no
key-follow, no loop detection).

## 4. Loop points ignore baked-in amplitude modulation

Reported on listening: "tremolo is really broken up" on the bank B strings,
worst on TremoloStrng.

Most JV patches carry an LFO in the SAMPLES -- it is only stripped when
decide_lfo_strip judges it strippable, so a great deal of periodic amplitude
modulation survives into the audio. find_loop scores candidates purely on
crossfade-region correlation and knows nothing about that modulation, so a
loop can land mid-tremolo and the modulation jumps phase on every pass.

Measured over one looped zone in each of 93 internal patches:

    strong periodic AM present          87  (94%)
    loop NOT a whole multiple of it     53  (61% of those)

    TremoloStrng   4.86 periods   <- worst, and the one reported
    Warm Strings  35.53 periods
    Slow Strings  22.04 periods   <- aligned, and sounds fine

Two caveats before acting on those numbers:

  - The worst offenders cluster at almost exactly 0.50 periods off, which is
    too systematic to be chance. The detector is probably locking onto the
    SECOND HARMONIC of the modulation (an amplitude envelope peaks twice per
    LFO cycle), so the true misalignment may differ. Verify the period
    detection before trusting the 61%.
  - Misalignment is not automatically audible. The crossfade runs up to 0.5 s
    and smears roughly nine cycles at 18 Hz, which can mask it. The reported
    case has the shortest loop (0.26 s) and therefore the least smearing.

Fix: constrain loop length to a whole multiple of any detected AM period,
alongside the existing crossfade-correlation score. This is the same
machinery item 2 needs for bar-aligned phrase loops -- do them together.

Cost is low: `postprocess --reloop` recomputes loop points from the encoded
FLACs with no re-render, the same route used to apply the release fix.

