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

## 2. Loop / phrase patches
Patches where holding ONE key plays a musical or rhythmic loop rather than a
sustained tone. "House Hunter" is likely one of these (user's read, and it
matches its behaviour: percussive, non-sustaining, reverbtime 0).

These break assumptions the pipeline makes everywhere:
  - the fixed 3.5 s hold cuts the phrase at an arbitrary point
  - loop-point detection looks for a steady sustain that does not exist
  - one sample every 3 semitones transposes the whole phrase's TEMPO, since
    pitch and playback rate are the same thing in a sampler

The user's guidance: handle them properly OR exclude them from the library.
Excluding is legitimate and much cheaper -- a phrase patch transposed across
25 root keys is 25 different tempos, which is not what the JV does.

Needs first: a detector. Candidate signal is a strongly periodic amplitude
envelope (rhythmic re-triggering) inside a single held note, which a
sustained or decaying patch does not have.

## 3. Drum kits / Rhythm Sets -- SKIPPED for now
Explicitly deferred by the user. Note these are NOT missing from the render by
accident: on the JV, drum kits live in **Rhythm Sets**, a separate ROM area
from Patches, and `src/jv_rom.cpp` enumerates Patches only. So no drum kit is
present in the 4,197-patch count at all.

If picked up later it is genuinely new scope: new ROM parsing, plus a
different sampling strategy (one sample per key, no pitch-tiling, no
key-follow, no loop detection).
