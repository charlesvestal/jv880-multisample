# Deferred work

Agreed with the user, not yet done. Recorded here so it survives a session
ending.

## 1. Monophonic patches
The JV's key-assign mode (poly/solo) is never read. `src/jv_patch.cpp` reads
the portamento bit (byte 24, bit 6) and `preprocess()` deliberately turns
portamento OFF for sampling -- correct for capturing individual notes, but the
emitted preset then plays every patch polyphonically, including ones the JV
plays solo with glide.

Both halves are now researched; only the wiring is left.

JV side -- patch-common bits, confirmed against the reference implementation's
own parameter table (`schwung-jv880/src/dsp/jv880_plugin.cpp`, PATCH_COMMON_PARAMS):

    keyassign          byte 24, bit 7      0 = Poly, 1 = Solo
    sololegato         byte 24, bit 5      0 = Off,  1 = On
    portamentoswitch   byte 24, bit 6
    portamentomode     byte 24, bit 4
    portamentotime     byte 25, bits 0-6   0-127
    portamentotype     byte 25, bit 7

DecentSampler side -- confirmed in the developer guide:

    glideTime   seconds; 0.0 = no portamento
    glideMode   "always" | "legato" (default) | "off"
    polyphony="1" via the TAG system gives monophonic behaviour; there is no
                  direct monophony attribute on <group>

Work needed:
  1. `src/jv_patch.cpp` -- read the six bits above into the patch struct.
  2. `src/jv_sampler.cpp` -- write them into patch.json.
  3. `tools/emit_presets.py` -- for keyassign=Solo emit a mono tag with
     polyphony="1", and map portamentotime to glideTime with
     glideMode="legato" when sololegato is on, "always" otherwise.

NOTE: step 2 changes jv_sampler, so the already-rendered patch.json files will
lack the fields. Either re-run a metadata-only pass, or add a small dump tool
and merge the values into existing patch.json files -- the samples themselves
do not need re-rendering, since each note is sampled in isolation and mono
behaviour is purely a playback-time property.

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
