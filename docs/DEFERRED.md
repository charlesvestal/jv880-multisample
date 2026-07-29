# Deferred work

Agreed with the user, not yet done. Recorded here so it survives a session
ending.

## 1. Monophonic patches
The JV's key-assign mode (poly/solo) is never read. `src/jv_patch.cpp` reads
the portamento bit (byte 24, bit 6) and `preprocess()` deliberately turns
portamento OFF for sampling -- correct for capturing individual notes, but the
emitted preset then plays every patch polyphonically, including ones the JV
plays solo with glide.

Needs: read key-assign from the patch bytes, and emit a monophonic group with
glide in DecentSampler for those patches.

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
