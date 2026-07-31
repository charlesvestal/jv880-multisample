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

## 3. Drum kits / Rhythm Sets -- DONE

Implemented. Rhythm Sets are a separate ROM region from Patches, which is why
none appeared among the 4,197 rendered patches. 52 kits exist: 3 internal
(Internal / Preset A / Preset B) and 49 across the 13 of 22 expansion boards
that carry them. Nine boards have none, including -- surprisingly -- Pop and
Vintage Synth, while Bass & Drum carries only 4.

A set is 61 keys (MIDI 36-96) x 44 bytes. Expansions declare rhythm_count at
header 0x68 and rhythm_offset at 0x90; the COUNT is authoritative, since Vocal
stores a non-zero offset while declaring zero sets.

Playback goes through the performance's rhythm part on MIDI channel 10, with
the wanted kit injected into the PRESET A rhythm region of an in-memory rom2
(never the file on disk). NVRAM 0x67f0 holds a copy of a kit but is NOT what
the firmware sounds -- writing there changes nothing.

Still open for kits:

  - They have NO names anywhere in the data. Scanning an expansion's whole
    tail region finds only the patch name table; the rhythm block sits in a
    stretch with no text at all. Kits are therefore named by board and index
    ("Rhythm 1".."Rhythm 4"), not by whatever Roland printed in the manual.
    If a name list is ever transcribed from the manuals, only the emitted
    preset names need to change -- no re-render.
  - Velocity is sampled at 4 fixed bands. A rhythm key holds a single tone
    with no velocity-switch points to derive layers from, unlike a patch.

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

## 5. DecentSampler lists presets in its own order -- NOT FIXABLE from here

Reported: presets in a loaded .dslibrary appear in the File Browser in no
useful order. They can be stepped through in order once the library is
loaded; it is the browser listing that is scrambled.

This is DecentSampler's behaviour, not ours. Measured on
"JV-880 Internal part1of6":

  - the archive stores presets in perfect order: A00, A01, A02, A03 ...
  - the File Browser shows: A16, A28, A05, A31, A20, A12, A11, A06 ...

Those names are already zero-padded and sort correctly, so this is neither a
filename-sort problem nor an archive-order problem. DecentSampler is using
neither.

Nothing in the format offers a lever:

  - no manifest. A commercial library (ASIMOV v1.0) contains only presets,
    Samples/ and background.png -- the same shape as ours.
  - no ordering attribute in the preset XML. ASIMOV's root element is a bare
    <DecentSampler>.
  - no naming convention that helps. ASIMOV uses UNPADDED names ("1 - Off
    World", "10 - Moog Town", "2 - Foundation"), so a commercial vendor has
    the same issue and has not solved it either.

A three-digit preset index was tried and REVERTED. It would give the chunker
tidier boundaries -- with two digits, chunk membership is lexicographic, which
is why Pop's patch 23 sits in part 3 of 5 -- but it does nothing for the
browser listing, and re-emitting into an already-rendered tree would leave
stale two-digit presets beside new three-digit ones for the chunker to
package twice.

If this is ever worth pursuing, it is a request to the DecentSampler author,
not a change to this repository.

