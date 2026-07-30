# jv880-multisample

Turn a set of Roland JV-880 ROMs into multisampled DecentSampler libraries.

Point it at your ROMs and it renders every patch on every expansion board,
measures the JV's own effects, and writes `.dslibrary` files:

```bash
python3 tools/build_library.py --roms /path/to/roms --out /path/to/output
```

The full run is 4,197 patches across 21 boards, about 5 hours of rendering
and ~150 GB. `--board "SR-JV80-04 Vintage Synth"` does one board.

The effect calibration ships with the repo, so the default run goes straight
to rendering. Pass `--stop-after calibrate` to re-measure it from your own
ROMs instead.

## What is and is not in this repository

**Not here:** any Roland PCM. No samples, no patch data. Those come from your
own ROMs, on your own machine.

**Here:** the measured effect calibration, including the reverb impulse
responses. These are the JV reverb's response to a synthetic impulse that this
project generates and injects into a copy of the wave ROM — no Roland waveform
is present in the signal, the same way an IR pack of a hardware reverb
contains none of the recordings ever played through it. Shipping them means
you get working presets without running the calibrate stage at all.

## Requirements

- **JV-880 ROM images**: `jv880_rom1.bin`, `jv880_rom2.bin`,
  `jv880_waverom1.bin`, `jv880_waverom2.bin`, `jv880_nvram.bin`, and optionally
  SR-JV80 expansion ROMs in an `expansions/` subdirectory.
- cmake, a C++17 compiler, Python 3 with `numpy scipy soundfile`.

The emulator core ([schwung-jv880](https://github.com/charlesvestal/schwung-jv880),
MAME licence, inherited from Nuked-SC55 and juce-jv880) is a submodule:

```bash
git clone --recursive https://github.com/charlesvestal/jv880-multisample.git
# or, in an existing clone:
git submodule update --init
```

Already have a checkout? `cmake -S . -B build -DJV_DSP=/path/to/src/dsp`.

## What it does

The JV is a PCM synth with a global effects section. Sampling it naively
captures the effects baked into every note, which cannot then be turned off.
So the renderer strips reverb, chorus and the LFOs, samples the patch dry, and
rebuilds the effects in DecentSampler from measurements of the hardware.

| stage | what happens |
|---|---|
| calibrate | Renders test signals to measure reverb impulse responses, chorus depth/rate, delay time and feedback, and portamento time. A synthetic impulse is injected directly into a copy of the wave ROM, so the IRs carry no source resonance. |
| render | Every patch, 25 keys × up to 5 velocity layers, split at each patch's real tone switch points. Dry. |
| phrases | Patches whose names carry a tempo (`125:BtMenu 1`) are re-rendered with a hold long enough for two full bars. |
| postprocess | 64 → 48 kHz, loop-point search scored over the crossfade region DecentSampler actually blends, release measurement, FLAC. |
| emit | `.dspreset` and `.sfz`. Reverb is a convolution on a parallel send bus, matching the JV's architecture; chorus and delay are inserts; monophonic patches get `polyphony="1"` and measured glide. |
| validate | Every zone, key range, velocity split and effect reference. |
| package | One `.dslibrary` per board. |

## Fidelity

Effects are measured rather than guessed, and the measurements are in
`calib/calibration.json`. Where a value could not be measured it is documented
as an assumption in the code rather than presented as fact.

`tools/audit_fidelity.py` compares emitted presets against the plugin itself
on separate axes — output level, wet/dry ratio, reverb pre-delay, decay shape,
spectral tilt — and fails per-metric. Aggregate scores hide exactly the
failures that matter: an 18 dB-quiet reverb once scored 0.95 on decay
correlation.

Known gaps are listed in `docs/DEFERRED.md`.

## Licence

The tooling is yours to use. The ROMs are not distributed here and are not
mine to license; the sounds they contain are Roland's, and Roland currently
sells them as software instruments.
