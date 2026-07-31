#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "jv_rom.h"
#include "jv_patch.h"

namespace jv {

static const int SAMPLE_RATE   = 64000;
// 100,000 warmup updateSC55(1) calls, matching the schwung-jv880 reference
// harness exactly. Do not "fix" this the way run_frames() fixes the render
// loops: each updateSC55(1) call empirically produces exactly 2 real
// frames (never drained, so the frame/sample-count bug never applied
// here), so this already yields 200,000 frames = 3.125s -- matching the
// reference harness's own comment ("~3 s worth of emulator ticks at
// 64 kHz") almost exactly. See jv_render.cpp's WARMUP_STEPS call site and
// run_frames() for the full measurement.
static const int WARMUP_STEPS  = 100000;
static const int NVRAM_PATCH_OFFSET = 0x0d70;
static const int NVRAM_MODE_OFFSET  = 0x11;

// Velocity is deliberately NOT part of GridSpec: lokey/hikey/key_step/
// hold/tail/settle/silence_db apply uniformly to every patch, but velocity
// layering is now patch-specific (see jv_patch.h's compute_velocity_regions)
// -- each patch's own tone velocity-switch points determine its layer
// count and placement, so it's computed fresh per patch by the caller
// (jv_sampler.cpp) rather than fixed here.
struct GridSpec {
    int lokey = 24, hikey = 96, key_step = 3;   // C1..C7 every 3 semitones
    double hold_seconds   = 3.5;
    double tail_seconds   = 2.5;
    double settle_seconds = 0.5;
    double silence_db     = -72.0;
};

// Drives the JV-880 emulator headlessly and deterministically: no threads,
// no wall-clock reads, no RNG. mcu_ is an opaque MCU* (declared in mcu.h,
// which lives outside this repo under JV_DSP) so this header never needs
// that include path — only jv_render.cpp does.
//
// Call-order contract: init() must be called first and must return true
// before load_patch_bytes() or render_note() are called — both dereference
// the MCU that init() allocates, so calling either one first (or after a
// failed init()) is undefined behavior (a null-pointer dereference in this
// implementation; debug builds assert on it). init() may be called again on
// an already-initialized Renderer — e.g. to reset emulator state — without
// leaking the previous MCU (several MB): it is freed before the new one
// replaces it.
class Renderer {
public:
    Renderer() = default;
    ~Renderer();
    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    // Boots the emulator (via MCU::startSC55) and runs the warmup once.
    // Returns false, and leaves any previously-initialized MCU untouched, if
    // startSC55 reports failure — callers must check the return value before
    // calling load_patch_bytes()/render_note().
    bool init(const RomSet &roms);

    // Loads the 362 preprocessed patch bytes into NVRAM, sends the Program
    // Change that makes the firmware reload from NVRAM, then settles once
    // (design note A: the settle belongs here, not in render_note, so it
    // runs once per patch rather than once per rendered cell).
    // Requires a prior successful init().
    void load_patch_bytes(const std::vector<uint8_t> &patch_bytes,
                          const GridSpec &g);

    // Boots the emulator with a RHYTHM SET (drum kit) in place, replacing any
    // prior init(). Rhythm sets cannot be loaded the way patches are: the
    // firmware sources the sounding kit from ROM, not from NVRAM, so a kit is
    // installed by INJECTING it into an in-memory copy of rom2 and rebooting.
    // (The ROM files on disk are never written -- same technique wave_inject
    // uses for the wave ROM.)
    //
    // Established by intervention rather than assumption: writing a kit into
    // NVRAM at 0x67f0 -- where the factory image demonstrably keeps a copy --
    // changed the output by nothing at all, identical to two decimal places.
    // Injecting a kit whose 61 keys all hold ONE tone collapsed every key to
    // identical audio only when written to the PRESET A rhythm region, which
    // is why that region is the injection target no matter which kit is
    // wanted.
    //
    // `kit` must point to RHYTHM_SET_BYTES readable bytes. Because installing
    // a kit means rebooting, this costs a full warmup (~3 s) per kit.
    // Returns false if the emulator failed to start, leaving any previous
    // MCU untouched.
    // exp_waves/exp_len install an expansion board's wave data as part of
    // the SAME boot. Do not call load_expansion_waves() after this instead:
    // that routine resets the emulator, which would discard the performance
    // selection init_rhythm() makes and leave the rhythm part unpointed.
    bool init_rhythm(const RomSet &roms, const uint8_t *kit, const GridSpec &g,
                     const uint8_t *exp_waves = nullptr, size_t exp_len = 0);

    // Renders one key/velocity cell: note-on, hold, note-off, tail (with
    // early truncation once the signal sits quietly below the noise floor
    // for a sustained run), then All Notes Off plus a short discarded flush
    // so this cell's decay never bleeds into the next one (design note B).
    // Requires a prior successful init().
    // Load an expansion board's WAVE data into the PCM chip.
    //
    // MUST be called before rendering any expansion patch. An expansion patch
    // addresses its waves through PCM banks 3-6, which map to the emulator's
    // waverom_exp -- a region startSC55() does not populate. Without this the
    // wave numbers still resolve, but against the INTERNAL wave ROM, so every
    // expansion patch plays the wrong sound entirely: strings come out as
    // acoustic piano. The whole first render of the 20 expansion boards was
    // wrong this way, and it is invisible to structural validation because
    // each patch still produces distinct, plausible audio.
    void load_expansion_waves(const uint8_t *data, size_t len);

    // Clear the expansion wave area, for rendering internal patches after an
    // expansion (stale waves would otherwise persist in the PCM chip).
    void clear_expansion_waves();

    std::vector<int16_t> render_note(int key, int velocity, const GridSpec &g);

    // Two overlapping notes, for measuring portamento. The JV only glides
    // BETWEEN notes, so a single-note render cannot show it at all: the
    // second note-on has to arrive while the first is still held, which is
    // also what triggers glide in Solo mode and in Legato portamento mode.
    // Both notes are released at the end. Renders a fixed length (no
    // silence-truncation) so the glide is never clipped by the tail
    // heuristic.
    std::vector<int16_t> render_glide(int key_from, int key_to, int velocity,
                                      double first_hold_s, double total_s);

private:
    void *mcu_ = nullptr;   // opaque MCU*
    // MIDI channel every note is sent on. Patches play on channel 1 (0);
    // a rhythm set is reached through the performance's rhythm part, which
    // is channel 10 (9). Set by load_patch_bytes / init_rhythm so
    // render_note() needs no separate rhythm variant.
    int   channel_ = 0;
    // Trimmed 12-char ROM name of the currently-loaded patch (cheap to pull
    // out of patch_bytes in load_patch_bytes), used only to identify which
    // patch a render_note() flush-cap diagnostic came from.
    std::string current_patch_name_;
};

} // namespace jv
