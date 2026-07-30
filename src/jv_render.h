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

    // Renders one key/velocity cell: note-on, hold, note-off, tail (with
    // early truncation once the signal sits quietly below the noise floor
    // for a sustained run), then All Notes Off plus a short discarded flush
    // so this cell's decay never bleeds into the next one (design note B).
    // Requires a prior successful init().
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
    // Trimmed 12-char ROM name of the currently-loaded patch (cheap to pull
    // out of patch_bytes in load_patch_bytes), used only to identify which
    // patch a render_note() flush-cap diagnostic came from.
    std::string current_patch_name_;
};

} // namespace jv
