#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "jv_rom.h"

namespace jv {

static const int TONE_COUNT  = 4;
static const int TONE_STRIDE = 84;
static const int TONE_BASE   = 26;   // offset of tone 0 within the patch

struct Effects {
    int reverb_type = 0, reverb_level = 0, reverb_time = 0, reverb_feedback = 0;
    int chorus_type = 0, chorus_level = 0, chorus_depth = 0, chorus_rate = 0;
    int chorus_feedback = 0, chorus_output = 0;  // output: 0 = Mix, 1 = Reverb
    int bend_up = 0, bend_down = 0;
    int portamento = 0;
    // Voice mode. The JV plays a patch either polyphonically or as a solo
    // (monophonic) voice, optionally gliding between notes. Sampling renders
    // every note in isolation, which is correct either way -- monophony only
    // matters when notes overlap -- so these are carried through to the
    // preset rather than baked into the audio.
    //   key_assign      0 = Poly, 1 = Solo
    //   solo_legato     glide only when the previous note is still held
    //   portamento_time 0-127, mapped to DecentSampler's glideTime in seconds
    int key_assign = 0, solo_legato = 0;
    int portamento_mode = 0, portamento_time = 0, portamento_type = 0;
    int reverb_send[TONE_COUNT] = {0, 0, 0, 0};
    int chorus_send[TONE_COUNT] = {0, 0, 0, 0};
    int tone_level[TONE_COUNT]  = {0, 0, 0, 0};
};

struct ToneLfo {
    int form = 0, rate = 0, delay = 0, fade = 0, sync = 0;
    int pitch_depth = 0, tvf_depth = 0, tva_depth = 0;   // signed -63..63
    bool any_depth() const {
        return pitch_depth != 0 || tvf_depth != 0 || tva_depth != 0;
    }
};

struct LfoDecision {
    bool strip = false;
    std::string reason;
    ToneLfo lfo;   // representative (mean of active tones) when strip is true
};

// Raw tone velocity-range bytes (offsets 3/4 within the 84-byte tone),
// exactly as stored -- not clamped or validated. Real hardware constrains
// these to 1-127 with lo <= hi, but nothing guarantees a ROM byte can't
// violate that; callers that need a safe interval (compute_velocity_regions)
// clamp it themselves rather than this raw accessor silently doing so.
struct ToneVelocityRange {
    int lo = 1, hi = 127;
};

// One contiguous MIDI velocity band, inclusive on both ends. A vector of
// these returned by compute_velocity_regions() always tiles 1..127
// exactly: the first region's lo is 1, the last region's hi is 127, and
// consecutive regions abut with no gap or overlap.
struct VelocityRegion {
    int lo = 1, hi = 127;
};

// Every `patch` pointer below must point to at least PATCH_SIZE (see
// jv_rom.h) readable bytes — a full 362-byte patch image, whether taken
// from PatchRef::data or a synthetic test buffer. None of these functions
// validate the buffer length themselves; the caller owns that guarantee
// (matching the pointer contract documented on jv_rom.h's PatchRef).
Effects     read_effects(const uint8_t *patch);
ToneLfo     read_tone_lfo(const uint8_t *patch, int tone, int lfo_index /*1 or 2*/);
bool        tone_active(const uint8_t *patch, int tone);
LfoDecision decide_lfo_strip(const uint8_t *patch, int lfo_index);

// Raw velocityrangelower/velocityrangeupper (tone bytes 3/4) for the given
// tone, regardless of whether that tone is active. See ToneVelocityRange's
// contract note above: unclamped.
ToneVelocityRange read_tone_velocity_range(const uint8_t *patch, int tone);

// Partitions MIDI velocity 1..127 into contiguous regions in which the SET
// of active (level > 0) tones sounding is constant, derived from every
// active tone's own [velocityrangelower, velocityrangeupper]. The raw
// switch-derived count is then adjusted to land in [min_layers,
// max_layers]:
//   - Too few (a patch with no velocity switching yields exactly one raw
//     region): the shortfall is apportioned across the existing regions
//     proportional to width, each gaining that many even internal
//     sub-splits -- a single whole-keyboard region reproduces today's
//     exact fixed-thirds default; a genuinely-switching patch keeps its
//     real boundaries intact and only subdivides further within whichever
//     side is widest. A real switch boundary is never discarded to make
//     room for the minimum.
//   - Too many: the narrowest ADJACENT PAIR (by combined width) is merged,
//     repeated until the count is within max_layers.
// Always returns a valid tiling of 1..127 (see VelocityRegion). See
// jv_patch.cpp for the full algorithm.
std::vector<VelocityRegion> compute_velocity_regions(const uint8_t *patch,
                                                      int min_layers = 3,
                                                      int max_layers = 5);

// Render-ready bytes: dry, portamento off, LFOs stripped per decision.
// `patch` must point to at least PATCH_SIZE readable bytes (see above).
std::vector<uint8_t> preprocess(const uint8_t *patch,
                                const LfoDecision &lfo1,
                                const LfoDecision &lfo2);

const char *reverb_type_name(int t);
const char *chorus_type_name(int t);
const char *lfo_form_name(int f);

} // namespace jv
