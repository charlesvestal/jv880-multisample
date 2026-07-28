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

Effects     read_effects(const uint8_t *patch);
ToneLfo     read_tone_lfo(const uint8_t *patch, int tone, int lfo_index /*1 or 2*/);
bool        tone_active(const uint8_t *patch, int tone);
LfoDecision decide_lfo_strip(const uint8_t *patch, int lfo_index);

// Render-ready bytes: dry, portamento off, LFOs stripped per decision.
std::vector<uint8_t> preprocess(const uint8_t *patch,
                                const LfoDecision &lfo1,
                                const LfoDecision &lfo2);

const char *reverb_type_name(int t);
const char *chorus_type_name(int t);
const char *lfo_form_name(int f);

} // namespace jv
