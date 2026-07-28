#include "jv_patch.h"
#include <stdlib.h>
#include <string.h>

namespace jv {

static int bits(const uint8_t *p, int off, int shift, int width) {
    return (p[off] >> shift) & ((1 << width) - 1);
}

static const uint8_t *tone_ptr(const uint8_t *patch, int tone) {
    return patch + TONE_BASE + tone * TONE_STRIDE;
}

bool tone_active(const uint8_t *patch, int tone) {
    return tone_ptr(patch, tone)[67] > 0;   // tone `level`
}

Effects read_effects(const uint8_t *p) {
    Effects e;
    e.reverb_type     = bits(p, 12, 0, 4);
    e.chorus_type     = bits(p, 12, 4, 2);
    e.reverb_level    = bits(p, 13, 0, 7);
    e.reverb_time     = bits(p, 14, 0, 7);
    e.reverb_feedback = bits(p, 15, 0, 7);
    e.chorus_level    = bits(p, 16, 0, 7);
    e.chorus_output   = bits(p, 16, 7, 1);
    e.chorus_depth    = bits(p, 17, 0, 7);
    e.chorus_rate     = bits(p, 18, 0, 7);
    e.chorus_feedback = bits(p, 19, 0, 7);
    e.bend_down       = bits(p, 23, 0, 7);
    e.bend_up         = bits(p, 24, 0, 4);
    e.portamento      = bits(p, 24, 6, 1);
    for (int t = 0; t < TONE_COUNT; t++) {
        const uint8_t *tp = tone_ptr(p, t);
        e.tone_level[t]  = tp[67];
        e.reverb_send[t] = tp[82];
        e.chorus_send[t] = tp[83];
    }
    return e;
}

ToneLfo read_tone_lfo(const uint8_t *patch, int tone, int lfo_index) {
    const uint8_t *tp = tone_ptr(patch, tone);
    ToneLfo l;
    if (lfo_index == 1) {
        l.form  = (tp[23] >> 0) & 0x07;
        l.sync  = (tp[23] >> 6) & 0x01;
        l.rate  = tp[24];
        l.delay = tp[25];
        l.fade  = tp[26];
        l.pitch_depth = (int8_t)tp[31];
        l.tvf_depth   = (int8_t)tp[32];
        l.tva_depth   = (int8_t)tp[33];
    } else {
        l.form  = (tp[27] >> 0) & 0x07;
        l.sync  = 0;                    // LFO2 has no sync bit
        l.rate  = tp[28];
        l.delay = tp[29];
        l.fade  = tp[30];
        l.pitch_depth = (int8_t)tp[34];
        l.tvf_depth   = (int8_t)tp[35];
        l.tva_depth   = (int8_t)tp[36];
    }
    return l;
}

static bool same_sign(int a, int b) {
    if (a == 0 && b == 0) return true;
    return (a >= 0) == (b >= 0);
}

LfoDecision decide_lfo_strip(const uint8_t *patch, int lfo_index) {
    LfoDecision d;
    std::vector<ToneLfo> active;
    for (int t = 0; t < TONE_COUNT; t++)
        if (tone_active(patch, t)) active.push_back(read_tone_lfo(patch, t, lfo_index));

    if (active.empty()) { d.reason = "no active tones"; return d; }

    bool any = false;
    for (const auto &l : active) if (l.any_depth()) any = true;
    if (!any) { d.reason = "no LFO depth"; return d; }

    for (const auto &l : active)
        if (l.form >= 4) { d.reason = "random waveform"; return d; }  // RND1/RND2

    const ToneLfo &ref = active[0];
    for (const auto &l : active) {
        if (l.form != ref.form)         { d.reason = "waveform mismatch"; return d; }
        if (l.sync != ref.sync)         { d.reason = "sync mismatch";     return d; }
        if (abs(l.rate - ref.rate) > 4) { d.reason = "rate mismatch";     return d; }
        if (abs(l.pitch_depth - ref.pitch_depth) > 6 ||
            !same_sign(l.pitch_depth, ref.pitch_depth)) { d.reason = "pitch depth mismatch"; return d; }
        if (abs(l.tvf_depth - ref.tvf_depth) > 6 ||
            !same_sign(l.tvf_depth, ref.tvf_depth))     { d.reason = "tvf depth mismatch";   return d; }
        if (abs(l.tva_depth - ref.tva_depth) > 6 ||
            !same_sign(l.tva_depth, ref.tva_depth))     { d.reason = "tva depth mismatch";   return d; }
    }

    long rate = 0, del = 0, fade = 0, pd = 0, fd = 0, ad = 0;
    for (const auto &l : active) {
        rate += l.rate; del += l.delay; fade += l.fade;
        pd += l.pitch_depth; fd += l.tvf_depth; ad += l.tva_depth;
    }
    int n = (int)active.size();
    d.lfo = ref;
    d.lfo.rate  = (int)(rate / n);
    d.lfo.delay = (int)(del  / n);
    d.lfo.fade  = (int)(fade / n);
    d.lfo.pitch_depth = (int)(pd / n);
    d.lfo.tvf_depth   = (int)(fd / n);
    d.lfo.tva_depth   = (int)(ad / n);
    d.strip  = true;
    d.reason = "strippable";
    return d;
}

std::vector<uint8_t> preprocess(const uint8_t *patch,
                                const LfoDecision &lfo1,
                                const LfoDecision &lfo2) {
    std::vector<uint8_t> out(patch, patch + PATCH_SIZE);

    out[13] = (uint8_t)(out[13] & ~0x7f);           // reverblevel = 0
    out[16] = (uint8_t)(out[16] & ~0x7f);           // choruslevel = 0, keep bit 7
    out[24] = (uint8_t)(out[24] & ~(1 << 6));       // portamento off

    for (int t = 0; t < TONE_COUNT; t++) {
        uint8_t *tp = out.data() + TONE_BASE + t * TONE_STRIDE;
        if (lfo1.strip) { tp[31] = 0; tp[32] = 0; tp[33] = 0; }
        if (lfo2.strip) { tp[34] = 0; tp[35] = 0; tp[36] = 0; }
    }
    return out;
}

const char *reverb_type_name(int t) {
    static const char *n[] = {"Room1","Room2","Stage1","Stage2",
                              "Hall1","Hall2","Delay","Pan-Dly"};
    return (t >= 0 && t < 8) ? n[t] : "Room1";
}
const char *chorus_type_name(int t) {
    static const char *n[] = {"Chorus1","Chorus2","Chorus3"};
    return (t >= 0 && t < 3) ? n[t] : "Chorus1";
}
const char *lfo_form_name(int f) {
    static const char *n[] = {"TRI","SIN","SAW","SQU","RND1","RND2"};
    return (f >= 0 && f < 6) ? n[f] : "TRI";
}

} // namespace jv
