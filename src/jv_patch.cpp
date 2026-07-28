#include "jv_patch.h"
#include <assert.h>

namespace jv {

static int bits(const uint8_t *p, int off, int shift, int width) {
    return (p[off] >> shift) & ((1 << width) - 1);
}

static const uint8_t *tone_ptr(const uint8_t *patch, int tone) {
    // tone must be in [0, TONE_COUNT). Tasks 3/4/6 call tone_active() and
    // read_tone_lfo() with their own loop indices; an out-of-range tone
    // would otherwise silently read into the START OF A NEIGHBORING PATCH
    // (or before this one) — no crash, no signal, just wrong data. assert()
    // catches the mistake immediately in debug builds; the clamp keeps
    // release builds (this project defaults CMAKE_BUILD_TYPE to Release,
    // which compiles asserts out) inside this patch's own bytes instead of
    // reading someone else's.
    assert(tone >= 0 && tone < TONE_COUNT);
    if (tone < 0) tone = 0;
    else if (tone >= TONE_COUNT) tone = TONE_COUNT - 1;
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

// True when every value in `vals` falls within a window of `tol` of every
// OTHER value (max - min <= tol) — i.e. compared pairwise against each
// other, not just against vals[0]. A "star" comparison (everything vs a
// single reference) lets two non-reference values drift up to 2x the
// stated tolerance apart while each still individually reads as "within
// tolerance of the reference".
//
// When check_sign is true, also require that not both a strictly positive
// and a strictly negative value are present. A value of exactly 0 sets
// neither has_pos nor has_neg, so a 0 depth never conflicts with a
// same-tolerance-window nonzero depth of either sign — a 0 measurement
// imposes no direction, so it shouldn't have a direction to be wrong
// about (this is the bug: comparing "a >= 0" made 0 read as "positive").
static bool spread_ok(const std::vector<int> &vals, int tol, bool check_sign) {
    int mn = vals[0], mx = vals[0];
    bool has_pos = false, has_neg = false;
    for (int v : vals) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (v > 0) has_pos = true;
        if (v < 0) has_neg = true;
    }
    if (mx - mn > tol) return false;
    if (check_sign && has_pos && has_neg) return false;
    return true;
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
    // Waveform and sync require exact equality, so comparing each tone
    // against the first is fine here: equality is transitive (if every
    // tone equals ref, every tone equals every other tone), unlike the
    // numeric tolerance checks below.
    for (const auto &l : active) {
        if (l.form != ref.form) { d.reason = "waveform mismatch"; return d; }
        if (l.sync != ref.sync) { d.reason = "sync mismatch";     return d; }
    }

    std::vector<int> rates, pitch, tvf, tva;
    for (const auto &l : active) {
        rates.push_back(l.rate);
        pitch.push_back(l.pitch_depth);
        tvf.push_back(l.tvf_depth);
        tva.push_back(l.tva_depth);
    }
    if (!spread_ok(rates, 4, /*check_sign=*/false)) { d.reason = "rate mismatch";        return d; }
    if (!spread_ok(pitch, 6, /*check_sign=*/true))  { d.reason = "pitch depth mismatch"; return d; }
    if (!spread_ok(tvf,   6, /*check_sign=*/true))  { d.reason = "tvf depth mismatch";   return d; }
    if (!spread_ok(tva,   6, /*check_sign=*/true))  { d.reason = "tva depth mismatch";   return d; }

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

// reverbtype is a 4-bit field (0-15) but only 0-7 are named Roland reverb
// types; chorustype is 2-bit (0-3) but only 0-2 are named; lfoform is
// 3-bit (0-7) but only 0-5 are named forms. The remaining values are
// reserved/undocumented on real hardware and should never silently read
// back as a plausible-looking name (that would render into Task 6's
// preset metadata as if it were a deliberate, meaningful value).
const char *reverb_type_name(int t) {
    static const char *n[] = {"Room1","Room2","Stage1","Stage2",
                              "Hall1","Hall2","Delay","Pan-Dly"};
    return (t >= 0 && t < 8) ? n[t] : "Unknown";
}
const char *chorus_type_name(int t) {
    static const char *n[] = {"Chorus1","Chorus2","Chorus3"};
    return (t >= 0 && t < 3) ? n[t] : "Unknown";
}
const char *lfo_form_name(int f) {
    static const char *n[] = {"TRI","SIN","SAW","SQU","RND1","RND2"};
    return (f >= 0 && f < 6) ? n[f] : "Unknown";
}

} // namespace jv
