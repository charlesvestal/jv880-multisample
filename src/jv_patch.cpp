#include "jv_patch.h"
#include <algorithm>
#include <assert.h>
#include <cmath>

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
    // reverbtype is THREE bits, not four.  The plugin's parameter table lists
    // width 4 but max 7, which is self-contradictory -- 0-7 needs only 3 bits.
    // Reading 4 bits picks up an adjacent unrelated flag (set on 40 of the 192
    // internal patches) and yields invalid types 8-15: A.Piano 1 decoded as 12
    // rather than Hall1, Clav 1 as 8 rather than Room1.
    //
    // Ground truth is the reference JUCE implementation, which reads
    //   reverbTypeComboBox.setSelectedItemIndex(patch->revChorConfig & 0x7)
    // (jv880_juce Source/ui/EditCommonTab.cpp:174).  Masking 3 bits yields an
    // all-valid, sensibly distributed set across every patch.
    e.reverb_type     = bits(p, 12, 0, 3);
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
    // Patch-common voice bits, offsets confirmed against the reference
    // implementation's own PATCH_COMMON_PARAMS table.
    e.key_assign      = bits(p, 24, 7, 1);   // 0 = Poly, 1 = Solo
    e.solo_legato     = bits(p, 24, 5, 1);
    e.portamento_mode = bits(p, 24, 4, 1);
    e.portamento_time = bits(p, 25, 0, 7);
    e.portamento_type = bits(p, 25, 7, 1);
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

ToneVelocityRange read_tone_velocity_range(const uint8_t *patch, int tone) {
    const uint8_t *tp = tone_ptr(patch, tone);
    ToneVelocityRange r;
    r.lo = tp[3];   // velocityrangelower
    r.hi = tp[4];   // velocityrangeupper
    return r;
}

namespace {

// Clamp+order a raw tone velocity range into a valid [1,127] interval with
// lo <= hi. Real hardware constrains velocityrangelower/upper to exactly
// that, but nothing stops a malformed byte from violating it, and
// compute_velocity_regions must never build an out-of-domain or inverted
// breakpoint from one.
VelocityRegion clamp_tone_range(ToneVelocityRange r) {
    int lo = r.lo, hi = r.hi;
    if (lo < 1) lo = 1;
    if (lo > 127) lo = 127;
    if (hi < 1) hi = 1;
    if (hi > 127) hi = 127;
    if (lo > hi) std::swap(lo, hi);
    return {lo, hi};
}

// Splits [r.lo, r.hi] into `count` contiguous, evenly-sized sub-bands using
// the same even-division formula the fixed-thirds default already used
// (1 + round(i * span / n)), so subdividing a single no-switching region
// into min_layers pieces reproduces the exact 1-42/43-85/86-127 split
// rather than a visibly different scheme.
std::vector<VelocityRegion> split_even(VelocityRegion r, int count) {
    if (count <= 1) return {r};
    int span = r.hi - r.lo + 1;
    std::vector<int> bounds(count + 1);
    for (int i = 0; i < count; i++)
        bounds[i] = r.lo + (int)std::lround(i * (double)span / count);
    bounds[count] = r.hi + 1;   // force the exact end, no rounding drift
    std::vector<VelocityRegion> out;
    out.reserve(count);
    for (int i = 0; i < count; i++)
        out.push_back({bounds[i], bounds[i + 1] - 1});
    return out;
}

} // namespace

std::vector<VelocityRegion> compute_velocity_regions(const uint8_t *patch,
                                                      int min_layers,
                                                      int max_layers) {
    // Defensive clamp mirroring tone_ptr's assert+clamp pattern: the
    // precondition is a caller bug (every call site in this codebase uses
    // sane literals), but Release builds compile asserts out, so a
    // release build must still behave sanely rather than produce garbage
    // (e.g. an inverted-range apportionment loop) if it's ever violated.
    assert(min_layers >= 1 && min_layers <= max_layers);
    if (max_layers < 1) max_layers = 1;
    if (min_layers < 1) min_layers = 1;
    if (min_layers > max_layers) min_layers = max_layers;

    // Step 1: raw regions where the SET of active tones is constant.
    // Every active tone contributes two breakpoints (its lo, and one past
    // its hi); 1 and 128 are always present as sentinels, so this tiles
    // 1..127 exactly even with zero active tones (a silent/degenerate
    // patch still needs a valid region to render *something* into).
    std::vector<int> breaks = {1, 128};
    for (int t = 0; t < TONE_COUNT; t++) {
        if (!tone_active(patch, t)) continue;
        VelocityRegion r = clamp_tone_range(read_tone_velocity_range(patch, t));
        breaks.push_back(r.lo);
        breaks.push_back(r.hi + 1);
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());

    std::vector<VelocityRegion> regions;
    regions.reserve(breaks.size());
    for (size_t i = 0; i + 1 < breaks.size(); i++)
        regions.push_back({breaks[i], breaks[i + 1] - 1});

    // Step 2: cap at max_layers. Repeatedly merge whichever ADJACENT PAIR
    // has the smallest combined width -- that pair loses the least
    // switch-point resolution of any available merge, and it's the pair
    // (not a lone region merged into a neighbor) that's "narrowest": the
    // measured corpus tops out at 4 raw regions per patch, so in practice
    // this is a safety net against a patch this codebase hasn't seen
    // rather than something that fires on the known data.
    while ((int)regions.size() > max_layers) {
        size_t best = 0;
        int best_width = regions[1].hi - regions[0].lo + 1;
        for (size_t i = 1; i + 1 < regions.size(); i++) {
            int width = regions[i + 1].hi - regions[i].lo + 1;
            if (width < best_width) { best_width = width; best = i; }
        }
        regions[best].hi = regions[best + 1].hi;
        regions.erase(regions.begin() + best + 1);
    }

    // Step 3: ensure at least min_layers. Velocity still modulates
    // level/filter cutoff continuously within a single tone's range, so
    // even a patch with a single switch-derived region needs more than
    // one sampled layer. The shortfall is apportioned across the EXISTING
    // regions proportional to width (greedy largest-quotient: repeatedly
    // give the next split to whichever region currently has the largest
    // width-per-sub-layer share), so a genuine switch boundary is never
    // discarded to make room -- each region only ever gains additional
    // even sub-splits inside itself. For the common case of one raw
    // region spanning the whole keyboard, every slot goes to that one
    // region, which is exactly today's fixed-thirds split (see
    // split_even's comment).
    int n = (int)regions.size();
    if (n < min_layers) {
        std::vector<int> sub_count(regions.size(), 1);
        int extra = min_layers - n;
        for (int k = 0; k < extra; k++) {
            // A region can never usefully take more sub-layers than its
            // own width (each sub-piece needs at least 1 velocity value);
            // skip any region already at that ceiling so split_even()
            // below can never be asked for more pieces than a region has
            // room for. For any min_layers actually used in this codebase
            // (<=5) this never triggers -- 1..127 always tiles exactly,
            // so pigeonhole guarantees the widest of n<5 regions has
            // ample width -- but it makes the function safe for
            // arbitrary/adversarial min_layers too.
            size_t best = 0;
            double best_share = -1.0;
            bool found = false;
            for (size_t i = 0; i < regions.size(); i++) {
                int width = regions[i].hi - regions[i].lo + 1;
                if (sub_count[i] >= width) continue;
                double share = (double)width / sub_count[i];
                if (share > best_share) { best_share = share; best = i; found = true; }
            }
            if (!found) break;   // every region already down to 1-wide sub-pieces
            sub_count[best]++;
        }
        std::vector<VelocityRegion> out;
        for (size_t i = 0; i < regions.size(); i++) {
            auto pieces = split_even(regions[i], sub_count[i]);
            out.insert(out.end(), pieces.begin(), pieces.end());
        }
        regions = std::move(out);
    }

    return regions;
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
