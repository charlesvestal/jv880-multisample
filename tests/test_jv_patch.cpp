#include "jv_patch.h"
#include <stdio.h>
#include <string.h>
#include <string>

static int failures = 0;
static void check(bool c, const char *what) {
    if (!c) { fprintf(stderr, "FAIL: %s\n", what); failures++; }
    else    { fprintf(stderr, "ok: %s\n", what); }
}

int main(int argc, char **argv) {
    const std::string roms_dir = (argc > 1)
        ? argv[1]
        : "/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880";
    std::string err;
    jv::RomSet roms;
    if (!roms.load(roms_dir, &err)) {
        fprintf(stderr, "FAIL: ROMs load (%s)\n", err.c_str());
        return 1;
    }
    check(true, "ROMs load");
    auto patches = jv::enumerate_internal(roms);
    check(patches.size() == 192, "192 internal patches available");

    uint8_t p[jv::PATCH_SIZE];
    memset(p, 0, sizeof(p));
    p[13] = 100;                      // reverblevel
    p[16] = 64;                       // choruslevel (bit 7 = chorusoutput = 0)
    p[14] = 70;                       // reverbtime
    p[12] = (uint8_t)(4 | (2 << 4));  // reverbtype=4 (Hall1), chorustype=2
    p[15] = 33;                       // reverbfeedback
    p[17] = 55;                       // chorusdepth
    p[18] = 47;                       // chorusrate
    p[19] = 77;                       // chorusfeedback
    p[23] = 45;                       // bendrangedown
    p[24] = (uint8_t)(9 | (1 << 6));  // bendrangeup=9 (bits0-3), portamentoswitch on (bit6)

    // Per-tone Effects fields (level/reverbsend/chorussend): distinct
    // sentinel per tone AND per field, so an offset swap between the two
    // adjacent single-byte sends (82/83) -- or between either send and
    // level (67) -- would be caught by a mismatched value, not masked by
    // every field happening to share one number.
    uint8_t *tone0 = p + jv::TONE_BASE + 0 * jv::TONE_STRIDE;
    uint8_t *tone1 = p + jv::TONE_BASE + 1 * jv::TONE_STRIDE;
    tone0[67] = 111; tone0[82] = 61; tone0[83] = 71;
    tone1[67] = 122; tone1[82] = 83; tone1[83] = 94;

    jv::Effects fx = jv::read_effects(p);
    check(fx.reverb_type == 4,        "reverbtype decodes from bits 0-3");
    check(fx.chorus_type == 2,        "chorustype decodes from bits 4-5");
    check(fx.reverb_level == 100,     "reverblevel reads offset 13");
    check(fx.reverb_time == 70,       "reverbtime reads offset 14");
    check(fx.reverb_feedback == 33,   "reverbfeedback reads offset 15");
    check(fx.chorus_level == 64,      "choruslevel masks off bit 7");
    check(fx.chorus_depth == 55,      "chorusdepth reads offset 17");
    check(fx.chorus_rate == 47,       "chorusrate reads offset 18");
    check(fx.chorus_feedback == 77,   "chorusfeedback reads offset 19");
    check(fx.bend_down == 45,         "bendrangedown reads offset 23");
    check(fx.bend_up == 9,            "bendrangeup reads bits 0-3 of offset 24");
    check(fx.portamento == 1,         "portamentoswitch reads bit 6 of offset 24");
    check(fx.tone_level[0] == 111 && fx.tone_level[1] == 122,
          "tone_level[] reads offset 67 per tone");
    check(fx.reverb_send[0] == 61 && fx.reverb_send[1] == 83,
          "reverb_send[] reads offset 82 per tone");
    check(fx.chorus_send[0] == 71 && fx.chorus_send[1] == 94,
          "chorus_send[] reads offset 83 per tone");

    for (int t = 0; t < 2; t++) {
        uint8_t *tone = p + jv::TONE_BASE + t * jv::TONE_STRIDE;
        tone[67] = 100;                    // level -> active
        tone[23] = 1;                      // form = SIN
        tone[24] = 60;                     // rate
        tone[33] = (uint8_t)(int8_t)20;    // lfo1tvadepth
    }
    jv::LfoDecision d1 = jv::decide_lfo_strip(p, 1);
    check(d1.strip, "identical LFO1 across active tones is strippable");
    check(d1.lfo.tva_depth == 20, "representative depth preserved");

    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 24] = 90;
    check(!jv::decide_lfo_strip(p, 1).strip, "diverging rate blocks strip");
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 24] = 60;

    p[jv::TONE_BASE + 0 * jv::TONE_STRIDE + 23] = 4;
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 23] = 4;
    check(!jv::decide_lfo_strip(p, 1).strip, "RND1 blocks strip");
    p[jv::TONE_BASE + 0 * jv::TONE_STRIDE + 23] = 1;
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 23] = 1;

    uint8_t *t3 = p + jv::TONE_BASE + 3 * jv::TONE_STRIDE;
    t3[67] = 0;
    t3[24] = 5;
    check(!jv::tone_active(p, 3), "level 0 tone is inactive");
    check(jv::decide_lfo_strip(p, 1).strip, "inactive tone ignored");

    // Three active tones with LFO1 rates 60 (first), 56, 64: a pairwise
    // min/max spread of 8, which must be rejected against a tolerance of
    // 4. A "star" comparison (everything vs tone 0 only) would wrongly
    // accept this, since |56-60|<=4 and |64-60|<=4 individually, even
    // though 56 and 64 are 8 apart.
    uint8_t *t2 = p + jv::TONE_BASE + 2 * jv::TONE_STRIDE;
    t2[67] = 100; t2[23] = 1; t2[33] = (uint8_t)(int8_t)20;
    p[jv::TONE_BASE + 0 * jv::TONE_STRIDE + 24] = 60;
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 24] = 56;
    t2[24] = 64;
    check(!jv::decide_lfo_strip(p, 1).strip,
          "star-topology: 56 and 64 are 8 apart, must block even though both are within 4 of 60");
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 24] = 60;
    t2[67] = 0;   // deactivate tone 2 again for the tests below

    // Signed depth handling: negative depths must round-trip.
    uint8_t *t0 = p + jv::TONE_BASE + 0 * jv::TONE_STRIDE;
    uint8_t *t1 = p + jv::TONE_BASE + 1 * jv::TONE_STRIDE;
    t0[33] = (uint8_t)(int8_t)-30;
    t1[33] = (uint8_t)(int8_t)-30;
    check(jv::read_tone_lfo(p, 0, 1).tva_depth == -30, "negative depth reads as signed");
    check(jv::decide_lfo_strip(p, 1).strip, "matching negative depths strippable");
    t1[33] = (uint8_t)(int8_t)30;
    check(!jv::decide_lfo_strip(p, 1).strip, "opposite-sign depths block strip");

    // Zero-sign-neutrality edge cases: a depth of exactly 0 must not read
    // as "positive" (the old `a >= 0` comparison did exactly that, so
    // (0,+5) passed but (0,-5) -- the same physical situation -- failed).
    t0[33] = (uint8_t)(int8_t)0;
    t1[33] = (uint8_t)(int8_t)5;
    check(jv::decide_lfo_strip(p, 1).strip, "(0, +5) depths strippable");
    t1[33] = (uint8_t)(int8_t)-5;
    check(jv::decide_lfo_strip(p, 1).strip,
          "(0, -5) depths strippable (was wrongly blocked before the zero-sign fix)");
    t0[33] = (uint8_t)(int8_t)-5;
    t1[33] = (uint8_t)(int8_t)5;
    check(!jv::decide_lfo_strip(p, 1).strip,
          "(-5, +5) depths blocked: spread of 10 exceeds the tolerance of 6");

    // The sign gate must do independent work, not merely be implied by
    // the spread check: -2 and +3 have a spread of 5, well inside the
    // +/-6 tolerance, so only the sign gate can block this pair. Without
    // this case the sign gate could be silently deleted and every other
    // test above would still pass (the earlier (-5,+5) case fails on
    // spread alone and proves nothing about sign specifically).
    t0[33] = (uint8_t)(int8_t)-2;
    t1[33] = (uint8_t)(int8_t)3;
    check(!jv::decide_lfo_strip(p, 1).strip,
          "(-2, +3) depths blocked by sign alone: spread of 5 is within the +/-6 tolerance");

    // Same idea right at the tolerance boundary: -3 and +3 have a spread
    // of exactly 6 (still within tolerance) but opposite sign.
    t0[33] = (uint8_t)(int8_t)-3;
    t1[33] = (uint8_t)(int8_t)3;
    check(!jv::decide_lfo_strip(p, 1).strip,
          "(-3, +3) depths blocked by sign alone: spread of 6 is at the tolerance boundary");

    t0[33] = (uint8_t)(int8_t)20;
    t1[33] = (uint8_t)(int8_t)20;

    jv::LfoDecision d2 = jv::decide_lfo_strip(p, 2);
    auto out = jv::preprocess(p, jv::decide_lfo_strip(p, 1), d2);
    check(out.size() == (size_t)jv::PATCH_SIZE, "preprocess returns full patch");
    check(out[13] == 0, "reverblevel zeroed");
    check((out[16] & 0x7f) == 0, "choruslevel zeroed");
    check((out[24] & (1 << 6)) == 0, "portamento cleared");
    check((out[12] & 0x0f) == 4, "reverbtype preserved for metadata");
    check((out[24] & 0x0f) == (p[24] & 0x0f), "bendrangeup bits untouched");
    for (int t = 0; t < 2; t++)
        check(out[jv::TONE_BASE + t * jv::TONE_STRIDE + 33] == 0,
              "lfo1 tva depth stripped");

    // Unstripped LFO2 depths must survive preprocess untouched.
    uint8_t q[jv::PATCH_SIZE];
    memcpy(q, p, sizeof(q));
    q[jv::TONE_BASE + 36] = (uint8_t)(int8_t)25;   // lfo2tvadepth, tone 0
    jv::LfoDecision none;
    auto out2 = jv::preprocess(q, none, none);
    check(out2[jv::TONE_BASE + 33] == q[jv::TONE_BASE + 33],
          "lfo1 depth kept when not stripped");
    check(out2[jv::TONE_BASE + 36] == (uint8_t)(int8_t)25,
          "lfo2 depth kept when not stripped");

    // Out-of-range enum values must return "Unknown", not a
    // plausible-looking default -- reverbtype is a 4-bit field (0-15) but
    // only 0-7 are named; chorustype is 2-bit (0-3) but only 0-2 are
    // named; lfoform is 3-bit (0-7) but only 0-5 are named.
    static const char *reverb_names[] = {"Room1","Room2","Stage1","Stage2",
                                         "Hall1","Hall2","Delay","Pan-Dly"};
    for (int i = 0; i < 16; i++) {
        const char *expect = (i < 8) ? reverb_names[i] : "Unknown";
        check(strcmp(jv::reverb_type_name(i), expect) == 0,
              "reverb_type_name covers the full 4-bit range");
    }
    static const char *chorus_names[] = {"Chorus1","Chorus2","Chorus3"};
    for (int i = 0; i < 4; i++) {
        const char *expect = (i < 3) ? chorus_names[i] : "Unknown";
        check(strcmp(jv::chorus_type_name(i), expect) == 0,
              "chorus_type_name covers the full 2-bit range");
    }
    static const char *lfo_names[] = {"TRI","SIN","SAW","SQU","RND1","RND2"};
    for (int i = 0; i < 8; i++) {
        const char *expect = (i < 6) ? lfo_names[i] : "Unknown";
        check(strcmp(jv::lfo_form_name(i), expect) == 0,
              "lfo_form_name covers the full 3-bit range");
    }

    int strippable1 = 0, strippable2 = 0, rnd1 = 0;
    for (const auto &pr : patches) {
        jv::LfoDecision r1 = jv::decide_lfo_strip(pr.data, 1);
        jv::LfoDecision r2 = jv::decide_lfo_strip(pr.data, 2);
        if (r1.strip) strippable1++;
        if (r2.strip) strippable2++;
        if (r1.reason == "random waveform") rnd1++;
    }
    fprintf(stderr,
            "info: %d/192 internal patches strippable on LFO1, %d/192 on LFO2, %d blocked by RND (LFO1)\n",
            strippable1, strippable2, rnd1);
    // Pinned to the exact corrected values (verified against the
    // min/max-spread fix for the star-topology bug and the zero-neutral
    // sign fix; both fixes were confirmed correct via direct reproduction
    // of the reviewer's cases, but neither changed these counts on this
    // specific 192-patch ROM corpus -- no factory patch happens to hit
    // the star-topology or zero-sign edge cases, so old and new code give
    // IDENTICAL results here). This assertion pins the aggregate outcome
    // across all 192 real patches so any future tolerance/semantics
    // change must be consciously acknowledged and re-verified, rather
    // than silently drifting -- it must never degrade back into a
    // tautology like `strippable >= 0 && strippable <= 192`. It is NOT a
    // star-topology regression guard: since old and new code agree on
    // this corpus, this assertion cannot by itself detect that bug
    // reappearing. The synthetic 56/64-vs-60 test above (search
    // "star-topology") is what actually guards that behavior.
    check(strippable1 == 6, "LFO1 strippable count pinned to corrected tolerance semantics (6/192)");
    check(strippable2 == 1, "LFO2 strippable count pinned to corrected tolerance semantics (1/192)");

    // preprocess must be safe on every real patch and never change size.
    bool preprocess_ok = true;
    for (const auto &pr : patches) {
        auto o = jv::preprocess(pr.data, jv::decide_lfo_strip(pr.data, 1),
                                         jv::decide_lfo_strip(pr.data, 2));
        if (o.size() != (size_t)jv::PATCH_SIZE) { preprocess_ok = false; break; }
        if ((o[13] & 0x7f) != 0 || (o[16] & 0x7f) != 0) { preprocess_ok = false; break; }
    }
    check(preprocess_ok, "preprocess dry on all 192 ROM patches");

    fprintf(stderr, failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
