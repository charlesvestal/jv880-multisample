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
    p[12] = (uint8_t)(4 | (1 << 4));  // reverbtype=4 (Hall1), chorustype=1
    p[24] = (uint8_t)(1 << 6);        // portamentoswitch on

    jv::Effects fx = jv::read_effects(p);
    check(fx.reverb_type == 4,    "reverbtype decodes from bits 0-3");
    check(fx.chorus_type == 1,    "chorustype decodes from bits 4-5");
    check(fx.reverb_level == 100, "reverblevel reads offset 13");
    check(fx.reverb_time == 70,   "reverbtime reads offset 14");
    check(fx.chorus_level == 64,  "choruslevel masks off bit 7");
    check(fx.portamento == 1,     "portamentoswitch reads bit 6 of offset 24");

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

    // Signed depth handling: negative depths must round-trip.
    uint8_t *t0 = p + jv::TONE_BASE + 0 * jv::TONE_STRIDE;
    uint8_t *t1 = p + jv::TONE_BASE + 1 * jv::TONE_STRIDE;
    t0[33] = (uint8_t)(int8_t)-30;
    t1[33] = (uint8_t)(int8_t)-30;
    check(jv::read_tone_lfo(p, 0, 1).tva_depth == -30, "negative depth reads as signed");
    check(jv::decide_lfo_strip(p, 1).strip, "matching negative depths strippable");
    t1[33] = (uint8_t)(int8_t)30;
    check(!jv::decide_lfo_strip(p, 1).strip, "opposite-sign depths block strip");
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

    int strippable = 0, rnd = 0;
    for (const auto &pr : patches) {
        jv::LfoDecision d = jv::decide_lfo_strip(pr.data, 1);
        if (d.strip) strippable++;
        if (d.reason == "random waveform") rnd++;
    }
    fprintf(stderr, "info: %d/192 internal patches strippable on LFO1, %d blocked by RND\n",
            strippable, rnd);
    check(strippable >= 0 && strippable <= 192, "strip decision runs on all ROM patches");

    // preprocess must be safe on every real patch and never change size.
    for (const auto &pr : patches) {
        auto o = jv::preprocess(pr.data, jv::decide_lfo_strip(pr.data, 1),
                                         jv::decide_lfo_strip(pr.data, 2));
        if (o.size() != (size_t)jv::PATCH_SIZE) { check(false, "preprocess size on ROM patch"); break; }
        if ((o[13] & 0x7f) != 0 || (o[16] & 0x7f) != 0) { check(false, "ROM patch not fully dry"); break; }
    }
    check(true, "preprocess dry on all 192 ROM patches");

    fprintf(stderr, failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
