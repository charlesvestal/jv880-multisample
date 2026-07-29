// calibrate — Task 4: render fixed sweeps of the JV-880's chorus and reverb
// parameters through a steady reference patch, so analyze_calibration.py can
// MEASURE the emulator's actual effect response instead of guessing a
// formula for Task 6.
//
// Usage:
//   calibrate --roms <dir> [--out <dir, default "calib">]
//
// One WAV per setting is written into --out. tools/analyze_calibration.py
// turns those renders into calib/calibration.json.

#include "jv_render.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace jv;

namespace {

void mkdirs(const std::string &p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) mkdir(cur.c_str(), 0755);
    }
}

void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --roms <dir> [--out <dir, default \"calib\">]\n", prog);
}

// Patch-common byte offsets (bit-packed), verified against real ROM data --
// see the task spec's layout table. Sweep values are poked into a COPY of
// the (already preprocess()'d) base patch, never into the original.
std::vector<uint8_t> with_reverb(std::vector<uint8_t> base, int type, int level, int time) {
    base[12] = (uint8_t)((base[12] & 0xF0) | (type & 0x0F));   // reverbtype: 4 bits
    base[13] = (uint8_t)(level & 0x7F);                        // reverblevel: 7 bits
    base[14] = (uint8_t)(time & 0x7F);                         // reverbtime: 7 bits
    return base;
}

std::vector<uint8_t> with_chorus(std::vector<uint8_t> base, int level, int depth, int rate) {
    base[16] = (uint8_t)((base[16] & 0x80) | (level & 0x7F));  // choruslevel, keep chorusoutput bit
    base[17] = (uint8_t)(depth & 0x7F);                        // chorusdepth
    base[18] = (uint8_t)(rate & 0x7F);                         // chorusrate
    return base;
}

// Like with_reverb, but also pokes reverbfeedback (byte 15, 7 bits) --
// needed for the Delay/Pan-Dly (types 6-7) feedback sweep. Kept as a
// separate function rather than adding a 4th parameter to with_reverb:
// every existing call site (types 0-5 chorus/reverb sweeps) intentionally
// leaves feedback at the base patch's own native value (see the
// chorusfeedback note above this function for why an earlier attempt at
// forcing effect-adjacent bytes to 0 backfired), so a shared function
// would need a magic "leave alone" sentinel for no benefit.
std::vector<uint8_t> with_reverb_fb(std::vector<uint8_t> base, int type, int level, int time, int feedback) {
    base[12] = (uint8_t)((base[12] & 0xF0) | (type & 0x0F));
    base[13] = (uint8_t)(level & 0x7F);
    base[14] = (uint8_t)(time & 0x7F);
    base[15] = (uint8_t)(feedback & 0x7F);
    return base;
}

// Mutes EVERY tone's dry path and routes it through the reverb send alone
// (reverb level forced to max), so the render contains ONLY the reverb
// algorithm's own output -- a short percussive strike through that IS an
// impulse response of the JV's real reverb. Per-tone byte offsets
// (drylevel=81, reverbsendlevel=82, chorussendlevel=83) confirmed against
// docs/superpowers/plans/2026-07-28-jv880-multisample.md's per-tone offset
// table and verified empirically: with these three bytes poked on every
// tone, a Marimba strike into Hall2 produces a clean decaying tail with a
// near-silent onset (no percussive attack bleeding through), vs. an
// audible attack transient when drylevel is left alone.
std::vector<uint8_t> pure_wet_reverb(std::vector<uint8_t> base, int type, int time) {
    base[12] = (uint8_t)((base[12] & 0xF0) | (type & 0x0F));
    base[13] = 127;                              // reverblevel: max send
    base[14] = (uint8_t)(time & 0x7F);
    for (int t = 0; t < TONE_COUNT; t++) {
        int off = TONE_BASE + t * TONE_STRIDE;
        base[off + 81] = 0;                       // drylevel -> mute direct signal
        base[off + 82] = 127;                     // reverbsendlevel -> full send
        base[off + 83] = 0;                        // chorussendlevel -> none
    }
    return base;
}

// 0, step, 2*step, ... through the largest multiple < 128, plus a final
// exact 127 if the sweep didn't already land there. Acceptance criteria
// compare measurements AT raw 0 and raw 127 specifically (e.g. "chorus rate
// at 127 is at least 2x that at 0"), so every sweep must actually render
// those two exact endpoints rather than stopping at whatever a fixed step
// happens to land on (e.g. step 16 would otherwise stop at 112).
std::vector<int> sweep(int step) {
    std::vector<int> v;
    for (int raw = 0; raw < 128; raw += step) v.push_back(raw);
    if (v.empty() || v.back() != 127) v.push_back(127);
    return v;
}

void render_to(Renderer &r, const GridSpec &g, const std::vector<uint8_t> &bytes,
               const std::string &path) {
    r.load_patch_bytes(bytes, g);
    std::vector<int16_t> pcm = r.render_note(60, 100, g);
    int frames = (int)(pcm.size() / 2);
    if (!wav_write_s16(path, pcm.data(), frames, 2, SAMPLE_RATE)) {
        fprintf(stderr, "failed to write %s\n", path.c_str());
        exit(1);
    }
    fprintf(stderr, "wrote %s (%d frames)\n", path.c_str(), frames);
}

} // namespace

int main(int argc, char **argv) {
    std::string roms_dir, out_dir = "calib";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--roms") && i + 1 < argc) {
            roms_dir = argv[++i];
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_dir = argv[++i];
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (roms_dir.empty()) {
        fprintf(stderr, "--roms is required\n");
        usage(argv[0]);
        return 1;
    }

    std::string err;
    RomSet roms;
    if (!roms.load(roms_dir, &err)) {
        fprintf(stderr, "failed to load ROMs: %s\n", err.c_str());
        return 1;
    }

    auto internal = enumerate_internal(roms);
    // "Pipe Organ 3", NOT "Pipe Organ 1" (internal index 24) as originally
    // specified. index 24 has LFO1 set to a RANDOM waveform (form 5 / RND2)
    // with real per-tone pitch/TVA depth (+-5 / +-1) on all four tones.
    // decide_lfo_strip() correctly refuses to strip random waveforms
    // ("random waveform" reason, strip=false), so that genuine random
    // modulation survives preprocess() and contaminates the amplitude
    // envelope -- confirmed by probing read_effects/read_tone_lfo/
    // decide_lfo_strip directly on index 24 and seeing exactly this. It
    // produced a non-monotonic, noisy chorus-rate measurement (see the
    // Task 4 report for the numbers).
    //
    // index 26 has LFO1 (SIN) and LFO2 (SIN) present on every tone but with
    // pitch/TVF/TVA depth == 0 across the board ("no LFO depth" reason) --
    // i.e. genuinely no modulation regardless of the strip decision. That
    // makes it the clean, flat-envelope patch the calibration sweep needs.
    const int BASE_PATCH_INDEX = 26;
    if ((size_t)BASE_PATCH_INDEX >= internal.size()) {
        fprintf(stderr, "internal patch %d out of range (have %zu)\n",
                BASE_PATCH_INDEX, internal.size());
        return 1;
    }
    const PatchRef &pr = internal[BASE_PATCH_INDEX];
    fprintf(stderr, "base patch: %s (bank %s, index %d)\n",
            pr.name.c_str(), pr.bank.c_str(), pr.index);

    LfoDecision d1 = decide_lfo_strip(pr.data, 1);
    LfoDecision d2 = decide_lfo_strip(pr.data, 2);
    // preprocess() zeroes reverblevel/choruslevel (dry rendering is its
    // job) -- that dry state IS the base we sweep away from. Every sweep
    // value below is poked in AFTER this call, never before: poking first
    // and calling preprocess() second would silently flatten the whole
    // sweep back to all-dry and make every render identical.
    std::vector<uint8_t> base = preprocess(pr.data, d1, d2);

    // NOTE: "Pipe Organ 3" ships with chorusfeedback=100/127 and we
    // deliberately leave it alone. An earlier version of this tool zeroed
    // it, hypothesising the feedback was causing a rate*feedback resonance
    // that explained a couple of non-monotonic chorus-rate readings.
    // Empirically that made things categorically worse: zeroing feedback
    // unmasked a MUCH stronger, suspiciously exact ~6.89 Hz artifact that
    // then dominated the L-R signal at most raw settings (identical to 4
    // significant figures across unrelated raw values -- not audio content,
    // some other fixed periodicity in the render/measurement path). With
    // feedback left at its native value, that artifact stays subordinate to
    // the genuine chorus-rate-driven signal across nearly the whole sweep.
    // See the Task 4 report for the actual numbers and remaining concerns.

    // Second base patch, used ONLY by the Delay/Pan-Dly (types 6-7) time/
    // feedback sweeps and the reverb impulse-response captures below.
    // "Marimba", internal index 18: percussive (decays on its own within a
    // few hundred ms regardless of hold length) and, like Pipe Organ 3
    // above, has "no LFO depth" on both LFO1/LFO2 (confirmed via
    // decide_lfo_strip), so no modulation contaminates the transient's
    // shape. A sustained tone (Pipe Organ 3) is the RIGHT choice for
    // measuring a reverb/chorus's steady-state response, but the wrong one
    // for locating discrete echoes: overlapping copies of a continuous
    // drone don't produce separable peaks, and the task spec's own guidance
    // is to use "a short percussive source" for exactly this reason.
    const int DELAY_BASE_PATCH_INDEX = 18;
    if ((size_t)DELAY_BASE_PATCH_INDEX >= internal.size()) {
        fprintf(stderr, "internal patch %d out of range (have %zu)\n",
                DELAY_BASE_PATCH_INDEX, internal.size());
        return 1;
    }
    const PatchRef &pr2 = internal[DELAY_BASE_PATCH_INDEX];
    fprintf(stderr, "delay base patch: %s (bank %s, index %d)\n",
            pr2.name.c_str(), pr2.bank.c_str(), pr2.index);
    LfoDecision dd1 = decide_lfo_strip(pr2.data, 1);
    LfoDecision dd2 = decide_lfo_strip(pr2.data, 2);
    std::vector<uint8_t> delay_base = preprocess(pr2.data, dd1, dd2);

    GridSpec g;
    g.hold_seconds = 4.0;
    g.tail_seconds = 4.0;
    g.silence_db   = -120.0;   // effectively never truncate: truncation would destroy RT60

    // The chorus-RATE sweep specifically needs a MUCH longer hold than the
    // other sweeps: measuring a slow LFO's frequency needs several full
    // cycles of data, and the suggested 0.05..12 Hz search band's low end
    // (0.05 Hz = 20s/cycle) cannot be resolved from a 4s hold at all -- a
    // direct empirical check (rendering the same sweep at hold=16s and
    // comparing) confirmed the low raw settings only became reliably
    // measurable once the hold window grew past a few seconds. 12s hold
    // keeps total render time trivial (the emulator runs far faster than
    // real time) while giving low rates enough cycles to autocorrelate
    // against. Depth/level/reverb sweeps stay on the spec's 4.0/4.0 `g`
    // (rate is the only measurement that is cycle-count-starved by it).
    GridSpec g_rate = g;
    g_rate.hold_seconds = 12.0;
    g_rate.tail_seconds = 1.0;   // chorus has no meaningful post-note-off tail of its own

    // Delay/Pan-Dly TIME sweep: only needs to capture the very first echo
    // of a single percussive strike (feedback=0), which measured well under
    // 0.5s across the whole raw range on Marimba -- 1.5s hold is generous
    // margin. Kept short so the sweep stays cheap.
    GridSpec g_delay_time = g;
    g_delay_time.hold_seconds = 1.5;
    g_delay_time.tail_seconds = 1.0;
    g_delay_time.silence_db = -120.0;   // never truncate: a real echo could arrive late

    // Delay/Pan-Dly FEEDBACK sweep: needs MANY repeats (analyze_calibration
    // .py's measure_delay_feedback_gain compares energy ~3-5 and ~15-17
    // repeat-periods out). render_note()'s tail early-exits after ~100ms of
    // near-silence, which a train of discrete echoes spaced tens-to-hundreds
    // of ms apart can trip well before a later repeat arrives -- but the
    // HOLD phase is always rendered in full regardless (see jv_render.cpp's
    // design note C), so hold is set long enough (5s, comfortably above 17x
    // the longest measured single-repeat period at time=64) to capture the
    // whole needed span deterministically rather than hoping the tail's
    // quiet-detection doesn't misfire on it.
    GridSpec g_delay_fb = g;
    g_delay_fb.hold_seconds = 5.0;
    g_delay_fb.tail_seconds = 1.0;
    g_delay_fb.silence_db = -120.0;

    // Reverb impulse-response capture (types 0-5, pure-wet via
    // pure_wet_reverb): 0.1s hold is enough for the reverb's early
    // reflections to register (confirmed empirically: unlike Delay/Pan-Dly,
    // a true reverb's response starts within the first ~100ms, so
    // render_note()'s hold-phase `peak` -- which gates the tail's
    // early-quiet-exit threshold -- is meaningfully nonzero here). -60dB is
    // a REAL trim threshold (unlike the -120dB "never truncate" used
    // elsewhere): it lets the render stop once genuinely inaudible so the
    // raw WAV IS already close to a trimmed IR, rather than shipping
    // multiple extra seconds of digital silence. 8s tail comfortably
    // exceeds every measured reverb_rt60 (max 5.84s, Hall2).
    GridSpec g_ir = g;
    g_ir.hold_seconds = 0.1;
    g_ir.tail_seconds = 8.0;
    g_ir.silence_db = -60.0;

    Renderer r;
    if (!r.init(roms)) {
        fprintf(stderr, "emulator init failed\n");
        return 1;
    }

    mkdirs(out_dir);

    // Reference: the base patch, fully dry. Every wet/depth measurement in
    // analyze_calibration.py is taken relative to this file.
    render_to(r, g, base, out_dir + "/dry.wav");

    for (int raw : sweep(8)) {
        std::vector<uint8_t> b = with_chorus(base, /*level=*/100, /*depth=*/100, /*rate=*/raw);
        char fn[64];
        snprintf(fn, sizeof(fn), "/chorus_rate_%03d.wav", raw);
        render_to(r, g_rate, b, out_dir + fn);
    }

    for (int raw : sweep(16)) {
        std::vector<uint8_t> b = with_chorus(base, /*level=*/100, /*depth=*/raw, /*rate=*/40);
        char fn[64];
        snprintf(fn, sizeof(fn), "/chorus_depth_%03d.wav", raw);
        render_to(r, g, b, out_dir + fn);
    }

    for (int raw : sweep(16)) {
        std::vector<uint8_t> b = with_chorus(base, /*level=*/raw, /*depth=*/80, /*rate=*/40);
        char fn[64];
        snprintf(fn, sizeof(fn), "/chorus_level_%03d.wav", raw);
        render_to(r, g, b, out_dir + fn);
    }

    for (int type = 0; type <= 5; type++) {
        for (int raw : sweep(16)) {
            std::vector<uint8_t> b = with_reverb(base, type, /*level=*/100, /*time=*/raw);
            char fn[64];
            snprintf(fn, sizeof(fn), "/reverb_t%d_time_%03d.wav", type, raw);
            render_to(r, g, b, out_dir + fn);
        }
    }

    for (int raw : sweep(16)) {
        std::vector<uint8_t> b = with_reverb(base, /*type=*/4, /*level=*/raw, /*time=*/80);
        char fn[64];
        snprintf(fn, sizeof(fn), "/reverb_level_%03d.wav", raw);
        render_to(r, g, b, out_dir + fn);
    }

    // Delay/Pan-Dly (types 6-7) TIME sweep, on the Marimba base patch with
    // dry PRESENT (not pure-wet): the echo's arrival is what's being
    // measured, and analyze_calibration.py locates it by cross-correlating
    // against the dry attack's own shape (delay_dry.wav below), which only
    // works if that same dry attack is actually present in these renders.
    // Reuses the exact "reverb_t{type}_time_{raw}.wav" naming the types 0-5
    // sweep above already uses -- analyze_calibration.py's existing
    // TYPE_RE/RAW_RE parse it unchanged; the type NUMBER (6/7, vs. 0-5) is
    // what routes it to delay-echo analysis instead of RT60. feedback=0:
    // only the FIRST echo is needed to locate the delay time.
    for (int type : {6, 7}) {
        for (int raw : sweep(16)) {
            std::vector<uint8_t> b = with_reverb_fb(delay_base, type, /*level=*/127,
                                                     /*time=*/raw, /*feedback=*/0);
            char fn[64];
            snprintf(fn, sizeof(fn), "/reverb_t%d_time_%03d.wav", type, raw);
            render_to(r, g_delay_time, b, out_dir + fn);
        }
    }

    // Reference (reverb level=0): the SAME Marimba base patch/hold/tail as
    // the time sweep above, with the effect send silenced. This is the
    // cross-correlation TEMPLATE source for locating echoes -- the shape of
    // Marimba's own un-effected attack transient.
    render_to(r, g_delay_time, with_reverb_fb(delay_base, /*type=*/6, /*level=*/0,
                                              /*time=*/0, /*feedback=*/0),
              out_dir + "/delay_dry.wav");

    // Delay/Pan-Dly FEEDBACK sweep. time fixed at 64 (mid-range raw value;
    // measured ~0.09s/0.19s single-repeat period for Pan-Dly/Delay
    // respectively) -- see g_delay_fb's comment above for why hold is 5s.
    for (int type : {6, 7}) {
        for (int raw : sweep(16)) {
            std::vector<uint8_t> b = with_reverb_fb(delay_base, type, /*level=*/127,
                                                     /*time=*/64, /*feedback=*/raw);
            char fn[64];
            snprintf(fn, sizeof(fn), "/delay_t%d_feedback_%03d.wav", type, raw);
            render_to(r, g_delay_fb, b, out_dir + fn);
        }
    }

    // Reverb impulse-response captures (types 0-5), pure-wet via
    // pure_wet_reverb on the Marimba base patch. Reuses the SAME 9
    // step-16 raw time values already proven adequate for reverb_rt60
    // (adjacent points differ well under 1% -- see analyze_calibration.py's
    // module docstring), so these IRs double as a direct RT60 cross-check.
    for (int type = 0; type <= 5; type++) {
        for (int raw : sweep(16)) {
            std::vector<uint8_t> b = pure_wet_reverb(delay_base, type, raw);
            char fn[64];
            snprintf(fn, sizeof(fn), "/ir_t%d_time_%03d.wav", type, raw);
            render_to(r, g_ir, b, out_dir + fn);
        }
    }

    fprintf(stderr, "calibration renders complete: %s\n", out_dir.c_str());
    return 0;
}
