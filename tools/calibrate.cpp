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

    fprintf(stderr, "calibration renders complete: %s\n", out_dir.c_str());
    return 0;
}
