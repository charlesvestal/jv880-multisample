// Probe: can we render a JV-880 Rhythm Set (drum kit) at all?
//
// Rhythm Sets live in a separate ROM area from Patches and are played through
// a different path, so none of the existing render machinery reaches them.
// This tool answers the one question everything else depends on: which boot
// mode actually makes the firmware sound a rhythm key.
//
// Two candidate paths are tried and MEASURED rather than assumed:
//   patch  - patch mode, note on channel 1 (how every Patch is rendered today)
//   perf   - performance mode, note on channel 10 (what the reference
//            harness's --seq drums does)
//
// Success is not "it made a noise". A wrong path can still produce sound --
// patch mode with a rhythm set loaded will happily play whatever patch is in
// the patch slot, on every key, pitched by key. So the probe checks the
// signature that only a real rhythm set has: DIFFERENT keys produce
// SPECTRALLY DIFFERENT samples that are NOT transpositions of each other.
// A kick and a hi-hat are different recordings; a piano's C2 and C3 are not.
//
// Usage: probe_rhythm <roms_dir> <out_dir>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "jv_rom.h"
#include "jv_render.h"
#include "wav.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "mcu.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace jv;

namespace {

const int SR = 64000;
const int CHUNK = 64;
const int WARMUP = 100000;

// Rhythm Set layout now lives in jv_rom.h; alias the two names this probe
// uses under its own shorter spellings.
const uint32_t ROM_RHYTHM_A = ROM_RHYTHM_PRESET_A;
const uint32_t ROM_RHYTHM_B = ROM_RHYTHM_PRESET_B;
const int      RHYTHM_BYTES = RHYTHM_SET_BYTES;
const int      NVRAM_RHYTHM_OFFSET = 0x67f0;

void run_frames(MCU *m, int frames) { m->updateSC55(frames * 2); }

void drain(MCU *m, std::vector<int16_t> &out, int n) {
    size_t at = out.size();
    out.resize(at + (size_t)n * 2);
    memcpy(out.data() + at, m->sample_buffer, (size_t)n * 2 * sizeof(int16_t));
}

std::vector<int16_t> render_note(MCU *m, int chan, int key, int vel,
                                 double hold_s, double tail_s) {
    std::vector<int16_t> out;
    uint8_t on[3] = {(uint8_t)(0x90 | chan), (uint8_t)key, (uint8_t)vel};
    m->postMidiSC55(on, 3);
    int hold = (int)(hold_s * SR);
    for (int p = 0; p < hold; p += CHUNK) {
        int n = std::min(CHUNK, hold - p);
        run_frames(m, n); drain(m, out, n);
    }
    uint8_t off[3] = {(uint8_t)(0x80 | chan), (uint8_t)key, 0};
    m->postMidiSC55(off, 3);
    int tail = (int)(tail_s * SR);
    for (int p = 0; p < tail; p += CHUNK) {
        int n = std::min(CHUNK, tail - p);
        run_frames(m, n); drain(m, out, n);
    }
    uint8_t all_off[3] = {(uint8_t)(0xB0 | chan), 0x7B, 0x00};
    m->postMidiSC55(all_off, 3);
    // Flush so this note's decay never bleeds into the next.
    for (int p = 0; p < SR / 4; p += CHUNK) run_frames(m, CHUNK);
    return out;
}

double rms_db(const std::vector<int16_t> &s) {
    if (s.empty()) return -200.0;
    double acc = 0;
    for (int16_t v : s) acc += (double)v * v;
    return 20.0 * std::log10(std::sqrt(acc / s.size()) / 32768.0 + 1e-12);
}

// Coarse log-spaced spectral envelope, for comparing timbre between keys.
// Deliberately coarse: we are asking "is this a different instrument", not
// "what pitch is it".
std::vector<double> spectrum(const std::vector<int16_t> &s, int bands = 16) {
    const int N = 4096;
    std::vector<double> mono(N, 0.0);
    int frames = (int)s.size() / 2;
    int take = std::min(frames, N);
    for (int i = 0; i < take; i++) {
        mono[i] = ((double)s[i * 2] + s[i * 2 + 1]) / 2.0 / 32768.0;
        mono[i] *= 0.5 - 0.5 * std::cos(2 * M_PI * i / (N - 1));   // Hann
    }
    std::vector<double> out(bands, 0.0);
    // Direct DFT at band centres -- N is small and this runs a handful of
    // times, so an FFT would be optimising the wrong thing.
    for (int b = 0; b < bands; b++) {
        double f0 = 40.0 * std::pow(2.0, b * 0.5);      // 40 Hz up, half-octave
        if (f0 > SR / 2.0) break;
        double re = 0, im = 0;
        double w = 2 * M_PI * f0 / SR;
        for (int i = 0; i < N; i++) { re += mono[i] * std::cos(w * i); im += mono[i] * std::sin(w * i); }
        out[b] = std::log10(std::sqrt(re * re + im * im) / N + 1e-9);
    }
    return out;
}

double spectral_distance(const std::vector<double> &a, const std::vector<double> &b) {
    double acc = 0; int n = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) { double d = a[i] - b[i]; acc += d * d; n++; }
    return n ? std::sqrt(acc / n) : 0.0;
}

}  // namespace

// Renders one rhythm set across every sounding key, back to back, into a
// single WAV for listening. Uses the path the injection experiment proved:
// the wanted kit is written into the PRESET-A rhythm region of an in-memory
// rom2, the firmware boots in performance mode, and notes are played on
// channel 10.
int demo(const RomSet &roms, uint32_t src_off, const std::string &out_path) {
    std::vector<uint8_t> rom2 = roms.rom2;          // in-memory copy only
    memcpy(&rom2[ROM_RHYTHM_A], roms.rom2.data() + src_off, RHYTHM_BYTES);

    MCU *m = new MCU();
    std::vector<uint8_t> nv = roms.nvram;
    nv[NVRAM_MODE_OFFSET] = 0;                      // performance mode
    if (m->startSC55(roms.rom1.data(), rom2.data(),
                     roms.waverom1.data(), roms.waverom2.data(), nv.data()) != 0) {
        fprintf(stderr, "startSC55 failed\n"); delete m; return 1;
    }
    for (int i = 0; i < WARMUP; i++) m->updateSC55(1);
    uint8_t bank[3] = {0xB0 | 0x0F, 0x00, 81};
    m->postMidiSC55(bank, 3);
    uint8_t pc[2] = {0xC0 | 0x0F, 0x00};
    m->postMidiSC55(pc, 2);
    for (int p = 0; p < SR / 2; p += CHUNK) run_frames(m, CHUNK);

    std::vector<int16_t> all;
    int sounding = 0;
    for (int key = RHYTHM_LOW_KEY; key < RHYTHM_LOW_KEY + RHYTHM_KEYS; key++) {
        auto s = render_note(m, 9, key, 110, 0.25, 0.55);
        if (rms_db(s) < -90.0) continue;            // unassigned key
        sounding++;
        all.insert(all.end(), s.begin(), s.end());
    }
    delete m;

    // Normalise for audition only. The real pipeline will keep absolute level.
    int16_t peak = 1;
    for (int16_t v : all) peak = std::max(peak, (int16_t)std::abs((int)v));
    double g = 28000.0 / peak;
    for (int16_t &v : all) v = (int16_t)std::lround(v * g);

    printf("  %d sounding keys, %.1f s -> %s\n",
           sounding, all.size() / 2.0 / SR, out_path.c_str());
    wav_write_s16(out_path, all.data(), (int)all.size() / 2, 2, SR);
    return 0;
}

// Fraction of total energy sitting in the LATE tail. A rhythm tone with a
// reverb/chorus send feeding an effect has a long tail; a dry one does not.
// This is what distinguishes a send byte from a level byte: a level byte
// scales everything and leaves this ratio alone.
double tail_ratio(const std::vector<int16_t> &s) {
    int frames = (int)s.size() / 2;
    if (frames < SR) return 0.0;
    auto energy = [&](int a, int b) {
        double acc = 0;
        for (int i = a * 2; i < b * 2 && i < (int)s.size(); i++) acc += (double)s[i] * s[i];
        return acc;
    };
    double early = energy(0, SR / 5);              // first 200 ms
    double late  = energy(frames - SR / 2, frames); // last 500 ms
    return late / (early + late + 1e-9);
}

// Which of the 44 rhythm-tone bytes control level and effect sends?
// Established by intervention, not by reading a parameter chart: set each
// byte in turn to 0 and to 127, render the same key, and watch what moves.
// A LEVEL byte swings RMS and leaves the tail ratio alone; a SEND byte
// collapses the tail ratio without gutting RMS.
int sweep(const RomSet &roms, int key) {
    const uint8_t *kit = roms.rom2.data() + ROM_RHYTHM_INTERNAL;
    const int slot = (key - RHYTHM_LOW_KEY) * RHYTHM_TONE_SIZE;

    struct Row { int idx; uint8_t orig; double db0, db127, tr0, tr127; };
    std::vector<Row> rows;
    double base_db = 0, base_tr = 0;

    for (int idx = -1; idx < RHYTHM_TONE_SIZE; idx++) {
        Row r{idx, idx >= 0 ? kit[slot + idx] : (uint8_t)0, 0, 0, 0, 0};
        for (int pass = 0; pass < (idx < 0 ? 1 : 2); pass++) {
            std::vector<uint8_t> rom2 = roms.rom2;
            memcpy(&rom2[ROM_RHYTHM_A], kit, RHYTHM_BYTES);
            if (idx >= 0) rom2[ROM_RHYTHM_A + slot + idx] = pass ? 127 : 0;

            MCU *m = new MCU();
            std::vector<uint8_t> nv = roms.nvram;
            nv[NVRAM_MODE_OFFSET] = 0;
            if (m->startSC55(roms.rom1.data(), rom2.data(), roms.waverom1.data(),
                             roms.waverom2.data(), nv.data()) != 0) { delete m; return 1; }
            for (int i = 0; i < WARMUP; i++) m->updateSC55(1);
            uint8_t bank[3] = {0xB0 | 0x0F, 0x00, 81}; m->postMidiSC55(bank, 3);
            uint8_t pc[2] = {0xC0 | 0x0F, 0x00};       m->postMidiSC55(pc, 2);
            for (int p = 0; p < SR / 2; p += CHUNK) run_frames(m, CHUNK);
            auto s = render_note(m, 9, key, 110, 0.3, 1.2);
            delete m;

            double db = rms_db(s), tr = tail_ratio(s);
            if (idx < 0) { base_db = db; base_tr = tr; }
            else if (pass) { r.db127 = db; r.tr127 = tr; }
            else           { r.db0 = db;  r.tr0 = tr; }
        }
        if (idx >= 0) rows.push_back(r);
        fprintf(stderr, "\r  sweeping byte %2d/%d ", idx + 1, RHYTHM_TONE_SIZE);
    }
    fprintf(stderr, "\r%*s\r", 40, "");

    printf("baseline: %.2f dBFS, tail ratio %.3f   (key %d)\n\n", base_db, base_tr, key);
    printf("byte orig    dB@0   dB@127 | dRMS  | tail@0 tail@127 | dTail | reading\n");
    for (const Row &r : rows) {
        double drms  = std::max(std::fabs(r.db0 - base_db), std::fabs(r.db127 - base_db));
        double dtail = std::max(std::fabs(r.tr0 - base_tr), std::fabs(r.tr127 - base_tr));
        const char *reading = "";
        if (drms > 6.0 && dtail < 0.10)      reading = "LEVEL";
        else if (dtail > 0.15 && drms < 6.0) reading = "SEND / effect";
        else if (drms > 6.0 && dtail > 0.15) reading = "level+effect";
        else if (drms > 1.5 || dtail > 0.05) reading = "minor";
        if (!*reading) continue;   // only print bytes that actually do something
        printf("%4d %4d  %7.2f %7.2f | %5.1f | %6.3f %8.3f | %5.3f | %s\n",
               r.idx, r.orig, r.db0, r.db127, drms, r.tr0, r.tr127, dtail, reading);
    }
    return 0;
}

// Autocorrelation fundamental. A spectral centroid was tried first and is
// NOT adequate here: it measures brightness, which a tuning change moves only
// incidentally, so a genuine octave shift can leave it almost still. This
// searches lag directly and reports the actual repeat period.
double f0_hz(const std::vector<int16_t> &s) {
    int frames = (int)s.size() / 2;
    // Skip the attack transient: percussion is at its most inharmonic there.
    int start = std::min(frames / 8, SR / 20), N = std::min(frames - start, SR / 4);
    if (N < 512) return 0.0;
    std::vector<double> x(N);
    for (int i = 0; i < N; i++)
        x[i] = ((double)s[(start + i) * 2] + s[(start + i) * 2 + 1]) / 2.0;
    double mean = 0; for (double v : x) mean += v; mean /= N;
    for (double &v : x) v -= mean;
    double e0 = 0; for (double v : x) e0 += v * v;
    if (e0 < 1e-6) return 0.0;
    // 50 Hz .. 4 kHz
    int lag_lo = SR / 4000, lag_hi = std::min(N / 2, SR / 50);
    double best = 0; int best_lag = 0;
    for (int lag = lag_lo; lag < lag_hi; lag++) {
        double acc = 0, e = 0;
        for (int i = 0; i + lag < N; i++) { acc += x[i] * x[i + lag]; e += x[i + lag] * x[i + lag]; }
        double norm = acc / (std::sqrt(e0 * e) + 1e-9);
        if (norm > best) { best = norm; best_lag = lag; }
    }
    // Below this the signal has no periodicity worth calling a pitch.
    if (best < 0.30 || best_lag == 0) return 0.0;
    return (double)SR / best_lag;
}

// Amplitude-weighted mean frequency. Coarse but entirely sufficient for the
// question asked below, which is "did this jump an octave", not "what note".
double centroid_hz(const std::vector<int16_t> &s) {
    const int N = 8192;
    std::vector<double> mono(N, 0.0);
    int frames = (int)s.size() / 2, take = std::min(frames, N);
    for (int i = 0; i < take; i++) {
        mono[i] = ((double)s[i * 2] + s[i * 2 + 1]) / 2.0 / 32768.0;
        mono[i] *= 0.5 - 0.5 * std::cos(2 * M_PI * i / (N - 1));
    }
    double num = 0, den = 0;
    for (int b = 1; b < 200; b++) {
        double f0 = b * 25.0;
        double re = 0, im = 0, w = 2 * M_PI * f0 / SR;
        for (int i = 0; i < N; i++) { re += mono[i] * std::cos(w * i); im += mono[i] * std::sin(w * i); }
        double mag = std::sqrt(re * re + im * im);
        num += mag * f0; den += mag;
    }
    return den > 0 ? num / den : 0.0;
}

// What do rhythm-tone bytes 3 and 31 actually control?
//
// The 44-byte sweep showed both MOVE the output, which is not the same as
// knowing what they are. Rendering a wave "neutral" means setting them, so
// guessing is not good enough: byte 3 is only usable as coarse tune if
// +12 really is an octave, and byte 31 is only pan if it really moves the
// image between the channels.
int params(const RomSet &roms) {
    const uint8_t *kit = roms.rom2.data() + ROM_RHYTHM_INTERNAL;
    const int key = 36, slot = (key - RHYTHM_LOW_KEY) * RHYTHM_TONE_SIZE;

    auto render_with = [&](int idx, int val) {
        std::vector<uint8_t> rom2 = roms.rom2;
        memcpy(&rom2[ROM_RHYTHM_A], kit, RHYTHM_BYTES);
        rom2[ROM_RHYTHM_A + slot + idx] = (uint8_t)val;
        MCU *m = new MCU();
        std::vector<uint8_t> nv = roms.nvram;
        nv[NVRAM_MODE_OFFSET] = 0;
        m->startSC55(roms.rom1.data(), rom2.data(), roms.waverom1.data(),
                     roms.waverom2.data(), nv.data());
        for (int i = 0; i < WARMUP; i++) m->updateSC55(1);
        uint8_t bank[3] = {0xB0 | 0x0F, 0x00, 81}; m->postMidiSC55(bank, 3);
        uint8_t pc[2] = {0xC0 | 0x0F, 0x00};       m->postMidiSC55(pc, 2);
        for (int p = 0; p < SR / 2; p += CHUNK) run_frames(m, CHUNK);
        auto s = render_note(m, 9, key, 110, 0.3, 1.0);
        delete m;
        return s;
    };

    printf("byte 3 -- is it COARSE TUNE in semitones?\n");
    printf("   (stored value is %d; if semitones, +12 should roughly double the centroid)\n",
           kit[slot + 3]);
    double base = 0;
    for (int v : {36, 48, 60, 72, 84}) {
        auto s = render_with(3, v);
        double c = centroid_hz(s);
        if (v == 60) base = c;
        printf("   byte3=%3d  centroid %8.1f Hz", v, c);
        if (base > 0 && v != 60) printf("   ratio vs 60: %.3f  (an octave step = %.1f)",
                                        c / base, std::pow(2.0, (v - 60) / 12.0));
        printf("\n");
    }

    printf("\nbyte 31 -- is it PAN?\n");
    printf("   (stored value is %d; 64 would be centre)\n", kit[slot + 31]);
    for (int v : {0, 32, 64, 96, 127}) {
        auto s = render_with(31, v);
        double l = 0, r = 0;
        for (size_t i = 0; i + 1 < s.size(); i += 2) {
            l += (double)s[i] * s[i];
            r += (double)s[i + 1] * s[i + 1];
        }
        double bal = 10.0 * std::log10((r + 1e-9) / (l + 1e-9));
        printf("   byte31=%3d  R-L balance %+7.2f dB %s\n", v, bal,
               bal < -3 ? "(left)" : bal > 3 ? "(right)" : "(centre)");
    }
    return 0;
}

// WHICH byte is coarse tune?
//
// Byte 3 was the obvious candidate -- it sits right after the wave number and
// its stored values cluster around 60 -- but measuring it on a kick showed no
// octave relationship at all (+12 gave a centroid ratio of 0.985 where an
// octave is 2.0). That could mean byte 3 is not tuning, or simply that a kick
// is too inharmonic for a spectral centroid to track. This settles it by
// scanning EVERY byte on a deliberately TONAL key and looking for the one
// where -12 and +12 land near half and double.
//
// Nothing here assumes a unit or an offset: it just looks for the byte that
// behaves like semitones.
int findtune(const RomSet &roms, int key) {
    const uint8_t *kit = roms.rom2.data() + ROM_RHYTHM_INTERNAL;
    const int slot = (key - RHYTHM_LOW_KEY) * RHYTHM_TONE_SIZE;

    auto render_with = [&](int idx, int val) {
        std::vector<uint8_t> rom2 = roms.rom2;
        memcpy(&rom2[ROM_RHYTHM_A], kit, RHYTHM_BYTES);
        if (idx >= 0) rom2[ROM_RHYTHM_A + slot + idx] = (uint8_t)val;
        MCU *m = new MCU();
        std::vector<uint8_t> nv = roms.nvram;
        nv[NVRAM_MODE_OFFSET] = 0;
        m->startSC55(roms.rom1.data(), rom2.data(), roms.waverom1.data(),
                     roms.waverom2.data(), nv.data());
        for (int i = 0; i < WARMUP; i++) m->updateSC55(1);
        uint8_t bank[3] = {0xB0 | 0x0F, 0x00, 81}; m->postMidiSC55(bank, 3);
        uint8_t pc[2] = {0xC0 | 0x0F, 0x00};       m->postMidiSC55(pc, 2);
        for (int p = 0; p < SR / 2; p += CHUNK) run_frames(m, CHUNK);
        auto s = render_note(m, 9, key, 110, 0.3, 0.8);
        delete m;
        return s;
    };

    double base = f0_hz(render_with(-1, 0));
    printf("key %d baseline f0 %.1f Hz\n", key, base);
    printf("looking for a byte where -12 -> ~%.0f Hz and +12 -> ~%.0f Hz\n\n",
           base / 2, base * 2);
    printf("byte orig   down12_Hz   up12_Hz   down_ratio  up_ratio  verdict\n");
    if (base <= 0) { printf("  (baseline has no detectable pitch -- pick a more tonal key)\n"); return 1; }

    for (int idx = 0; idx < RHYTHM_TONE_SIZE; idx++) {
        int orig = kit[slot + idx];
        int lo = orig - 12, hi = orig + 12;
        if (lo < 0 || hi > 127) continue;      // cannot test a clean +/-12 here
        double d = f0_hz(render_with(idx, lo));
        double u = f0_hz(render_with(idx, hi));
        double dr = base > 0 ? d / base : 0, ur = base > 0 ? u / base : 0;
        // Semitone tuning means the two ratios straddle 1 in opposite
        // directions AND the up-step is a large rise. Deliberately loose:
        // a centroid is not a pitch tracker, so this only has to nominate a
        // candidate, which is then confirmed by eye from the numbers.
        bool cand = ur > 1.6 && dr < 0.7;
        printf("%4d %4d  %9.1f %9.1f   %8.3f  %8.3f  %s\n",
               idx, orig, d, u, dr, ur, cand ? "<== COARSE TUNE?" : "");
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: probe_rhythm <roms_dir> <out_dir> [--demo|--sweep|--params|--findtune KEY]\n"); return 2; }
    std::string romdir = argv[1], outdir = argv[2];

    RomSet roms; std::string err;
    if (!roms.load(romdir, &err)) { fprintf(stderr, "rom load: %s\n", err.c_str()); return 1; }

    if (argc > 3 && strcmp(argv[3], "--findtune") == 0) {
        return findtune(roms, argc > 4 ? atoi(argv[4]) : 56);
    }

    if (argc > 3 && strcmp(argv[3], "--params") == 0) {
        return params(roms);
    }

    if (argc > 3 && strcmp(argv[3], "--sweep") == 0) {
        return sweep(roms, argc > 4 ? atoi(argv[4]) : 36);
    }

    if (argc > 3 && strcmp(argv[3], "--demo") == 0) {
        struct Kit { const char *name; uint32_t off; };
        const Kit kits[] = {{"Internal", ROM_RHYTHM_INTERNAL},
                            {"PresetA",  ROM_RHYTHM_A},
                            {"PresetB",  ROM_RHYTHM_B}};
        for (const Kit &k : kits) {
            printf("rhythm set %s:\n", k.name);
            if (demo(roms, k.off, outdir + "/kit_" + k.name + ".wav") != 0) return 1;
        }
        return 0;
    }

    // The rhythm set we want to hear, straight out of rom2.
    const uint8_t *rhythm = roms.rom2.data() + ROM_RHYTHM_INTERNAL;

    struct Mode { const char *name; bool perf; int chan; };
    const Mode modes[] = { {"perf", true, 9} };

    // Where to write the uniform kit. Writing it into NVRAM proved to change
    // NOTHING (identical output to the stock kit, to two decimals), so the
    // firmware is not sourcing the sounding rhythm set from NVRAM 0x67f0 --
    // that address is merely where the factory image happens to keep a copy.
    // The remaining candidate is the ROM itself, so inject into an IN-MEMORY
    // copy of rom2 (never the file on disk) the way wave_inject already does
    // for the wave ROM, and find which of the three regions is live.
    struct Target { const char *name; int32_t rom_off; int nvram_off; };
    const Target targets[] = {
        {"none (control)",  -1,       -1},
        {"nvram 0x67f0",    -1,       NVRAM_RHYTHM_OFFSET},
        {"rom Internal",    (int32_t)ROM_RHYTHM_INTERNAL, -1},
        {"rom PresetA",     (int32_t)ROM_RHYTHM_A,        -1},
        {"rom PresetB",     (int32_t)ROM_RHYTHM_B,        -1},
    };

    // Keys chosen to be different INSTRUMENTS in the GM-style map, not just
    // different pitches: kick, snare, closed hat, and a conga well up the
    // keyboard. If a path is really playing the rhythm set these four are
    // four unrelated recordings.
    const int keys[] = {36, 40, 44, 64};
    const char *labels[] = {"36_kick", "40_snare", "44_hat", "64_conga"};

    // THE DECISIVE TEST. Comparing spectra across keys does NOT prove we are
    // playing the rhythm set: different keys of an ordinary PATCH also differ
    // spectrally, because they are different pitches. (That confusion already
    // produced one wrong result in this project, when a drum-kit detector
    // flagged 441/447 patches by comparing FFT bins across keys.)
    //
    // So instead of observing, intervene: build a UNIFORM kit in which every
    // one of the 61 keys holds a copy of the same rhythm tone. If the firmware
    // is really sourcing its sound from the bytes we wrote, every key must now
    // sound IDENTICAL -- no pitch tracking, no variation. If it is playing
    // anything else, the keys keep differing and the mode is wrong.
    // The uniform kit: 61 copies of the closed-hat tone.
    std::vector<uint8_t> uniform_kit(RHYTHM_BYTES);
    {
        const uint8_t *src = rhythm + (44 - RHYTHM_LOW_KEY) * RHYTHM_TONE_SIZE;
        for (int k = 0; k < RHYTHM_KEYS; k++)
            memcpy(&uniform_kit[k * RHYTHM_TONE_SIZE], src, RHYTHM_TONE_SIZE);
    }

    for (const Target &tgt : targets) {
    printf("\n########## uniform kit -> %s ##########\n", tgt.name);
    for (const Mode &mode : modes) {

        MCU *m = new MCU();
        std::vector<uint8_t> nv = roms.nvram;
        nv[NVRAM_MODE_OFFSET] = mode.perf ? 0 : 1;
        if (tgt.nvram_off >= 0)
            memcpy(&nv[tgt.nvram_off], uniform_kit.data(), RHYTHM_BYTES);

        // In-memory ROM copy only. The ROM files on disk are never written.
        std::vector<uint8_t> rom2 = roms.rom2;
        if (tgt.rom_off >= 0)
            memcpy(&rom2[tgt.rom_off], uniform_kit.data(), RHYTHM_BYTES);

        if (m->startSC55(roms.rom1.data(), rom2.data(),
                         roms.waverom1.data(), roms.waverom2.data(), nv.data()) != 0) {
            fprintf(stderr, "startSC55 failed\n"); delete m; return 1;
        }
        for (int i = 0; i < WARMUP; i++) m->updateSC55(1);

        if (mode.perf) {
            // Preset-A performance 0 via Bank Select MSB 81 + PC on ch16.
            uint8_t bank[3] = {0xB0 | 0x0F, 0x00, 81};
            m->postMidiSC55(bank, 3);
            uint8_t pc[2] = {0xC0 | 0x0F, 0x00};
            m->postMidiSC55(pc, 2);
        } else {
            uint8_t pc[2] = {0xC0, 0x00};
            m->postMidiSC55(pc, 2);
        }
        for (int p = 0; p < SR / 2; p += CHUNK) run_frames(m, CHUNK);

        std::vector<std::vector<double>> spectra;
        for (int i = 0; i < 4; i++) {
            auto s = render_note(m, mode.chan, keys[i], 100, 0.5, 1.5);
            double db = rms_db(s);
            spectra.push_back(spectrum(s));
            char path[1024];
            char tag[64]; snprintf(tag, sizeof tag, "%s", tgt.name);
            for (char *c = tag; *c; c++) if (*c == ' ' || *c == '(' || *c == ')') *c = '_';
            snprintf(path, sizeof path, "%s/%s_%s_%s.wav", outdir.c_str(),
                     tag, mode.name, labels[i]);
            wav_write_s16(path, s.data(), (int)s.size() / 2, 2, SR);
            printf("  key %3d %-9s  %7.2f dBFS  -> %s\n", keys[i], labels[i], db, path);
        }

        // The discriminating test: are these four different instruments?
        double dmin = 1e9, dmax = 0;
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++) {
                double d = spectral_distance(spectra[i], spectra[j]);
                dmin = std::min(dmin, d); dmax = std::max(dmax, d);
            }
        printf("  pairwise spectral distance: min %.3f  max %.3f\n", dmin, dmax);
        delete m;
    }
    }
    printf("\nVERDICT RULE: the live region is the one whose distances COLLAPSE\n"
           "toward 0 relative to the control. Any target that leaves the control\n"
           "numbers unchanged is not the region the firmware reads.\n");
    return 0;
}
