// render_waves -- render each distinct WAVE referenced by a board's drum kits
// once, at neutral settings, as a plain one-shot sample library.
//
// A kit's 61 keys do not hold 61 different waves: they hold about 39-103
// distinct ones, reused at different tunings (one tom wave becomes high, mid
// and low) and shared between the kits on a board. Across the whole library
// 3,172 key references resolve to 1,076 distinct waves, so rendering waves
// rather than keys is about a third of the work and yields the raw material
// instead of Roland's arrangement of it.
//
// NEUTRAL means three bytes are overridden, each verified by measurement
// rather than read off a chart:
//
//   byte 3  coarse tune -> 60 (centre). Confirmed by scanning every byte on a
//           tonal key with an autocorrelation pitch detector: -12 gives
//           exactly half the fundamental and +12 exactly double. An earlier
//           attempt using a spectral centroid on a KICK concluded byte 3 was
//           not tuning at all -- centroid measures brightness, which a tuning
//           change barely moves, and a kick is too inharmonic to track.
//   byte 31 pan -> 64 (centre). Confirmed monotonic: 0 is hard left, 127 hard
//           right, 64 measures +0.23 dB R-L.
//   byte 41 level -> 127 (full). Its stored values vary per key (110, 127)
//           and setting it to 0 silences the tone.
//
// Everything else -- envelope, filter, velocity response -- is left exactly as
// the source tone has it. Only bytes whose meaning was actually established
// are touched; guessing at the rest would quietly colour 1,076 renders.
//
// Usage:
//   render_waves --roms <dir> --board <name> --out <dir> [--velocity N]

#include "jv_render.h"
#include "wav.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace jv;

namespace {

const char *BOARD_INTERNAL = "JV-880 Internal";

// Verified rhythm-tone byte meanings (see the file comment).
const int TONE_ENABLE_BYTE = 0;    // bit 7
const int WAVE_LOW_BYTE    = 1;
const int WAVE_GROUP_BYTE  = 2;
const int COARSE_TUNE_BYTE = 3;
const int PAN_BYTE         = 31;
const int LEVEL_BYTE       = 41;
const int COARSE_TUNE_CENTRE = 60;
const int PAN_CENTRE         = 64;
const int LEVEL_MAX          = 127;

// Internal wave names live in a table at the very start of rom2: a 12-char
// name at offset 4 of each 0x3c-byte record. Expansion boards have no
// equivalent table -- the offset in their header at 0x84 points at the end of
// the patch block, not at wave names -- so expansion waves are identified
// numerically.
const uint32_t WAVE_NAME_TABLE   = 0x000004;
const uint32_t WAVE_NAME_STRIDE  = 0x3c;

std::string internal_wave_name(const RomSet &roms, int wave) {
    size_t off = WAVE_NAME_TABLE + (size_t)wave * WAVE_NAME_STRIDE;
    if (off + 12 > roms.rom2.size()) return "";
    std::string s((const char *)roms.rom2.data() + off, 12);
    while (!s.empty() && (s.back() == ' ' || (unsigned char)s.back() >= 0x7f)) s.pop_back();
    for (char &c : s) if ((unsigned char)c < 0x20) c = ' ';
    return s;
}

std::string sanitize(const std::string &s) {
    std::string o;
    for (unsigned char c : s)
        if (std::isalnum(c) || c == ' ' || c == '-' || c == '_' || c == '.') o += (char)c;
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o;
}

std::string json_escape(const std::string &s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o;
}

bool mkdirs(const std::string &p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size())
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return true;
}

struct WaveRef {
    int group = 0, wave = 0;
    std::vector<uint8_t> tone;   // RHYTHM_TONE_SIZE bytes, already neutralised
};

// One neutralised copy of the source tone, so the wave sounds at centre
// pitch, centre pan and full level.
std::vector<uint8_t> neutralise(const uint8_t *src) {
    std::vector<uint8_t> t(src, src + RHYTHM_TONE_SIZE);
    t[TONE_ENABLE_BYTE] |= 0x80;
    t[COARSE_TUNE_BYTE] = COARSE_TUNE_CENTRE;
    t[PAN_BYTE]         = PAN_CENTRE;
    t[LEVEL_BYTE]       = LEVEL_MAX;
    return t;
}

// Every distinct wave a set of kits references, in first-seen order, each
// carrying the first tone that used it as its template.
std::vector<WaveRef> collect_waves(const std::vector<RhythmRef> &kits) {
    std::map<std::pair<int, int>, size_t> seen;
    std::vector<WaveRef> out;
    for (const RhythmRef &k : kits) {
        for (int i = 0; i < RHYTHM_KEYS; i++) {
            const uint8_t *t = k.data + (size_t)i * RHYTHM_TONE_SIZE;
            if (!(t[TONE_ENABLE_BYTE] & 0x80)) continue;
            auto id = std::make_pair((int)t[WAVE_GROUP_BYTE], (int)t[WAVE_LOW_BYTE]);
            if (seen.count(id)) continue;
            seen[id] = out.size();
            WaveRef w;
            w.group = id.first;
            w.wave  = id.second;
            w.tone  = neutralise(t);
            out.push_back(w);
        }
    }
    return out;
}

void usage(const char *p) {
    fprintf(stderr, "Usage: %s --roms <dir> --board <name> --out <dir> [--velocity N]\n", p);
}

}  // namespace

int main(int argc, char **argv) {
    std::string roms_dir, board, out_dir;
    int velocity = 112;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--roms") && i + 1 < argc) roms_dir = argv[++i];
        else if (!strcmp(argv[i], "--board") && i + 1 < argc) board = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--velocity") && i + 1 < argc) velocity = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (roms_dir.empty() || board.empty() || out_dir.empty()) { usage(argv[0]); return 1; }

    RomSet roms;
    std::string err;
    if (!roms.load(roms_dir, &err)) { fprintf(stderr, "rom load: %s\n", err.c_str()); return 1; }

    std::vector<Expansion> expansions;
    std::vector<RhythmRef> kits;
    const Expansion *sel = nullptr;
    if (board == BOARD_INTERNAL) {
        kits = enumerate_internal_rhythm(roms);
    } else {
        expansions = scan_expansions(roms_dir + "/expansions");
        for (const auto &e : expansions)
            if (e.usable && e.name == board) { kits = enumerate_expansion_rhythm(e); sel = &e; break; }
    }
    if (kits.empty()) { fprintf(stderr, "%s has no rhythm sets\n", board.c_str()); return 0; }

    std::vector<WaveRef> waves = collect_waves(kits);
    fprintf(stderr, "%s: %zu kits -> %zu distinct waves\n",
            board.c_str(), kits.size(), waves.size());

    if (!mkdirs(out_dir)) { fprintf(stderr, "cannot create %s\n", out_dir.c_str()); return 1; }

    GridSpec g;
    g.hold_seconds = 0.5;
    g.tail_seconds = 6.0;

    Renderer r;
    const uint8_t *exp_data = sel ? sel->unscrambled.data() : nullptr;
    size_t exp_len = sel ? sel->unscrambled.size() : 0;

    std::string manifest;
    int rendered = 0, silent = 0;

    // A synthetic kit holds 61 waves at a time -- the firmware only reads a
    // kit at boot, so packing the keyboard full means one boot per 61 waves
    // instead of one per wave.
    for (size_t base = 0; base < waves.size(); base += RHYTHM_KEYS) {
        size_t n = std::min((size_t)RHYTHM_KEYS, waves.size() - base);
        std::vector<uint8_t> kit((size_t)RHYTHM_SET_BYTES, 0);
        for (size_t i = 0; i < n; i++)
            memcpy(&kit[i * RHYTHM_TONE_SIZE], waves[base + i].tone.data(), RHYTHM_TONE_SIZE);

        if (!r.init_rhythm(roms, kit.data(), g, exp_data, exp_len)) {
            fprintf(stderr, "emulator init failed\n");
            return 1;
        }

        for (size_t i = 0; i < n; i++) {
            const WaveRef &w = waves[base + i];
            std::vector<int16_t> pcm = r.render_note(RHYTHM_LOW_KEY + (int)i, velocity, g);

            bool sounds = false;
            for (int16_t v : pcm) if (std::abs((int)v) > 64) { sounds = true; break; }
            if (!sounds) { silent++; continue; }

            std::string name = (board == BOARD_INTERNAL)
                                   ? sanitize(internal_wave_name(roms, w.wave))
                                   : std::string();
            char fn[256];
            if (name.empty())
                snprintf(fn, sizeof fn, "wave_%d_%03d.wav", w.group, w.wave);
            else
                snprintf(fn, sizeof fn, "wave_%d_%03d %s.wav", w.group, w.wave, name.c_str());

            if (!wav_write_s16(out_dir + "/" + fn, pcm.data(), (int)pcm.size() / 2, 2, SAMPLE_RATE)) {
                fprintf(stderr, "failed to write %s\n", fn);
                return 1;
            }
            char b[512];
            snprintf(b, sizeof b,
                     "%s\n    {\"group\": %d, \"wave\": %d, \"name\": \"%s\", "
                     "\"frames\": %d, \"file\": \"%s\"}",
                     manifest.empty() ? "" : ",", w.group, w.wave,
                     json_escape(name).c_str(), (int)pcm.size() / 2, json_escape(fn).c_str());
            manifest += b;
            rendered++;
        }
        fprintf(stderr, "  %zu/%zu waves\n", std::min(base + n, waves.size()), waves.size());
    }

    std::string mp = out_dir + "/waves.json";
    FILE *f = fopen(mp.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", mp.c_str()); return 1; }
    int wrote = fprintf(f,
        "{\n  \"board\": \"%s\", \"sample_rate\": %d, \"velocity\": %d,\n"
        "  \"neutral\": {\"coarse_tune\": %d, \"pan\": %d, \"level\": %d},\n"
        "  \"waves\": [%s\n  ]\n}\n",
        json_escape(board).c_str(), SAMPLE_RATE, velocity,
        COARSE_TUNE_CENTRE, PAN_CENTRE, LEVEL_MAX, manifest.c_str());
    int rc = fclose(f);
    if (wrote < 0 || rc != 0) { fprintf(stderr, "failed writing %s\n", mp.c_str()); return 1; }

    fprintf(stderr, "%s: rendered %d waves (%d silent, skipped)\n",
            board.c_str(), rendered, silent);
    return 0;
}
