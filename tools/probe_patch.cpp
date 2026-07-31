// probe_patch -- render one patch with arbitrary byte overrides, to find out
// WHY it behaves as it does.
//
// Written for SR-JV80-01 Pop patch 23 "Snow Bells", the only patch of 4,197
// that renders completely silent: every zone, every key from 0 to 127, at
// velocity 127, with and without the board's expansion waves. Reading its
// bytes narrows the suspects but cannot settle which one is responsible, so
// this changes them one at a time and listens.
//
// Overrides are given as tone:byte=value, or c:byte=value for patch common:
//   probe_patch --roms R --board B --patch 23 --set 3:0=128
//     turns tone 3's switch bit on (tone byte 0, bit 7).
//
// Usage:
//   probe_patch --roms <dir> --board <name> --patch N [--set S]... [--key K]
//               [--velocity V]

#include "jv_render.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace jv;

namespace {

const char *BOARD_INTERNAL = "JV-880 Internal";

struct Override { int tone; int byte; int value; };   // tone < 0 == common

bool parse_set(const char *s, Override *o) {
    // "<tone|c>:<byte>=<value>"
    const char *colon = strchr(s, ':');
    const char *eq = strchr(s, '=');
    if (!colon || !eq || eq < colon) return false;
    o->tone  = (s[0] == 'c' || s[0] == 'C') ? -1 : atoi(s);
    o->byte  = atoi(colon + 1);
    o->value = atoi(eq + 1);
    if (o->tone >= TONE_COUNT) return false;
    if (o->byte < 0) return false;
    if (o->tone < 0 && o->byte >= TONE_BASE) return false;
    if (o->tone >= 0 && o->byte >= TONE_STRIDE) return false;
    if (o->value < 0 || o->value > 255) return false;
    return true;
}

double peak_db(const std::vector<int16_t> &s) {
    int pk = 0;
    for (int16_t v : s) pk = std::max(pk, std::abs((int)v));
    return 20.0 * std::log10(pk / 32768.0 + 1e-12);
}

}  // namespace

int main(int argc, char **argv) {
    std::string roms_dir, board;
    int patch_index = -1, key = 60, velocity = 127;
    std::vector<Override> sets;
    bool raw = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--roms") && i + 1 < argc) roms_dir = argv[++i];
        else if (!strcmp(argv[i], "--board") && i + 1 < argc) board = argv[++i];
        else if (!strcmp(argv[i], "--patch") && i + 1 < argc) patch_index = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--key") && i + 1 < argc) key = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--velocity") && i + 1 < argc) velocity = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--raw")) raw = true;   // skip preprocess()
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            Override o;
            if (!parse_set(argv[++i], &o)) { fprintf(stderr, "bad --set %s\n", argv[i]); return 1; }
            sets.push_back(o);
        } else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 1; }
    }
    if (roms_dir.empty() || board.empty() || patch_index < 0) {
        fprintf(stderr, "need --roms, --board and --patch\n");
        return 1;
    }

    RomSet roms;
    std::string err;
    if (!roms.load(roms_dir, &err)) { fprintf(stderr, "rom load: %s\n", err.c_str()); return 1; }

    std::vector<Expansion> expansions;
    std::vector<PatchRef> patches;
    const Expansion *sel = nullptr;
    if (board == BOARD_INTERNAL) {
        patches = enumerate_internal(roms);
    } else {
        expansions = scan_expansions(roms_dir + "/expansions");
        for (const auto &e : expansions)
            if (e.usable && e.name == board) { patches = enumerate_expansion(e); sel = &e; break; }
    }
    if (patch_index >= (int)patches.size()) { fprintf(stderr, "patch out of range\n"); return 1; }
    const PatchRef &pr = patches[patch_index];

    LfoDecision d1 = decide_lfo_strip(pr.data, 1);
    LfoDecision d2 = decide_lfo_strip(pr.data, 2);
    std::vector<uint8_t> bytes = raw
        ? std::vector<uint8_t>(pr.data, pr.data + PATCH_SIZE)
        : preprocess(pr.data, d1, d2);

    for (const Override &o : sets) {
        size_t at = (o.tone < 0) ? (size_t)o.byte
                                 : (size_t)TONE_BASE + (size_t)o.tone * TONE_STRIDE + o.byte;
        printf("set %s byte %d: %d -> %d\n",
               o.tone < 0 ? "common" : ("tone" + std::to_string(o.tone)).c_str(),
               o.byte, bytes[at], o.value);
        bytes[at] = (uint8_t)o.value;
    }

    GridSpec g;
    Renderer r;
    if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
    if (sel) r.load_expansion_waves(sel->unscrambled.data(), sel->unscrambled.size());
    r.load_patch_bytes(bytes, g);

    auto s = r.render_note(key, velocity, g);
    printf("%s patch %d '%s'  key %d vel %d%s : peak %.2f dBFS  (%zu frames)\n",
           board.c_str(), patch_index, pr.name.c_str(), key, velocity,
           raw ? " [raw]" : "", peak_db(s), s.size() / 2);
    return 0;
}
