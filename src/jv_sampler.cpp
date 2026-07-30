// jv_sampler — CLI: render one JV-880 patch's full key/velocity grid to WAV
// files plus a patch.json metadata sidecar, deterministically.
//
// Usage:
//   jv_sampler --roms <dir> --board <name> --out <dir> [--patch N] [--list]
//
// --list prints the available boards (one per line, "<name>\t<patch_count>")
// and exits; Task 7's orchestrator parses this to discover boards.
// Without --patch, every patch on the selected board is rendered.

#include "jv_render.h"
#include "wav.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace jv;

namespace {

// Named so a typo in either the comparison or the --list header line can't
// silently break board selection by disagreeing with each other.
const char *BOARD_INTERNAL = "JV-880 Internal";

const char *NOTE_NAMES[12] = {"C", "C#", "D", "D#", "E", "F",
                              "F#", "G", "G#", "A", "A#", "B"};

std::string note_name(int midi) {
    return std::string(NOTE_NAMES[midi % 12]) + std::to_string(midi / 12 - 1);
}

// Filesystem-safe patch name: keep alphanumerics, space, dash, underscore,
// dot; drop everything else. Patch names in the ROM include things like
// "X/Y/Z", "Slap !!!", "P-P-P-Puff", "*Tr.Rhodes".
std::string sanitize(const std::string &s) {
    std::string o;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == ' ' || c == '-' || c == '_' || c == '.')
            o += (char)c;
    }
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o.empty() ? "patch" : o;
}

// Minimal JSON string escaping so ROM-derived names can never break the
// generated patch.json even if they contain a stray quote/backslash/control
// byte.
std::string json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

// Recursively creates directory p (like `mkdir -p`). Returns false on the
// first real failure (anything but EEXIST — each intermediate segment may
// legitimately already exist). Checking this upfront, before the render,
// means an unwritable output root is caught immediately instead of only
// being discovered after paying for a full settle plus a 75-cell render,
// when the first WAV write fails.
bool mkdirs(const std::string &p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) {
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "failed to create directory %s: %s\n",
                        cur.c_str(), strerror(errno));
                return false;
            }
        }
    }
    return true;
}

void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --roms <dir> [--board <name> --out <dir> [--patch N] | --list]\n",
            prog);
}

} // namespace

int main(int argc, char **argv) {
    std::string roms_dir, board, out_dir;
    int only_patch = -1;
    bool dump_voice = false;
    // Phrase patches need a longer hold than the default grid: a 2-bar loop
    // at 61 BPM runs almost 8 s, and the standard 3.5 s cuts it mid-bar.
    double hold_seconds = -1.0;
    bool do_list = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--roms") && i + 1 < argc) {
            roms_dir = argv[++i];
        } else if (!strcmp(argv[i], "--board") && i + 1 < argc) {
            board = argv[++i];
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (!strcmp(argv[i], "--patch") && i + 1 < argc) {
            only_patch = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--hold") && i + 1 < argc) {
            hold_seconds = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--dump-voice")) {
            dump_voice = true;
        } else if (!strcmp(argv[i], "--list")) {
            do_list = true;
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

    if (do_list) {
        auto internal = enumerate_internal(roms);
        printf("%s\t%zu\n", BOARD_INTERNAL, internal.size());
        for (const auto &e : scan_expansions(roms_dir + "/expansions"))
            if (e.usable) printf("%s\t%d\n", e.name.c_str(), e.patch_count);
        return 0;
    }

    if (board.empty()) {
        fprintf(stderr, "--board is required (see --list)\n");
        usage(argv[0]);
        return 1;
    }
    // --dump-voice reports each patch's voice mode as JSON on stdout and
    // renders nothing. The already-rendered library needs these fields, but
    // rewriting patch.json wholesale would destroy the zone metadata
    // postprocess added (kind, loop points, release) -- so this dumps and a
    // separate merge step folds it in.
    if (out_dir.empty() && !dump_voice) {
        fprintf(stderr, "--out is required\n");
        usage(argv[0]);
        return 1;
    }

    // Expansion storage must outlive `patches`: PatchRef::data points into
    // Expansion::unscrambled, so the Expansion has to stay alive for the
    // whole render loop below, not just this lookup.
    std::vector<Expansion> expansions;
    std::vector<PatchRef> patches;
    const Expansion *selected_expansion = nullptr;
    if (board == BOARD_INTERNAL) {
        patches = enumerate_internal(roms);
    } else {
        expansions = scan_expansions(roms_dir + "/expansions");
        for (const auto &e : expansions) {
            if (e.usable && e.name == board) {
                patches = enumerate_expansion(e);
                selected_expansion = &e;
                break;
            }
        }
        if (patches.empty()) {
            fprintf(stderr, "board not found or unusable: %s\n", board.c_str());
            return 1;
        }
    }

    if (only_patch >= 0 && (size_t)only_patch >= patches.size()) {
        fprintf(stderr, "patch index %d out of range (0..%zu)\n",
                only_patch, patches.size() - 1);
        return 1;
    }

    if (dump_voice) {
        for (size_t pi = 0; pi < patches.size(); pi++) {
            const PatchRef &pr = patches[pi];
            Effects fx = read_effects(pr.data);
            printf("{\"index\":%d,\"bank\":\"%s\",\"name\":\"%s\","
                   "\"voice\":{\"key_assign\":\"%s\",\"solo_legato\":%s,"
                   "\"portamento\":%s,\"portamento_time\":%d,"
                   "\"portamento_mode\":\"%s\"},\"pitch_keyfollow\":[%d,%d,%d,%d],\"tone_level\":[%d,%d,%d,%d]}\n",
                   pr.index, json_escape(pr.bank).c_str(), json_escape(pr.name).c_str(),
                   fx.key_assign ? "Solo" : "Poly",
                   fx.solo_legato ? "true" : "false",
                   fx.portamento ? "true" : "false",
                   fx.portamento_time,
                   fx.portamento_mode ? "Legato" : "Normal",
                   fx.pitch_keyfollow[0], fx.pitch_keyfollow[1],
                   fx.pitch_keyfollow[2], fx.pitch_keyfollow[3],
                   fx.tone_level[0], fx.tone_level[1],
                   fx.tone_level[2], fx.tone_level[3]);
        }
        return 0;
    }

    GridSpec grid;
    if (hold_seconds > 0.0) grid.hold_seconds = hold_seconds;
    Renderer r;
    if (!r.init(roms)) {
        fprintf(stderr, "emulator init failed\n");
        return 1;
    }
    // An expansion patch addresses its waves through PCM banks 3-6, which map
    // to waverom_exp -- a region startSC55() does not fill. Without this the
    // wave numbers resolve against the INTERNAL wave ROM instead, and every
    // expansion patch renders the wrong instrument while still sounding like
    // a plausible, distinct patch. That is exactly how the first full render
    // of all 20 expansion boards came out silently wrong.
    if (selected_expansion) {
        r.load_expansion_waves(selected_expansion->unscrambled.data(),
                               selected_expansion->unscrambled.size());
        fprintf(stderr, "loaded %zu bytes of expansion wave data for %s\n",
                selected_expansion->unscrambled.size(), board.c_str());
    } else {
        r.clear_expansion_waves();
    }

    for (size_t pi = 0; pi < patches.size(); pi++) {
        if (only_patch >= 0 && (int)pi != only_patch) continue;
        const PatchRef &pr = patches[pi];

        Effects fx = read_effects(pr.data);
        LfoDecision d1 = decide_lfo_strip(pr.data, 1);
        LfoDecision d2 = decide_lfo_strip(pr.data, 2);
        std::vector<uint8_t> bytes = preprocess(pr.data, d1, d2);

        // Adaptive velocity layers: derive this patch's own layer count and
        // boundaries from its tones' velocity-switch points (jv_patch.h),
        // instead of always sampling 3 fixed thirds. Read from the
        // original ROM bytes, not `bytes` -- preprocess() never touches
        // the velocity-range bytes, but the ranges are patch identity, not
        // a render-time transform, so reading the source of truth directly
        // is the more obviously-correct choice regardless.
        std::vector<VelocityRegion> regions = compute_velocity_regions(pr.data);

        char idx[16];
        snprintf(idx, sizeof(idx), "%03d", (int)pi);
        std::string pdir = out_dir + "/" + idx + "_" + sanitize(pr.name);
        if (!mkdirs(pdir)) {
            fprintf(stderr, "failed to prepare output directory for patch %03d_%s\n",
                    (int)pi, pr.name.c_str());
            return 1;
        }

        r.load_patch_bytes(bytes, grid);

        std::string zones;
        int zone_count = 0;
        for (int key = grid.lokey; key <= grid.hikey; key += grid.key_step) {
            for (size_t v = 0; v < regions.size(); v++) {
                const VelocityRegion &region = regions[v];
                // Representative velocity: the region's midpoint,
                // floor-biased on odd widths (matches jv_patch.cpp's
                // split_even, which produced this region in the first
                // place) -- simple, deterministic, and centered so a
                // single sample best represents the whole band it covers.
                int velocity = region.lo + (region.hi - region.lo) / 2;

                std::vector<int16_t> pcm = r.render_note(key, velocity, grid);
                int frames = (int)(pcm.size() / 2);

                std::string fn = note_name(key) + "_v" + std::to_string(v + 1) + ".wav";
                if (!wav_write_s16(pdir + "/" + fn, pcm.data(), frames, 2, SAMPLE_RATE)) {
                    fprintf(stderr, "failed to write %s\n", (pdir + "/" + fn).c_str());
                    return 1;
                }

                char zbuf[512];
                snprintf(zbuf, sizeof(zbuf),
                         "%s{\"key\":%d,\"velocity\":%d,\"layer\":%d,\"lovel\":%d,\"hivel\":%d,"
                         "\"frames\":%d,\"file\":\"%s\"}",
                         zones.empty() ? "" : ", ", key, velocity, (int)v + 1,
                         region.lo, region.hi, frames, json_escape(fn).c_str());
                zones += zbuf;
                zone_count++;
            }
        }

        std::string json_path = pdir + "/patch.json";
        FILE *jf = fopen(json_path.c_str(), "w");
        if (!jf) {
            fprintf(stderr, "failed to open patch.json for %03d_%s (%s): %s\n",
                    (int)pi, pr.name.c_str(), json_path.c_str(), strerror(errno));
            return 1;
        }
        // Every downstream task (5, 6, 7) depends on patch.json existing and
        // being complete — a silently-dropped or truncated write here is
        // worse than a loud crash, especially unattended across ~4,197
        // patches. Check fprintf's return (negative means the stream hit an
        // error) AND fclose's return: stdio buffers writes, so a small file
        // like this can have fprintf report success (the data only reached
        // the userspace buffer) while a late error — e.g. ENOSPC — only
        // surfaces when fclose performs the final flush. Either failure
        // means this patch must NOT be reported as complete.
        int written = fprintf(jf,
            "{\n"
            "  \"name\": \"%s\", \"bank\": \"%s\", \"index\": %d, \"sample_rate\": %d,\n"
            "  \"effects\": {\n"
            "    \"reverb\": {\"type\":\"%s\",\"level\":%d,\"time\":%d,\"feedback\":%d},\n"
            "    \"chorus\": {\"type\":\"%s\",\"level\":%d,\"depth\":%d,\"rate\":%d,\"feedback\":%d,\"output\":\"%s\"},\n"
            "    \"reverb_send\": [%d,%d,%d,%d], \"chorus_send\": [%d,%d,%d,%d], \"tone_level\": [%d,%d,%d,%d],\n"
            "    \"bend_up\": %d, \"bend_down\": %d\n"
            "  },\n"
            "  \"voice\": {\"key_assign\":\"%s\",\"solo_legato\":%s,\"portamento\":%s,"
            "\"portamento_time\":%d,\"portamento_mode\":\"%s\"},\n"
            "  \"lfo1\": {\"stripped\":%s,\"reason\":\"%s\",\"form\":\"%s\",\"rate\":%d,\"delay\":%d,\"sync\":%d,\"pitch\":%d,\"tvf\":%d,\"tva\":%d},\n"
            "  \"lfo2\": {\"stripped\":%s,\"reason\":\"%s\",\"form\":\"%s\",\"rate\":%d,\"delay\":%d,\"sync\":%d,\"pitch\":%d,\"tvf\":%d,\"tva\":%d},\n"
            "  \"zones\": [%s]\n"
            "}\n",
            json_escape(pr.name).c_str(), json_escape(pr.bank).c_str(), pr.index, SAMPLE_RATE,
            reverb_type_name(fx.reverb_type), fx.reverb_level, fx.reverb_time, fx.reverb_feedback,
            chorus_type_name(fx.chorus_type), fx.chorus_level, fx.chorus_depth, fx.chorus_rate,
            fx.chorus_feedback, fx.chorus_output ? "Reverb" : "Mix",
            fx.reverb_send[0], fx.reverb_send[1], fx.reverb_send[2], fx.reverb_send[3],
            fx.chorus_send[0], fx.chorus_send[1], fx.chorus_send[2], fx.chorus_send[3],
            fx.tone_level[0], fx.tone_level[1], fx.tone_level[2], fx.tone_level[3],
            fx.bend_up, fx.bend_down,
            fx.key_assign ? "Solo" : "Poly",
            fx.solo_legato ? "true" : "false",
            fx.portamento ? "true" : "false",
            fx.portamento_time,
            fx.portamento_mode ? "Legato" : "Normal",
            d1.strip ? "true" : "false", json_escape(d1.reason).c_str(), lfo_form_name(d1.lfo.form),
            d1.lfo.rate, d1.lfo.delay, d1.lfo.sync, d1.lfo.pitch_depth, d1.lfo.tvf_depth, d1.lfo.tva_depth,
            d2.strip ? "true" : "false", json_escape(d2.reason).c_str(), lfo_form_name(d2.lfo.form),
            d2.lfo.rate, d2.lfo.delay, d2.lfo.sync, d2.lfo.pitch_depth, d2.lfo.tvf_depth, d2.lfo.tva_depth,
            zones.c_str());
        int close_rc = fclose(jf);
        if (written < 0 || close_rc != 0) {
            fprintf(stderr,
                    "failed to write patch.json for %03d_%s (%s): fprintf=%d fclose=%d (%s)\n",
                    (int)pi, pr.name.c_str(), json_path.c_str(), written, close_rc,
                    strerror(errno));
            return 1;
        }

        fprintf(stderr, "rendered %03d_%s (%d zones, %d velocity layer%s)\n",
                (int)pi, pr.name.c_str(), zone_count,
                (int)regions.size(), regions.size() == 1 ? "" : "s");
    }

    return 0;
}
