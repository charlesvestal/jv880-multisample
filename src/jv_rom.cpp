#include "jv_rom.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <algorithm>

namespace jv {

static bool read_file(const std::string &path, std::vector<uint8_t> *out,
                      size_t expected, std::string *err) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) {
        if (err) *err = "ftell failed " + path;
        fclose(f);
        return false;
    }
    fseek(f, 0, SEEK_SET);
    if (expected && (size_t)sz != expected) {
        if (err) *err = "size mismatch " + path;
        fclose(f);
        return false;
    }
    out->resize((size_t)sz);
    size_t got = fread(out->data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { if (err) *err = "short read " + path; return false; }
    return true;
}

bool RomSet::load(const std::string &dir, std::string *err) {
    if (!read_file(dir + "/jv880_rom1.bin",     &rom1,     ROM1_BYTES, err)) return false;
    if (!read_file(dir + "/jv880_rom2.bin",     &rom2,     ROM2_BYTES, err)) return false;
    if (!read_file(dir + "/jv880_waverom1.bin", &waverom1, WAVE_BYTES, err)) return false;
    if (!read_file(dir + "/jv880_waverom2.bin", &waverom2, WAVE_BYTES, err)) return false;
    // NVRAM is optional; default to 0xFF fill. If present but the wrong size,
    // say so rather than silently dropping it.
    nvram.assign(NVRAM_BYTES, 0xFF);
    std::vector<uint8_t> nv;
    std::string nv_err;
    if (read_file(dir + "/jv880_nvram.bin", &nv, NVRAM_BYTES, &nv_err)) {
        nvram = std::move(nv);
    } else if (nv_err.rfind("cannot open", 0) != 0) {
        fprintf(stderr, "warning: %s; using default nvram\n", nv_err.c_str());
    }
    return true;
}

std::string trim_patch_name(const uint8_t *patch) {
    std::string s((const char *)patch, 12);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\0')) s.pop_back();
    return s;
}

std::vector<PatchRef> enumerate_internal(const RomSet &roms) {
    struct Bank { const char *name; uint32_t off; };
    static const Bank banks[] = {
        {"A",        0x010ce0},
        {"B",        0x018ce0},
        {"Internal", 0x008ce0},
    };
    // Guard against a partially-loaded RomSet (e.g. caller ignored a false
    // return from RomSet::load): verify every bank fits inside rom2 before
    // doing any pointer arithmetic on it.
    for (const auto &b : banks) {
        size_t need = (size_t)b.off + (size_t)PATCHES_PER_BANK * PATCH_SIZE;
        if (roms.rom2.size() < need) return {};
    }
    std::vector<PatchRef> out;
    for (const auto &b : banks) {
        for (int i = 0; i < PATCHES_PER_BANK; i++) {
            const uint8_t *p = roms.rom2.data() + b.off + (uint32_t)i * PATCH_SIZE;
            PatchRef r;
            r.name  = trim_patch_name(p);
            r.bank  = b.name;
            r.index = i;
            r.data  = p;
            out.push_back(r);
        }
    }
    return out;
}

void unscramble_rom(const uint8_t *src, uint8_t *dst, size_t len) {
    static const int aa[20] = {2, 0, 3, 4, 1, 9, 13, 10, 18, 17,
                               6, 15, 11, 16, 8, 5, 12, 7, 14, 19};
    static const int dd[8]  = {2, 0, 4, 5, 7, 6, 3, 1};
    for (size_t i = 0; i < len; i++) {
        size_t address = i & ~(size_t)0xfffff;
        for (int j = 0; j < 20; j++)
            if (i & ((size_t)1 << j)) address |= (size_t)1 << aa[j];
        uint8_t s = src[address], d = 0;
        for (int j = 0; j < 8; j++)
            if (s & (1 << dd[j])) d |= (uint8_t)(1 << j);
        dst[i] = d;
    }
}

// "SR-JV80-01 Pop - CS 0x3F1CF705.bin" -> "SR-JV80-01 Pop"
// "SR-JV80-99_Experience 0x0FC21498.BIN" -> "SR-JV80-99 Experience"
static std::string board_name_from_filename(const std::string &fn) {
    std::string base = fn;
    size_t slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    size_t cut = base.find(" - CS ");
    if (cut == std::string::npos) {
        cut = base.find_last_of('.');
        size_t hex = base.find(" 0x");
        if (hex != std::string::npos && hex < cut) cut = hex;
    }
    if (cut != std::string::npos) base = base.substr(0, cut);
    while (!base.empty() && base.back() == ' ') base.pop_back();
    std::replace(base.begin(), base.end(), '_', ' ');
    return base;
}

bool load_expansion(const std::string &path, Expansion *out, std::string *err) {
    std::vector<uint8_t> scrambled;
    if (!read_file(path, &scrambled, 0, err)) return false;

    out->path = path;
    out->name = board_name_from_filename(path);
    out->unscrambled.resize(scrambled.size());
    unscramble_rom(scrambled.data(), out->unscrambled.data(), scrambled.size());

    const uint8_t *u = out->unscrambled.data();
    out->patch_count    = (int)u[0x67] | ((int)u[0x66] << 8);
    out->patches_offset = ((uint32_t)u[0x8c] << 24) | ((uint32_t)u[0x8d] << 16) |
                          ((uint32_t)u[0x8e] << 8)  |  (uint32_t)u[0x8f];

    size_t need = (size_t)out->patches_offset +
                  (size_t)out->patch_count * PATCH_SIZE;
    out->usable = out->patch_count > 0 && out->patch_count <= MAX_EXPANSION_PATCHES &&
                  out->patches_offset < out->unscrambled.size() &&
                  need <= out->unscrambled.size();

    // usable == false is a legitimate, expected outcome for some boards
    // (e.g. SR-JV80-97/98 report patch_count == 0) — it is not a load
    // failure, so `err` is left untouched and the function still returns
    // true. Callers that want a diagnostic for unusable boards should check
    // Expansion::usable themselves (scan_expansions does this).
    return true;
}

std::vector<Expansion> scan_expansions(const std::string &dir) {
    std::vector<std::string> files;
    DIR *d = opendir(dir.c_str());
    if (!d) return {};
    while (struct dirent *e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() < 4) continue;
        std::string ext = n.substr(n.size() - 4);
        for (auto &c : ext) c = (char)tolower((unsigned char)c);
        if (ext != ".bin") continue;
        std::string upper = n;
        for (auto &c : upper) c = (char)toupper((unsigned char)c);
        if (upper.find("SR-JV80") == std::string::npos) continue;
        files.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(files.begin(), files.end());

    std::vector<Expansion> out;
    for (const auto &f : files) {
        Expansion e;
        std::string err;
        if (!load_expansion(f, &e, &err)) {
            fprintf(stderr, "skip %s: %s\n", f.c_str(), err.c_str());
            continue;
        }
        if (!e.usable)
            fprintf(stderr, "unusable: %s (patch_count=%d)\n",
                    e.name.c_str(), e.patch_count);
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<PatchRef> enumerate_expansion(const Expansion &exp) {
    std::vector<PatchRef> out;
    if (!exp.usable) return out;
    for (int i = 0; i < exp.patch_count; i++) {
        const uint8_t *p = exp.unscrambled.data() + exp.patches_offset +
                           (size_t)i * PATCH_SIZE;
        PatchRef r;
        r.name  = trim_patch_name(p);
        r.bank  = exp.name;
        r.index = i;
        r.data  = p;
        out.push_back(r);
    }
    return out;
}

} // namespace jv
