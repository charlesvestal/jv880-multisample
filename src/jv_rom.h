#pragma once
#include <stdint.h>
#include <string>
#include <vector>

namespace jv {

static const int    PATCH_SIZE   = 0x16a;   // 362 = 26 common + 4 * 84 tone
static const size_t ROM1_BYTES   = 0x8000;
static const size_t ROM2_BYTES   = 0x40000;
static const size_t WAVE_BYTES   = 0x200000;
static const size_t NVRAM_BYTES  = 0x8000;

struct PatchRef {
    std::string name;       // trimmed 12-char ROM name
    std::string bank;       // "A", "B", "Internal", or board name
    int         index = 0;  // index within its bank
    const uint8_t *data = nullptr;  // PATCH_SIZE bytes
};

struct Expansion {
    std::string          name;        // e.g. "SR-JV80-01 Pop"
    std::string          path;
    bool                 usable = false;
    int                  patch_count = 0;
    uint32_t             patches_offset = 0;
    std::vector<uint8_t> unscrambled;
};

struct RomSet {
    std::vector<uint8_t> rom1, rom2, waverom1, waverom2, nvram;
    bool load(const std::string &dir, std::string *err);
};

std::vector<PatchRef> enumerate_internal(const RomSet &roms);
bool load_expansion(const std::string &path, Expansion *out, std::string *err);
std::vector<Expansion> scan_expansions(const std::string &dir);
std::vector<PatchRef> enumerate_expansion(const Expansion &exp);
void unscramble_rom(const uint8_t *src, uint8_t *dst, size_t len);
std::string trim_patch_name(const uint8_t *patch);

} // namespace jv
