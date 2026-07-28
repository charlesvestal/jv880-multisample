#pragma once
#include <stdint.h>
#include <string>
#include <vector>

namespace jv {

static const int    PATCH_SIZE            = 0x16a;   // 362 = 26 common + 4 * 84 tone
static const size_t ROM1_BYTES            = 0x8000;
static const size_t ROM2_BYTES            = 0x40000;
static const size_t WAVE_BYTES            = 0x200000;
static const size_t NVRAM_BYTES           = 0x8000;
static const int    PATCHES_PER_BANK      = 64;   // Preset A / Preset B / Internal
static const int    MAX_EXPANSION_PATCHES = 256;  // sanity bound on a parsed patch_count

// PatchRef::data points into memory owned by the RomSet (for internal
// patches) or Expansion (for expansion-board patches) that produced it.
// The pointer is valid only as long as that owning object is alive and has
// not been reloaded, reassigned, or moved from. In particular, taking
// PatchRef's from a temporary Expansion (e.g. an rvalue that is never
// stored) leaves them dangling once the temporary is destroyed.
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
