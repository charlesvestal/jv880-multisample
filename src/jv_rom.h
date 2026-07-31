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

// ---- Rhythm Sets (drum kits) -----------------------------------------------
//
// Rhythm Sets live in a DIFFERENT ROM region from Patches, which is why
// enumerating patches never yields a drum kit. One set covers 61 keys (MIDI
// 36..96), each key holding a single 44-byte rhythm tone whose byte 1 is a
// wave number -- the internal set decodes to a GM-style map (36 Bright Kick,
// 40 LA Snare, 44 Closed HAT).
//
// Rhythm sets carry NO name field. Scanning an expansion's whole tail region
// finds only the patch name table; the rhythm block sits in a stretch with no
// text at all. Names below are therefore synthesized from bank and index.
static const int      RHYTHM_KEYS         = 61;
static const int      RHYTHM_LOW_KEY      = 36;      // C2
static const int      RHYTHM_TONE_SIZE    = 44;
static const int      RHYTHM_SET_BYTES    = RHYTHM_KEYS * RHYTHM_TONE_SIZE;  // 2684
static const int      MAX_EXPANSION_RHYTHM = 16;     // sanity bound

// Internal rhythm sets: one per patch bank, immediately after that bank's
// patches.
static const uint32_t ROM_RHYTHM_INTERNAL = 0x00e760;
static const uint32_t ROM_RHYTHM_PRESET_A = 0x016760;
static const uint32_t ROM_RHYTHM_PRESET_B = 0x01e760;

struct Expansion {
    std::string          name;        // e.g. "SR-JV80-01 Pop"
    std::string          path;
    bool                 usable = false;
    int                  patch_count = 0;
    uint32_t             patches_offset = 0;
    // Rhythm sets, when the board has any. The COUNT is authoritative, not
    // the offset: SR-JV80-13 Vocal stores a non-zero rhythm offset while
    // declaring zero sets, so trusting the offset alone invents kits that do
    // not exist. 13 of 22 boards carry rhythm sets, 49 in total.
    int                  rhythm_count = 0;
    uint32_t             rhythm_offset = 0;
    std::vector<uint8_t> unscrambled;
};

// Same pointer-lifetime contract as PatchRef: `data` points into memory owned
// by the RomSet or Expansion that produced it.
struct RhythmRef {
    std::string name;       // synthesized, e.g. "Internal" or "Dance 3"
    std::string bank;       // "Internal" or board name
    int         index = 0;
    const uint8_t *data = nullptr;   // RHYTHM_SET_BYTES bytes
};

struct RomSet {
    std::vector<uint8_t> rom1, rom2, waverom1, waverom2, nvram;
    bool load(const std::string &dir, std::string *err);
};

std::vector<PatchRef> enumerate_internal(const RomSet &roms);
bool load_expansion(const std::string &path, Expansion *out, std::string *err);
std::vector<Expansion> scan_expansions(const std::string &dir);
std::vector<PatchRef> enumerate_expansion(const Expansion &exp);

// The three internal rhythm sets (Internal, Preset A, Preset B), and any an
// expansion board declares. Both return an empty vector rather than failing
// when the source has none.
std::vector<RhythmRef> enumerate_internal_rhythm(const RomSet &roms);
std::vector<RhythmRef> enumerate_expansion_rhythm(const Expansion &exp);
void unscramble_rom(const uint8_t *src, uint8_t *dst, size_t len);
std::string trim_patch_name(const uint8_t *patch);

} // namespace jv
