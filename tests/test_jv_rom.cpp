#include "jv_rom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

static int failures = 0;

static void check(bool cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); failures++; }
    else       { fprintf(stderr, "ok: %s\n", what); }
}

int main(int argc, char **argv) {
    const std::string roms_dir = (argc > 1)
        ? argv[1]
        : "";   // no default: set JV880_ROMS (see README)

    std::string err;
    jv::RomSet roms;
    if (!roms.load(roms_dir, &err)) {
        // A missing/broken ROM directory must produce a clean failing test,
        // never a segfault from downstream enumeration on a partial RomSet.
        fprintf(stderr, "FAIL: ROM set loads (%s)\n", err.c_str());
        fprintf(stderr, "\n1 FAILURES\n");
        return 1;
    }
    check(true, "ROM set loads");

    auto internal = jv::enumerate_internal(roms);
    check(internal.size() == 192, "192 internal patches");
    check(internal[0].name   == "A.Piano 1",  "internal[0] is A.Piano 1");
    check(internal[64].name  == "Pizzicato",  "internal[64] is Pizzicato");
    check(internal[128].name == "JV Strings", "internal[128] is JV Strings");

    // Every internal patch must expose a data pointer, and within each
    // 64-patch bank the pointers must be contiguous PATCH_SIZE strides
    // (the three banks live at different rom2 offsets, so stride only
    // holds within a bank, not across all 192 patches).
    bool all_have_data = true;
    for (const auto &p : internal)
        if (p.data == nullptr) all_have_data = false;
    check(all_have_data, "every internal patch has a data pointer");

    bool stride_ok = true;
    for (int bank = 0; bank < 3; bank++) {
        const uint8_t *base = internal[bank * 64].data;
        for (int i = 0; i < 64; i++)
            if (internal[bank * 64 + i].data != base + i * jv::PATCH_SIZE)
                stride_ok = false;
    }
    check(stride_ok, "internal patch pointers are PATCH_SIZE strides within each bank");

    auto exps = jv::scan_expansions(roms_dir + "/expansions");
    int usable = 0, total = 0;
    for (const auto &e : exps) if (e.usable) { usable++; total += e.patch_count; }
    check(usable == 20,   "20 usable expansion boards");
    check(total  == 4005, "4005 expansion patches");

    bool checked_expansion_data = false;
    for (const auto &e : exps) {
        if (e.name.find("SR-JV80-01") != std::string::npos) {
            check(e.patch_count == 145, "board 01 has 145 patches");
            auto p = jv::enumerate_expansion(e);
            check(p.size() == 145 && p[0].name == "770 Grand 1",
                  "board 01 patch 0 is '770 Grand 1'");

            bool exp_has_data = true;
            for (const auto &pr : p)
                if (pr.data == nullptr) exp_has_data = false;
            check(exp_has_data, "every SR-JV80-01 patch has a data pointer");
            checked_expansion_data = true;
        }
        if (e.name.find("SR-JV80-97") != std::string::npos ||
            e.name.find("SR-JV80-98") != std::string::npos)
            check(!e.usable, "boards 97/98 reported unusable");
    }
    check(checked_expansion_data,
          "at least one usable expansion board's patches were checked for data pointers");

    // ---- Rhythm sets (drum kits) -------------------------------------------
    auto irhy = jv::enumerate_internal_rhythm(roms);
    check(irhy.size() == 3, "three internal rhythm sets (Internal / Preset A / B)");
    if (irhy.size() == 3) {
        check(irhy[0].name == "Internal" && irhy[1].name == "Preset A" &&
              irhy[2].name == "Preset B", "internal rhythm sets are named by bank");
        bool all_data = true;
        for (const auto &r : irhy) if (!r.data) all_data = false;
        check(all_data, "every internal rhythm set has a data pointer");
        // The three must be genuinely different kits, not one kit read three
        // times from the same place -- the exact mistake that would make a
        // wrong offset look like success.
        check(memcmp(irhy[0].data, irhy[1].data, jv::RHYTHM_SET_BYTES) != 0 &&
              memcmp(irhy[1].data, irhy[2].data, jv::RHYTHM_SET_BYTES) != 0,
              "the three internal rhythm sets differ from each other");
    }

    // A rhythm set is 61 keys x 44 bytes. Its first record must have the
    // tone-enable bit set; that is what distinguishes real key data from the
    // 0xFF padding that follows key 61.
    if (!irhy.empty())
        check((irhy[0].data[0] & 0x80) != 0, "rhythm key 0 has its tone-enable bit set");

    // Expansion rhythm sets: the COUNT is authoritative, not the offset.
    // Vocal declares zero sets while still storing a non-zero rhythm offset,
    // so an offset-driven enumerator would fabricate kits for it.
    bool saw_rhythm_board = false, saw_rhythmless_board = false;
    for (const auto &e : exps) {
        auto rs = jv::enumerate_expansion_rhythm(e);
        check((int)rs.size() == (e.usable ? e.rhythm_count : 0),
              ("rhythm set count matches header for " + e.name).c_str());
        if (e.name.find("SR-JV80-06") != std::string::npos) {
            check(e.rhythm_count == 4, "board 06 Dance declares 4 rhythm sets");
            // Independent confirmation of BOTH header fields and the 2684-byte
            // set size: the gap between the rhythm block and the patch block
            // must divide by the set size to exactly the declared count.
            check(e.patches_offset > e.rhythm_offset &&
                  (e.patches_offset - e.rhythm_offset) ==
                      (uint32_t)e.rhythm_count * jv::RHYTHM_SET_BYTES,
                  "board 06 rhythm block size == count x 2684 exactly");
            saw_rhythm_board = true;
        }
        if (e.name.find("SR-JV80-13") != std::string::npos) {
            check(e.rhythm_count == 0 && rs.empty(),
                  "board 13 Vocal declares no rhythm sets despite a non-zero offset");
            saw_rhythmless_board = true;
        }
    }
    check(saw_rhythm_board, "a board with rhythm sets was checked");
    check(saw_rhythmless_board, "a board without rhythm sets was checked");

    fprintf(stderr, failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
