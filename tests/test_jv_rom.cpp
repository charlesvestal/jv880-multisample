#include "jv_rom.h"
#include <stdio.h>
#include <stdlib.h>
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

    fprintf(stderr, failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
