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
        : "/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880";

    std::string err;
    jv::RomSet roms;
    check(roms.load(roms_dir, &err), "ROM set loads");

    auto internal = jv::enumerate_internal(roms);
    check(internal.size() == 192, "192 internal patches");
    check(internal[0].name   == "A.Piano 1",  "internal[0] is A.Piano 1");
    check(internal[64].name  == "Pizzicato",  "internal[64] is Pizzicato");
    check(internal[128].name == "JV Strings", "internal[128] is JV Strings");
    check(internal[0].data != nullptr, "patch data pointer present");

    auto exps = jv::scan_expansions(roms_dir + "/expansions");
    int usable = 0, total = 0;
    for (const auto &e : exps) if (e.usable) { usable++; total += e.patch_count; }
    check(usable == 20,   "20 usable expansion boards");
    check(total  == 4005, "4005 expansion patches");

    for (const auto &e : exps) {
        if (e.name.find("SR-JV80-01") != std::string::npos) {
            check(e.patch_count == 145, "board 01 has 145 patches");
            auto p = jv::enumerate_expansion(e);
            check(p.size() == 145 && p[0].name == "770 Grand 1",
                  "board 01 patch 0 is '770 Grand 1'");
        }
        if (e.name.find("SR-JV80-97") != std::string::npos ||
            e.name.find("SR-JV80-98") != std::string::npos)
            check(!e.usable, "boards 97/98 reported unusable");
    }

    fprintf(stderr, failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
