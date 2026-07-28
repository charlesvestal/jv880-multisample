# JV-880 Multisample Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render 4,197 Roland JV-880 patches into DecentSampler and SFZ multisample libraries, one library per expansion board, using the headless emulator core.

**Architecture:** A C++ batch renderer links the existing JV-880 emulator sources (`mcu.cpp`, `mcu_opcodes.cpp`, `pcm.cpp`) from the `schwung-jv880` repo and drives them deterministically — no threads, no wall clock. Patch bytes are preprocessed in NVRAM before rendering (effects zeroed, LFOs conditionally stripped, portamento forced off), then each key/velocity cell is rendered to raw WAV. Python stages then resample, detect loops, encode FLAC, and emit both preset formats from a shared metadata model. Stages communicate through files so any stage can re-run without redoing the ones before it.

**Tech Stack:** C++17 (clang 17), Python 3.14 with numpy 2.4 / scipy 1.17 / soundfile 0.13, flac 1.5.0, ffmpeg 8.0.1.

**User decisions (already made):**
- Scope: everything — 192 internal patches plus all expansion boards.
- Grid: 3 semitones x 3 velocity layers.
- Target is desktop DecentSampler / SFZ players, **not** the Schwung Multisampler on Move. Design for fidelity and flexibility.
- Render **fully dry**; capture reverb *and* chorus settings per preset and faithfully recreate them in DecentSampler.
- LFOs: conditional strip — recreate in DS where representable, bake where not.
- Loops: smart auto-loop for sustainers plus release tails.
- Libraries organized per board.

**Reference spec:** `docs/superpowers/specs/2026-07-28-jv880-multisample-design.md`

---

## Verified Facts (measured, not assumed)

These were confirmed on 2026-07-28 before planning. Do not re-derive them.

**Emulator harness** — `tools/render_test/render_test.cpp` in `schwung-jv880` builds in ~2s and renders 5.75s of audio in 0.28s user time (~20x realtime).

**Paths:**
- Emulator sources: `/Volumes/ExtFS/charlesvestal/github/schwung-parent/schwung-jv880/src/dsp`
- Reference harness: `/Volumes/ExtFS/charlesvestal/github/schwung-parent/schwung-jv880/tools/render_test/render_test.cpp`
- ROMs: `/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880`
- Output root: `/Volumes/ExtFS/charlesvestal/JV-880 Multisamples` (**never** the internal drive — 22 GB free)

**Patch memory layout** (verified against `jv880_plugin.cpp`):
- `PATCH_SIZE = 0x16a` (362) = 26 bytes patch-common + 4 tones x 84 bytes.
- `NVRAM_PATCH_OFFSET = 0x0d70`, `NVRAM_MODE_OFFSET = 0x11` (1 = patch mode).
- Tone base: `NVRAM_PATCH_OFFSET + 26 + (toneIdx * 84)`.
- ROM2 bank offsets: Preset A `0x010ce0`, Preset B `0x018ce0`, Internal `0x008ce0`, 64 patches each.
- Patch name: 12 ASCII bytes at patch offset 0.

**Patch-common offsets** (`PatchCommonParam` = name, sysex, **nvramOffset**, shift, width, min, max):

| Param | Offset | Shift | Width |
|---|---|---|---|
| `reverbtype` | 12 | 0 | 4 |
| `chorustype` | 12 | 4 | 2 |
| `reverblevel` | 13 | 0 | 7 |
| `reverbtime` | 14 | 0 | 7 |
| `reverbfeedback` | 15 | 0 | 7 |
| `choruslevel` | 16 | 0 | 7 |
| `chorusoutput` | 16 | 7 | 1 |
| `chorusdepth` | 17 | 0 | 7 |
| `chorusrate` | 18 | 0 | 7 |
| `chorusfeedback` | 19 | 0 | 7 |
| `portamentoswitch` | 24 | 6 | 1 |
| `bendrangeup` | 24 | 0 | 4 |
| `bendrangedown` | 23 | 0 | 7 |

**Tone offsets** (`ToneParamEntry` = name, **nvram_offset within tone**, sysex_idx, …). The first numeric column is the NVRAM offset; the second is the SysEx index. Do not confuse them.

| Param | Tone offset | Notes |
|---|---|---|
| `lfo1form` | 23 | shift 0, mask 0x07 |
| `lfo1offset` | 23 | shift 3, mask 0x07 |
| `lfo1synchro` | 23 | bit 6 |
| `lfo1rate` | 24 | |
| `lfo1delay` | 25 | |
| `lfo1fadetime` | 26 | |
| `lfo2form` | 27 | shift 0, mask 0x07 |
| `lfo2rate` | 28 | |
| `lfo2delay` | 29 | |
| `lfo2fadetime` | 30 | |
| `lfo1pitchdepth` | 31 | signed int8 |
| `lfo1tvfdepth` | 32 | signed int8 |
| `lfo1tvadepth` | 33 | signed int8 |
| `lfo2pitchdepth` | 34 | signed int8 |
| `lfo2tvfdepth` | 35 | signed int8 |
| `lfo2tvadepth` | 36 | signed int8 |
| `cutofffrequency` | 52 | |
| `level` | 67 | tone output level; 0 = inactive |
| `drylevel` | 81 | |
| `reverbsendlevel` | 82 | |
| `chorussendlevel` | 83 | |

Enums: `reverbtype` = Room1, Room2, Stage1, Stage2, Hall1, Hall2, Delay, Pan-Dly. `chorustype` = Chorus1, Chorus2, Chorus3. LFO `form` = TRI, SIN, SAW, SQU, RND1, RND2.

**Expansion ROM format** (validated against all 22 boards):
- Unscramble with the address permutation `aa = [2,0,3,4,1,9,13,10,18,17,6,15,11,16,8,5,12,7,14,19]` and data-bit permutation `dd = [2,0,4,5,7,6,3,1]` (from `jv880_plugin.cpp:496`).
- `patch_count` = `u[0x67] | (u[0x66] << 8)`.
- `patches_offset` = big-endian u32 at `u[0x8c..0x8f]`.
- Patches are `0x16a`-strided from `patches_offset`, 12-byte name at each start.
- **20 of 22 boards parse cleanly, totalling 4,005 patches.** Boards 97 (Experience III) and 98 (Experience II) report `patch_count = 0` and are excluded.

**MCU interface** (`mcu.h`): `startSC55(rom1, rom2, waverom1, waverom2, nvram)`, `updateSC55(nSamples)`, `postMidiSC55(msg, len)`, `SC55_Reset()`, `uint8_t nvram[]`, `int16_t sample_buffer[4096]`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/jv_rom.h` / `.cpp` | ROM loading, internal bank enumeration, expansion unscramble + patch table parsing. Knows nothing about rendering. |
| `src/jv_patch.h` / `.cpp` | Patch byte reading/writing: effect params, per-tone LFO params, active-tone detection, strip decision, preprocessing mutations. Pure byte manipulation, no emulator. |
| `src/jv_render.h` / `.cpp` | Emulator lifecycle and grid rendering: warmup, patch load, note render, decay truncation. |
| `src/jv_sampler.cpp` | CLI entry point wiring the above; renders one board (or one patch range) to raw WAV + `patch.json`. |
| `src/wav.h` / `.cpp` | Minimal WAV read/write (16-bit stereo at 64 kHz in, used by both renderer and calibration). |
| `tools/calibrate.cpp` | Renders effect parameter sweeps to WAV for measurement. |
| `tools/analyze_calibration.py` | Measures sweeps, writes `calibration.json`. |
| `tools/postprocess.py` | Resample to 48 kHz, loop detection, release measurement, FLAC encode, per-patch metadata. |
| `tools/emit_presets.py` | Generate `.dspreset` and `.sfz` from metadata + calibration. |
| `tools/run_batch.py` | Parallel orchestration across boards/cores. |
| `tests/` | pytest for Python stages; C++ assertions run as standalone binaries. |
| `CMakeLists.txt` | Build for all C++ targets. |

---

### Task 1: ROM loading and patch enumeration

**Goal:** A library that enumerates every patch (internal + expansion) with correct names and counts.

**Files:**
- Create: `src/jv_rom.h`, `src/jv_rom.cpp`
- Create: `tests/test_jv_rom.cpp`
- Create: `CMakeLists.txt`

**Acceptance Criteria:**
- [ ] Internal enumeration returns exactly 192 patches; index 0 is `A.Piano 1`, index 64 is `Pizzicato`, index 128 is `JV Strings`.
- [ ] Expansion scan of the ROM directory returns 20 usable boards totalling 4,005 patches.
- [ ] SR-JV80-01 reports 145 patches with patch 0 named `770 Grand 1`.
- [ ] Boards 97 and 98 are reported as unusable, not crashed on.
- [ ] Every returned patch exposes a 362-byte data pointer.

**Verify:** `cmake --build build --target test_jv_rom && ./build/test_jv_rom` → `ALL TESTS PASSED`

**Steps:**

- [ ] **Step 1: Write `src/jv_rom.h`**

```cpp
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

// Raw ROM images required to boot the emulator.
struct RomSet {
    std::vector<uint8_t> rom1, rom2, waverom1, waverom2, nvram;
    bool load(const std::string &dir, std::string *err);
};

// 192 internal patches: Preset A (0-63), Preset B (64-127), Internal (128-191).
std::vector<PatchRef> enumerate_internal(const RomSet &roms);

// Unscramble one expansion image and parse its patch table.
bool load_expansion(const std::string &path, Expansion *out, std::string *err);

// Scan a directory for SR-JV80 images, sorted by filename.
std::vector<Expansion> scan_expansions(const std::string &dir);

std::vector<PatchRef> enumerate_expansion(const Expansion &exp);

void unscramble_rom(const uint8_t *src, uint8_t *dst, size_t len);

std::string trim_patch_name(const uint8_t *patch);

} // namespace jv
```

- [ ] **Step 2: Write the failing test `tests/test_jv_rom.cpp`**

```cpp
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
```

- [ ] **Step 3: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.20)
project(jv880_multisample CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(JV_DSP "/Volumes/ExtFS/charlesvestal/github/schwung-parent/schwung-jv880/src/dsp"
    CACHE PATH "Path to schwung-jv880 emulator sources")

add_library(jv_core STATIC
    src/jv_rom.cpp src/jv_patch.cpp src/wav.cpp)
target_include_directories(jv_core PUBLIC src)

add_library(jv_emu STATIC
    ${JV_DSP}/mcu.cpp ${JV_DSP}/mcu_opcodes.cpp ${JV_DSP}/pcm.cpp)
target_include_directories(jv_emu PUBLIC ${JV_DSP})

add_executable(test_jv_rom tests/test_jv_rom.cpp)
target_link_libraries(test_jv_rom jv_core)

add_executable(test_jv_patch tests/test_jv_patch.cpp)
target_link_libraries(test_jv_patch jv_core)

add_library(jv_render STATIC src/jv_render.cpp)
target_link_libraries(jv_render jv_core jv_emu)

add_executable(jv_sampler src/jv_sampler.cpp)
target_link_libraries(jv_sampler jv_render)

add_executable(calibrate tools/calibrate.cpp)
target_link_libraries(calibrate jv_render)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cmake -B build -S . && cmake --build build --target test_jv_rom`
Expected: FAIL — link errors for undefined `jv::RomSet::load` etc., because `jv_rom.cpp` does not exist yet.

- [ ] **Step 5: Implement `src/jv_rom.cpp`**

```cpp
#include "jv_rom.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

namespace jv {

static bool read_file(const std::string &path, std::vector<uint8_t> *out,
                      size_t expected, std::string *err) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
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
    // NVRAM is optional; default to 0xFF fill.
    nvram.assign(NVRAM_BYTES, 0xFF);
    std::vector<uint8_t> nv;
    if (read_file(dir + "/jv880_nvram.bin", &nv, 0, nullptr) && nv.size() == NVRAM_BYTES)
        nvram = nv;
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
    std::vector<PatchRef> out;
    for (const auto &b : banks) {
        for (int i = 0; i < 64; i++) {
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
static std::string board_name_from_filename(const std::string &fn) {
    std::string base = fn;
    size_t slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    size_t cut = base.find(" - CS ");
    if (cut == std::string::npos) {
        cut = base.find_last_of('.');
        // Trailing checksum with no " - CS " separator, e.g. "..._Experience 0x0FC21498.BIN"
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

    // A board is usable only if the table is sane AND fits inside the image.
    size_t need = (size_t)out->patches_offset +
                  (size_t)out->patch_count * PATCH_SIZE;
    out->usable = out->patch_count > 0 && out->patch_count <= 256 &&
                  out->patches_offset < out->unscrambled.size() &&
                  need <= out->unscrambled.size();

    if (!out->usable && err)
        *err = out->name + ": unusable (patch_count=" +
               std::to_string(out->patch_count) + ")";
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
        for (auto &c : ext) c = (char)tolower(c);
        if (ext != ".bin") continue;
        if (n.find("SR-JV80") == std::string::npos) continue;
        files.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(files.begin(), files.end());

    std::vector<Expansion> out;
    for (const auto &f : files) {
        Expansion e;
        std::string err;
        if (load_expansion(f, &e, &err)) out.push_back(std::move(e));
        else fprintf(stderr, "skip %s: %s\n", f.c_str(), err.c_str());
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cmake --build build --target test_jv_rom && ./build/test_jv_rom`
Expected: PASS — `ALL TESTS PASSED`, including `20 usable expansion boards` and `4005 expansion patches`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/jv_rom.h src/jv_rom.cpp tests/test_jv_rom.cpp
git commit -m "feat: ROM loading and patch enumeration for internal + expansion banks"
```

---

### Task 2: Patch preprocessing — effects capture, LFO strip decision, portamento

**Goal:** Read every effect and LFO parameter from a patch, decide whether LFOs can be stripped, and produce the mutated patch bytes used for dry rendering.

**Files:**
- Create: `src/jv_patch.h`, `src/jv_patch.cpp`
- Create: `tests/test_jv_patch.cpp`

**Acceptance Criteria:**
- [ ] `read_effects()` returns reverb type/level/time/feedback and chorus type/level/depth/rate/feedback/output from the documented offsets.
- [ ] `read_tone_lfo()` returns per-tone LFO form/rate/delay/fade/sync and signed pitch/TVF/TVA depths.
- [ ] A tone with `level == 0` is not counted active.
- [ ] `decide_lfo_strip()` returns strip=true only when all active tones match within tolerance (waveform identical, rate +/-4, each depth +/-6 with matching sign, sync identical) and form is not RND1/RND2.
- [ ] `preprocess()` zeroes `reverblevel` and `choruslevel`, clears `portamentoswitch`, and zeroes the six LFO depth bytes on every tone when and only when strip was decided.
- [ ] `preprocess()` never modifies bits outside the documented fields (verified byte-diff).

**Verify:** `cmake --build build --target test_jv_patch && ./build/test_jv_patch` → `ALL TESTS PASSED`

**Steps:**

- [ ] **Step 1: Write `src/jv_patch.h`**

```cpp
#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "jv_rom.h"

namespace jv {

static const int TONE_COUNT  = 4;
static const int TONE_STRIDE = 84;
static const int TONE_BASE   = 26;   // offset of tone 0 within the patch

struct Effects {
    int reverb_type = 0, reverb_level = 0, reverb_time = 0, reverb_feedback = 0;
    int chorus_type = 0, chorus_level = 0, chorus_depth = 0, chorus_rate = 0;
    int chorus_feedback = 0, chorus_output = 0;  // output: 0 = Mix, 1 = Reverb
    int bend_up = 0, bend_down = 0;
    int portamento = 0;
    // Per-tone sends, indexed by tone.
    int reverb_send[TONE_COUNT] = {0, 0, 0, 0};
    int chorus_send[TONE_COUNT] = {0, 0, 0, 0};
    int tone_level[TONE_COUNT]  = {0, 0, 0, 0};
};

struct ToneLfo {
    int form = 0, rate = 0, delay = 0, fade = 0, sync = 0;
    int pitch_depth = 0, tvf_depth = 0, tva_depth = 0;   // signed -63..63
    bool any_depth() const {
        return pitch_depth != 0 || tvf_depth != 0 || tva_depth != 0;
    }
};

struct LfoDecision {
    bool strip = false;
    std::string reason;
    // Representative (mean of active tones) values used when strip is true.
    ToneLfo lfo;
};

Effects    read_effects(const uint8_t *patch);
ToneLfo    read_tone_lfo(const uint8_t *patch, int tone, int lfo_index /*1 or 2*/);
bool       tone_active(const uint8_t *patch, int tone);
LfoDecision decide_lfo_strip(const uint8_t *patch, int lfo_index);

// Produce render-ready bytes: dry, portamento off, LFOs stripped per decision.
std::vector<uint8_t> preprocess(const uint8_t *patch,
                                const LfoDecision &lfo1,
                                const LfoDecision &lfo2);

const char *reverb_type_name(int t);
const char *chorus_type_name(int t);
const char *lfo_form_name(int f);

} // namespace jv
```

- [ ] **Step 2: Write the failing test `tests/test_jv_patch.cpp`**

```cpp
#include "jv_patch.h"
#include <stdio.h>
#include <string.h>
#include <string>

static int failures = 0;
static void check(bool c, const char *what) {
    if (!c) { fprintf(stderr, "FAIL: %s\n", what); failures++; }
    else    { fprintf(stderr, "ok: %s\n", what); }
}

int main(int argc, char **argv) {
    const std::string roms_dir = (argc > 1)
        ? argv[1]
        : "/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880";
    std::string err;
    jv::RomSet roms;
    check(roms.load(roms_dir, &err), "ROMs load");
    auto patches = jv::enumerate_internal(roms);

    // Synthetic patch: full control over every field under test.
    uint8_t p[jv::PATCH_SIZE];
    memset(p, 0, sizeof(p));
    p[13] = 100;                 // reverblevel
    p[16] = 64;                  // choruslevel (bit 7 = chorusoutput = 0)
    p[14] = 70;                  // reverbtime
    p[12] = (uint8_t)(4 | (1 << 4));  // reverbtype=4 (Hall1), chorustype=1
    p[24] = (uint8_t)(1 << 6);   // portamentoswitch on

    jv::Effects fx = jv::read_effects(p);
    check(fx.reverb_type == 4,   "reverbtype decodes from bits 0-3");
    check(fx.chorus_type == 1,   "chorustype decodes from bits 4-5");
    check(fx.reverb_level == 100, "reverblevel reads offset 13");
    check(fx.reverb_time == 70,  "reverbtime reads offset 14");
    check(fx.chorus_level == 64, "choruslevel masks off bit 7");
    check(fx.portamento == 1,    "portamentoswitch reads bit 6 of offset 24");

    // Two active tones with identical LFO1 -> strippable.
    for (int t = 0; t < 2; t++) {
        uint8_t *tone = p + jv::TONE_BASE + t * jv::TONE_STRIDE;
        tone[67] = 100;                    // level -> active
        tone[23] = 1;                      // form = SIN
        tone[24] = 60;                     // rate
        tone[33] = (uint8_t)(int8_t)20;    // lfo1tvadepth
    }
    jv::LfoDecision d1 = jv::decide_lfo_strip(p, 1);
    check(d1.strip, "identical LFO1 across active tones is strippable");
    check(d1.lfo.tva_depth == 20, "representative depth preserved");

    // Diverging rate beyond tolerance -> not strippable.
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 24] = 90;
    check(!jv::decide_lfo_strip(p, 1).strip, "diverging rate blocks strip");
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 24] = 60;

    // RND1 waveform -> not strippable even when identical.
    p[jv::TONE_BASE + 0 * jv::TONE_STRIDE + 23] = 4;
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 23] = 4;
    check(!jv::decide_lfo_strip(p, 1).strip, "RND1 blocks strip");
    p[jv::TONE_BASE + 0 * jv::TONE_STRIDE + 23] = 1;
    p[jv::TONE_BASE + 1 * jv::TONE_STRIDE + 23] = 1;

    // Inactive tone must not veto the decision.
    uint8_t *t3 = p + jv::TONE_BASE + 3 * jv::TONE_STRIDE;
    t3[67] = 0;         // inactive
    t3[24] = 5;         // wildly different rate
    check(!jv::tone_active(p, 3), "level 0 tone is inactive");
    check(jv::decide_lfo_strip(p, 1).strip, "inactive tone ignored");

    // preprocess() mutations.
    jv::LfoDecision d2 = jv::decide_lfo_strip(p, 2);
    auto out = jv::preprocess(p, jv::decide_lfo_strip(p, 1), d2);
    check(out.size() == jv::PATCH_SIZE, "preprocess returns full patch");
    check(out[13] == 0, "reverblevel zeroed");
    check((out[16] & 0x7f) == 0, "choruslevel zeroed");
    check((out[24] & (1 << 6)) == 0, "portamento cleared");
    check((out[12] & 0x0f) == 4, "reverbtype preserved (type kept for metadata)");
    for (int t = 0; t < 2; t++)
        check(out[jv::TONE_BASE + t * jv::TONE_STRIDE + 33] == 0,
              "lfo1 tva depth stripped");

    // Real ROM patches must all decode without crashing.
    int strippable = 0;
    for (const auto &pr : patches)
        if (jv::decide_lfo_strip(pr.data, 1).strip) strippable++;
    fprintf(stderr, "info: %d/192 internal patches strippable on LFO1\n", strippable);
    check(strippable >= 0 && strippable <= 192, "strip decision runs on all ROM patches");

    fprintf(stderr, failures ? "\n%d FAILURES\n" : "\nALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cmake --build build --target test_jv_patch`
Expected: FAIL — undefined references to `jv::read_effects`, `jv::decide_lfo_strip`, `jv::preprocess`.

- [ ] **Step 4: Implement `src/jv_patch.cpp`**

```cpp
#include "jv_patch.h"
#include <math.h>
#include <string.h>

namespace jv {

static int bits(const uint8_t *p, int off, int shift, int width) {
    return (p[off] >> shift) & ((1 << width) - 1);
}

static const uint8_t *tone_ptr(const uint8_t *patch, int tone) {
    return patch + TONE_BASE + tone * TONE_STRIDE;
}

bool tone_active(const uint8_t *patch, int tone) {
    return tone_ptr(patch, tone)[67] > 0;   // tone `level`
}

Effects read_effects(const uint8_t *p) {
    Effects e;
    e.reverb_type     = bits(p, 12, 0, 4);
    e.chorus_type     = bits(p, 12, 4, 2);
    e.reverb_level    = bits(p, 13, 0, 7);
    e.reverb_time     = bits(p, 14, 0, 7);
    e.reverb_feedback = bits(p, 15, 0, 7);
    e.chorus_level    = bits(p, 16, 0, 7);
    e.chorus_output   = bits(p, 16, 7, 1);
    e.chorus_depth    = bits(p, 17, 0, 7);
    e.chorus_rate     = bits(p, 18, 0, 7);
    e.chorus_feedback = bits(p, 19, 0, 7);
    e.bend_down       = bits(p, 23, 0, 7);
    e.bend_up         = bits(p, 24, 0, 4);
    e.portamento      = bits(p, 24, 6, 1);
    for (int t = 0; t < TONE_COUNT; t++) {
        const uint8_t *tp = tone_ptr(p, t);
        e.tone_level[t]  = tp[67];
        e.reverb_send[t] = tp[82];
        e.chorus_send[t] = tp[83];
    }
    return e;
}

ToneLfo read_tone_lfo(const uint8_t *patch, int tone, int lfo_index) {
    const uint8_t *tp = tone_ptr(patch, tone);
    ToneLfo l;
    if (lfo_index == 1) {
        l.form  = (tp[23] >> 0) & 0x07;
        l.sync  = (tp[23] >> 6) & 0x01;
        l.rate  = tp[24];
        l.delay = tp[25];
        l.fade  = tp[26];
        l.pitch_depth = (int8_t)tp[31];
        l.tvf_depth   = (int8_t)tp[32];
        l.tva_depth   = (int8_t)tp[33];
    } else {
        l.form  = (tp[27] >> 0) & 0x07;
        l.sync  = 0;                    // LFO2 has no sync bit in the table
        l.rate  = tp[28];
        l.delay = tp[29];
        l.fade  = tp[30];
        l.pitch_depth = (int8_t)tp[34];
        l.tvf_depth   = (int8_t)tp[35];
        l.tva_depth   = (int8_t)tp[36];
    }
    return l;
}

static bool same_sign(int a, int b) {
    if (a == 0 && b == 0) return true;
    return (a >= 0) == (b >= 0);
}

LfoDecision decide_lfo_strip(const uint8_t *patch, int lfo_index) {
    LfoDecision d;
    std::vector<ToneLfo> active;
    for (int t = 0; t < TONE_COUNT; t++)
        if (tone_active(patch, t)) active.push_back(read_tone_lfo(patch, t, lfo_index));

    if (active.empty())            { d.reason = "no active tones";      return d; }

    bool any = false;
    for (const auto &l : active) if (l.any_depth()) any = true;
    if (!any)                      { d.reason = "no LFO depth";         return d; }

    // RND1 (4) and RND2 (5) have no DecentSampler equivalent.
    for (const auto &l : active)
        if (l.form >= 4)           { d.reason = "random waveform";      return d; }

    const ToneLfo &ref = active[0];
    for (const auto &l : active) {
        if (l.form != ref.form)                 { d.reason = "waveform mismatch"; return d; }
        if (l.sync != ref.sync)                 { d.reason = "sync mismatch";     return d; }
        if (abs(l.rate - ref.rate) > 4)         { d.reason = "rate mismatch";     return d; }
        if (abs(l.pitch_depth - ref.pitch_depth) > 6 ||
            !same_sign(l.pitch_depth, ref.pitch_depth)) { d.reason = "pitch depth mismatch"; return d; }
        if (abs(l.tvf_depth - ref.tvf_depth) > 6 ||
            !same_sign(l.tvf_depth, ref.tvf_depth))     { d.reason = "tvf depth mismatch";   return d; }
        if (abs(l.tva_depth - ref.tva_depth) > 6 ||
            !same_sign(l.tva_depth, ref.tva_depth))     { d.reason = "tva depth mismatch";   return d; }
    }

    // Representative = mean over active tones.
    long rate = 0, del = 0, fade = 0, pd = 0, fd = 0, ad = 0;
    for (const auto &l : active) {
        rate += l.rate; del += l.delay; fade += l.fade;
        pd += l.pitch_depth; fd += l.tvf_depth; ad += l.tva_depth;
    }
    int n = (int)active.size();
    d.lfo = ref;
    d.lfo.rate  = (int)(rate / n);
    d.lfo.delay = (int)(del  / n);
    d.lfo.fade  = (int)(fade / n);
    d.lfo.pitch_depth = (int)(pd / n);
    d.lfo.tvf_depth   = (int)(fd / n);
    d.lfo.tva_depth   = (int)(ad / n);
    d.strip  = true;
    d.reason = "strippable";
    return d;
}

std::vector<uint8_t> preprocess(const uint8_t *patch,
                                const LfoDecision &lfo1,
                                const LfoDecision &lfo2) {
    std::vector<uint8_t> out(patch, patch + PATCH_SIZE);

    out[13] = (uint8_t)(out[13] & ~0x7f);           // reverblevel = 0
    out[16] = (uint8_t)(out[16] & ~0x7f);           // choruslevel = 0, keep bit 7
    out[24] = (uint8_t)(out[24] & ~(1 << 6));       // portamento off

    for (int t = 0; t < TONE_COUNT; t++) {
        uint8_t *tp = out.data() + TONE_BASE + t * TONE_STRIDE;
        if (lfo1.strip) { tp[31] = 0; tp[32] = 0; tp[33] = 0; }
        if (lfo2.strip) { tp[34] = 0; tp[35] = 0; tp[36] = 0; }
    }
    return out;
}

const char *reverb_type_name(int t) {
    static const char *n[] = {"Room1","Room2","Stage1","Stage2",
                              "Hall1","Hall2","Delay","Pan-Dly"};
    return (t >= 0 && t < 8) ? n[t] : "Room1";
}
const char *chorus_type_name(int t) {
    static const char *n[] = {"Chorus1","Chorus2","Chorus3"};
    return (t >= 0 && t < 3) ? n[t] : "Chorus1";
}
const char *lfo_form_name(int f) {
    static const char *n[] = {"TRI","SIN","SAW","SQU","RND1","RND2"};
    return (f >= 0 && f < 6) ? n[f] : "TRI";
}

} // namespace jv
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cmake --build build --target test_jv_patch && ./build/test_jv_patch`
Expected: PASS — `ALL TESTS PASSED`, plus an informational line reporting how many of the 192 internal patches are LFO1-strippable.

- [ ] **Step 6: Commit**

```bash
git add src/jv_patch.h src/jv_patch.cpp tests/test_jv_patch.cpp CMakeLists.txt
git commit -m "feat: patch effect/LFO reading, strip decision, and dry preprocessing"
```

---

### Task 3: Grid renderer

**Goal:** Render one patch's full 25 x 3 grid to raw WAV files plus a metadata JSON, deterministically.

**Files:**
- Create: `src/wav.h`, `src/wav.cpp`
- Create: `src/jv_render.h`, `src/jv_render.cpp`
- Create: `src/jv_sampler.cpp`

**Acceptance Criteria:**
- [ ] Rendering the same patch twice produces byte-identical WAV output.
- [ ] Each rendered cell is stereo 16-bit at 64000 Hz.
- [ ] A dry-rendered patch that originally had reverb shows a shorter tail than the same patch rendered wet (proves the reverb bypass works).
- [ ] Percussive patches are truncated at decay; a sustained patch is not.
- [ ] `patch.json` records name, bank, index, effects, and both LFO decisions.
- [ ] `jv_sampler --board Internal --patch 0` completes in under 90 seconds.

**Verify:** `./build/jv_sampler --roms "$ROMS" --board "JV-880 Internal" --patch 0 --out /tmp/jvtest && ls /tmp/jvtest/*/*.wav | wc -l` → `75`

**Steps:**

- [ ] **Step 1: Write `src/wav.h` and `src/wav.cpp`**

```cpp
// src/wav.h
#pragma once
#include <stdint.h>
#include <vector>
#include <string>
namespace jv {
bool wav_write_s16(const std::string &path, const int16_t *interleaved,
                   int frames, int channels, int sample_rate);
bool wav_read_s16(const std::string &path, std::vector<int16_t> *out,
                  int *channels, int *sample_rate);
}
```

```cpp
// src/wav.cpp
#include "wav.h"
#include <stdio.h>
#include <string.h>
namespace jv {

bool wav_write_s16(const std::string &path, const int16_t *data,
                   int frames, int channels, int sample_rate) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t data_bytes = (uint32_t)frames * channels * 2;
    uint32_t riff = 36 + data_bytes;
    uint32_t fmt_size = 16, byte_rate = (uint32_t)sample_rate * channels * 2;
    uint16_t fmt = 1, ch = (uint16_t)channels, align = (uint16_t)(channels * 2), bits = 16;
    uint32_t sr = (uint32_t)sample_rate;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmt_size, 4, 1, f);
    fwrite(&fmt, 2, 1, f);   fwrite(&ch, 2, 1, f);   fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_bytes, 4, 1, f);
    fwrite(data, 2, (size_t)frames * channels, f);
    fclose(f);
    return true;
}

bool wav_read_s16(const std::string &path, std::vector<int16_t> *out,
                  int *channels, int *sample_rate) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return false; }
    if (channels)    *channels    = hdr[22] | (hdr[23] << 8);
    if (sample_rate) *sample_rate = hdr[24] | (hdr[25] << 8) |
                                    (hdr[26] << 16) | (hdr[27] << 24);
    uint32_t data_bytes = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);
    out->resize(data_bytes / 2);
    size_t got = fread(out->data(), 2, out->size(), f);
    out->resize(got);
    fclose(f);
    return true;
}

} // namespace jv
```

- [ ] **Step 2: Write `src/jv_render.h`**

```cpp
#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "jv_rom.h"
#include "jv_patch.h"

namespace jv {

static const int SAMPLE_RATE   = 64000;   // emulator native
static const int WARMUP_STEPS  = 100000;  // matches the plugin's boot warmup
static const int NVRAM_PATCH_OFFSET = 0x0d70;
static const int NVRAM_MODE_OFFSET  = 0x11;

struct GridSpec {
    int lokey = 24, hikey = 96, key_step = 3;   // C1..C7 every 3 semitones
    std::vector<int> velocities = {32, 72, 110};
    double hold_seconds    = 3.5;
    double tail_seconds    = 2.5;
    double settle_seconds  = 0.5;
    // Truncate once RMS falls this far below the sample's peak.
    double silence_db      = -72.0;
};

struct RenderedCell {
    int key = 0, velocity = 0, vel_layer = 0;
    int frames = 0;
    std::string filename;
};

class Renderer {
public:
    bool init(const RomSet &roms);                 // boot + warmup once
    void load_patch_bytes(const std::vector<uint8_t> &patch_bytes);
    // Render a single note; returns interleaved stereo at SAMPLE_RATE.
    std::vector<int16_t> render_note(int key, int velocity, const GridSpec &g);
    ~Renderer();
private:
    void *mcu_ = nullptr;   // opaque MCU*
};

} // namespace jv
```

- [ ] **Step 3: Implement `src/jv_render.cpp`**

```cpp
#include "jv_render.h"
#include "mcu.h"
#include <math.h>
#include <string.h>

namespace jv {

bool Renderer::init(const RomSet &roms) {
    MCU *m = new MCU();
    // startSC55 takes ownership of nothing; it copies what it needs.
    std::vector<uint8_t> nv = roms.nvram;
    nv[NVRAM_MODE_OFFSET] = 1;   // patch mode
    m->startSC55(roms.rom1.data(), roms.rom2.data(),
                 roms.waverom1.data(), roms.waverom2.data(), nv.data());
    for (int i = 0; i < WARMUP_STEPS; i++) m->updateSC55(1);
    mcu_ = m;
    return true;
}

Renderer::~Renderer() { delete (MCU *)mcu_; }

void Renderer::load_patch_bytes(const std::vector<uint8_t> &patch_bytes) {
    MCU *m = (MCU *)mcu_;
    memcpy(&m->nvram[NVRAM_PATCH_OFFSET], patch_bytes.data(), PATCH_SIZE);
    m->nvram[NVRAM_MODE_OFFSET] = 1;
    uint8_t pc[2] = {0xC0, 0x00};    // program change reloads from NVRAM
    m->postMidiSC55(pc, 2);
}

// Drain exactly n frames the emulator just produced.
static void drain(MCU *m, std::vector<int16_t> &out, int n) {
    size_t at = out.size();
    out.resize(at + (size_t)n * 2);
    memcpy(out.data() + at, m->sample_buffer, (size_t)n * 2 * sizeof(int16_t));
}

std::vector<int16_t> Renderer::render_note(int key, int velocity, const GridSpec &g) {
    MCU *m = (MCU *)mcu_;
    const int CHUNK = 64;

    // Settle after the program change so the first samples are clean.
    int settle = (int)(g.settle_seconds * SAMPLE_RATE);
    for (int i = 0; i < settle; i += CHUNK) m->updateSC55(CHUNK);

    std::vector<int16_t> out;
    int hold   = (int)(g.hold_seconds * SAMPLE_RATE);
    int tail   = (int)(g.tail_seconds * SAMPLE_RATE);

    uint8_t on[3]  = {0x90, (uint8_t)key, (uint8_t)velocity};
    m->postMidiSC55(on, 3);
    for (int i = 0; i < hold; i += CHUNK) { m->updateSC55(CHUNK); drain(m, out, CHUNK); }

    uint8_t off[3] = {0x80, (uint8_t)key, 0};
    m->postMidiSC55(off, 3);

    // Render the tail, stopping early once it decays below the floor.
    int peak = 1;
    for (size_t i = 0; i < out.size(); i++) peak = std::max(peak, abs((int)out[i]));
    const double floor_amp = peak * pow(10.0, g.silence_db / 20.0);
    int quiet_run = 0;
    for (int i = 0; i < tail; i += CHUNK) {
        m->updateSC55(CHUNK);
        drain(m, out, CHUNK);
        int blk = 0;
        for (int k = 0; k < CHUNK * 2; k++)
            blk = std::max(blk, abs((int)out[out.size() - CHUNK * 2 + k]));
        quiet_run = (blk < floor_amp) ? quiet_run + CHUNK : 0;
        if (quiet_run >= SAMPLE_RATE / 10) break;   // 100 ms below floor
    }

    // Silence any hanging voice before the next note.
    uint8_t allnotes[3] = {0xB0, 123, 0};
    m->postMidiSC55(allnotes, 3);
    for (int i = 0; i < SAMPLE_RATE / 4; i += CHUNK) m->updateSC55(CHUNK);

    return out;
}

} // namespace jv
```

- [ ] **Step 4: Implement `src/jv_sampler.cpp` (CLI)**

```cpp
#include "jv_render.h"
#include "wav.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <string>

using namespace jv;

static const char *NOTE_NAMES[12] =
    {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

static std::string note_name(int midi) {
    return std::string(NOTE_NAMES[midi % 12]) + std::to_string(midi / 12 - 1);
}

// Filesystem-safe patch name.
static std::string sanitize(const std::string &s) {
    std::string o;
    for (char c : s) {
        if (isalnum((unsigned char)c)) o += c;
        else if (c == ' ' || c == '-' || c == '_' || c == '.') o += c;
    }
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o.empty() ? "patch" : o;
}

static void mkdirs(const std::string &p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) mkdir(cur.c_str(), 0755);
    }
}

int main(int argc, char **argv) {
    std::string roms_dir, board = "JV-880 Internal", out_dir = "out";
    int only_patch = -1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--roms")  && i + 1 < argc) roms_dir = argv[++i];
        else if (!strcmp(argv[i], "--board") && i + 1 < argc) board = argv[++i];
        else if (!strcmp(argv[i], "--out")   && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--patch") && i + 1 < argc) only_patch = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (roms_dir.empty()) { fprintf(stderr, "--roms required\n"); return 1; }

    std::string err;
    RomSet roms;
    if (!roms.load(roms_dir, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }

    std::vector<PatchRef> patches;
    if (board == "JV-880 Internal") {
        patches = enumerate_internal(roms);
    } else {
        for (auto &e : scan_expansions(roms_dir + "/expansions"))
            if (e.usable && e.name == board) patches = enumerate_expansion(e);
        if (patches.empty()) { fprintf(stderr, "board not found: %s\n", board.c_str()); return 1; }
    }

    GridSpec grid;
    Renderer r;
    r.init(roms);

    for (size_t pi = 0; pi < patches.size(); pi++) {
        if (only_patch >= 0 && (int)pi != only_patch) continue;
        const PatchRef &pr = patches[pi];

        Effects fx = read_effects(pr.data);
        LfoDecision d1 = decide_lfo_strip(pr.data, 1);
        LfoDecision d2 = decide_lfo_strip(pr.data, 2);
        auto bytes = preprocess(pr.data, d1, d2);

        std::string slug = sanitize(pr.name);
        char idx[16]; snprintf(idx, sizeof(idx), "%03d", (int)pi);
        std::string pdir = out_dir + "/" + std::string(idx) + "_" + slug;
        mkdirs(pdir);

        r.load_patch_bytes(bytes);

        std::string zones;
        for (int key = grid.lokey; key <= grid.hikey; key += grid.key_step) {
            for (size_t v = 0; v < grid.velocities.size(); v++) {
                auto pcm = r.render_note(key, grid.velocities[v], grid);
                int frames = (int)(pcm.size() / 2);
                char fn[256];
                snprintf(fn, sizeof(fn), "%s_v%d.wav", note_name(key).c_str(), (int)v + 1);
                wav_write_s16(pdir + "/" + fn, pcm.data(), frames, 2, SAMPLE_RATE);
                char buf[512];
                snprintf(buf, sizeof(buf),
                    "%s{\"key\":%d,\"velocity\":%d,\"layer\":%d,\"frames\":%d,\"file\":\"%s\"}",
                    zones.empty() ? "" : ",", key, grid.velocities[v], (int)v + 1, frames, fn);
                zones += buf;
            }
        }

        FILE *jf = fopen((pdir + "/patch.json").c_str(), "w");
        fprintf(jf,
            "{\n"
            "  \"name\": \"%s\",\n  \"bank\": \"%s\",\n  \"index\": %d,\n"
            "  \"sample_rate\": %d,\n"
            "  \"effects\": {\n"
            "    \"reverb\": {\"type\":\"%s\",\"level\":%d,\"time\":%d,\"feedback\":%d},\n"
            "    \"chorus\": {\"type\":\"%s\",\"level\":%d,\"depth\":%d,\"rate\":%d,"
            "\"feedback\":%d,\"output\":\"%s\"},\n"
            "    \"reverb_send\": [%d,%d,%d,%d],\n"
            "    \"chorus_send\": [%d,%d,%d,%d],\n"
            "    \"tone_level\": [%d,%d,%d,%d],\n"
            "    \"bend_up\": %d, \"bend_down\": %d\n  },\n"
            "  \"lfo1\": {\"stripped\":%s,\"reason\":\"%s\",\"form\":\"%s\",\"rate\":%d,"
            "\"delay\":%d,\"sync\":%d,\"pitch\":%d,\"tvf\":%d,\"tva\":%d},\n"
            "  \"lfo2\": {\"stripped\":%s,\"reason\":\"%s\",\"form\":\"%s\",\"rate\":%d,"
            "\"delay\":%d,\"sync\":%d,\"pitch\":%d,\"tvf\":%d,\"tva\":%d},\n"
            "  \"zones\": [%s]\n}\n",
            pr.name.c_str(), pr.bank.c_str(), pr.index, SAMPLE_RATE,
            reverb_type_name(fx.reverb_type), fx.reverb_level, fx.reverb_time, fx.reverb_feedback,
            chorus_type_name(fx.chorus_type), fx.chorus_level, fx.chorus_depth, fx.chorus_rate,
            fx.chorus_feedback, fx.chorus_output ? "Reverb" : "Mix",
            fx.reverb_send[0], fx.reverb_send[1], fx.reverb_send[2], fx.reverb_send[3],
            fx.chorus_send[0], fx.chorus_send[1], fx.chorus_send[2], fx.chorus_send[3],
            fx.tone_level[0], fx.tone_level[1], fx.tone_level[2], fx.tone_level[3],
            fx.bend_up, fx.bend_down,
            d1.strip ? "true" : "false", d1.reason.c_str(), lfo_form_name(d1.lfo.form),
            d1.lfo.rate, d1.lfo.delay, d1.lfo.sync,
            d1.lfo.pitch_depth, d1.lfo.tvf_depth, d1.lfo.tva_depth,
            d2.strip ? "true" : "false", d2.reason.c_str(), lfo_form_name(d2.lfo.form),
            d2.lfo.rate, d2.lfo.delay, d2.lfo.sync,
            d2.lfo.pitch_depth, d2.lfo.tvf_depth, d2.lfo.tva_depth,
            zones.c_str());
        fclose(jf);
        fprintf(stderr, "rendered %s (%zu zones)\n", pr.name.c_str(),
                grid.velocities.size() * (size_t)((grid.hikey - grid.lokey) / grid.key_step + 1));
    }
    return 0;
}
```

- [ ] **Step 5: Build and verify determinism and cell count**

```bash
ROMS="/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880"
cmake --build build --target jv_sampler
./build/jv_sampler --roms "$ROMS" --board Internal --patch 0 --out /tmp/jvtest_a
./build/jv_sampler --roms "$ROMS" --board Internal --patch 0 --out /tmp/jvtest_b
ls /tmp/jvtest_a/000_A.Piano 1/*.wav | wc -l     # expect 75
diff -r /tmp/jvtest_a /tmp/jvtest_b && echo "DETERMINISTIC"
```

Expected: `75`, then `DETERMINISTIC`.

- [ ] **Step 6: Verify the dry render actually removed reverb**

```bash
python3 - <<'EOF'
import soundfile as sf, numpy as np, glob
f = sorted(glob.glob('/tmp/jvtest_a/*/C4_v2.wav'))[0]
x, sr = sf.read(f)
env = np.abs(x).max(axis=1)
peak = env.max()
tail = env[int(len(env)*0.8):].mean()
print(f"peak={peak:.4f} tail_mean={tail:.6f} ratio={tail/peak:.5f}")
assert tail/peak < 0.05, "tail too loud - reverb may still be present"
print("DRY OK")
EOF
```

Expected: `DRY OK`.

- [ ] **Step 7: Commit**

```bash
git add src/wav.h src/wav.cpp src/jv_render.h src/jv_render.cpp src/jv_sampler.cpp CMakeLists.txt
git commit -m "feat: deterministic grid renderer with decay truncation and patch metadata"
```

---

### Task 4: Effect calibration

**Goal:** Measure the emulator's actual chorus rate/depth and reverb decay response, producing `calibration.json` for faithful DecentSampler parameter mapping.

**Files:**
- Create: `tools/calibrate.cpp`
- Create: `tools/analyze_calibration.py`
- Create: `tests/test_calibration.py`

**Acceptance Criteria:**
- [ ] `calibrate` renders sweeps of `chorusrate` (0-127 step 8), `chorusdepth` (0-127 step 16), `reverbtime` (0-127 step 16) per reverb type 0-5, and `reverblevel` (0-127 step 16).
- [ ] Measured chorus rate is monotonically non-decreasing with the raw value.
- [ ] Measured chorus rate at raw 127 is greater than at raw 0 by at least 2x.
- [ ] Measured RT60 is monotonically non-decreasing with `reverbtime`.
- [ ] `calibration.json` contains `chorus_rate_hz`, `chorus_depth_norm`, `chorus_mix`, `reverb_rt60`, and `reverb_wet` tables.

**Verify:** `python3 -m pytest tests/test_calibration.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write `tools/calibrate.cpp`**

Renders a fixed reference note with one effect parameter varied, writing one WAV per setting. Uses a patch with a simple sustained tone (Preset A `Pipe Organ 1`, internal index 24) so the modulation is easy to measure.

```cpp
#include "jv_render.h"
#include "wav.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

using namespace jv;

// Render one note with the given patch bytes and write it.
static void render_one(Renderer &r, const std::vector<uint8_t> &bytes,
                       const std::string &path) {
    GridSpec g;
    g.hold_seconds = 4.0;
    g.tail_seconds = 4.0;
    g.silence_db   = -120.0;   // never truncate during calibration
    r.load_patch_bytes(bytes);
    auto pcm = r.render_note(60, 100, g);
    wav_write_s16(path, pcm.data(), (int)(pcm.size() / 2), 2, SAMPLE_RATE);
}

int main(int argc, char **argv) {
    std::string roms_dir, out_dir = "calib";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--roms") && i + 1 < argc) roms_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
    }
    if (roms_dir.empty()) { fprintf(stderr, "--roms required\n"); return 1; }
    mkdir(out_dir.c_str(), 0755);

    std::string err;
    RomSet roms;
    if (!roms.load(roms_dir, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto patches = enumerate_internal(roms);
    const uint8_t *base = patches[24].data;   // Pipe Organ 1: steady sustain

    Renderer r;
    r.init(roms);
    char path[512];

    // Dry reference.
    {
        LfoDecision none;
        auto b = preprocess(base, none, none);
        snprintf(path, sizeof(path), "%s/dry.wav", out_dir.c_str());
        render_one(r, b, path);
    }

    // Chorus rate sweep (chorus fully wet, reverb off).
    for (int raw = 0; raw <= 127; raw += 8) {
        LfoDecision none;
        auto b = preprocess(base, none, none);
        b[16] = (uint8_t)((b[16] & 0x80) | 100);   // choruslevel = 100
        b[17] = 100;                                // chorusdepth
        b[18] = (uint8_t)raw;                       // chorusrate
        snprintf(path, sizeof(path), "%s/chorus_rate_%03d.wav", out_dir.c_str(), raw);
        render_one(r, b, path);
    }

    // Chorus depth sweep.
    for (int raw = 0; raw <= 127; raw += 16) {
        LfoDecision none;
        auto b = preprocess(base, none, none);
        b[16] = (uint8_t)((b[16] & 0x80) | 100);
        b[17] = (uint8_t)raw;
        b[18] = 40;
        snprintf(path, sizeof(path), "%s/chorus_depth_%03d.wav", out_dir.c_str(), raw);
        render_one(r, b, path);
    }

    // Chorus level sweep (for mix mapping).
    for (int raw = 0; raw <= 127; raw += 16) {
        LfoDecision none;
        auto b = preprocess(base, none, none);
        b[16] = (uint8_t)((b[16] & 0x80) | raw);
        b[17] = 80; b[18] = 40;
        snprintf(path, sizeof(path), "%s/chorus_level_%03d.wav", out_dir.c_str(), raw);
        render_one(r, b, path);
    }

    // Reverb time sweep per reverb type 0-5 (6/7 are delays, handled separately).
    for (int type = 0; type <= 5; type++) {
        for (int raw = 0; raw <= 127; raw += 16) {
            LfoDecision none;
            auto b = preprocess(base, none, none);
            b[12] = (uint8_t)((b[12] & ~0x0f) | type);
            b[13] = 100;             // reverblevel
            b[14] = (uint8_t)raw;    // reverbtime
            snprintf(path, sizeof(path), "%s/reverb_t%d_time_%03d.wav",
                     out_dir.c_str(), type, raw);
            render_one(r, b, path);
        }
    }

    // Reverb level sweep (wet mapping), type Hall1.
    for (int raw = 0; raw <= 127; raw += 16) {
        LfoDecision none;
        auto b = preprocess(base, none, none);
        b[12] = (uint8_t)((b[12] & ~0x0f) | 4);
        b[13] = (uint8_t)raw;
        b[14] = 80;
        snprintf(path, sizeof(path), "%s/reverb_level_%03d.wav", out_dir.c_str(), raw);
        render_one(r, b, path);
    }

    fprintf(stderr, "calibration renders complete in %s\n", out_dir.c_str());
    return 0;
}
```

- [ ] **Step 2: Write `tools/analyze_calibration.py`**

```python
#!/usr/bin/env python3
"""Measure calibration renders and emit calibration.json.

Chorus rate is recovered from the amplitude-envelope modulation spectrum;
chorus depth from the peak-to-peak envelope excursion; reverb RT60 from the
decay slope of the tail after note-off.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

SR_NATIVE = 64000


def load_mono(path):
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def envelope(x, sr, hop=64):
    """Amplitude envelope, decimated, DC-removed."""
    n = (len(x) // hop) * hop
    env = np.abs(x[:n]).reshape(-1, hop).max(axis=1)
    return env - env.mean(), sr / hop


def measure_mod_rate(path):
    """Dominant modulation frequency of the amplitude envelope, in Hz."""
    x, sr = load_mono(path)
    # Use the sustained middle so attack/release don't dominate.
    x = x[int(0.7 * sr):int(3.5 * sr)]
    if len(x) < 1024:
        return 0.0
    env, env_sr = envelope(x, sr)
    f, p = welch(env, fs=env_sr, nperseg=min(4096, len(env)))
    band = (f > 0.05) & (f < 12.0)
    if not band.any():
        return 0.0
    return float(f[band][np.argmax(p[band])])


def measure_mod_depth(path, dry_path):
    """Envelope excursion relative to the dry reference (0..1)."""
    x, sr = load_mono(path)
    d, _ = load_mono(dry_path)
    seg = slice(int(0.7 * sr), int(3.5 * sr))
    ex, _ = envelope(x[seg], sr)
    dx, _ = envelope(d[seg], sr)
    ref = np.abs(d[seg]).max() or 1.0
    return float(np.clip((ex.std() - dx.std()) / ref * 4.0, 0.0, 1.0))


def measure_wet_ratio(path, dry_path):
    """Extra energy over the dry reference, normalized to 0..1."""
    x, _ = load_mono(path)
    d, _ = load_mono(dry_path)
    n = min(len(x), len(d))
    ex, dx = float(np.sum(x[:n] ** 2)), float(np.sum(d[:n] ** 2))
    if dx <= 0:
        return 0.0
    return float(np.clip((ex - dx) / dx, 0.0, 1.0))


def measure_rt60(path, note_off_s=4.0):
    """RT60 from the decay slope after note-off (seconds)."""
    x, sr = load_mono(path)
    tail = x[int(note_off_s * sr):]
    if len(tail) < sr // 10:
        return 0.0
    env, env_sr = envelope(tail, sr)
    env = np.abs(env) + 1e-12
    db = 20 * np.log10(env / env.max())
    idx = np.where(db < -5)[0]
    if len(idx) < 10:
        return 0.0
    lo = idx[0]
    hi_c = np.where(db < -35)[0]
    hi = hi_c[0] if len(hi_c) else len(db) - 1
    if hi <= lo + 2:
        return 0.0
    t = np.arange(lo, hi) / env_sr
    slope = np.polyfit(t, db[lo:hi], 1)[0]      # dB per second
    if slope >= -1e-6:
        return 0.0
    return float(-60.0 / slope)


def sweep(calib, pattern):
    out = {}
    rx = re.compile(pattern)
    for p in sorted(calib.glob("*.wav")):
        m = rx.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def main():
    calib = Path(sys.argv[1] if len(sys.argv) > 1 else "calib")
    dry = calib / "dry.wav"
    if not dry.exists():
        sys.exit(f"missing {dry}")

    result = {
        "chorus_rate_hz": {},
        "chorus_depth_norm": {},
        "chorus_mix": {},
        "reverb_rt60": {},
        "reverb_wet": {},
    }

    for raw, p in sweep(calib, r"chorus_rate_(\d+)\.wav").items():
        result["chorus_rate_hz"][str(raw)] = round(measure_mod_rate(p), 4)
    for raw, p in sweep(calib, r"chorus_depth_(\d+)\.wav").items():
        result["chorus_depth_norm"][str(raw)] = round(measure_mod_depth(p, dry), 4)
    for raw, p in sweep(calib, r"chorus_level_(\d+)\.wav").items():
        result["chorus_mix"][str(raw)] = round(measure_wet_ratio(p, dry), 4)
    for raw, p in sweep(calib, r"reverb_level_(\d+)\.wav").items():
        result["reverb_wet"][str(raw)] = round(measure_wet_ratio(p, dry), 4)

    for t in range(6):
        tab = {}
        for raw, p in sweep(calib, rf"reverb_t{t}_time_(\d+)\.wav").items():
            tab[str(raw)] = round(measure_rt60(p), 4)
        result["reverb_rt60"][str(t)] = tab

    out = calib / "calibration.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    for k in result:
        print(f"  {k}: {len(result[k])} entries")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `tests/test_calibration.py`**

```python
import json
import subprocess
from pathlib import Path

import pytest

CALIB = Path("calib")
JSON = CALIB / "calibration.json"


@pytest.fixture(scope="module")
def cal():
    if not JSON.exists():
        pytest.skip("run calibrate + analyze_calibration.py first")
    return json.loads(JSON.read_text())


def _monotonic(pairs, tol=0.0):
    vals = [v for _, v in sorted(pairs, key=lambda kv: int(kv[0]))]
    return all(b >= a - tol for a, b in zip(vals, vals[1:]))


def test_tables_present(cal):
    for key in ("chorus_rate_hz", "chorus_depth_norm", "chorus_mix",
                "reverb_rt60", "reverb_wet"):
        assert key in cal and cal[key], f"{key} missing or empty"


def test_chorus_rate_monotonic(cal):
    assert _monotonic(cal["chorus_rate_hz"].items(), tol=0.15)


def test_chorus_rate_spans_range(cal):
    t = cal["chorus_rate_hz"]
    lo, hi = t[min(t, key=int)], t[max(t, key=int)]
    assert hi > lo * 2, f"rate barely changes: {lo} -> {hi}"


def test_reverb_rt60_monotonic(cal):
    for typ, tab in cal["reverb_rt60"].items():
        assert _monotonic(tab.items(), tol=0.25), f"type {typ} not monotonic"


def test_reverb_wet_increases(cal):
    t = cal["reverb_wet"]
    assert t[max(t, key=int)] > t[min(t, key=int)]
```

- [ ] **Step 4: Run calibration and the tests**

```bash
ROMS="/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880"
cmake --build build --target calibrate
./build/calibrate --roms "$ROMS" --out calib
python3 tools/analyze_calibration.py calib
python3 -m pytest tests/test_calibration.py -v
```

Expected: all tests pass. If `test_chorus_rate_monotonic` fails, the measurement window in `measure_mod_rate` is picking up the patch's own LFO — switch the base patch to a different steady tone (try internal index 26, `Pipe Organ 3`) and re-run.

- [ ] **Step 5: Commit**

```bash
git add tools/calibrate.cpp tools/analyze_calibration.py tests/test_calibration.py CMakeLists.txt
git commit -m "feat: measure emulator effect response to calibrate DS parameter mapping"
```

---

### Task 5: Post-processing — resample, loop detection, release, FLAC

**Goal:** Turn raw 64 kHz WAV renders into 48 kHz 24-bit FLAC with loop points and measured release times.

**Files:**
- Create: `tools/postprocess.py`
- Create: `tests/test_postprocess.py`

**Acceptance Criteria:**
- [ ] Output FLAC is 48000 Hz, 24-bit, stereo.
- [ ] A synthetic steady sine is classified sustaining and gets a loop whose start/end land on the same phase (endpoint discontinuity below 1% of peak).
- [ ] A synthetic exponentially decaying burst is classified decaying and gets no loop.
- [ ] `loop_crossfade` never exceeds `min(2000, loop_start // 4, (loop_end - loop_start) // 4)`.
- [ ] Release time is measured from the post-note-off decay and written per zone.
- [ ] A `patch.json` gains a `zones[].loop` and `zones[].release` for every zone.

**Verify:** `python3 -m pytest tests/test_postprocess.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write `tools/postprocess.py`**

```python
#!/usr/bin/env python3
"""Resample renders to 48 kHz, detect loops, measure release, encode FLAC."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR_IN = 64000
SR_OUT = 48000
MAX_XFADE = 2000


def resample_to_48k(x):
    # 64000 -> 48000 is exactly 3/4.
    return resample_poly(x, up=3, down=4, axis=0)


def classify(x, hold_frames):
    """Sustaining if energy just before note-off is still substantial."""
    peak = np.abs(x).max()
    if peak <= 0:
        return "silent", 0.0
    lo = max(0, hold_frames - int(0.25 * SR_OUT))
    hi = min(len(x), hold_frames)
    if hi <= lo:
        return "decaying", 0.0
    sustain = float(np.sqrt(np.mean(x[lo:hi] ** 2)))
    ratio = sustain / peak
    return ("sustaining" if ratio > 0.18 else "decaying"), ratio


def estimate_period(mono, sr):
    """Fundamental period in frames via autocorrelation."""
    seg = mono - mono.mean()
    if len(seg) < 4096:
        return 0
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(sr / 1200), int(sr / 40)     # 40 Hz .. 1200 Hz
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return 0
    return int(lo + np.argmax(ac[lo:hi]))


def find_loop(x, sr, hold_frames):
    """Correlation-matched loop points inside the steady-state region."""
    mono = x.mean(axis=1) if x.ndim > 1 else x
    start_lo = int(1.0 * sr)
    region_hi = min(hold_frames, len(mono)) - int(0.05 * sr)
    if region_hi - start_lo < int(0.3 * sr):
        return None

    period = estimate_period(mono[start_lo:region_hi], sr)
    if period <= 0:
        return None

    win = max(period * 2, 256)
    loop_start = start_lo
    best, best_end = None, None
    # Try loop lengths from 8 to 60 periods; longer loops sound less static.
    for n_per in range(8, 61):
        end = loop_start + period * n_per
        if end + win >= region_hi:
            break
        a = mono[loop_start:loop_start + win]
        b = mono[end:end + win]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        score = float(np.dot(a, b) / denom)
        if best is None or score > best:
            best, best_end = score, end

    if best is None or best < 0.90:
        return None

    length = best_end - loop_start
    xfade = int(min(MAX_XFADE, loop_start // 4, length // 4))
    return {"enabled": True, "start": int(loop_start), "end": int(best_end),
            "crossfade": int(max(0, xfade)), "score": round(best, 4)}


def measure_release(x, sr, hold_frames):
    """Seconds for the post-note-off tail to fall 60 dB."""
    tail = x[hold_frames:]
    if len(tail) < sr // 20:
        return 0.1
    mono = np.abs(tail.mean(axis=1) if tail.ndim > 1 else tail)
    hop = 64
    n = (len(mono) // hop) * hop
    env = mono[:n].reshape(-1, hop).max(axis=1) + 1e-12
    db = 20 * np.log10(env / env.max())
    below = np.where(db < -60)[0]
    if len(below):
        return float(max(0.05, below[0] * hop / sr))
    return float(len(mono) / sr)


def process_patch(pdir: Path, hold_seconds=3.5):
    meta = json.loads((pdir / "patch.json").read_text())
    hold_out = int(hold_seconds * SR_OUT)

    for z in meta["zones"]:
        src = pdir / z["file"]
        if not src.exists():
            continue
        x, sr = sf.read(str(src), always_2d=True)
        assert sr == SR_IN, f"unexpected rate {sr} in {src}"
        y = resample_to_48k(x)

        kind, ratio = classify(y, hold_out)
        loop = find_loop(y, SR_OUT, hold_out) if kind == "sustaining" else None

        dst = src.with_suffix(".flac")
        sf.write(str(dst), y, SR_OUT, subtype="PCM_24")
        src.unlink()

        z["file"] = dst.name
        z["frames"] = int(len(y))
        z["kind"] = kind
        z["sustain_ratio"] = round(ratio, 4)
        z["loop"] = loop or {"enabled": False}
        z["release"] = round(measure_release(y, SR_OUT, hold_out), 4)

    meta["sample_rate"] = SR_OUT
    (pdir / "patch.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    root = Path(sys.argv[1])
    dirs = sorted(p for p in root.iterdir() if (p / "patch.json").exists())
    for i, d in enumerate(dirs, 1):
        m = process_patch(d)
        looped = sum(1 for z in m["zones"] if z["loop"]["enabled"])
        print(f"[{i}/{len(dirs)}] {m['name']}: {looped}/{len(m['zones'])} looped")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/test_postprocess.py`**

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import postprocess as pp  # noqa: E402

SR = pp.SR_OUT
HOLD = int(3.5 * SR)


def make_stereo(mono):
    return np.stack([mono, mono], axis=1)


def test_steady_sine_is_sustaining_and_loops():
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    mono[HOLD:] *= np.exp(-6.0 * (t[HOLD:] - t[HOLD]))
    x = make_stereo(mono)

    kind, _ = pp.classify(x, HOLD)
    assert kind == "sustaining"

    loop = pp.find_loop(x, SR, HOLD)
    assert loop and loop["enabled"]

    a = x[loop["start"], 0]
    b = x[loop["end"], 0]
    assert abs(a - b) < 0.01 * np.abs(x).max(), "loop endpoints discontinuous"


def test_decaying_burst_gets_no_loop():
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.8 * np.sin(2 * np.pi * 440.0 * t) * np.exp(-6.0 * t)
    x = make_stereo(mono)
    kind, _ = pp.classify(x, HOLD)
    assert kind == "decaying"


def test_crossfade_bounded():
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    x = make_stereo(mono)
    loop = pp.find_loop(x, SR, HOLD)
    assert loop is not None
    length = loop["end"] - loop["start"]
    assert loop["crossfade"] <= min(pp.MAX_XFADE, loop["start"] // 4, length // 4)


def test_resample_ratio():
    x = np.zeros((pp.SR_IN, 2))
    y = pp.resample_to_48k(x)
    assert abs(len(y) - pp.SR_OUT) <= 2


def test_release_measured():
    t = np.arange(int(6.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    mono[HOLD:] *= np.exp(-10.0 * (t[HOLD:] - t[HOLD]))
    r = pp.measure_release(make_stereo(mono), SR, HOLD)
    assert 0.1 < r < 2.0
```

- [ ] **Step 3: Run tests to verify they fail, then pass**

Run: `python3 -m pytest tests/test_postprocess.py -v`
Expected first run: FAIL (`ModuleNotFoundError` or assertion errors) until `tools/postprocess.py` exists and is correct. After Step 1 is in place: all 5 tests PASS.

- [ ] **Step 4: Run against the real pilot render**

```bash
python3 tools/postprocess.py /tmp/jvtest_a
python3 -c "
import json,glob
m=json.load(open(glob.glob('/tmp/jvtest_a/*/patch.json')[0]))
z=m['zones']
print('zones',len(z),'looped',sum(1 for x in z if x['loop']['enabled']))
print('rate',m['sample_rate'])
"
```

Expected: `rate 48000` and a nonzero looped count for sustaining patches.

- [ ] **Step 5: Commit**

```bash
git add tools/postprocess.py tests/test_postprocess.py
git commit -m "feat: resample to 48k, detect loops, measure release, encode 24-bit FLAC"
```

---

### Task 6: Preset emitters

**Goal:** Generate `.dspreset` and `.sfz` for every patch from its metadata plus the calibration table.

**Files:**
- Create: `tools/emit_presets.py`
- Create: `tests/test_emit_presets.py`

**Acceptance Criteria:**
- [ ] Emitted `.dspreset` is well-formed XML with a `<groups>` containing one `<sample>` per zone.
- [ ] Zone key ranges tile the keyboard with no gaps: the lowest zone starts at `loNote=0`, the highest ends at `hiNote=127`, and consecutive zones are contiguous.
- [ ] Velocity ranges tile 1-127 with no gaps or overlaps.
- [ ] Reverb types 0-5 emit `<effect type="reverb">`; types 6-7 emit `<effect type="delay">`.
- [ ] Chorus emits `<effect type="chorus">` with `modRate` taken from the calibration table.
- [ ] A patch with `lfo1.stripped == true` emits a `<modulators>` LFO bound to the right target, with `scope="voice"` when `sync == 1` and `"global"` otherwise.
- [ ] RND1/RND2 patches emit no LFO modulator.
- [ ] `.sfz` contains one `<region>` per zone with matching key/vel ranges, loop points, and `ampeg_release`.
- [ ] Both files reference sample paths that exist on disk.

**Verify:** `python3 -m pytest tests/test_emit_presets.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write `tools/emit_presets.py`**

```python
#!/usr/bin/env python3
"""Emit DecentSampler .dspreset and SFZ .sfz from patch metadata."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REVERB_DELAY_TYPES = {6, 7}          # Delay, Pan-Dly
REVERB_NAMES = ["Room1", "Room2", "Stage1", "Stage2",
                "Hall1", "Hall2", "Delay", "Pan-Dly"]
# DS supports sine/square/saw only.
LFO_SHAPE = {"TRI": "sine", "SIN": "sine", "SAW": "saw", "SQU": "square"}


def interp_table(table, raw):
    """Linear interpolation over a {str(raw): value} calibration table."""
    if not table:
        return 0.0
    keys = sorted(int(k) for k in table)
    if raw <= keys[0]:
        return float(table[str(keys[0])])
    if raw >= keys[-1]:
        return float(table[str(keys[-1])])
    for a, b in zip(keys, keys[1:]):
        if a <= raw <= b:
            fa, fb = float(table[str(a)]), float(table[str(b)])
            t = (raw - a) / (b - a) if b != a else 0.0
            return fa + (fb - fa) * t
    return float(table[str(keys[-1])])


def key_ranges(zone_keys):
    """Contiguous key spans covering 0..127, centered on each sampled key."""
    keys = sorted(set(zone_keys))
    spans = {}
    for i, k in enumerate(keys):
        lo = 0 if i == 0 else (keys[i - 1] + k) // 2 + 1
        hi = 127 if i == len(keys) - 1 else (k + keys[i + 1]) // 2
        spans[k] = (lo, hi)
    return spans


def vel_ranges(n_layers):
    """Tile 1..127 across n layers with no gaps."""
    out, lo = [], 1
    for i in range(n_layers):
        hi = 127 if i == n_layers - 1 else round(127 * (i + 1) / n_layers)
        out.append((lo, hi))
        lo = hi + 1
    return out


def effective_send(meta, which):
    """Average a send across active tones only (tone level > 0)."""
    fx = meta["effects"]
    sends = fx[f"{which}_send"]
    levels = fx["tone_level"]
    active = [s for s, l in zip(sends, levels) if l > 0]
    return sum(active) / len(active) if active else 0.0


def build_effects(meta, cal):
    """Return list of (type, attrib-dict) in chain order."""
    fx = meta["effects"]
    out = []

    rv, ch = fx["reverb"], fx["chorus"]
    rtype = REVERB_NAMES.index(rv["type"]) if rv["type"] in REVERB_NAMES else 0

    chorus_el = None
    if ch["level"] > 0:
        send = effective_send(meta, "chorus") / 127.0
        mix = interp_table(cal.get("chorus_mix", {}), ch["level"]) or (ch["level"] / 127.0)
        chorus_el = ("chorus", {
            "mix": f"{min(1.0, mix * max(send, 0.25)):.3f}",
            "modDepth": f"{interp_table(cal.get('chorus_depth_norm', {}), ch['depth']) or ch['depth']/127.0:.3f}",
            "modRate": f"{interp_table(cal.get('chorus_rate_hz', {}), ch['rate']) or 0.5:.3f}",
        })

    reverb_el = None
    if rv["level"] > 0:
        send = effective_send(meta, "reverb") / 127.0
        wet = interp_table(cal.get("reverb_wet", {}), rv["level"]) or (rv["level"] / 127.0)
        wet = min(1.0, wet * max(send, 0.25))
        if rtype in REVERB_DELAY_TYPES:
            # Delay / Pan-Dly: reverbtime maps to delay time, feedback direct.
            reverb_el = ("delay", {
                "delayTime": f"{0.05 + (rv['time'] / 127.0) * 0.55:.3f}",
                "feedback": f"{rv['feedback'] / 127.0:.3f}",
                "stereoOffset": "0.4" if rtype == 7 else "0",
                "wetLevel": f"{wet:.3f}",
            })
        else:
            rt60 = interp_table(cal.get("reverb_rt60", {}).get(str(rtype), {}), rv["time"])
            room = min(1.0, rt60 / 6.0) if rt60 else rv["time"] / 127.0
            damp = {0: 0.55, 1: 0.45, 2: 0.35, 3: 0.30, 4: 0.22, 5: 0.15}.get(rtype, 0.3)
            reverb_el = ("reverb", {
                "roomSize": f"{room:.3f}",
                "damping": f"{damp:.3f}",
                "wetLevel": f"{wet:.3f}",
            })

    # chorus_output "Reverb" routes chorus INTO the reverb, so chorus must come
    # first in the chain. "Mix" runs them independently; reverb last reads more
    # naturally as a final space.
    if chorus_el and reverb_el:
        if fx["chorus"]["output"] == "Reverb":
            out = [chorus_el, reverb_el]
        else:
            out = [reverb_el, chorus_el]
    else:
        out = [e for e in (chorus_el, reverb_el) if e]
    return out


def build_dspreset(meta, cal, sample_prefix):
    root = ET.Element("DecentSampler", {"pluginVersion": "1"})
    ET.SubElement(root, "ui", {"width": "812", "height": "375"})

    groups = ET.SubElement(root, "groups", {"attack": "0.0", "decay": "0.0",
                                            "sustain": "1.0", "release": "0.4"})
    group = ET.SubElement(groups, "group")

    spans = key_ranges([z["key"] for z in meta["zones"]])
    layers = sorted({z["layer"] for z in meta["zones"]})
    vr = vel_ranges(len(layers))

    for z in meta["zones"]:
        lo, hi = spans[z["key"]]
        vlo, vhi = vr[z["layer"] - 1]
        attrs = {
            "path": f"{sample_prefix}/{z['file']}",
            "rootNote": str(z["key"]),
            "loNote": str(lo), "hiNote": str(hi),
            "loVel": str(vlo), "hiVel": str(vhi),
        }
        loop = z.get("loop", {})
        if loop.get("enabled"):
            attrs.update({
                "loopEnabled": "1",
                "loopStart": str(loop["start"]),
                "loopEnd": str(loop["end"]),
                "loopCrossfade": str(loop.get("crossfade", 0)),
            })
        ET.SubElement(group, "sample", attrs)

    effects = build_effects(meta, cal)
    if effects:
        fx_el = ET.SubElement(root, "effects")
        for etype, attrs in effects:
            ET.SubElement(fx_el, "effect", {"type": etype, **attrs})

    # LFO modulator, only when the renderer stripped it.
    lfo = meta.get("lfo1", {})
    if lfo.get("stripped") and lfo.get("form") in LFO_SHAPE:
        rate_hz = interp_table(cal.get("chorus_rate_hz", {}), lfo["rate"]) or 1.0
        mods = ET.SubElement(root, "modulators")
        el = ET.SubElement(mods, "lfo", {
            "shape": LFO_SHAPE[lfo["form"]],
            "frequency": f"{rate_hz:.3f}",
            "scope": "voice" if lfo.get("sync") else "global",
            "delayTime": f"{lfo.get('delay', 0) / 127.0 * 2.0:.3f}",
            "modAmount": f"{max(abs(lfo.get('tva', 0)), abs(lfo.get('pitch', 0)), abs(lfo.get('tvf', 0))) / 63.0:.3f}",
        })
        if lfo.get("tva"):
            ET.SubElement(el, "binding", {"type": "amp", "level": "group", "position": "0",
                                          "parameter": "AMP_VOLUME", "modBehavior": "modulate"})
        elif lfo.get("pitch"):
            ET.SubElement(el, "binding", {"type": "general", "level": "group", "position": "0",
                                          "parameter": "GROUP_TUNING", "modBehavior": "modulate"})
        elif lfo.get("tvf"):
            ET.SubElement(el, "binding", {"type": "effect", "level": "instrument", "position": "0",
                                          "parameter": "FX_FILTER_FREQUENCY", "modBehavior": "modulate"})

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def build_sfz(meta, sample_prefix):
    fx = meta["effects"]
    lines = [
        f"// {meta['name']} — JV-880 {meta['bank']} #{meta['index']}",
        "// SFZ has no standard reverb/chorus; these are recorded for reference:",
        f"//   reverb: {fx['reverb']}",
        f"//   chorus: {fx['chorus']}",
        f"//   lfo1:   {meta.get('lfo1', {})}",
        "",
        "<control>",
        f"default_path={sample_prefix}/",
        "",
        "<global>",
        "ampeg_attack=0.001",
        "",
    ]
    spans = key_ranges([z["key"] for z in meta["zones"]])
    layers = sorted({z["layer"] for z in meta["zones"]})
    vr = vel_ranges(len(layers))

    for z in meta["zones"]:
        lo, hi = spans[z["key"]]
        vlo, vhi = vr[z["layer"] - 1]
        lines.append("<region>")
        lines.append(f"sample={z['file']}")
        lines.append(f"lokey={lo} hikey={hi} pitch_keycenter={z['key']}")
        lines.append(f"lovel={vlo} hivel={vhi}")
        lines.append(f"ampeg_release={z.get('release', 0.4)}")
        loop = z.get("loop", {})
        if loop.get("enabled"):
            lines.append("loop_mode=loop_continuous")
            lines.append(f"loop_start={loop['start']} loop_end={loop['end']}")
        else:
            lines.append("loop_mode=no_loop")
        lines.append("")
    return "\n".join(lines)


def main():
    root = Path(sys.argv[1])
    cal_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("calib/calibration.json")
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}

    for pdir in sorted(p for p in root.iterdir() if (p / "patch.json").exists()):
        meta = json.loads((pdir / "patch.json").read_text())
        prefix = f"Samples/{pdir.name}"
        safe = "".join(c for c in meta["name"] if c.isalnum() or c in " -_.").strip()
        stem = f"{meta['bank']}{meta['index']:03d} {safe or 'patch'}"
        (root / f"{stem}.dspreset").write_text(build_dspreset(meta, cal, prefix))
        (root / f"{stem}.sfz").write_text(build_sfz(meta, prefix))
        print(f"emitted {stem}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/test_emit_presets.py`**

```python
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import emit_presets as ep  # noqa: E402


def make_meta(reverb_type="Hall1", lfo_form="SIN", stripped=True, sync=1):
    zones = []
    for key in range(24, 97, 3):
        for layer in (1, 2, 3):
            zones.append({
                "key": key, "velocity": 32 * layer, "layer": layer,
                "frames": 200000, "file": f"n{key}_v{layer}.flac",
                "kind": "sustaining",
                "loop": {"enabled": True, "start": 48000, "end": 158000,
                         "crossfade": 500},
                "release": 0.5,
            })
    return {
        "name": "Test Patch", "bank": "A", "index": 0, "sample_rate": 48000,
        "effects": {
            "reverb": {"type": reverb_type, "level": 80, "time": 64, "feedback": 20},
            "chorus": {"type": "Chorus1", "level": 40, "depth": 30, "rate": 50,
                       "feedback": 0, "output": "Mix"},
            "reverb_send": [100, 100, 0, 0], "chorus_send": [80, 80, 0, 0],
            "tone_level": [100, 100, 0, 0], "bend_up": 2, "bend_down": 2,
        },
        "lfo1": {"stripped": stripped, "reason": "strippable", "form": lfo_form,
                 "rate": 60, "delay": 0, "sync": sync,
                 "pitch": 0, "tvf": 0, "tva": 20},
        "lfo2": {"stripped": False, "reason": "no LFO depth", "form": "TRI",
                 "rate": 0, "delay": 0, "sync": 0, "pitch": 0, "tvf": 0, "tva": 0},
        "zones": zones,
    }


CAL = {
    "chorus_rate_hz": {"0": 0.1, "64": 1.5, "127": 6.0},
    "chorus_depth_norm": {"0": 0.0, "127": 1.0},
    "chorus_mix": {"0": 0.0, "127": 0.8},
    "reverb_wet": {"0": 0.0, "127": 0.7},
    "reverb_rt60": {"4": {"0": 0.5, "64": 2.4, "127": 5.0}},
}


def parse(meta, cal=CAL):
    return ET.fromstring(ep.build_dspreset(meta, cal, "Samples/x"))


def test_dspreset_is_valid_xml_with_all_zones():
    root = parse(make_meta())
    samples = root.findall(".//sample")
    assert len(samples) == 25 * 3


def test_key_ranges_tile_without_gaps():
    root = parse(make_meta())
    spans = sorted({(int(s.get("loNote")), int(s.get("hiNote")))
                    for s in root.findall(".//sample")})
    assert spans[0][0] == 0
    assert spans[-1][1] == 127
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert lo == hi + 1, f"gap between {hi} and {lo}"


def test_velocity_ranges_tile_1_to_127():
    root = parse(make_meta())
    vr = sorted({(int(s.get("loVel")), int(s.get("hiVel")))
                 for s in root.findall(".//sample")})
    assert vr[0][0] == 1 and vr[-1][1] == 127
    for (_, hi), (lo, _) in zip(vr, vr[1:]):
        assert lo == hi + 1


def test_reverb_type_uses_reverb_effect():
    root = parse(make_meta(reverb_type="Hall1"))
    types = [e.get("type") for e in root.findall(".//effect")]
    assert "reverb" in types and "delay" not in types


def test_delay_reverb_types_use_delay_effect():
    root = parse(make_meta(reverb_type="Pan-Dly"))
    types = [e.get("type") for e in root.findall(".//effect")]
    assert "delay" in types and "reverb" not in types


def test_chorus_rate_comes_from_calibration():
    root = parse(make_meta())
    ch = [e for e in root.findall(".//effect") if e.get("type") == "chorus"][0]
    # raw rate 50 interpolates between 0.1 @0 and 1.5 @64
    assert 0.5 < float(ch.get("modRate")) < 1.5


def test_stripped_lfo_emits_voice_scope_when_synced():
    root = parse(make_meta(stripped=True, sync=1))
    lfo = root.find(".//lfo")
    assert lfo is not None and lfo.get("scope") == "voice"


def test_free_running_lfo_is_global():
    root = parse(make_meta(stripped=True, sync=0))
    assert root.find(".//lfo").get("scope") == "global"


def test_random_waveform_emits_no_lfo():
    root = parse(make_meta(lfo_form="RND1"))
    assert root.find(".//lfo") is None


def test_unstripped_lfo_emits_no_modulator():
    root = parse(make_meta(stripped=False))
    assert root.find(".//lfo") is None


def test_loop_attributes_present():
    root = parse(make_meta())
    s = root.find(".//sample")
    assert s.get("loopEnabled") == "1"
    assert int(s.get("loopCrossfade")) <= 2000


def test_sfz_has_region_per_zone():
    sfz = ep.build_sfz(make_meta(), "Samples/x")
    assert sfz.count("<region>") == 25 * 3
    assert "loop_mode=loop_continuous" in sfz
    assert "ampeg_release=" in sfz
```

- [ ] **Step 3: Run the tests**

Run: `python3 -m pytest tests/test_emit_presets.py -v`
Expected: 12 tests PASS.

- [ ] **Step 4: Emit for the pilot patch and confirm files resolve**

```bash
python3 tools/emit_presets.py /tmp/jvtest_a calib/calibration.json
ls /tmp/jvtest_a/*.dspreset /tmp/jvtest_a/*.sfz
python3 -c "
import xml.etree.ElementTree as ET, glob, os
f=glob.glob('/tmp/jvtest_a/*.dspreset')[0]
r=ET.parse(f).getroot()
base=os.path.dirname(f)
missing=[s.get('path') for s in r.findall('.//sample')
         if not os.path.exists(os.path.join(base,s.get('path')))]
print('missing samples:', missing[:5], 'count', len(missing))
assert not missing
print('ALL SAMPLE PATHS RESOLVE')
"
```

Expected: `ALL SAMPLE PATHS RESOLVE`.

- [ ] **Step 5: Commit**

```bash
git add tools/emit_presets.py tests/test_emit_presets.py
git commit -m "feat: emit DecentSampler and SFZ presets with calibrated effect mapping"
```

---

### Task 7: Batch orchestrator

**Goal:** Render a whole board (or all boards) in parallel across cores, with resumability.

**Files:**
- Create: `tools/run_batch.py`

**Acceptance Criteria:**
- [ ] `run_batch.py --board "SR-JV80-01 Pop"` renders every patch on that board into the output root.
- [ ] Work is distributed across `os.cpu_count()` processes by default, overridable with `--jobs`.
- [ ] A patch directory containing a complete `patch.json` with FLAC zones is skipped on re-run (resumable).
- [ ] `--list` prints all board names with patch counts and exits.
- [ ] Output goes under `/Volumes/ExtFS/charlesvestal/JV-880 Multisamples/<board>/`.
- [ ] A failing patch logs and continues rather than aborting the board.

**Verify:** `python3 tools/run_batch.py --list` → prints 21 libraries totalling 4,197 patches

**Steps:**

- [ ] **Step 1: Write `tools/run_batch.py`**

```python
#!/usr/bin/env python3
"""Render JV-880 boards in parallel, resumably."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROMS = Path("/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880")
OUT_ROOT = Path("/Volumes/ExtFS/charlesvestal/JV-880 Multisamples")
SAMPLER = Path("build/jv_sampler")


def list_boards():
    """Ask the sampler binary for boards; falls back to Internal only."""
    out = subprocess.run([str(SAMPLER), "--roms", str(ROMS), "--list"],
                         capture_output=True, text=True)
    boards = []
    for line in out.stdout.splitlines():
        if "\t" in line:
            name, count = line.rsplit("\t", 1)
            boards.append((name.strip(), int(count)))
    return boards


def patch_done(pdir: Path) -> bool:
    j = pdir / "patch.json"
    if not j.exists():
        return False
    try:
        meta = json.loads(j.read_text())
    except json.JSONDecodeError:
        return False
    zones = meta.get("zones", [])
    if not zones:
        return False
    return all((pdir / z["file"]).exists() and z["file"].endswith(".flac")
               for z in zones)


def render_one(board: str, index: int, out_dir: str) -> tuple:
    cmd = [str(SAMPLER), "--roms", str(ROMS), "--board", board,
           "--patch", str(index), "--out", out_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (board, index, r.returncode, r.stderr[-400:] if r.returncode else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", help="board name, or 'all'")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="render only first N patches")
    args = ap.parse_args()

    boards = list_boards()
    if args.list:
        total = 0
        for name, count in boards:
            print(f"{name}\t{count}")
            total += count
        print(f"\n{len(boards)} libraries, {total} patches")
        return

    if not args.board:
        sys.exit("--board required (or --list)")

    targets = boards if args.board == "all" else [b for b in boards if b[0] == args.board]
    if not targets:
        sys.exit(f"board not found: {args.board}")

    for name, count in targets:
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = min(count, args.limit) if args.limit else count

        todo = [i for i in range(n)
                if not any(patch_done(p) for p in out_dir.glob(f"{i:03d}_*"))]
        print(f"{name}: {len(todo)}/{n} patches to render")

        failures = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(render_one, name, i, str(out_dir)): i for i in todo}
            for k, f in enumerate(as_completed(futs), 1):
                _, idx, rc, err = f.result()
                if rc != 0:
                    failures += 1
                    print(f"  FAIL patch {idx}: {err}", file=sys.stderr)
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)}")
        print(f"{name}: done ({failures} failures)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `--list` support to `src/jv_sampler.cpp`**

Insert into the argument loop and before rendering:

```cpp
        else if (!strcmp(argv[i], "--list")) list_only = true;
```

Declare `bool list_only = false;` alongside the other locals, and after loading ROMs:

```cpp
    if (list_only) {
        printf("JV-880 Internal\t192\n");
        for (auto &e : scan_expansions(roms_dir + "/expansions"))
            if (e.usable) printf("%s\t%d\n", e.name.c_str(), e.patch_count);
        return 0;
    }
```

- [ ] **Step 3: Verify the board listing**

```bash
cmake --build build --target jv_sampler
python3 tools/run_batch.py --list
```

Expected: 21 lines then `21 libraries, 4197 patches`.

- [ ] **Step 4: Commit**

```bash
git add tools/run_batch.py src/jv_sampler.cpp
git commit -m "feat: parallel resumable batch orchestrator"
```

---

### Task 8: Pilot validation gate

**Goal:** Render the pilot set, then confirm quality against measurable criteria and by ear before committing to the full ~5-6 hour, ~300 GB run.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create: `tools/validate_pilot.py`

**Acceptance Criteria:**
- [ ] Pilot rendered: `JV-880 Internal` (192 patches) and `SR-JV80-04 Vintage Synth` (255 patches).
- [ ] Every patch directory has a `patch.json` whose zone count is exactly 75.
- [ ] Every referenced FLAC exists, is 48000 Hz, 24-bit, stereo.
- [ ] Across the pilot, at least 25% of zones are looped (proves loop detection is finding real loops, not silently failing everywhere).
- [ ] For every looped zone, the amplitude discontinuity at the loop point is under 5% of that sample's peak.
- [ ] Every `.dspreset` parses as XML; key ranges tile 0-127 with no gaps; every sample path resolves.
- [ ] At least one preset with a `Delay`/`Pan-Dly` reverb type emits a `delay` effect, and at least one with a Hall/Room type emits a `reverb` effect.
- [ ] **Human check:** user loads at least 3 pilot presets in DecentSampler, confirms they play across the keyboard, pads sustain without cutting off, and the reconstructed reverb/chorus sounds plausible versus the JV-880 plugin.

**Verify:** `python3 tools/validate_pilot.py "/Volumes/ExtFS/charlesvestal/JV-880 Multisamples"` → `PILOT VALIDATION PASSED`

**Steps:**

- [ ] **Step 1: Render the pilot**

```bash
python3 tools/run_batch.py --board "JV-880 Internal" --jobs 8
python3 tools/run_batch.py --board "SR-JV80-04 Vintage Synth" --jobs 8
```

Note: the sampler writes raw WAV; run post-processing next.

- [ ] **Step 2: Post-process and emit presets**

```bash
ROOT="/Volumes/ExtFS/charlesvestal/JV-880 Multisamples"
for b in "JV-880 Internal" "SR-JV80-04 Vintage Synth"; do
  python3 tools/postprocess.py "$ROOT/$b"
  python3 tools/emit_presets.py "$ROOT/$b" calib/calibration.json
done
```

- [ ] **Step 3: Write `tools/validate_pilot.py`**

```python
#!/usr/bin/env python3
"""Validate a rendered library against the pilot acceptance criteria."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

EXPECTED_ZONES = 75
errors, warnings = [], []


def validate_library(lib: Path):
    pdirs = sorted(p for p in lib.iterdir() if (p / "patch.json").exists())
    if not pdirs:
        errors.append(f"{lib.name}: no patch directories")
        return 0, 0

    looped = total = 0
    for pdir in pdirs:
        meta = json.loads((pdir / "patch.json").read_text())
        zones = meta.get("zones", [])
        if len(zones) != EXPECTED_ZONES:
            errors.append(f"{pdir.name}: {len(zones)} zones, expected {EXPECTED_ZONES}")

        for z in zones:
            total += 1
            f = pdir / z["file"]
            if not f.exists():
                errors.append(f"{pdir.name}/{z['file']}: missing")
                continue
            info = sf.info(str(f))
            if info.samplerate != 48000:
                errors.append(f"{f.name}: {info.samplerate} Hz")
            if info.channels != 2:
                errors.append(f"{f.name}: {info.channels} channels")
            if "24" not in info.subtype:
                errors.append(f"{f.name}: subtype {info.subtype}")

            loop = z.get("loop", {})
            if loop.get("enabled"):
                looped += 1
                x, _ = sf.read(str(f), always_2d=True)
                s, e = loop["start"], loop["end"]
                if e >= len(x) or s >= e:
                    errors.append(f"{f.name}: loop {s}-{e} out of range ({len(x)})")
                    continue
                peak = np.abs(x).max() or 1.0
                disc = float(np.abs(x[s] - x[e]).max())
                if disc > 0.05 * peak:
                    errors.append(f"{f.name}: loop discontinuity {disc/peak:.1%}")
    return looped, total


def validate_presets(lib: Path):
    dsp = sorted(lib.glob("*.dspreset"))
    if not dsp:
        errors.append(f"{lib.name}: no .dspreset emitted")
        return
    saw_reverb = saw_delay = False
    for f in dsp:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError as ex:
            errors.append(f"{f.name}: XML parse error {ex}")
            continue
        types = {e.get("type") for e in root.findall(".//effect")}
        saw_reverb |= "reverb" in types
        saw_delay |= "delay" in types

        spans = sorted({(int(s.get("loNote")), int(s.get("hiNote")))
                        for s in root.findall(".//sample")})
        if not spans:
            errors.append(f"{f.name}: no samples")
            continue
        if spans[0][0] != 0 or spans[-1][1] != 127:
            errors.append(f"{f.name}: key range {spans[0][0]}-{spans[-1][1]} not full")
        for (_, hi), (lo, _) in zip(spans, spans[1:]):
            if lo != hi + 1:
                errors.append(f"{f.name}: key gap {hi}->{lo}")
                break
        for s in root.findall(".//sample"):
            if not (f.parent / s.get("path")).exists():
                errors.append(f"{f.name}: unresolved {s.get('path')}")
                break
    if not saw_reverb:
        warnings.append(f"{lib.name}: no preset emitted a reverb effect")
    if not saw_delay:
        warnings.append(f"{lib.name}: no preset emitted a delay effect")


def main():
    root = Path(sys.argv[1])
    libs = [p for p in root.iterdir() if p.is_dir()]
    looped = total = 0
    for lib in sorted(libs):
        l, t = validate_library(lib)
        validate_presets(lib)
        looped += l
        total += t

    ratio = looped / total if total else 0
    print(f"zones: {total}, looped: {looped} ({ratio:.1%})")
    if total and ratio < 0.25:
        errors.append(f"only {ratio:.1%} of zones looped (expected >= 25%)")

    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for e in errors[:40]:
            print(f"  {e}")
        sys.exit(1)
    print("\nPILOT VALIDATION PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run validation and capture output**

```bash
python3 tools/validate_pilot.py "/Volumes/ExtFS/charlesvestal/JV-880 Multisamples" 2>&1 | tee pilot_validation.log
```

Expected: `PILOT VALIDATION PASSED`. Any errors must be fixed and the affected stage re-run before this task closes.

- [ ] **Step 5: Human listening check — ask the user**

Present the user with at least 3 pilot presets to load in DecentSampler, chosen to cover different behaviours:
- a sustaining pad (Internal `Mighty Pad` or `Analog Pad 1`) — verify it holds indefinitely without a loop click,
- a percussive patch (Preset A `A.Piano 1`) — verify natural decay and no loop,
- an effect-heavy patch (any with a Hall reverb) — verify reconstructed reverb sounds plausible against the JV-880 plugin.

Ask explicitly whether the results are acceptable. Do not proceed to Task 9 without an affirmative answer.

- [ ] **Step 6: Commit**

```bash
git add tools/validate_pilot.py pilot_validation.log
git commit -m "test: pilot validation harness and captured results"
```

---

### Task 9: Full library render

**Goal:** Render the remaining 19 boards and emit all presets.

**Files:** none created; runs existing tooling.

**Acceptance Criteria:**
- [ ] All 21 libraries present under the output root.
- [ ] `validate_pilot.py` passes across the entire output root.
- [ ] Total patch directories equal 4,197.
- [ ] Disk usage is recorded and within the ExtFS free space.

**Verify:** `python3 tools/validate_pilot.py "/Volumes/ExtFS/charlesvestal/JV-880 Multisamples"` → `PILOT VALIDATION PASSED`

**Steps:**

- [ ] **Step 1: Confirm free space before starting**

```bash
df -h /Volumes/ExtFS
```
Expected: at least 400 GB available. If not, stop and report — do not partially fill the disk.

- [ ] **Step 2: Render all remaining boards**

```bash
python3 tools/run_batch.py --board all --jobs 8
```

This is resumable: re-running skips completed patches. Expect ~5-6 hours.

- [ ] **Step 3: Post-process and emit for every library**

```bash
ROOT="/Volumes/ExtFS/charlesvestal/JV-880 Multisamples"
for d in "$ROOT"/*/; do
  python3 tools/postprocess.py "$d"
  python3 tools/emit_presets.py "$d" calib/calibration.json
done
```

- [ ] **Step 4: Validate everything and record size**

```bash
python3 tools/validate_pilot.py "$ROOT" 2>&1 | tee full_validation.log
du -sh "$ROOT"
find "$ROOT" -name patch.json | wc -l    # expect 4197
```

- [ ] **Step 5: Commit**

```bash
git add full_validation.log
git commit -m "chore: full library render validated"
```

---

### Task 10: Investigate boards 97 and 98

**Goal:** Determine whether SR-JV80-97 (Experience III) and 98 (Experience II) can be parsed, and either include them or document why not.

**Files:**
- Create: `tools/probe_expansion.py`

**Acceptance Criteria:**
- [ ] Probe reports header bytes, candidate patch counts, and candidate table offsets for both boards.
- [ ] Either both boards enumerate patches with printable names and are added to the pipeline, or a written conclusion in the spec explains why they cannot be.
- [ ] Board 99 (Experience, 64 patches, parses correctly) is used as the reference for what a working 2 MB board looks like.

**Verify:** `python3 tools/probe_expansion.py` → prints a conclusion for each of 97 and 98

**Steps:**

- [ ] **Step 1: Write `tools/probe_expansion.py`**

```python
#!/usr/bin/env python3
"""Probe non-parsing expansion ROMs for their patch table."""
from pathlib import Path

import numpy as np

EXP = Path("/Users/charlesvestal/Documents/_Songs/Move ROMs/Roland JV880/expansions")
PATCH_SIZE = 0x16a
AA = [2, 0, 3, 4, 1, 9, 13, 10, 18, 17, 6, 15, 11, 16, 8, 5, 12, 7, 14, 19]
DD = [2, 0, 4, 5, 7, 6, 3, 1]


def unscramble(src):
    n = len(src)
    i = np.arange(n, dtype=np.int64)
    addr = (i & ~0xFFFFF).astype(np.int64)
    for j in range(20):
        addr |= ((i >> j) & 1) << AA[j]
    s = src[addr]
    out = np.zeros(n, dtype=np.uint8)
    for j in range(8):
        out |= (((s >> DD[j]) & 1) << j).astype(np.uint8)
    return out


def printable_run(u, off, count=4):
    """True if `count` consecutive PATCH_SIZE-strided names are printable."""
    for k in range(count):
        blk = u[off + k * PATCH_SIZE: off + k * PATCH_SIZE + 12]
        if len(blk) < 12 or not all(32 <= c < 127 for c in blk):
            return False
    return True


def probe(path: Path):
    u = unscramble(np.fromfile(path, dtype=np.uint8))
    print(f"\n=== {path.name} ({len(u)} bytes) ===")
    print("  0x66-0x67:", " ".join(f"{b:02X}" for b in u[0x66:0x68]))
    print("  0x8c-0x8f:", " ".join(f"{b:02X}" for b in u[0x8C:0x90]))

    # Scan the whole image on 2-byte alignment for a run of printable names.
    hits = []
    for off in range(0, len(u) - PATCH_SIZE * 4, 2):
        if printable_run(u, off):
            hits.append(off)
            if len(hits) >= 5:
                break
    if not hits:
        print("  CONCLUSION: no printable patch table found — likely a different")
        print("  ROM layout or an incomplete dump. Exclude from the pipeline.")
        return
    for off in hits:
        names = [bytes(u[off + k * PATCH_SIZE: off + k * PATCH_SIZE + 12])
                 .decode("ascii", "replace") for k in range(4)]
        print(f"  candidate @0x{off:x}: {names}")
    print("  CONCLUSION: candidate table(s) found — verify names look like patch")
    print("  names (not random ASCII) before wiring into jv_rom.cpp.")


def main():
    for pat in ("*97*", "*98*", "*99*"):
        for f in sorted(EXP.glob(pat)):
            probe(f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe**

```bash
python3 tools/probe_expansion.py
```

Compare 97/98 output against 99, which is known good (64 patches at `0x1f9866`, first patch `*Tr.Rhodes`). Note that an earlier scan of 97/98 found only random-looking ASCII (`3!3#2##2#"32`), which suggests no real patch table — expect to confirm exclusion rather than rescue them.

- [ ] **Step 3: Record the conclusion in the spec**

Append the finding to `docs/superpowers/specs/2026-07-28-jv880-multisample-design.md` under the expansion section — either the working offsets, or a statement that these two boards are excluded and why.

- [ ] **Step 4: Commit**

```bash
git add tools/probe_expansion.py docs/superpowers/specs/2026-07-28-jv880-multisample-design.md
git commit -m "chore: probe and document non-parsing Experience boards"
```

---

## Notes for the implementer

- **Never write output to the internal drive.** It has ~22 GB free. Everything goes to `/Volumes/ExtFS`.
- The emulator sources are consumed from the `schwung-jv880` checkout via the `JV_DSP` CMake variable. Do not copy or modify them.
- `render_note` relies on the emulator's `sample_buffer` being drained every `updateSC55(n)` call; the buffer is 4096 int16 (2048 stereo frames), so chunks must stay well under that. 64 is the value the reference harness uses.
- If a whole board renders silent, the most likely cause is the patch bytes not reaching NVRAM before the program change — check `load_patch_bytes` ordering.
- The tone parameter table's first numeric column is the NVRAM offset within the tone, the second is the SysEx index. Mixing them up produces plausible-looking but wrong values.
