#include "jv_render.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// mcu.h is third-party (schwung-jv880) and not ours to fix; pcm.h, which it
// pulls in, trips -Wmissing-braces/-Wmissing-field-initializers under our
// -Wall -Wextra. Suppress just those two, just for this include, so this
// translation unit's build stays warning-free without weakening -Wall
// -Wextra for our own code below.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "mcu.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace jv {

namespace {

const int CHUNK = 64;   // matches the reference harness / audio_buffer_size headroom

// Appends n stereo frames (n*2 int16 values) drained from mcu->sample_buffer
// to out. Must be called right after updateSC55(n): sample_write_ptr resets
// to 0 at the start of every updateSC55 call, so the buffer holds exactly n
// frames starting at index 0.
void drain(MCU *m, std::vector<int16_t> &out, int n) {
    size_t at = out.size();
    out.resize(at + (size_t)n * 2);
    memcpy(out.data() + at, m->sample_buffer, (size_t)n * 2 * sizeof(int16_t));
}

// True when every sample in mcu->sample_buffer[0, n*2) has |amplitude| <= floor.
bool chunk_is_quiet(const MCU *m, int n, double floor) {
    for (int i = 0; i < n * 2; i++) {
        if (std::abs((int)m->sample_buffer[i]) > floor) return false;
    }
    return true;
}

} // namespace

Renderer::~Renderer() {
    delete (MCU *)mcu_;
}

bool Renderer::init(const RomSet &roms) {
    MCU *m = new MCU();
    std::vector<uint8_t> nv = roms.nvram;
    nv[NVRAM_MODE_OFFSET] = 1;   // patch mode
    m->startSC55(roms.rom1.data(), roms.rom2.data(),
                 roms.waverom1.data(), roms.waverom2.data(), nv.data());
    for (int i = 0; i < WARMUP_STEPS; i++) m->updateSC55(1);
    mcu_ = m;
    return true;
}

void Renderer::load_patch_bytes(const std::vector<uint8_t> &bytes, const GridSpec &g) {
    MCU *m = (MCU *)mcu_;
    memcpy(&m->nvram[NVRAM_PATCH_OFFSET], bytes.data(), (size_t)PATCH_SIZE);
    m->nvram[NVRAM_MODE_OFFSET] = 1;
    uint8_t pc[2] = {0xC0, 0x00};
    m->postMidiSC55(pc, 2);

    // Cheap to pull out of bytes we already have; used only to identify the
    // patch in the flush-cap diagnostic below (design note B follow-up).
    current_patch_name_ = trim_patch_name(bytes.data());

    // Settle ONCE here (design note A), not per note: 75 render_note calls
    // per patch would otherwise each pay this cost, wasting emulator time.
    int settle = (int)(g.settle_seconds * SAMPLE_RATE);
    for (int pos = 0; pos < settle; pos += CHUNK) {
        int n = std::min(CHUNK, settle - pos);
        m->updateSC55(n);
    }
}

std::vector<int16_t> Renderer::render_note(int key, int velocity, const GridSpec &g) {
    MCU *m = (MCU *)mcu_;
    std::vector<int16_t> out;

    const int quiet_run_needed = (int)(0.1 * SAMPLE_RATE);   // ~100 ms

    uint8_t note_on[3] = {0x90, (uint8_t)key, (uint8_t)velocity};
    m->postMidiSC55(note_on, 3);

    // Hold: always rendered in full (design note C truncates only the tail).
    int hold_samples = (int)(g.hold_seconds * SAMPLE_RATE);
    int peak = 0;
    for (int pos = 0; pos < hold_samples; pos += CHUNK) {
        int n = std::min(CHUNK, hold_samples - pos);
        m->updateSC55(n);
        size_t before = out.size();
        drain(m, out, n);
        for (size_t i = before; i < out.size(); i++)
            peak = std::max(peak, std::abs((int)out[i]));
    }

    uint8_t note_off[3] = {0x80, (uint8_t)key, 0};
    m->postMidiSC55(note_off, 3);

    // Tail: track running peak, stop once a ~100ms run sits below
    // peak * 10^(silence_db/20). Sustained patches simply never hit that
    // run within tail_seconds and render in full.
    int tail_samples = (int)(g.tail_seconds * SAMPLE_RATE);
    int quiet_run = 0;
    for (int pos = 0; pos < tail_samples; pos += CHUNK) {
        int n = std::min(CHUNK, tail_samples - pos);
        m->updateSC55(n);
        size_t before = out.size();
        drain(m, out, n);

        double floor = (double)peak * std::pow(10.0, g.silence_db / 20.0);
        bool quiet = true;
        for (size_t i = before; i < out.size(); i++) {
            int a = std::abs((int)out[i]);
            peak = std::max(peak, a);
            if ((double)a > floor) quiet = false;
        }

        if (quiet) {
            quiet_run += n;
            if (quiet_run >= quiet_run_needed) break;
        } else {
            quiet_run = 0;
        }
    }

    // Design note B: force release and drain-but-discard until the voice is
    // actually quiet (or a safety cap elapses), so this cell's decay cannot
    // bleed into the next cell's attack.
    //
    // The cap must be generous: measured pad/string patches (e.g. "JV Heaven",
    // internal index 161) take up to ~3.5s of *additional* time beyond the
    // standard 2.5s tail to actually cross the -72dB floor — a 1s cap was
    // verified (via back-to-back render_note calls) to leave audible residual
    // energy at the start of the next cell. 5s covers every pad/string patch
    // measured with headroom, and the loop below still exits early for the
    // vast majority of (fast-decaying) cells, so this mainly costs time on
    // the genuinely slow-release patches that need it.
    uint8_t all_off[3] = {0xB0, 0x7B, 0x00};
    m->postMidiSC55(all_off, 3);

    double floor = (double)peak * std::pow(10.0, g.silence_db / 20.0);
    const int flush_cap = 5 * SAMPLE_RATE;   // 5 s safety bound
    int flush_quiet_run = 0;
    bool flushed_quiet = false;
    for (int pos = 0; pos < flush_cap; pos += CHUNK) {
        int n = std::min(CHUNK, flush_cap - pos);
        m->updateSC55(n);
        if (chunk_is_quiet(m, n, floor)) {
            flush_quiet_run += n;
            if (flush_quiet_run >= quiet_run_needed) { flushed_quiet = true; break; }
        } else {
            flush_quiet_run = 0;
        }
    }

    // The 5s cap is calibrated against measured pad/string patches, but the
    // full library is ~4,197 patches across 20 expansion boards — if some
    // patch's release genuinely outlasts the cap, the original bleed bug
    // (design note B) silently returns for the next cell. Surface it on
    // stderr rather than let that happen unnoticed; this should be rare
    // enough to be a real signal, not routine noise.
    if (!flushed_quiet) {
        fprintf(stderr,
                "warning: flush cap reached without quiet (patch=%s key=%d vel=%d)"
                " — possible bleed into next note\n",
                current_patch_name_.c_str(), key, velocity);
    }

    return out;
}

} // namespace jv
