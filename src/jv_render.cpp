#include "jv_render.h"

#include <algorithm>
#include <cassert>
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

// CRITICAL: MCU::updateSC55(nSamples) does NOT produce nSamples stereo
// frames — it is misleadingly named. mcu.h's MCU_PostSample() bumps
// sample_write_ptr once for the L value and once for the R value (see
// mcu.h ~line 396), so sample_write_ptr counts int16 VALUES posted, not
// frames, and it resets to 0 at the top of every updateSC55() call. The
// while-loop condition is `sample_write_ptr < nSamples`, so requesting
// nSamples actually yields ceil(nSamples/2) frames in the naive model —
// and empirically (verified via direct instrumentation: calling
// updateSC55(n) for n in {1..128} and reading sample_write_ptr straight
// after) the real emulator rounds UP to the next multiple of 4 int16
// values (pcm.cpp's PCM_Update posts two frames per triggered oversampled
// DAC batch), i.e. updateSC55(n) always yields exactly ceil(n/4)*4 int16
// values — never fewer, occasionally up to 1 extra frame.
//
// This function is the ONE place that convention is applied: pass the
// number of stereo FRAMES you want; it requests frames*2 from updateSC55
// so the emulator writes at least `frames` real frames into sample_buffer
// before returning, which is what drain()/chunk_is_quiet() below assume.
//
// Before this fix, every render-loop call site passed the frame count
// directly as nSamples (i.e. requested n, not n*2), so updateSC55 only
// produced n/2 real frames while drain() still copied n frames worth of
// buffer — the trailing half was stale leftover content from the
// PREVIOUS updateSC55 call. That silently doubled every rendered note's
// duration and halved its pitch (measured: MIDI 60 rendering at 131 Hz
// instead of 261.63 Hz, ratio 0.501 — exactly a time-stretch-by-2 /
// octave-down error). See tests/test_pitch.py for the regression test.
void run_frames(MCU *m, int frames) {
    m->updateSC55(frames * 2);
}

// Appends n stereo frames (n*2 int16 values) drained from mcu->sample_buffer
// to out. Must be called right after run_frames(m, n): sample_write_ptr
// resets to 0 at the start of every updateSC55() call, and run_frames(m, n)
// guarantees at least n real frames are written starting at index 0 (see
// run_frames's comment for why passing n directly to updateSC55 would be
// wrong).
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
    int rc = m->startSC55(roms.rom1.data(), roms.rom2.data(),
                          roms.waverom1.data(), roms.waverom2.data(), nv.data());
    if (rc != 0) {
        // Leave any previously-initialized mcu_ untouched on failure: a
        // failed re-init should not destroy a still-good prior instance.
        delete m;
        return false;
    }
    // NOT run_frames(m, 1): this intentionally matches the reference
    // harness's literal updateSC55(1) call, not the frames-requested
    // convention used elsewhere. Warmup never reads sample_buffer (no
    // drain), so the n->frames bug doesn't apply here — the only thing
    // that matters is total warmup duration, and updateSC55(1) always
    // yields exactly 2 real frames per call (see run_frames's comment on
    // the ceil(n/4)*4 quantization: n=1 rounds up to 4 int16 values).
    // Verified by direct instrumentation: 100,000 calls x 2 frames =
    // 200,000 frames = 3.125s, matching this constant's origin (the
    // schwung-jv880 reference harness's own comment: "Warmup: ~3 s worth
    // of emulator ticks at 64 kHz"). So WARMUP_STEPS already produces the
    // originally-intended ~3s warmup and needs no adjustment for this fix.
    for (int i = 0; i < WARMUP_STEPS; i++) m->updateSC55(1);

    // Guard against double-init() leaking the previous MCU (several MB):
    // free it before mcu_ is replaced. Safe when mcu_ is still null (delete
    // on nullptr is a no-op).
    delete (MCU *)mcu_;
    mcu_ = m;
    return true;
}

bool Renderer::init_rhythm(const RomSet &roms, const uint8_t *kit, const GridSpec &g,
                           const uint8_t *exp_waves, size_t exp_len) {
    // In-memory copy of rom2 with the wanted kit dropped into the Preset A
    // rhythm region. The file on disk is never touched.
    std::vector<uint8_t> rom2 = roms.rom2;
    if (rom2.size() < (size_t)ROM_RHYTHM_PRESET_A + RHYTHM_SET_BYTES) return false;
    memcpy(&rom2[ROM_RHYTHM_PRESET_A], kit, (size_t)RHYTHM_SET_BYTES);

    MCU *m = new MCU();
    std::vector<uint8_t> nv = roms.nvram;
    nv[NVRAM_MODE_OFFSET] = 0;   // performance mode: rhythm plays on part 10
    if (m->startSC55(roms.rom1.data(), rom2.data(),
                     roms.waverom1.data(), roms.waverom2.data(), nv.data()) != 0) {
        delete m;
        return false;
    }
    // Expansion waves must go in HERE -- after startSC55, before the warmup
    // and before the performance is selected. Calling load_expansion_waves()
    // afterwards instead would work for the waves but silently undo this
    // function's work: that routine performs an SC55_Reset, which discards
    // the performance selection made below and leaves the rhythm part
    // pointing wherever the firmware defaults to. Doing it in this order
    // means exactly one reset, with the performance chosen after it.
    if (exp_waves && exp_len) {
        const size_t cap = sizeof(m->pcm.waverom_exp);
        std::memset(m->pcm.waverom_exp, 0, cap);
        std::memcpy(m->pcm.waverom_exp, exp_waves, std::min(exp_len, cap));
        m->SC55_Reset();
    }

    for (int i = 0; i < WARMUP_STEPS; i++) m->updateSC55(1);

    // Preset A performance 0 -- the performance whose rhythm part reads the
    // region we just injected into.
    uint8_t bank[3] = {0xB0 | 0x0F, 0x00, 81};
    m->postMidiSC55(bank, 3);
    uint8_t pc[2] = {0xC0 | 0x0F, 0x00};
    m->postMidiSC55(pc, 2);

    delete (MCU *)mcu_;
    mcu_ = m;
    channel_ = 9;                // MIDI channel 10
    current_patch_name_ = "rhythm";

    int settle = (int)(g.settle_seconds * SAMPLE_RATE);
    for (int pos = 0; pos < settle; pos += CHUNK) {
        int n = std::min(CHUNK, settle - pos);
        run_frames(m, n);
    }
    return true;
}

void Renderer::load_patch_bytes(const std::vector<uint8_t> &bytes, const GridSpec &g) {
    assert(mcu_ != nullptr && "Renderer::load_patch_bytes called before a successful init()");
    MCU *m = (MCU *)mcu_;
    channel_ = 0;
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
        run_frames(m, n);
    }
}

std::vector<int16_t> Renderer::render_glide(int key_from, int key_to, int velocity,
                                            double first_hold_s, double total_s) {
    assert(mcu_ != nullptr && "Renderer::render_glide called before a successful init()");
    MCU *m = (MCU *)mcu_;

    int first_samples = (int)(first_hold_s * SAMPLE_RATE);
    int total_samples = (int)(total_s * SAMPLE_RATE);
    if (total_samples < first_samples) total_samples = first_samples;

    std::vector<int16_t> out;
    out.reserve((size_t)total_samples * 2);

    uint8_t on_from[3] = {0x90, (uint8_t)key_from, (uint8_t)velocity};
    m->postMidiSC55(on_from, 3);
    for (int pos = 0; pos < first_samples; pos += CHUNK) {
        int n = std::min(CHUNK, first_samples - pos);
        run_frames(m, n);
        drain(m, out, n);
    }

    // Second note ON while the first is still held -- this is the transition
    // the glide happens across.
    uint8_t on_to[3] = {0x90, (uint8_t)key_to, (uint8_t)velocity};
    m->postMidiSC55(on_to, 3);
    int rest = total_samples - first_samples;
    for (int pos = 0; pos < rest; pos += CHUNK) {
        int n = std::min(CHUNK, rest - pos);
        run_frames(m, n);
        drain(m, out, n);
    }

    uint8_t off_to[3]   = {0x80, (uint8_t)key_to, 0};
    uint8_t off_from[3] = {0x80, (uint8_t)key_from, 0};
    m->postMidiSC55(off_to, 3);
    m->postMidiSC55(off_from, 3);
    uint8_t all_off[3] = {0xB0, 0x7B, 0x00};
    m->postMidiSC55(all_off, 3);
    return out;
}


void Renderer::load_expansion_waves(const uint8_t *data, size_t len) {
    assert(mcu_ != nullptr && "load_expansion_waves called before init()");
    MCU *m = (MCU *)mcu_;
    const size_t cap = sizeof(m->pcm.waverom_exp);
    std::memset(m->pcm.waverom_exp, 0, cap);
    std::memcpy(m->pcm.waverom_exp, data, std::min(len, cap));

    // RESET AND RE-WARM. This is not optional and it is the whole reason an
    // earlier attempt failed: init() boots the firmware and warms it for ~3 s
    // with waverom_exp still empty, so the firmware has already decided what
    // waves exist. Filling the array afterwards changes nothing it will look
    // at, and every expansion patch keeps playing internal waves -- a Rhodes
    // comes out as an acoustic piano. The reference implementation says it
    // plainly: "The emulator can't handle waveform ROM swaps with active
    // voices", and resets after every expansion swap.
    m->SC55_Reset();
    for (int i = 0; i < WARMUP_STEPS; i++) m->updateSC55(1);
}

void Renderer::clear_expansion_waves() {
    assert(mcu_ != nullptr && "clear_expansion_waves called before init()");
    MCU *m = (MCU *)mcu_;
    if (!std::memchr(m->pcm.waverom_exp, 1, 0)) { /* no-op guard for clarity */ }
    std::memset(m->pcm.waverom_exp, 0, sizeof(m->pcm.waverom_exp));
    // Same reasoning as load_expansion_waves: the firmware must re-boot to
    // notice the change.
    m->SC55_Reset();
    for (int i = 0; i < WARMUP_STEPS; i++) m->updateSC55(1);
}

std::vector<int16_t> Renderer::render_note(int key, int velocity, const GridSpec &g) {
    assert(mcu_ != nullptr && "Renderer::render_note called before a successful init()");
    MCU *m = (MCU *)mcu_;

    const int quiet_run_needed = (int)(0.1 * SAMPLE_RATE);   // ~100 ms

    // Hold and tail lengths are both known up front, so the output vector's
    // max possible size is too (the tail may truncate early, but never
    // grows past this) — reserve it once instead of letting repeated
    // push-driven resize()s in drain() reallocate/copy as the note grows.
    int hold_samples = (int)(g.hold_seconds * SAMPLE_RATE);
    int tail_samples = (int)(g.tail_seconds * SAMPLE_RATE);
    std::vector<int16_t> out;
    out.reserve((size_t)(hold_samples + tail_samples) * 2);

    uint8_t note_on[3] = {(uint8_t)(0x90 | channel_), (uint8_t)key, (uint8_t)velocity};
    m->postMidiSC55(note_on, 3);

    // Hold: always rendered in full (design note C truncates only the tail).
    int peak = 0;
    for (int pos = 0; pos < hold_samples; pos += CHUNK) {
        int n = std::min(CHUNK, hold_samples - pos);
        run_frames(m, n);
        size_t before = out.size();
        drain(m, out, n);
        for (size_t i = before; i < out.size(); i++)
            peak = std::max(peak, std::abs((int)out[i]));
    }

    uint8_t note_off[3] = {(uint8_t)(0x80 | channel_), (uint8_t)key, 0};
    m->postMidiSC55(note_off, 3);

    // Tail: track running peak, stop once a ~100ms run sits below
    // peak * 10^(silence_db/20). Sustained patches simply never hit that
    // run within tail_seconds and render in full.
    int quiet_run = 0;
    for (int pos = 0; pos < tail_samples; pos += CHUNK) {
        int n = std::min(CHUNK, tail_samples - pos);
        run_frames(m, n);
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
    uint8_t all_off[3] = {(uint8_t)(0xB0 | channel_), 0x7B, 0x00};
    m->postMidiSC55(all_off, 3);

    double floor = (double)peak * std::pow(10.0, g.silence_db / 20.0);
    const int flush_cap = 5 * SAMPLE_RATE;   // 5 s safety bound
    int flush_quiet_run = 0;
    bool flushed_quiet = false;
    for (int pos = 0; pos < flush_cap; pos += CHUNK) {
        int n = std::min(CHUNK, flush_cap - pos);
        run_frames(m, n);
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
