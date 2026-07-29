// wave_inject — reverse-engineered JV-880 wave-ROM sample injection.
//
// Goal: excite the emulated reverb with a mathematically exact impulse (or
// a known sine) instead of a musical note, by writing synthetic DPCM data
// directly into an in-memory copy of the wave ROM at an existing wave's
// slot, then rendering through the real emulator (src/jv_render.h's
// jv::Renderer / jv::RomSet). See the module-level comment blocks below for
// the encoding this was reverse-engineered against
// (schwung-jv880/src/dsp/pcm.cpp, pcm.h — read-only, never modified).
//
// Subcommands:
//   probe        — empirically locate a (wavegroup,wavenumber)'s ROM
//                  start/end/loop address and bank, by watching the live
//                  pcm_t voice registers right after note-on. This is how
//                  the wave table (address/length/loop per wave number) was
//                  resolved: it is not a flat table this tool parses
//                  statically, it's read back from the running emulator's
//                  own per-voice state after the firmware's wave lookup has
//                  already happened.
//   impulse      — inject a single-sample DPCM impulse at a wave slot,
//                  render DRY, write a WAV. This is the flatness proof.
//   sine         — inject a known-frequency sine (in the DPCM "reference"
//                  domain), render DRY, write a WAV. Frequency proof.
//   capture-irs  — inject the impulse once, then loop reverb type x time
//                  pure-wet renders (the wet render IS the IR, no
//                  deconvolution needed), writing WAVs.
//   capture-delay — like capture-irs, but for Delay/Pan-Dly (reverb types
//                  6-7): pure-wet impulse renders whose taps ARE the echo
//                  train directly (time sweep at feedback=0, feedback sweep
//                  at a fixed time), for direct peak-picking instead of
//                  cross-correlation against a note's own attack shape.
//   capture-chorus-depth — pure-wet chorus, excited by an injected sine
//                  (not the impulse -- depth needs a sustained signal to
//                  track modulation over multiple LFO cycles). Writes a dry
//                  reference plus one wet render per depth sweep step.
//   groundtruth  — render a REAL patch (e.g. A.Piano 1) both wet (native
//                  effect intact, chorus disabled) and dry (reverb zeroed),
//                  using the UNMODIFIED ROM (no injection) — for
//                  tools/ir_capture.py to validate a captured IR or the
//                  parametric delay mapping against real hardware. Also
//                  prints the patch's own native reverb_type/level/time/
//                  feedback as one JSON line on stdout.
//   effects      — dump read_effects() for one (--patch-index) or every
//                  internal patch as JSON lines on stdout; used to find
//                  which patches actually use a given reverb/delay type.
//
// Usage:
//   wave_inject probe        --roms <dir> --wavegroup G --wavenumber N [--key 60]
//   wave_inject impulse      --roms <dir> --wavegroup G --wavenumber N --out f.wav [--key 60] [--amp 127]
//   wave_inject sine         --roms <dir> --wavegroup G --wavenumber N --out f.wav --freq HZ [--key 60] [--amp 100]
//   wave_inject capture-irs  --roms <dir> --wavegroup G --wavenumber N --out-dir DIR [--key 60]
//   wave_inject capture-delay --roms <dir> --wavegroup G --wavenumber N --out-dir DIR [--key 60]
//   wave_inject groundtruth  --roms <dir> --patch-index N --out-dir DIR
//   wave_inject effects      --roms <dir> [--patch-index N]

#include "jv_render.h"
#include "jv_rom.h"
#include "jv_patch.h"
#include "wav.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "mcu.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace jv;

namespace {

// =============================================================================
// ROM SCRAMBLE INVERSION
//
// MCU::startSC55 (mcu.cpp) does NOT hand the raw jv880_waverom{1,2}.bin bytes
// straight to the PCM decoder: it runs them through unscramble() first
// (mcu.cpp:454, byte-identical to jv_rom.cpp's unscramble_rom — same aa[20]
// address-bit permutation, same dd[8] data-bit permutation) into
// pcm.waverom1/waverom2, and THOSE are what PCM_ReadROM(address) actually
// indexes. So injecting a byte at "clean" (post-unscramble, i.e. the address
// space PCM_ReadROM operates in) address `i` means writing the RAW/scrambled
// byte at unscramble's own address(i) into RomSet::waverom1/2 — the vectors
// read directly from the .bin files — such that startSC55's unscramble()
// call reconstructs our intended byte at i.
//
// unscramble(src,dst,len) computes, for each dst index i:
//   address = (i & ~0xfffff) | permute_bits(i & 0xfffff, aa)   // 20 low bits permuted, bit 20 kept
//   dst[i]  = permute_bits(src[address], dd)                    // 8-bit value permuted
//
// Both permutations are simple bit-position remaps (not lossy transforms —
// aa is a bijection on {0..19}, dd a bijection on {0..7}), so this is fully
// invertible: for a target dst[i] == V, precompute address(i) the SAME way
// unscramble() does (no inversion needed for the address part — it's
// already "which raw byte does dst[i] come from"), then invert dd to find
// which raw byte value produces V after dd's forward permutation.
//
// Sanity check (below, run once at startup): after poking, actually calling
// jv::unscramble_rom (jv_rom.h — the SAME permutation tables, already
// reviewed/committed code) on the poked buffer must reproduce the intended
// bytes exactly. This catches any transcription slip in SCRAMBLE_AA/DD
// before it can silently corrupt an injection.
static const int SCRAMBLE_AA[20] = {2, 0, 3, 4, 1, 9, 13, 10, 18, 17,
                                     6, 15, 11, 16, 8, 5, 12, 7, 14, 19};
static const int SCRAMBLE_DD[8] = {2, 0, 4, 5, 7, 6, 3, 1};

uint32_t scrambled_address_for(uint32_t i) {
    uint32_t address = i & ~0xfffffu;
    for (int j = 0; j < 20; j++)
        if (i & (1u << j)) address |= 1u << SCRAMBLE_AA[j];
    return address;
}

uint8_t scrambled_byte_for(uint8_t data) {
    uint8_t srcdata = 0;
    for (int j = 0; j < 8; j++)
        if (data & (1 << j)) srcdata |= (uint8_t)(1 << SCRAMBLE_DD[j]);
    return srcdata;
}

// Writes `value` into `rom` (RomSet::waverom1 or waverom2 — the raw bytes
// read straight off jv880_waverom{1,2}.bin) such that after
// MCU::startSC55's unscramble(), PCM_ReadROM((bank<<21)|clean_addr) returns
// exactly `value`. clean_addr must be < WAVE_BYTES (0x200000).
void poke_clean_byte(std::vector<uint8_t> &rom, uint32_t clean_addr, uint8_t value) {
    uint32_t scrambled = scrambled_address_for(clean_addr);
    rom[scrambled] = scrambled_byte_for(value);
}

// Self-check: round-trip poke_clean_byte through jv::unscramble_rom (the
// already-reviewed, byte-identical permutation used by the emulator itself)
// on a handful of addresses/values, aborting loudly if it doesn't hold.
void selftest_scramble_inverse() {
    static const uint32_t addrs[] = {0, 1, 5, 0x123, 0xfffff, 0x100000,
                                      0x1a2b3c, 0x1fffff, 0xabcde, 0x55555};
    static const uint8_t vals[] = {0x00, 0xff, 0x17, 0x80, 0x7f, 0x55, 0xaa, 0x01};
    std::vector<uint8_t> raw(WAVE_BYTES, 0);
    std::vector<uint8_t> expect(WAVE_BYTES, 0xEE);   // sentinel: untouched addrs unchecked
    for (size_t a = 0; a < sizeof(addrs) / sizeof(addrs[0]); a++) {
        uint8_t v = vals[a % (sizeof(vals) / sizeof(vals[0]))];
        poke_clean_byte(raw, addrs[a], v);
        expect[addrs[a]] = v;
    }
    std::vector<uint8_t> clean(WAVE_BYTES, 0);
    unscramble_rom(raw.data(), clean.data(), WAVE_BYTES);
    for (size_t a = 0; a < sizeof(addrs) / sizeof(addrs[0]); a++) {
        uint8_t want = vals[a % (sizeof(vals) / sizeof(vals[0]))];
        if (clean[addrs[a]] != want) {
            fprintf(stderr,
                    "FATAL: scramble self-test failed at addr 0x%x: got 0x%02x want 0x%02x\n",
                    addrs[a], clean[addrs[a]], want);
            exit(1);
        }
    }
    fprintf(stderr, "scramble self-test: OK (%zu addresses round-tripped)\n",
            sizeof(addrs) / sizeof(addrs[0]));
}

// =============================================================================
// DPCM ENCODER
//
// Reverse-engineered from pcm.cpp:1092-1372 (PCM_Update's per-voice address
// generator + "dpcm" block). Per ROM address step k (an 8-bit signed sample
// byte read at that address), the decoder does:
//
//   preshift = samp[k] << 10
//   shift    = (10 - nibble) & 15                      // nibble: 4-bit exponent, shared per 16-address block
//   shifted  = (preshift << 1) >> shift
//   reference = addclip20(reference, shifted >> 1, shifted & 1)   // 20-bit signed clipped add
//
// For nibble in [0,10] (shift = 10-nibble, non-negative, no & 15 wraparound):
//   shifted = samp[k] << (nibble + 1)   — always even, so shifted&1 == 0 —
//   shifted >> 1 = samp[k] << nibble    — EXACT, no truncation.
// So for nibble in [0,10]:  delta[k] = samp[k] * 2^nibble, exactly.
// (nibble 11-15 wrap to small negative shifts and are deliberately never
// used here — see the comment on ENCODE_MAX_NIBBLE below.)
//
// `reference` (ram1[voice][5]) is reset to 0 whenever the voice is idle, so
// a fresh note starts from ref[-1] = 0 (pcm.cpp ~1046/1436). The encoder
// below reproduces this exactly, in software, using the same addclip20
// 20-bit-signed-clip arithmetic, so the byte stream it produces is
// bit-exact against the real decoder for every value in [0,10] (no rounding
// ambiguity — the C++ software model matches the hardware model exactly,
// there's no "approximately" here for the pure-impulse case).
//
// The nibble/exponent stream lives in a SEPARATE, smaller region of the
// SAME clean ROM half: for absolute ROM address `a` (the raw 20-bit
// "address" register value, before the hiaddr/bank OR), the nibble byte
// address is a>>5, and within that byte, bit4 of `a` selects low nibble
// (0) or high nibble (1) — i.e. one nibble byte covers 32 consecutive
// sample addresses as two 16-address blocks. (pcm.cpp:1080-1099.)
//
// nibble in [0,10] gives an exact multiplier from 2^0=1 up to 2^10=1024;
// combined with samp in [-128,127], the largest single-step delta
// representable is 127*1024 = 130048, comfortably inside addclip20's
// [-0x80000, 0x7ffff] (±524288-ish) 20-bit clip range.
const int ENCODE_MAX_NIBBLE = 10;

int32_t addclip20_sw(int32_t add1, int32_t add2, int32_t cin) {
    uint32_t sum = ((uint32_t)add1 + (uint32_t)add2 + (uint32_t)cin) & 0xfffffu;
    bool neg1 = (add1 & 0x80000) != 0, neg2 = (add2 & 0x80000) != 0;
    bool negs = (sum & 0x80000) != 0;
    if (neg1 && neg2 && !negs) sum = 0x80000;
    else if (!neg1 && !neg2 && negs) sum = 0x7ffff;
    // sign-extend the 20-bit result back to a normal 32-bit int for the
    // caller's convenience (matches sx20() elsewhere in pcm.cpp).
    return ((int32_t)(sum << 12)) >> 12;
}

struct DpcmBlock {
    std::vector<int8_t> samp;   // one signed byte per ROM address in this block
    int nibble = 0;             // shared exponent, 0..ENCODE_MAX_NIBBLE
};

// Encodes a target "reference" trajectory (ref[k] = desired decoder
// accumulator value after processing ROM address start_addr+k, 0-based,
// ref[-1] implicitly 0) into DPCM sample bytes + shared per-block nibble
// exponents, grouped by ABSOLUTE 16-address blocks (block index =
// (start_addr+k) >> 4) — NOT by k's own 0-based index, since the nibble
// stream's block boundaries are fixed to absolute ROM address, and
// start_addr is not guaranteed to be 16-aligned. Returns per-address samp
// bytes and, separately, the packed nibble byte stream (one byte per 32
// consecutive addresses, low nibble = even absolute block, high nibble =
// odd), plus a software re-simulation of the exact decode (for a
// pre-render sanity check) in *reconstructed.
void encode_dpcm(uint32_t start_addr, const std::vector<int64_t> &ref,
                  std::vector<uint8_t> *samp_bytes,
                  std::vector<uint8_t> *nibble_bytes /* keyed by absolute nibble-byte index */,
                  uint32_t *nibble_base_byte_index,
                  std::vector<int64_t> *reconstructed) {
    size_t n = ref.size();
    samp_bytes->assign(n, 0);
    reconstructed->assign(n, 0);

    // Group k=0..n-1 by absolute block index (start_addr+k)>>4.
    uint32_t first_block = (start_addr) >> 4;
    uint32_t last_block = (start_addr + (uint32_t)(n - 1)) >> 4;
    uint32_t nblocks = last_block - first_block + 1;

    std::vector<int> block_nibble(nblocks, 0);

    for (uint32_t b = 0; b < nblocks; b++) {
        uint32_t abs_block = first_block + b;
        uint32_t blk_addr_lo = abs_block << 4;
        uint32_t blk_addr_hi = blk_addr_lo + 15;
        // k range covered by this block, intersected with [0, n)
        int64_t k_lo = (int64_t)blk_addr_lo - (int64_t)start_addr;
        int64_t k_hi = (int64_t)blk_addr_hi - (int64_t)start_addr;
        if (k_lo < 0) k_lo = 0;
        if (k_hi > (int64_t)n - 1) k_hi = (int64_t)n - 1;
        if (k_lo > k_hi) continue;

        // Deltas for this block (against the PREVIOUS ref value, ref[-1]=0).
        int64_t max_abs = 0;
        for (int64_t k = k_lo; k <= k_hi; k++) {
            int64_t prev = (k == 0) ? 0 : ref[(size_t)k - 1];
            int64_t d = ref[(size_t)k] - prev;
            max_abs = std::max(max_abs, std::llabs(d));
        }
        int nib = 0;
        while (nib < ENCODE_MAX_NIBBLE &&
               (max_abs + (1LL << nib) / 2) / (1LL << nib) > 127) {
            nib++;
        }
        // If still not representable (shouldn't happen for our callers —
        // callers keep peak deltas within 127*1024), clamp at max nibble.
        block_nibble[b] = nib;
    }

    // Now actually encode samp bytes AND re-simulate via the real formula
    // (addclip20_sw) so *reconstructed is a byte-exact prediction of what
    // the hardware will decode -- this lets callers assert correctness
    // BEFORE spending time on a full emulator render.
    int64_t reference = 0;
    for (size_t k = 0; k < n; k++) {
        uint32_t abs_addr = start_addr + (uint32_t)k;
        uint32_t b = (abs_addr >> 4) - first_block;
        int nib = block_nibble[b];
        int64_t prev = (k == 0) ? 0 : ref[k - 1];
        int64_t want_delta = ref[k] - prev;
        int64_t divisor = 1LL << nib;
        int64_t q = (want_delta >= 0) ? (want_delta + divisor / 2) / divisor
                                       : -((-want_delta + divisor / 2) / divisor);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        (*samp_bytes)[k] = (int8_t)q;

        // Exact hardware formula for nibble in [0, 10]: delta = samp << nibble.
        int64_t delta = q << nib;
        int32_t shifted_half = (int32_t)(delta);   // already the post->>1 value
        reference = addclip20_sw((int32_t)reference, (int32_t)shifted_half, 0);
        (*reconstructed)[k] = reference;
    }

    // Pack nibble bytes. Byte index (absolute) = abs_block >> 1; low nibble
    // = even abs_block, high nibble = odd abs_block.
    uint32_t nib_byte_lo = first_block >> 1;
    uint32_t nib_byte_hi = last_block >> 1;
    *nibble_base_byte_index = nib_byte_lo;
    nibble_bytes->assign(nib_byte_hi - nib_byte_lo + 1, 0);
    for (uint32_t b = 0; b < nblocks; b++) {
        uint32_t abs_block = first_block + b;
        uint32_t byte_idx = (abs_block >> 1) - nib_byte_lo;
        int shift = (abs_block & 1) ? 4 : 0;
        (*nibble_bytes)[byte_idx] |= (uint8_t)((block_nibble[b] & 0xF) << shift);
    }
}

// =============================================================================
// SYNTHETIC SINGLE-TONE PATCH BUILDER
//
// Tone byte offsets confirmed against jv880_plugin.cpp's TONE_PARAMS table
// (nvram_offset column) and this repo's src/jv_patch.cpp (which already
// independently confirmed level=67, reverbsendlevel=82, and — via
// tools/calibrate.cpp's pure_wet_reverb — drylevel=81, chorussendlevel=83).
// wavegroup = tone byte 0 bits 0-1 (TP_BITFIELD, mask 0x03); wavenumber is
// a PLAIN full byte at tone offset 1 (TP_BYTE — the "2-byte" flag in the
// plugin's table means "needs 2-byte nibblized SysEx on the wire", a MIDI
// transport detail; it is a single NVRAM byte, giving 0-255 wave numbers,
// matching "all 255 waves scanned" in the task brief).
struct ToneOffsets {
    static const int TONESWITCH = 0;        // bit7 (wavegroup shares this same byte, bits0-1)
    static const int WAVENUMBER = 1;        // full byte
    static const int VEL_LO = 3, VEL_HI = 4;
    static const int FXM_SWITCH = 2;        // bit7
    static const int PITCH_COARSE = 37;     // signed
    static const int PITCH_FINE = 38;       // signed
    static const int RANDOM_PITCH_DEPTH = 39;  // bits0-3
    static const int LFO1_PITCH = 31, LFO1_TVF = 32, LFO1_TVA = 33;
    static const int LFO2_PITCH = 34, LFO2_TVF = 35, LFO2_TVA = 36;
    static const int CUTOFF = 52;
    static const int RESONANCE = 53;
    static const int TVF_ENV_DEPTH = 58;
    static const int TVF_T1 = 59, TVF_L1 = 60, TVF_T2 = 61, TVF_L2 = 62;
    static const int TVF_T3 = 63, TVF_L3 = 64, TVF_T4 = 65, TVF_L4 = 66;
    static const int LEVEL = 67;
    static const int PAN = 68;
    static const int TONE_DELAY_TIME = 69;
    static const int TVA_T1 = 74, TVA_L1 = 75, TVA_T2 = 76, TVA_L2 = 77;
    static const int TVA_T3 = 78, TVA_L3 = 79, TVA_T4 = 80;
    static const int DRY_LEVEL = 81;
    static const int REVERB_SEND = 82;
    static const int CHORUS_SEND = 83;
};

// Builds a 362-byte patch with exactly ONE active tone (tone 0), pointed at
// (wavegroup, wavenumber), amplitude envelope forced to instant-max-sustain
// (so our injected impulse/sine isn't attenuated by an attack ramp), filter
// forced fully open (cutoff=127, resonance=0, no TVF envelope depth — the
// decoder's own internal one-pole "reconstruction" smoothing filter,
// pcm.cpp's mult1..mult5 chain, still applies regardless: it is not gated
// by this patch-level filter at all, see the module comment on
// decode_pcm_notes below), no LFO modulation, no FXM, centered pan.
// `drylevel`/`reverbsend`/`chorussend` are left to the caller: proof
// renders want dry (drylevel=127, reverbsend=0) while reverb-IR captures
// want pure-wet (drylevel=0, reverbsend=127) exactly like
// tools/calibrate.cpp's pure_wet_reverb().
std::vector<uint8_t> build_probe_patch(const std::vector<uint8_t> &tmpl, int wavegroup,
                                        int wavenumber, uint8_t dry_level, uint8_t reverb_send,
                                        uint8_t chorus_send = 0) {
    std::vector<uint8_t> p = tmpl;   // start from a real patch so untouched bytes (e.g. name) are sane
    for (int t = 0; t < TONE_COUNT; t++) {
        uint8_t *tp = p.data() + TONE_BASE + t * TONE_STRIDE;
        if (t != 0) {
            tp[ToneOffsets::TONESWITCH] &= (uint8_t)~0x80;   // toneswitch off
            tp[ToneOffsets::LEVEL] = 0;
            continue;
        }
        tp[ToneOffsets::TONESWITCH] = (uint8_t)((tp[ToneOffsets::TONESWITCH] & ~0x83) | 0x80 |
                                                 (wavegroup & 0x03));
        tp[ToneOffsets::WAVENUMBER] = (uint8_t)wavenumber;
        tp[ToneOffsets::VEL_LO] = 1;
        tp[ToneOffsets::VEL_HI] = 127;
        tp[ToneOffsets::FXM_SWITCH] &= (uint8_t)~0x80;
        tp[ToneOffsets::PITCH_COARSE] = 0;
        tp[ToneOffsets::PITCH_FINE] = 0;
        tp[ToneOffsets::RANDOM_PITCH_DEPTH] &= (uint8_t)~0x0F;
        tp[ToneOffsets::LFO1_PITCH] = 0;
        tp[ToneOffsets::LFO1_TVF] = 0;
        tp[ToneOffsets::LFO1_TVA] = 0;
        tp[ToneOffsets::LFO2_PITCH] = 0;
        tp[ToneOffsets::LFO2_TVF] = 0;
        tp[ToneOffsets::LFO2_TVA] = 0;
        tp[ToneOffsets::CUTOFF] = 127;
        tp[ToneOffsets::RESONANCE] = 0;
        tp[ToneOffsets::TVF_ENV_DEPTH] = 64;   // signed byte, 0 depth == raw 64 (matches +64 sign convention)
        tp[ToneOffsets::TVF_T1] = 0; tp[ToneOffsets::TVF_L1] = 127;
        tp[ToneOffsets::TVF_T2] = 0; tp[ToneOffsets::TVF_L2] = 127;
        tp[ToneOffsets::TVF_T3] = 0; tp[ToneOffsets::TVF_L3] = 127;
        tp[ToneOffsets::TVF_T4] = 0; tp[ToneOffsets::TVF_L4] = 127;
        tp[ToneOffsets::LEVEL] = 127;
        tp[ToneOffsets::PAN] = 64;   // center
        tp[ToneOffsets::TONE_DELAY_TIME] = 0;
        tp[ToneOffsets::TVA_T1] = 0; tp[ToneOffsets::TVA_L1] = 127;
        tp[ToneOffsets::TVA_T2] = 0; tp[ToneOffsets::TVA_L2] = 127;
        tp[ToneOffsets::TVA_T3] = 0; tp[ToneOffsets::TVA_L3] = 127;
        tp[ToneOffsets::TVA_T4] = 0;
        tp[ToneOffsets::DRY_LEVEL] = dry_level;
        tp[ToneOffsets::REVERB_SEND] = reverb_send;
        tp[ToneOffsets::CHORUS_SEND] = chorus_send;
    }
    p[24] = (uint8_t)(p[24] & (uint8_t)~(1 << 6));   // portamento off (patch-common)
    p[16] = (uint8_t)(p[16] & ~0x7f);                 // choruslevel = 0 (set_chorus, if used, overrides this)
    return p;
}

void set_reverb(std::vector<uint8_t> &p, int type, int level, int time, int feedback) {
    p[12] = (uint8_t)((p[12] & ~0x07) | (type & 0x07));   // reverbtype: 3 bits (jv_patch.cpp)
    p[13] = (uint8_t)(level & 0x7f);
    p[14] = (uint8_t)(time & 0x7f);
    p[15] = (uint8_t)(feedback & 0x7f);
}

// Pokes patch-common chorus bytes 16-19 (level/depth/rate/feedback), keeping
// byte16's own chorusoutput bit7 (Mix vs Reverb routing) untouched -- this
// tool never needs that bit set, but zeroing it unconditionally would be an
// unrelated side effect of a "set the chorus" call.
void set_chorus(std::vector<uint8_t> &p, int level, int depth, int rate, int feedback) {
    p[16] = (uint8_t)((p[16] & 0x80) | (level & 0x7f));
    p[17] = (uint8_t)(depth & 0x7f);
    p[18] = (uint8_t)(rate & 0x7f);
    p[19] = (uint8_t)(feedback & 0x7f);
}

// =============================================================================
// LOW-LEVEL EMULATOR DRIVER (for `probe`, which needs live pcm_t register
// access jv::Renderer intentionally hides — see src/jv_render.h's own
// comment on mcu_ being opaque). Mirrors src/jv_render.cpp's init/warmup/
// load/settle sequence exactly (same constants), just without the
// encapsulation, so `probe`'s timing matches what `impulse`/`sine`/
// `capture-irs` (which DO go through jv::Renderer) actually experience.

void raw_init(MCU *m, const RomSet &roms) {
    std::vector<uint8_t> nv = roms.nvram;
    nv[NVRAM_MODE_OFFSET] = 1;
    int rc = m->startSC55(roms.rom1.data(), roms.rom2.data(), roms.waverom1.data(),
                          roms.waverom2.data(), nv.data());
    if (rc != 0) {
        fprintf(stderr, "startSC55 failed\n");
        exit(1);
    }
    for (int i = 0; i < WARMUP_STEPS; i++) m->updateSC55(1);
}

void raw_load_patch(MCU *m, const std::vector<uint8_t> &bytes, double settle_seconds) {
    memcpy(&m->nvram[NVRAM_PATCH_OFFSET], bytes.data(), (size_t)PATCH_SIZE);
    m->nvram[NVRAM_MODE_OFFSET] = 1;
    uint8_t pc[2] = {0xC0, 0x00};
    m->postMidiSC55(pc, 2);
    int settle = (int)(settle_seconds * SAMPLE_RATE);
    for (int pos = 0; pos < settle; pos += 64) {
        int n = std::min(64, settle - pos);
        m->updateSC55(n * 2);   // run_frames() convention, see jv_render.cpp
    }
}

// =============================================================================
// PROBE: empirically locate a wave's ROM start/end/loop address + bank.
//
// Method: trigger a note on a synthetic single-tone patch pointed at
// (wavegroup, wavenumber), then single-step the emulator and watch every
// voice slot's pcm_t.ram2[v][7] bit 5 ("key", pcm.cpp ~1421/1427) for a
// 0->1 transition — that identifies which of the 32 hardware voices got
// allocated. At the moment of that transition ("kon" tick: pcm.cpp's
// `kon = key && !okey`), `active = okey && key` is FALSE (okey is the
// PREVIOUS tick's copy of the key bit), so the address-generator's
// `if (active && pcm.nfs) ram1[4] = next_address;` does NOT fire — meaning
// ram1[v][4] at that exact tick still holds whatever the FIRMWARE wrote as
// the wave's start address, before the decode loop has advanced it at all.
// ram1[v][0] (end) and ram1[v][2] (loop) are NEVER written by the decode
// loop itself (grep confirms — only ram1[v][1,3,4,5] are), so they're
// stable/exact at any point after note-on, not just the kon tick.
// ram2[v][7] bits 8-11 ("hiaddr") select the ROM bank: bank = hiaddr>>1,
// and hiaddr&1 is the top bit (bit 20) of the within-bank offset —
// (pcm.cpp:1094/1114/1140/1169/1198, PCM_ReadROM's own (address>>21)&7
// bank dispatch in pcm.h).
struct ProbeResult {
    bool found = false;
    int voice = -1;
    int hiaddr = 0;
    int bank = 0;    // 0=waverom1, 1=waverom2, 2=waverom_card, 3-6=waverom_exp
    int half = 0;    // hiaddr & 1 -> bit 20 of the within-bank clean address
    uint32_t start = 0, end = 0, loop = 0;   // 20-bit "address" register values
    int b6 = 0, b7 = 0;                      // loop-mode flags (ram2[v][7] bits 6/7)
    uint32_t min_addr = 0, max_addr = 0;     // observed over the trace window
    std::vector<uint32_t> trace;             // ram1[v][4] for the first ~40 ticks after kon
};

ProbeResult probe_wave(const RomSet &roms, const std::vector<uint8_t> &patch, int key, int vel,
                       int trace_ticks = 39) {
    ProbeResult result;
    MCU *m = new MCU();
    raw_init(m, roms);
    raw_load_patch(m, patch, 0.5);

    bool okey_before[32];
    for (int v = 0; v < 32; v++) okey_before[v] = (m->pcm.pcm.ram2[v][7] & 0x20) != 0;

    uint8_t note_on[3] = {0x90, (uint8_t)key, (uint8_t)vel};
    m->postMidiSC55(note_on, 3);

    const int MAX_TICKS = 40000;
    int voice = -1;
    for (int tick = 0; tick < MAX_TICKS && voice < 0; tick++) {
        m->updateSC55(2);   // smallest granularity that reliably advances the PCM engine
        for (int v = 0; v < 32; v++) {
            bool key_bit = (m->pcm.pcm.ram2[v][7] & 0x20) != 0;
            if (key_bit && !okey_before[v]) {
                voice = v;
                break;
            }
        }
        if (voice < 0) {
            for (int v = 0; v < 32; v++) okey_before[v] = (m->pcm.pcm.ram2[v][7] & 0x20) != 0;
        }
    }

    if (voice < 0) {
        delete m;
        return result;
    }

    result.found = true;
    result.voice = voice;
    result.hiaddr = (m->pcm.pcm.ram2[voice][7] >> 8) & 15;
    result.bank = result.hiaddr >> 1;
    result.half = result.hiaddr & 1;
    result.start = m->pcm.pcm.ram1[voice][4] & 0xfffff;
    result.end = m->pcm.pcm.ram1[voice][0] & 0xfffff;
    result.loop = m->pcm.pcm.ram1[voice][2] & 0xfffff;
    result.b6 = (m->pcm.pcm.ram2[voice][7] >> 6) & 1;
    result.b7 = (m->pcm.pcm.ram2[voice][7] >> 7) & 1;
    result.trace.push_back(result.start);
    result.min_addr = result.max_addr = result.start;

    for (int i = 0; i < trace_ticks; i++) {
        m->updateSC55(2);
        uint32_t a = m->pcm.pcm.ram1[voice][4] & 0xfffff;
        result.trace.push_back(a);
        result.min_addr = std::min(result.min_addr, a);
        result.max_addr = std::max(result.max_addr, a);
    }

    delete m;
    return result;
}

void cmd_probe(const RomSet &roms, int wavegroup, int wavenumber, int key) {
    auto internal = enumerate_internal(roms);
    if (internal.empty()) { fprintf(stderr, "no internal patches\n"); exit(1); }
    std::vector<uint8_t> tmpl(internal[0].data, internal[0].data + PATCH_SIZE);
    std::vector<uint8_t> patch = build_probe_patch(tmpl, wavegroup, wavenumber, 127, 0);

    ProbeResult r = probe_wave(roms, patch, key, 100, 8000);
    if (!r.found) {
        printf("{\"found\":false}\n");
        return;
    }
    printf("{\"found\":true,\"voice\":%d,\"hiaddr\":%d,\"bank\":%d,\"half\":%d,"
           "\"start\":%u,\"end\":%u,\"loop\":%u,\"length\":%d,\"b6\":%d,\"b7\":%d,"
           "\"min_addr\":%u,\"max_addr\":%u,\"trace_step\":[",
           r.voice, r.hiaddr, r.bank, r.half, r.start, r.end, r.loop,
           (int)r.end - (int)r.start, r.b6, r.b7, r.min_addr, r.max_addr);
    size_t show = std::min<size_t>(60, r.trace.size() > 0 ? r.trace.size() - 1 : 0);
    for (size_t i = 1; i <= show; i++) {
        printf("%d%s", (int)r.trace[i] - (int)r.trace[i - 1], i < show ? "," : "");
    }
    printf("]}\n");
    // Find where address crosses `end` or `loop` within the traced window,
    // for diagnosing loop-wrap behavior.
    for (size_t i = 0; i < r.trace.size(); i++) {
        if (r.trace[i] == r.end || r.trace[i] == r.loop) {
            fprintf(stderr, "trace[%zu] = %u (== %s)\n", i, r.trace[i],
                    r.trace[i] == r.end ? "end" : "loop");
        }
    }
}

// =============================================================================
// INJECTION: given a probed wave slot, write synthetic DPCM content into a
// COPY of RomSet's waverom bytes (never the real files on disk — RomSet is
// loaded fresh per invocation from --roms and only ever mutated in memory).
// `max_rel_err`: the encoder is bit-exact (0 error) for signals whose
// per-block deltas are exact multiples of that block's chosen power-of-two
// scale — always true for the impulse (2 nonzero deltas total, scale chosen
// to fit exactly). A smoothly-varying signal like a sine has NO such
// guarantee (its deltas are generic reals rounded to the nearest
// representable multiple), so callers encoding one pass a tolerance
// (fraction of peak |ref|) instead of demanding exactness outright.
void inject_wave(RomSet *roms, const ProbeResult &probe, const std::vector<int64_t> &ref,
                 double max_rel_err = 0.0) {
    std::vector<uint8_t> &rom = (probe.bank == 0) ? roms->waverom1 : roms->waverom2;
    if (probe.bank > 1) {
        fprintf(stderr, "FATAL: probed wave lives in bank %d (card/expansion) — "
                        "this RomSet only carries waverom1/waverom2\n", probe.bank);
        exit(1);
    }

    // The valid/read address range is [start, end] INCLUSIVE — confirmed
    // empirically via `probe`'s extended address trace on wave13: the
    // "address" register visits the value `end` itself (a real
    // PCM_ReadROM(...,address_cnt) happens there) before the wrap-to-loop
    // logic substitutes `loop` for the FOLLOWING read. An earlier version
    // of this function used exclusive `end - start`, leaving address `end`
    // un-zeroed — the single leftover original-wave byte there got read
    // once per loop cycle (~1348 ticks) and, smeared by interpolation,
    // showed up as a faint (~-46dBFS) recurring artifact over the whole
    // hold+tail. Confirmed fixed by making this inclusive.
    uint32_t max_len = probe.end - probe.start + 1;
    if (ref.size() > max_len) {
        fprintf(stderr, "FATAL: requested %zu samples but wave only spans %u (start=%u end=%u)\n",
                ref.size(), max_len, probe.start, probe.end);
        exit(1);
    }

    std::vector<uint8_t> samp_bytes, nibble_bytes;
    uint32_t nibble_base;
    std::vector<int64_t> reconstructed;
    encode_dpcm(probe.start, ref, &samp_bytes, &nibble_bytes, &nibble_base, &reconstructed);

    // Verify the software model against the target before touching ROM.
    int64_t max_err = 0, peak_ref = 1;
    for (size_t k = 0; k < ref.size(); k++) {
        max_err = std::max(max_err, std::llabs(reconstructed[k] - ref[k]));
        peak_ref = std::max(peak_ref, std::llabs(ref[k]));
    }
    double rel_err = (double)max_err / (double)peak_ref;
    fprintf(stderr,
            "encode_dpcm: %zu samples, max reconstruction error = %lld (%.4f%% of peak %lld)\n",
            ref.size(), (long long)max_err, 100.0 * rel_err, (long long)peak_ref);
    if (rel_err > max_rel_err) {
        fprintf(stderr,
                "FATAL: DPCM encoder error %.4f%% exceeds tolerance %.4f%% — refusing to inject\n",
                100.0 * rel_err, 100.0 * max_rel_err);
        exit(1);
    }

    // Silence the tail explicitly: any sample beyond ref.size() up to
    // max_len is left as whatever the ORIGINAL wave had, which is wrong —
    // zero it (sample byte 0, nibble irrelevant since delta 0*anything=0)
    // so nothing but our intended signal is ever heard, however long the
    // host tone is held.
    for (uint32_t k = (uint32_t)ref.size(); k < max_len; k++) {
        uint32_t abs_addr = probe.start + k;
        uint32_t clean = ((uint32_t)probe.half << 20) | (abs_addr & 0xfffff);
        poke_clean_byte(rom, clean, 0);
    }
    // Also zero the nibble bytes covering the silenced tail (belt-and-
    // braces: a 0 sample byte is silence regardless of nibble/exponent, but
    // zeroing both leaves no ambiguity for anyone reading the injected ROM
    // region later).
    for (uint32_t k = (uint32_t)ref.size(); k < max_len; k++) {
        uint32_t abs_addr = probe.start + k;
        uint32_t nib_idx = abs_addr >> 5;
        uint32_t clean = ((uint32_t)probe.half << 20) | (nib_idx & 0xfffff);
        poke_clean_byte(rom, clean, 0);
    }

    for (size_t k = 0; k < samp_bytes.size(); k++) {
        uint32_t abs_addr = probe.start + (uint32_t)k;
        uint32_t clean = ((uint32_t)probe.half << 20) | (abs_addr & 0xfffff);
        poke_clean_byte(rom, clean, samp_bytes[k]);
    }
    for (size_t i = 0; i < nibble_bytes.size(); i++) {
        uint32_t clean = ((uint32_t)probe.half << 20) | ((nibble_base + (uint32_t)i) & 0xfffff);
        poke_clean_byte(rom, clean, nibble_bytes[i]);
    }

    fprintf(stderr, "injected %zu sample bytes + %zu nibble bytes at bank %d clean [%u,%u)\n",
            samp_bytes.size(), nibble_bytes.size(), probe.bank, probe.start,
            probe.start + (uint32_t)samp_bytes.size());
}

// =============================================================================
// SIGNAL BUILDERS

std::vector<int64_t> build_impulse_ref(int n, int64_t amp) {
    std::vector<int64_t> ref(n, 0);
    if (n >= 1) ref[0] = amp;
    if (n >= 2) ref[1] = 0;   // returns to 0 immediately -> delta[1] = -amp
    return ref;
}

// The DPCM "reference" is a running INTEGRATOR (see encode_dpcm's module
// comment): a delta of 0 means "hold the current value", not "go to zero".
// A plain sine ends at whatever phase sin(2*pi*f*(n-1)/sr) happens to land
// on — essentially never an exact zero-crossing — so without an explicit
// taper, everything from k=n onward (the zero-filled tail inject_wave
// appends up to the host wave's own length) would decode as a sustained DC
// step at that leftover amplitude instead of silence. fade_samples linearly
// ramps the last stretch of the sine down to exactly 0 so the signal HANDS
// OFF cleanly to the caller's zero-fill.
std::vector<int64_t> build_sine_ref(int n, double amp, double freq_hz, double sample_rate,
                                    int fade_samples = 64) {
    std::vector<int64_t> ref(n, 0);
    for (int k = 0; k < n; k++)
        ref[k] = (int64_t)std::llround(amp * std::sin(2.0 * M_PI * freq_hz * (double)k / sample_rate));
    fade_samples = std::min(fade_samples, n);
    for (int i = 0; i < fade_samples; i++) {
        int k = n - fade_samples + i;
        double g = 1.0 - (double)(i + 1) / (double)fade_samples;   // 1 -> 0, excludes a final 1.0 step
        ref[k] = (int64_t)std::llround((double)ref[k] * g);
    }
    if (n > 0) ref[n - 1] = 0;   // exact zero at the handoff point
    return ref;
}

// =============================================================================
// COMMANDS THAT USE jv::Renderer

void write_render(Renderer &r, const std::vector<uint8_t> &patch, const GridSpec &g, int key,
                   int vel, const std::string &out) {
    r.load_patch_bytes(patch, g);
    std::vector<int16_t> pcm = r.render_note(key, vel, g);
    int frames = (int)(pcm.size() / 2);
    if (!wav_write_s16(out, pcm.data(), frames, 2, SAMPLE_RATE)) {
        fprintf(stderr, "failed to write %s\n", out.c_str());
        exit(1);
    }
    fprintf(stderr, "wrote %s (%d frames)\n", out.c_str(), frames);
}

void mkdirs(const std::string &p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); i++) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) mkdir(cur.c_str(), 0755);
    }
}

struct Args {
    std::string roms, out, out_dir;
    int wavegroup = 0, wavenumber = 17, key = 60, vel = 100, patch_index = 0;
    double freq = 440.0, amp = 100000.0;
    int amp_i = 127;
    int n_samples = 4000;
};

Args parse(int argc, char **argv, int start) {
    Args a;
    for (int i = start; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--roms") a.roms = next();
        else if (k == "--out") a.out = next();
        else if (k == "--out-dir") a.out_dir = next();
        else if (k == "--wavegroup") a.wavegroup = atoi(next().c_str());
        else if (k == "--wavenumber") a.wavenumber = atoi(next().c_str());
        else if (k == "--key") a.key = atoi(next().c_str());
        else if (k == "--vel") a.vel = atoi(next().c_str());
        else if (k == "--patch-index") a.patch_index = atoi(next().c_str());
        else if (k == "--freq") a.freq = atof(next().c_str());
        else if (k == "--amp") a.amp = atof(next().c_str());
        else if (k == "--amp-i") a.amp_i = atoi(next().c_str());
        else if (k == "--n") a.n_samples = atoi(next().c_str());
        else { fprintf(stderr, "unknown argument: %s\n", k.c_str()); exit(1); }
    }
    return a;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <probe|impulse|sine|capture-irs|groundtruth> --roms <dir> ...\n",
                argv[0]);
        return 1;
    }
    std::string cmd = argv[1];
    Args a = parse(argc, argv, 2);
    if (a.roms.empty()) { fprintf(stderr, "--roms is required\n"); return 1; }

    selftest_scramble_inverse();

    RomSet roms;
    std::string err;
    if (!roms.load(a.roms, &err)) {
        fprintf(stderr, "failed to load ROMs: %s\n", err.c_str());
        return 1;
    }

    if (cmd == "probe") {
        cmd_probe(roms, a.wavegroup, a.wavenumber, a.key);
        return 0;
    }

    if (cmd == "effects") {
        // Dumps read_effects() (src/jv_patch.cpp) for every internal patch
        // (or just --patch-index, if given) as one JSON object per line on
        // stdout. Used by tools/ir_capture.py / ad-hoc scripts to look up a
        // ground-truth test patch's OWN native reverb/delay type+time+
        // feedback (instead of hardcoding it), and to scan the internal
        // bank for patches that actually use Delay/Pan-Dly (reverb types
        // 6/7) for the Gap 3 validation set — this repo has no static table
        // of "which patch uses which effect", so the only honest way to
        // find one is to read every patch's own bytes.
        auto internal = enumerate_internal(roms);
        // --patch-index 0 is also Args's unset default, so "was it actually
        // passed on the command line" can't be read off a.patch_index alone
        // -- scan argv directly to disambiguate "print everything" (no flag)
        // from "print just index 0" (flag given, value 0).
        bool have_index = false;
        for (int i = 0; i < argc; i++) if (!strcmp(argv[i], "--patch-index")) have_index = true;
        for (size_t i = 0; i < internal.size(); i++) {
            if (have_index && (int)i != a.patch_index) continue;
            const PatchRef &pr = internal[i];
            Effects e = read_effects(pr.data);
            printf("{\"index\":%zu,\"bank\":\"%s\",\"name\":\"%s\",\"reverb_type\":%d,"
                   "\"reverb_level\":%d,\"reverb_time\":%d,\"reverb_feedback\":%d,"
                   "\"chorus_level\":%d,\"chorus_depth\":%d,\"chorus_rate\":%d,"
                   "\"chorus_output\":%d}\n",
                   i, pr.bank.c_str(), pr.name.c_str(), e.reverb_type, e.reverb_level,
                   e.reverb_time, e.reverb_feedback, e.chorus_level, e.chorus_depth,
                   e.chorus_rate, e.chorus_output);
        }
        return 0;
    }

    if (cmd == "playwave") {
        // Renders a wave slot's ORIGINAL, unmodified ROM content (no
        // injection at all) — a same-methodology baseline for comparing
        // measured spectral flatness against the injected impulse/sine.
        auto internal = enumerate_internal(roms);
        std::vector<uint8_t> tmpl(internal[0].data, internal[0].data + PATCH_SIZE);
        std::vector<uint8_t> dry_patch = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 127, 0);
        Renderer r;
        if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
        GridSpec g;
        g.hold_seconds = 0.6;
        g.tail_seconds = 0.5;
        g.silence_db = -120.0;
        write_render(r, dry_patch, g, a.key, a.vel, a.out);
        return 0;
    }

    if (cmd == "impulse" || cmd == "sine") {
        auto internal = enumerate_internal(roms);
        std::vector<uint8_t> tmpl(internal[0].data, internal[0].data + PATCH_SIZE);
        std::vector<uint8_t> probe_patch = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 127, 0);
        ProbeResult pr = probe_wave(roms, probe_patch, a.key, a.vel);
        if (!pr.found) { fprintf(stderr, "probe failed to find a voice\n"); return 1; }
        fprintf(stderr, "probe: bank=%d half=%d start=%u end=%u loop=%u length=%u\n", pr.bank,
                pr.half, pr.start, pr.end, pr.loop, pr.end - pr.start);

        std::vector<int64_t> ref;
        double output_samples_per_rom_step = 2.0;   // overwritten for `sine` below
        if (cmd == "impulse") {
            ref = build_impulse_ref(a.n_samples, (int64_t)a.amp_i * (1 << ENCODE_MAX_NIBBLE));
        } else {
            // ROM-address step k does NOT advance 1:1 with output samples:
            // pcm.cpp's PCM_Update posts 2 output frames per triggered
            // internal DSP tick (confirmed both by jv_render.cpp's own
            // run_frames() comment and directly here — see the impulse
            // proof, where >=96% of consecutive OUTPUT sample pairs are
            // bit-identical). `probe`'s own address trace (pr.trace, 39
            // ticks of updateSC55(2) = 2 output frames each) gives a
            // direct, empirical ROM-steps-per-tick rate; build_sine_ref's
            // `k` argument is a ROM-step index, so the frequency handed to
            // it must be pre-scaled by (2 / avg_rom_steps_per_tick) to land
            // the RENDERED sine at the caller's requested --freq, not at
            // --freq * avg_rom_steps_per_tick / 2.
            double avg_step = 1.0;
            if (pr.trace.size() > 1) {
                double sum = 0;
                for (size_t i = 1; i < pr.trace.size(); i++)
                    sum += (double)pr.trace[i] - (double)pr.trace[i - 1];
                avg_step = sum / (double)(pr.trace.size() - 1);
            }
            output_samples_per_rom_step = 2.0 / std::max(1e-6, avg_step);
            double effective_freq = a.freq * output_samples_per_rom_step;
            fprintf(stderr,
                    "sine: target=%.2f Hz, avg_rom_step/tick=%.4f, output_samples/rom_step=%.4f, "
                    "encoder freq=%.2f Hz\n",
                    a.freq, avg_step, output_samples_per_rom_step, effective_freq);
            ref = build_sine_ref(a.n_samples, a.amp, effective_freq, SAMPLE_RATE);
        }

        // Impulse must be bit-exact (tolerance 0) — that exactness IS the
        // proof. Sine is a smoothly-varying signal with no exact DPCM
        // representation in general; 2% of peak is a tight but achievable
        // per-sample quantization bound (11-bit-ish effective resolution
        // per block) that still leaves the spectral PEAK's frequency/shape
        // clean, which is what the frequency proof actually checks.
        inject_wave(&roms, pr, ref, cmd == "impulse" ? 0.0 : 0.02);

        Renderer r;
        if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
        std::vector<uint8_t> dry_patch = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 127, 0);
        GridSpec g;
        double render_seconds = (double)a.n_samples * output_samples_per_rom_step / SAMPLE_RATE;
        g.hold_seconds = std::max(0.5, render_seconds + 0.05);
        g.tail_seconds = 0.5;
        g.silence_db = -120.0;
        write_render(r, dry_patch, g, a.key, a.vel, a.out);
        return 0;
    }

    if (cmd == "capture-chorus-depth") {
        // Pure-wet chorus, excited by an injected sine (not the impulse):
        // chorus is a delay MODULATED over time, and measuring its depth
        // needs a signal that's actually PRESENT for multiple LFO cycles to
        // track how its delay lag wanders -- a single impulse's response
        // would just be one more (single, static) delay tap, indistinguishable
        // from an unmodulated echo. A held sine, cross-correlated against its
        // own dry copy in short sliding windows, recovers the wet signal's
        // instantaneous LAG directly (see tools/ir_capture.py's
        // measure_chorus_depth_excursion) -- a genuine physical quantity (a
        // delay-time excursion in ms), unlike the three earlier failed
        // attempts (RMS excursion / autocorrelation pitch-tracking / Hilbert
        // instantaneous frequency), none of which tracked delay time itself.
        //
        // rate is fixed at 127 (empirically ~9.4Hz, see calibration.json's
        // chorus_rate_hz) purely so multiple LFO cycles fit inside the
        // ~0.2-0.3s of unique content available from injecting into a single
        // wave slot (this ROM has no usable loop for indefinite sustain --
        // confirmed empirically: held well past its native length, this same
        // host wave decays to near-silence rather than repeating). depth is
        // independent of rate on the JV's own effect parameters (separate
        // patch bytes), so measuring depth's effect at a fast, cycle-dense
        // rate is a valid substitute for the sweep's own eventual (slower,
        // patch-typical) rate value.
        auto internal = enumerate_internal(roms);
        std::vector<uint8_t> tmpl(internal[0].data, internal[0].data + PATCH_SIZE);
        std::vector<uint8_t> probe_patch = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 127, 0);
        ProbeResult pr = probe_wave(roms, probe_patch, a.key, a.vel);
        if (!pr.found) { fprintf(stderr, "probe failed to find a voice\n"); return 1; }
        fprintf(stderr, "probe: bank=%d half=%d start=%u end=%u loop=%u length=%u\n", pr.bank,
                pr.half, pr.start, pr.end, pr.loop, pr.end - pr.start);

        double avg_step = 1.0;
        if (pr.trace.size() > 1) {
            double sum = 0;
            for (size_t i = 1; i < pr.trace.size(); i++)
                sum += (double)pr.trace[i] - (double)pr.trace[i - 1];
            avg_step = sum / (double)(pr.trace.size() - 1);
        }
        double output_samples_per_rom_step = 2.0 / std::max(1e-6, avg_step);

        // Use every available ROM step in the wave slot -- more unique
        // content means more LFO cycles observable in the fixed excitation
        // window, which is the scarce resource here (see comment above).
        uint32_t max_len = pr.end - pr.start + 1;
        int n_samples = (int)max_len;
        double effective_freq = a.freq * output_samples_per_rom_step;
        std::vector<int64_t> ref = build_sine_ref(n_samples, a.amp, effective_freq, SAMPLE_RATE);
        inject_wave(&roms, pr, ref, 0.02);

        Renderer r;
        if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
        mkdirs(a.out_dir);

        double render_seconds = (double)n_samples * output_samples_per_rom_step / SAMPLE_RATE;
        GridSpec g;
        g.hold_seconds = render_seconds + 0.02;
        g.tail_seconds = 0.1;
        g.silence_db = -120.0;

        std::vector<uint8_t> dry_patch = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 127, 0, 0);
        write_render(r, dry_patch, g, a.key, a.vel, a.out_dir + "/chorus_depth_dry.wav");

        std::vector<int> steps;
        for (int raw = 0; raw < 128; raw += 16) steps.push_back(raw);
        if (steps.back() != 127) steps.push_back(127);
        const int FIXED_LEVEL = 100, FIXED_RATE = 127, FIXED_FEEDBACK = 0;
        for (int raw : steps) {
            std::vector<uint8_t> p = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 0, 0, 127);
            set_chorus(p, FIXED_LEVEL, raw, FIXED_RATE, FIXED_FEEDBACK);
            char fn[128];
            snprintf(fn, sizeof(fn), "/chorus_depth_wet_%03d.wav", raw);
            write_render(r, p, g, a.key, a.vel, a.out_dir + fn);
        }
        fprintf(stderr, "chorus depth capture: %.4fs unique content, effective sine %.2fHz\n",
                render_seconds, effective_freq);
        return 0;
    }

    if (cmd == "capture-irs") {
        auto internal = enumerate_internal(roms);
        std::vector<uint8_t> tmpl(internal[0].data, internal[0].data + PATCH_SIZE);
        std::vector<uint8_t> probe_patch = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 127, 0);
        ProbeResult pr = probe_wave(roms, probe_patch, a.key, a.vel);
        if (!pr.found) { fprintf(stderr, "probe failed to find a voice\n"); return 1; }
        fprintf(stderr, "probe: bank=%d half=%d start=%u end=%u loop=%u length=%u\n", pr.bank,
                pr.half, pr.start, pr.end, pr.loop, pr.end - pr.start);

        std::vector<int64_t> ref = build_impulse_ref(a.n_samples, (int64_t)a.amp_i * (1 << ENCODE_MAX_NIBBLE));
        inject_wave(&roms, pr, ref);

        Renderer r;
        if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
        mkdirs(a.out_dir);

        GridSpec g;
        g.hold_seconds = 0.1;
        g.tail_seconds = 8.0;
        g.silence_db = -60.0;

        for (int type = 0; type <= 5; type++) {
            std::vector<int> steps;
            for (int raw = 0; raw < 128; raw += 16) steps.push_back(raw);
            if (steps.back() != 127) steps.push_back(127);
            for (int raw : steps) {
                std::vector<uint8_t> p = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 0, 127);
                set_reverb(p, type, 127, raw, 0);
                char fn[128];
                snprintf(fn, sizeof(fn), "/ir_t%d_time_%03d.wav", type, raw);
                write_render(r, p, g, a.key, a.vel, a.out_dir + fn);
            }
        }
        return 0;
    }

    if (cmd == "capture-delay") {
        // Pure-wet, impulse-excited captures of Delay/Pan-Dly (reverb types
        // 6-7). A delay's pure-wet response to a true impulse IS its taps:
        // a spike at t=delayTime, then (if feedback>0) further decaying
        // spikes at every multiple of the repeat period -- no
        // cross-correlation against a note's own attack shape is needed
        // (that was the old calibrate.cpp/analyze_calibration.py method,
        // which excited with a Marimba strike and had to matched-filter the
        // echo back out of a smeared, harmonically-beating decay). Direct
        // peak-picking on these captures is both simpler and more precise.
        //
        // GOTCHA (see task brief): jv_render.cpp's render_note() tail
        // early-exit threshold is derived from the HOLD phase's own peak.
        // With drylevel=0 the hold phase is silent until the first tap
        // arrives, so if the first tap fell in the TAIL instead of the
        // HOLD, peak would still be 0 there, floor would be exactly 0, and
        // the tail's quiet-run counter would hit its ~100ms threshold and
        // truncate the render to nothing before the first (real, nonzero)
        // tap ever arrived -- confirmed to reproduce a silent capture
        // exactly this way in an earlier attempt. The fix used here is
        // structural, not a threshold tweak: HOLD is always rendered in
        // full regardless of silence_db (jv_render.cpp design note C), so
        // every capture below sets hold_seconds long enough to contain
        // every tap this capture needs to measure, and keeps tail_seconds
        // short -- silence_db is irrelevant to correctness here as long as
        // hold covers the content.
        auto internal = enumerate_internal(roms);
        std::vector<uint8_t> tmpl(internal[0].data, internal[0].data + PATCH_SIZE);
        std::vector<uint8_t> probe_patch = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 127, 0);
        ProbeResult pr = probe_wave(roms, probe_patch, a.key, a.vel);
        if (!pr.found) { fprintf(stderr, "probe failed to find a voice\n"); return 1; }
        fprintf(stderr, "probe: bank=%d half=%d start=%u end=%u loop=%u length=%u\n", pr.bank,
                pr.half, pr.start, pr.end, pr.loop, pr.end - pr.start);

        std::vector<int64_t> ref = build_impulse_ref(a.n_samples, (int64_t)a.amp_i * (1 << ENCODE_MAX_NIBBLE));
        inject_wave(&roms, pr, ref);

        Renderer r;
        if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
        mkdirs(a.out_dir);

        std::vector<int> steps;
        for (int raw = 0; raw < 128; raw += 16) steps.push_back(raw);
        if (steps.back() != 127) steps.push_back(127);

        // TIME sweep: feedback=0 (only the first tap matters), so a hold
        // comfortably above the longest plausible single delay time is
        // enough. 1.0s is >2x the largest raw=127 time this brief's own
        // prior (note-based) measurement found (~0.49s, type 6) -- and even
        // if the impulse remeasurement below moves that number, a delay
        // that took over half a second for its FIRST repeat would not read
        // as "delay" on a percussive instrument at all.
        GridSpec g_time;
        g_time.hold_seconds = 1.0;
        g_time.tail_seconds = 0.3;
        g_time.silence_db = -90.0;

        for (int type : {6, 7}) {
            for (int raw : steps) {
                std::vector<uint8_t> p = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 0, 127);
                set_reverb(p, type, 127, raw, 0);
                char fn[128];
                snprintf(fn, sizeof(fn), "/delay_t%d_time_%03d.wav", type, raw);
                write_render(r, p, g_time, a.key, a.vel, a.out_dir + fn);
            }
        }

        // FEEDBACK sweep: fixed time=64 (same convention calibrate.cpp
        // already used), raw feedback 0..127. Needs many repeat periods
        // (measure_delay_feedback_gain-equivalent analysis below looks out
        // to ~17 periods) all captured within HOLD -- a repeat period at
        // time=64 measured well under 0.3s previously, so 6s of hold gives
        // >20 periods of margin even before this task's remeasurement.
        GridSpec g_fb;
        g_fb.hold_seconds = 6.0;
        g_fb.tail_seconds = 0.3;
        g_fb.silence_db = -90.0;

        for (int type : {6, 7}) {
            for (int raw : steps) {
                std::vector<uint8_t> p = build_probe_patch(tmpl, a.wavegroup, a.wavenumber, 0, 127);
                set_reverb(p, type, 127, 64, raw);
                char fn[128];
                snprintf(fn, sizeof(fn), "/delay_t%d_feedback_%03d.wav", type, raw);
                write_render(r, p, g_fb, a.key, a.vel, a.out_dir + fn);
            }
        }
        return 0;
    }

    if (cmd == "groundtruth") {
        auto internal = enumerate_internal(roms);
        if ((size_t)a.patch_index >= internal.size()) {
            fprintf(stderr, "patch index out of range\n");
            return 1;
        }
        const PatchRef &pr = internal[a.patch_index];
        fprintf(stderr, "groundtruth patch: %s (bank %s, index %d)\n", pr.name.c_str(),
                pr.bank.c_str(), pr.index);
        std::vector<uint8_t> raw(pr.data, pr.data + PATCH_SIZE);

        // Machine-readable echo of this patch's own native effect bytes, on
        // stdout, so callers (tools/ir_capture.py's multi-patch validation)
        // can pick the right IR/type/time/feedback to compare against
        // without a second `effects` subprocess call or hardcoding a
        // patch's settings by hand (which is exactly how the original
        // single-patch groundtruth check came to be hardwired to "A.Piano 1,
        // Hall1/type4, time 64" — a value that isn't even A.Piano 1's real
        // reverb time).
        {
            Effects e = read_effects(pr.data);
            // tone_level/reverb_send arrays let a caller reproduce
            // tools/emit_presets.py's effective_send() (average send across
            // active tones only) exactly, so a reconstruction-vs-groundtruth
            // comparison can use the SAME wet/mix fraction the shipped
            // preset would actually use, not an arbitrary stand-in.
            printf("{\"index\":%d,\"name\":\"%s\",\"reverb_type\":%d,\"reverb_level\":%d,"
                   "\"reverb_time\":%d,\"reverb_feedback\":%d,"
                   "\"tone_level\":[%d,%d,%d,%d],\"reverb_send\":[%d,%d,%d,%d]}\n",
                   a.patch_index, pr.name.c_str(), e.reverb_type, e.reverb_level,
                   e.reverb_time, e.reverb_feedback,
                   e.tone_level[0], e.tone_level[1], e.tone_level[2], e.tone_level[3],
                   e.reverb_send[0], e.reverb_send[1], e.reverb_send[2], e.reverb_send[3]);
        }

        LfoDecision d1 = decide_lfo_strip(pr.data, 1);
        LfoDecision d2 = decide_lfo_strip(pr.data, 2);
        std::vector<uint8_t> stripped = preprocess(pr.data, d1, d2);   // LFO-stripped, reverb ZEROED

        // WET: LFO-stripped (for a clean comparison) but with the patch's
        // OWN native reverb bytes restored (preprocess() zeroed them).
        std::vector<uint8_t> wet = stripped;
        wet[13] = raw[13];   // reverblevel
        wet[14] = raw[14];   // reverbtime
        wet[12] = raw[12];   // reverbtype/chorustype nibble
        wet[15] = raw[15];   // reverbfeedback

        Renderer r;
        if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
        mkdirs(a.out_dir);

        GridSpec g;
        g.hold_seconds = 3.5;
        g.tail_seconds = 6.0;
        g.silence_db = -80.0;

        write_render(r, wet, g, 60, 100, a.out_dir + "/groundtruth_wet.wav");
        write_render(r, stripped, g, 60, 100, a.out_dir + "/groundtruth_dry.wav");
        return 0;
    }

    if (cmd == "abcompare") {
        // Renders the two sides of a listening A/B for one patch at one key:
        //   full.wav -- the patch with its COMPLETE native effect chain
        //               (reverb AND chorus), i.e. what the JV actually sounds
        //               like. `groundtruth` deliberately restores reverb only,
        //               because isolating one effect is right for MEASURING
        //               it; for a listening test the whole chain is the point.
        //   dry.wav  -- exactly what this pipeline samples: effects zeroed.
        // LFO is stripped on BOTH sides. Our shipped samples are LFO-stripped
        // and re-add modulation via DecentSampler <modulators>, so leaving LFO
        // in would compare our effect chain against effects PLUS a modulation
        // path the reconstruction never claimed to cover, and the resulting
        // mismatch would be unattributable.
        auto internal = enumerate_internal(roms);
        if ((size_t)a.patch_index >= internal.size()) {
            fprintf(stderr, "patch index out of range\n");
            return 1;
        }
        const PatchRef &pr = internal[a.patch_index];
        std::vector<uint8_t> raw(pr.data, pr.data + PATCH_SIZE);

        LfoDecision d1 = decide_lfo_strip(pr.data, 1);
        LfoDecision d2 = decide_lfo_strip(pr.data, 2);
        std::vector<uint8_t> dry = preprocess(pr.data, d1, d2);

        std::vector<uint8_t> full = dry;
        full[12] = raw[12];   // reverbtype / chorustype nibble
        full[13] = raw[13];   // reverblevel  (preprocess zeroed)
        full[14] = raw[14];   // reverbtime
        full[15] = raw[15];   // reverbfeedback
        full[16] = raw[16];   // choruslevel + chorusoutput (preprocess zeroed)
        // chorus depth/rate/feedback (17-19) are never zeroed by preprocess.

        Effects e = read_effects(pr.data);
        printf("{\"index\":%d,\"name\":\"%s\",\"key\":%d,\"reverb_type\":%d,"
               "\"reverb_level\":%d,\"reverb_time\":%d,\"reverb_feedback\":%d,"
               "\"chorus_level\":%d,\"chorus_depth\":%d,\"chorus_rate\":%d,"
               "\"tone_level\":[%d,%d,%d,%d],\"reverb_send\":[%d,%d,%d,%d],"
               "\"chorus_send\":[%d,%d,%d,%d]}\n",
               a.patch_index, pr.name.c_str(), a.key, e.reverb_type,
               e.reverb_level, e.reverb_time, e.reverb_feedback,
               e.chorus_level, e.chorus_depth, e.chorus_rate,
               e.tone_level[0], e.tone_level[1], e.tone_level[2], e.tone_level[3],
               e.reverb_send[0], e.reverb_send[1], e.reverb_send[2], e.reverb_send[3],
               e.chorus_send[0], e.chorus_send[1], e.chorus_send[2], e.chorus_send[3]);

        Renderer r;
        if (!r.init(roms)) { fprintf(stderr, "emulator init failed\n"); return 1; }
        mkdirs(a.out_dir);

        GridSpec g;
        g.hold_seconds = 3.5;
        g.tail_seconds = 6.0;
        g.silence_db = -80.0;

        char buf[512];
        snprintf(buf, sizeof(buf), "%s/full_k%d.wav", a.out_dir.c_str(), a.key);
        write_render(r, full, g, a.key, 100, buf);
        snprintf(buf, sizeof(buf), "%s/dry_k%d.wav", a.out_dir.c_str(), a.key);
        write_render(r, dry, g, a.key, 100, buf);
        return 0;
    }

    fprintf(stderr, "unknown subcommand: %s\n", cmd.c_str());
    return 1;
}
