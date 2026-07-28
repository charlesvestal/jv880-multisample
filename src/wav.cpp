#include "wav.h"
#include <stdio.h>
#include <string.h>

namespace jv {

bool wav_write_s16(const std::string &path, const int16_t *interleaved,
                    int frames, int channels, int sample_rate) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;

    uint32_t data_bytes = (uint32_t)frames * (uint32_t)channels * sizeof(int16_t);
    uint32_t riff_size = 36 + data_bytes;

    bool ok = true;
    ok &= fwrite("RIFF", 1, 4, f) == 4;
    ok &= fwrite(&riff_size, 4, 1, f) == 1;
    ok &= fwrite("WAVE", 1, 4, f) == 4;

    ok &= fwrite("fmt ", 1, 4, f) == 4;
    uint32_t fmt_size = 16;
    ok &= fwrite(&fmt_size, 4, 1, f) == 1;
    uint16_t audio_fmt = 1;   // PCM
    ok &= fwrite(&audio_fmt, 2, 1, f) == 1;
    uint16_t ch = (uint16_t)channels;
    ok &= fwrite(&ch, 2, 1, f) == 1;
    uint32_t sr = (uint32_t)sample_rate;
    ok &= fwrite(&sr, 4, 1, f) == 1;
    uint32_t byte_rate = sr * (uint32_t)channels * sizeof(int16_t);
    ok &= fwrite(&byte_rate, 4, 1, f) == 1;
    uint16_t block_align = (uint16_t)(channels * (int)sizeof(int16_t));
    ok &= fwrite(&block_align, 2, 1, f) == 1;
    uint16_t bits = 16;
    ok &= fwrite(&bits, 2, 1, f) == 1;

    ok &= fwrite("data", 1, 4, f) == 4;
    ok &= fwrite(&data_bytes, 4, 1, f) == 1;
    if (frames > 0 && channels > 0) {
        size_t n = (size_t)frames * (size_t)channels;
        ok &= fwrite(interleaved, sizeof(int16_t), n, f) == n;
    }

    // fclose's result must be folded into the returned status, not just the
    // fwrite checks above: stdio buffers writes, so a small/short file can
    // have every fwrite() report success (the data only reached the
    // userspace buffer) while the real error — e.g. ENOSPC — only surfaces
    // when fclose() performs the final flush. Ignoring it here would let
    // this function return true for a file that was never fully written.
    if (fclose(f) != 0) ok = false;
    return ok;
}

bool wav_read_s16(const std::string &path, std::vector<int16_t> *out,
                   int *channels, int *sample_rate) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;

    char riff[4], wave[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return false;
    }
    uint32_t riff_size = 0;
    if (fread(&riff_size, 4, 1, f) != 1) { fclose(f); return false; }
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return false;
    }

    bool have_fmt = false, have_data = false;
    uint16_t fmt_channels = 0, fmt_bits = 0, fmt_audio_fmt = 0;
    uint32_t fmt_sample_rate = 0;
    std::vector<int16_t> samples;

    while (true) {
        char chunk_id[4];
        size_t got = fread(chunk_id, 1, 4, f);
        if (got != 4) break;   // EOF: no more chunks
        uint32_t chunk_size = 0;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            long chunk_start = ftell(f);
            if (chunk_size < 16) { fclose(f); return false; }
            uint16_t block_align = 0;
            if (fread(&fmt_audio_fmt, 2, 1, f) != 1) { fclose(f); return false; }
            if (fread(&fmt_channels, 2, 1, f) != 1) { fclose(f); return false; }
            if (fread(&fmt_sample_rate, 4, 1, f) != 1) { fclose(f); return false; }
            uint32_t byte_rate = 0;
            if (fread(&byte_rate, 4, 1, f) != 1) { fclose(f); return false; }
            if (fread(&block_align, 2, 1, f) != 1) { fclose(f); return false; }
            if (fread(&fmt_bits, 2, 1, f) != 1) { fclose(f); return false; }
            have_fmt = true;
            // Seek past any extra fmt bytes and the chunk padding byte.
            fseek(f, chunk_start + (long)chunk_size + (long)(chunk_size & 1), SEEK_SET);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            samples.resize(chunk_size / sizeof(int16_t));
            if (chunk_size > 0) {
                size_t n = chunk_size / sizeof(int16_t);
                if (fread(samples.data(), sizeof(int16_t), n, f) != n) {
                    fclose(f);
                    return false;
                }
            }
            have_data = true;
            if (chunk_size & 1) fseek(f, 1, SEEK_CUR);   // padding byte
        } else {
            // Skip unknown chunk (plus padding byte for odd sizes).
            fseek(f, (long)chunk_size + (long)(chunk_size & 1), SEEK_CUR);
        }
    }

    fclose(f);
    if (!have_fmt || !have_data) return false;
    if (fmt_audio_fmt != 1 || fmt_bits != 16) return false;   // PCM 16-bit only

    *out = std::move(samples);
    if (channels) *channels = fmt_channels;
    if (sample_rate) *sample_rate = (int)fmt_sample_rate;
    return true;
}

} // namespace jv
