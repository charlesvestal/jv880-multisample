#pragma once
#include <stdint.h>
#include <string>
#include <vector>

namespace jv {

// Standard 44-byte RIFF/WAVE PCM writer/reader, 16-bit signed samples.
bool wav_write_s16(const std::string &path, const int16_t *interleaved,
                    int frames, int channels, int sample_rate);
bool wav_read_s16(const std::string &path, std::vector<int16_t> *out,
                   int *channels, int *sample_rate);

} // namespace jv
