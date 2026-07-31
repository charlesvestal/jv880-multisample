cmake_minimum_required(VERSION 3.20)
project(jv880_multisample CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -Wall -Wextra is applied per-target below (via JV_WARN_FLAGS), not globally
# with add_compile_options: this directory also builds jv_emu from
# third-party schwung-jv880 sources we must not modify, and a directory-wide
# add_compile_options() would flood every build log with ~60 warnings from
# that code, burying any genuine new warning in our own sources.
set(JV_WARN_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(JV_WARN_FLAGS -Wall -Wextra)
endif()

# The cycle-accurate JV-880 emulator core (mcu/pcm), from schwung-jv880. It is
# a SUBMODULE rather than vendored: it is third-party in origin (MAME licence,
# inherited from Nuked-SC55 and juce-jv880) and has its own history worth
# keeping separate. `git clone --recursive` gets it, and so does
# `git submodule update --init`.
#
# Override only if you already have a checkout elsewhere:
#   cmake -S . -B build -DJV_DSP=/path/to/schwung-jv880/src/dsp
if(NOT JV_DSP AND DEFINED ENV{JV880_DSP})
  set(JV_DSP "$ENV{JV880_DSP}")
endif()
if(NOT JV_DSP)
  set(JV_DSP "${CMAKE_CURRENT_SOURCE_DIR}/external/schwung-jv880/src/dsp")
endif()
set(JV_DSP "${JV_DSP}" CACHE PATH "Path to schwung-jv880 emulator sources (src/dsp)")
if(NOT EXISTS "${JV_DSP}/mcu.cpp")
  message(FATAL_ERROR
    "Emulator sources not found at ${JV_DSP}.\n"
    "Fetch the submodule:  git submodule update --init\n"
    "Or point at your own checkout:  -DJV_DSP=/path/to/schwung-jv880/src/dsp")
endif()

add_library(jv_core STATIC src/jv_rom.cpp)
target_include_directories(jv_core PUBLIC src)
target_compile_options(jv_core PRIVATE ${JV_WARN_FLAGS})

add_executable(test_jv_rom tests/test_jv_rom.cpp)
target_link_libraries(test_jv_rom jv_core)
target_compile_options(test_jv_rom PRIVATE ${JV_WARN_FLAGS})

enable_testing()
add_test(NAME test_jv_rom COMMAND test_jv_rom)

# Task 2: patch preprocessing (src/jv_patch.h/.cpp, tests/test_jv_patch.cpp).
# jv_core is already declared above — append its new sources with
# target_sources() rather than calling add_library(jv_core ...) again,
# which errors with "add_library cannot create target ... because another
# target with the same name already exists".
target_sources(jv_core PRIVATE src/jv_patch.cpp)
add_executable(test_jv_patch tests/test_jv_patch.cpp)
target_link_libraries(test_jv_patch jv_core)
target_compile_options(test_jv_patch PRIVATE ${JV_WARN_FLAGS})
add_test(NAME test_jv_patch COMMAND test_jv_patch)

# Task 3: grid renderer, links the schwung-jv880 emulator core via JV_DSP.
# jv_emu is third-party (schwung-jv880) code we must not modify, so it
# deliberately does NOT get JV_WARN_FLAGS — its warnings are not ours to fix
# and shouldn't drown out warnings from our own targets.
add_library(jv_emu STATIC
    ${JV_DSP}/mcu.cpp ${JV_DSP}/mcu_opcodes.cpp ${JV_DSP}/pcm.cpp)
target_include_directories(jv_emu PUBLIC ${JV_DSP})

# wav.cpp lives in jv_render (not jv_core): it's WAV I/O for the renderer's
# output, unrelated to jv_core's ROM/patch-parsing scope.
add_library(jv_render STATIC src/jv_render.cpp src/wav.cpp)
target_link_libraries(jv_render jv_core jv_emu)
target_compile_options(jv_render PRIVATE ${JV_WARN_FLAGS})

add_executable(jv_sampler src/jv_sampler.cpp)
target_link_libraries(jv_sampler jv_render)
target_compile_options(jv_sampler PRIVATE ${JV_WARN_FLAGS})

# Task 4: effect calibration (tools/calibrate.cpp).
add_executable(calibrate tools/calibrate.cpp)
target_link_libraries(calibrate jv_render)
target_compile_options(calibrate PRIVATE ${JV_WARN_FLAGS})

# Wave-ROM injection harness (tools/wave_inject.cpp): renders effect IRs by
# substituting a synthetic impulse into an in-memory copy of the wave ROM, and
# renders the two sides of a listening A/B. Previously compiled ad hoc, which
# left no reproducible way to rebuild it after a source change.
add_executable(wave_inject tools/wave_inject.cpp)
target_link_libraries(wave_inject jv_render)
target_compile_options(wave_inject PRIVATE ${JV_WARN_FLAGS})

# Rhythm Set (drum kit) probe (tools/probe_rhythm.cpp): determines which boot
# mode actually sounds a rhythm key, by measurement rather than assumption.
add_executable(probe_rhythm tools/probe_rhythm.cpp)
target_link_libraries(probe_rhythm jv_render)
target_compile_options(probe_rhythm PRIVATE ${JV_WARN_FLAGS})
