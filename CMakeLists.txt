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

set(JV_DSP "/Volumes/ExtFS/charlesvestal/github/schwung-parent/schwung-jv880/src/dsp"
    CACHE PATH "Path to schwung-jv880 emulator sources")

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
