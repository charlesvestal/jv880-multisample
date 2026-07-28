cmake_minimum_required(VERSION 3.20)
project(jv880_multisample CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

set(JV_DSP "/Volumes/ExtFS/charlesvestal/github/schwung-parent/schwung-jv880/src/dsp"
    CACHE PATH "Path to schwung-jv880 emulator sources")

add_library(jv_core STATIC src/jv_rom.cpp)
target_include_directories(jv_core PUBLIC src)

add_executable(test_jv_rom tests/test_jv_rom.cpp)
target_link_libraries(test_jv_rom jv_core)

enable_testing()
add_test(NAME test_jv_rom COMMAND test_jv_rom)

# Task 2: patch preprocessing (src/jv_patch.h/.cpp, tests/test_jv_patch.cpp).
# jv_core is already declared above — append its new sources with
# target_sources() rather than calling add_library(jv_core ...) again,
# which errors with "add_library cannot create target ... because another
# target with the same name already exists".
# target_sources(jv_core PRIVATE src/jv_patch.cpp src/wav.cpp)
# add_executable(test_jv_patch tests/test_jv_patch.cpp)
# target_link_libraries(test_jv_patch jv_core)
# add_test(NAME test_jv_patch COMMAND test_jv_patch)

# Task 3: grid renderer, links the schwung-jv880 emulator core via JV_DSP.
# Uncomment once src/jv_render.cpp, src/jv_sampler.cpp, src/wav.cpp exist.
# add_library(jv_emu STATIC
#     ${JV_DSP}/mcu.cpp ${JV_DSP}/mcu_opcodes.cpp ${JV_DSP}/pcm.cpp)
# target_include_directories(jv_emu PUBLIC ${JV_DSP})
#
# add_library(jv_render STATIC src/jv_render.cpp)
# target_link_libraries(jv_render jv_core jv_emu)
#
# add_executable(jv_sampler src/jv_sampler.cpp)
# target_link_libraries(jv_sampler jv_render)

# Task 4: effect calibration (tools/calibrate.cpp).
# Uncomment once that source exists.
# add_executable(calibrate tools/calibrate.cpp)
# target_link_libraries(calibrate jv_render)
