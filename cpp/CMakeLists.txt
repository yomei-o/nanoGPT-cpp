# Cross-platform build for the nanoGPT C++ port.
# Works with MSVC (Visual Studio 2022), GCC and Clang on any OS.
# OpenMP is used automatically if available (optional -- builds fine without).
#
#   cmake -S . -B build
#   cmake --build build --config Release
#
cmake_minimum_required(VERSION 3.15)
project(nanogpt_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(OpenMP QUIET)

function(configure tgt)
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(${tgt} PRIVATE -O3 -ffast-math)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# float + OpenMP: the training / sampling driver (char-level GPT)
add_executable(nanogpt main.cpp)
configure(nanogpt)

# float + OpenMP: real GPT-2 weight-loading inference
add_executable(gpt2 gpt2.cpp)
configure(gpt2)

# double precision: authoritative gradient check (backprop correctness)
add_executable(nanogpt_gradcheck test_grad.cpp)
target_compile_definitions(nanogpt_gradcheck PRIVATE GPT_USE_DOUBLE)
configure(nanogpt_gradcheck)

# BPE tokenizer verification against known GPT-2 token ids
add_executable(bpe_test bpe_test.cpp)
configure(bpe_test)
