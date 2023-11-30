# Copyright (C) 2023-2024 Intel Corporation
cmake_minimum_required(VERSION 3.15.1)

# sudo dnf install clang
option(ENABLE_CLANG ">>> Enable clang" OFF)

if(ENABLE_CLANG)
    # clang version 16.0.6
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
else()
    set(CMAKE_C_COMPILER "/usr/local/gcc-12.3.0/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/local/gcc-12.3.0/bin/g++")
endif()

project(xDNN)

# Check GCC version
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
if(GCC_VERSION VERSION_LESS 12)
    message(FATAL_ERROR "ERROR: GCC version MUST >= 12 to support 'constexpr if'")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)  # Disable GNU extensions
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fthread-jumps -falign-functions -falign-jumps -falign-loops  -falign-labels -fcaller-saves -fcrossjumping -fcse-follow-jumps  -fcse-skip-blocks -fdelete-null-pointer-checks -fdevirtualize -fdevirtualize-speculatively  -fexpensive-optimizations -fgcse  -fgcse-lm -fhoist-adjacent-loads -finline-small-functions -findirect-inlining -fipa-cp -fipa-bit-cp -fipa-vrp -fipa-sra -fipa-icf -fisolate-erroneous-paths-dereference -flra-remat -foptimize-sibling-calls -foptimize-strlen -fpartial-inlining -fpeephole2 -freorder-functions -frerun-cse-after-loop -fsched-interblock -fstore-merging -fstrict-aliasing -ftree-builtin-call-dce -ftree-switch-conversion -ftree-tail-merge -fcode-hoisting -ftree-vrp -fipa-ra -mfma -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vbmi2 -mavx512fp16 -mavx512bf16 -funroll-loops -fopenmp -Wno-ignored-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fopenmp -march=sapphirerapids -mtune=sapphirerapids -Wno-ignored-attributes")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Notice: Using Debug mode.")
    set(CMAKE_C_FLAGS "-O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    set(CMAKE_C_FLAGS "-O2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

set(XDNN_LIB "")
option(BUILD_WITH_SHARED_LIBS "Build with shared libraries" OFF)
if(BUILD_WITH_SHARED_LIBS)
    message(STATUS "Notice: Building with shared libraries.")
    list(APPEND XDNN_LIB "xdnn")
else()
    message(STATUS "Notice: Building with static libraries.")
    list(APPEND XDNN_LIB "xdnn_static")
endif()

add_subdirectory(unit_test)
add_subdirectory(perf_test)