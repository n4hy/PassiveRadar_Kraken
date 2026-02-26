# GPU-Accelerated Kernels for PassiveRadar_Kraken
# Copyright (c) 2026 Dr Robert W McGwier, PhD
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (supports Jetson Orin and desktop GPUs)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;87;89;120" CACHE STRING "CUDA architectures")
    # 75: Turing (RTX 2000, GTX 1660)
    # 86: Ampere (RTX 3000, A100)
    # 87: Jetson Orin
    # 89: Ada Lovelace (RTX 4000)
    # 120: Blackwell (RTX 5000)
endif()

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Enable position-independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "Building GPU kernels for architectures: ${CMAKE_CUDA_ARCHITECTURES}")

#===============================================================================
# GPU Runtime Library (foundation for all GPU kernels)
#===============================================================================

add_library(kraken_gpu_runtime SHARED
    gpu_common.cu
    gpu_runtime.cu
    gpu_memory.cu
)

target_include_directories(kraken_gpu_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kraken_gpu_runtime
    PUBLIC
        CUDA::cudart
        CUDA::cufft
)

set_target_properties(kraken_gpu_runtime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "kraken_gpu_runtime"
)

# Install GPU runtime library
install(TARGETS kraken_gpu_runtime
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

#===============================================================================
# GPU-Accelerated Kernels
#===============================================================================

# CAF GPU kernel (Phase 1, Week 2)
add_library(kraken_caf_gpu SHARED caf_gpu.cu)

target_include_directories(kraken_caf_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kraken_caf_gpu
    PUBLIC
        kraken_gpu_runtime
        CUDA::cufft
)

set_target_properties(kraken_caf_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "kraken_caf_gpu"
)

install(TARGETS kraken_caf_gpu
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# Doppler GPU kernel (Phase 2, Week 5)
add_library(kraken_doppler_gpu SHARED doppler_gpu.cu)

target_include_directories(kraken_doppler_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kraken_doppler_gpu
    PUBLIC
        kraken_gpu_runtime
        CUDA::cufft
)

set_target_properties(kraken_doppler_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "kraken_doppler_gpu"
)

install(TARGETS kraken_doppler_gpu
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# CFAR GPU kernel (Phase 2, Week 6)
add_library(kraken_cfar_gpu SHARED cfar_gpu.cu)

target_include_directories(kraken_cfar_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kraken_cfar_gpu
    PUBLIC
        kraken_gpu_runtime
        CUDA::cudart
)

set_target_properties(kraken_cfar_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "kraken_cfar_gpu"
)

install(TARGETS kraken_cfar_gpu
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# ECA-B GPU kernel (adaptive clutter cancellation)
add_library(kraken_eca_gpu SHARED eca_gpu.cu)

target_include_directories(kraken_eca_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kraken_eca_gpu
    PUBLIC
        kraken_gpu_runtime
        CUDA::cudart
)

set_target_properties(kraken_eca_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "kraken_eca_gpu"
)

install(TARGETS kraken_eca_gpu
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

#===============================================================================
# Build Summary
#===============================================================================

message(STATUS "GPU libraries configured:")
message(STATUS "  - kraken_gpu_runtime (device detection, memory management)")
message(STATUS "  - kraken_caf_gpu (GPU-accelerated CAF with batched cuFFT)")
message(STATUS "  - kraken_doppler_gpu (GPU-accelerated Doppler with batched 2D FFT)")
message(STATUS "  - kraken_cfar_gpu (GPU-accelerated 2D CA-CFAR detection)")
message(STATUS "  - kraken_eca_gpu (GPU-accelerated ECA-B clutter canceller with cuBLAS/cuSOLVER)")
