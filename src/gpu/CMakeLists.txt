# GPU-Accelerated Kernels for PassiveRadar_Kraken
# Copyright (c) 2026 Dr Robert W McGwier, PhD
# SPDX-License-Identifier: MIT
#
# CUDA Support: 11.8+ with enhanced features for CUDA 12/13
# - Backward compatible with CUDA 11.8 (Turing+)
# - CUDA 12+ enables Hopper (sm_90) and Ada Lovelace (sm_89)
# - CUDA 13+ enables Blackwell (sm_100, sm_101, sm_103)
# - Stream-ordered memory pools (CUDA 11.2+)
# - Cooperative groups for efficient inter-warp communication
# - cuSOLVER integration for linear system solving

cmake_minimum_required(VERSION 3.18)

# Detect CUDA Toolkit version for conditional features
if(DEFINED CUDAToolkit_VERSION)
    set(CUDA_VERSION_DETECTED ${CUDAToolkit_VERSION})
else()
    set(CUDA_VERSION_DETECTED "11.8")  # Assume minimum
endif()

# CUDA configuration - C++17 baseline, C++20 for CUDA 12+
if(CUDA_VERSION_DETECTED VERSION_GREATER_EQUAL "12.0")
    set(CMAKE_CUDA_STANDARD 20)
else()
    set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures - build for all supported architectures
# Filter based on CUDA version to avoid unsupported arch errors
if(NOT CMAKE_CUDA_ARCHITECTURES)
    # Base architectures (CUDA 11.8+)
    set(_CUDA_ARCHS "75;80;86;87")
    # 75: Turing (RTX 2000, GTX 1660)
    # 80: Ampere (A100, A30)
    # 86: Ampere (RTX 3000)
    # 87: Jetson Orin

    # CUDA 11.8+ architectures
    if(CUDA_VERSION_DETECTED VERSION_GREATER_EQUAL "11.8")
        list(APPEND _CUDA_ARCHS "89")
        # 89: Ada Lovelace (RTX 4000)
    endif()

    # CUDA 12.0+ architectures
    if(CUDA_VERSION_DETECTED VERSION_GREATER_EQUAL "12.0")
        list(APPEND _CUDA_ARCHS "90")
        # 90: Hopper (H100, H200)
    endif()

    # CUDA 12.4+ architectures (Blackwell virtual)
    if(CUDA_VERSION_DETECTED VERSION_GREATER_EQUAL "12.4")
        list(APPEND _CUDA_ARCHS "120")
        # 120: Blackwell (virtual PTX)
    endif()

    # CUDA 13.0+ architectures (Blackwell native)
    if(CUDA_VERSION_DETECTED VERSION_GREATER_EQUAL "13.0")
        list(REMOVE_ITEM _CUDA_ARCHS "120")  # Replace virtual with native
        list(APPEND _CUDA_ARCHS "100;101;103")
        # 100: Blackwell (B100)
        # 101: Blackwell (B200)
        # 103: Blackwell (B300)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${_CUDA_ARCHS}" CACHE STRING "CUDA architectures")
endif()

# CUDA compilation flags - common for all versions
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

# CUDA 11.5+ features
if(CUDA_VERSION_DETECTED VERSION_GREATER_EQUAL "11.5")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# CUDA 11.0+ features - extended lambda for cooperative groups
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Define version macros for conditional compilation
add_compile_definitions(
    $<$<COMPILE_LANGUAGE:CUDA>:CUDA_VERSION_DETECTED=${CUDA_VERSION_DETECTED}>
)

# Enable position-independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "========================================")
message(STATUS "CUDA 13 GPU Kernel Configuration")
message(STATUS "Building GPU kernels for architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA C++ Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "========================================")

#===============================================================================
# GPU Runtime Library (foundation for all GPU kernels)
# CUDA 13: Enhanced with stream-ordered memory, cuSOLVER, cooperative groups
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
        CUDA::cusolver
        CUDA::cublas
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
# CUDA 13: Uses cuSOLVER for Cholesky decomposition, cuBLAS for matrix ops
add_library(kraken_eca_gpu SHARED eca_gpu.cu)

target_include_directories(kraken_eca_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kraken_eca_gpu
    PUBLIC
        kraken_gpu_runtime
        CUDA::cudart
        CUDA::cusolver
        CUDA::cublas
)

set_target_properties(kraken_eca_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "kraken_eca_gpu"
)

install(TARGETS kraken_eca_gpu
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# UKF/SRUKF GPU kernel (Kalman filter acceleration)
# CUDA 13: Cooperative groups for efficient sigma point propagation
add_library(kraken_ukf_gpu SHARED ukf_gpu.cu)

target_include_directories(kraken_ukf_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kraken_ukf_gpu
    PUBLIC
        kraken_gpu_runtime
        CUDA::cudart
        CUDA::cublas
)

set_target_properties(kraken_ukf_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "kraken_ukf_gpu"
)

install(TARGETS kraken_ukf_gpu
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

#===============================================================================
# Build Summary (CUDA 13)
#===============================================================================

message(STATUS "")
message(STATUS "GPU libraries configured (CUDA 13):")
message(STATUS "  - kraken_gpu_runtime    (device detection, stream-ordered memory pools)")
message(STATUS "  - kraken_caf_gpu        (GPU-accelerated CAF with batched cuFFT)")
message(STATUS "  - kraken_doppler_gpu    (GPU-accelerated Doppler with batched 2D FFT)")
message(STATUS "  - kraken_cfar_gpu       (GPU-accelerated 2D CA-CFAR detection)")
message(STATUS "  - kraken_eca_gpu        (GPU-accelerated ECA-B with cuSOLVER Cholesky)")
message(STATUS "  - kraken_ukf_gpu        (GPU-accelerated UKF sigma point propagation)")
message(STATUS "")
