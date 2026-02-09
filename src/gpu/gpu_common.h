/**
 * GPU Common Utilities for PassiveRadar_Kraken
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Shared CUDA utilities, error checking, and common definitions.
 */

#ifndef KRAKEN_GPU_COMMON_H
#define KRAKEN_GPU_COMMON_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CUDA Error Checking Macros
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT Error in %s:%d: Code %d\n", \
                    __FILE__, __LINE__, err); \
            return cudaErrorUnknown; \
        } \
    } while (0)

/**
 * Utility Functions
 */

// Get human-readable device name
const char* kraken_gpu_get_error_string(cudaError_t error);

// Print GPU device information
void kraken_gpu_print_device_info(int device_id);

// Query CUDA compute capability
int kraken_gpu_get_compute_capability(int device_id);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_GPU_COMMON_H
