/**
 * GPU Common Utilities Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "gpu_common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * kraken_gpu_get_error_string - Convert CUDA error code to human-readable string
 *
 * Technique: Wraps cudaGetErrorString for C API exposure.
 */
extern "C"
const char* kraken_gpu_get_error_string(cudaError_t error) {
    return cudaGetErrorString(error);
}

/**
 * kraken_gpu_print_device_info - Print detailed GPU device properties to stdout
 *
 * Technique: Queries cudaDeviceProp for compute capability, memory size,
 * multiprocessor count, warp size, memory bus width, and L2 cache size.
 * Handles CUDA 12+ deprecation of memoryClockRate via version guard.
 */
extern "C"
void kraken_gpu_print_device_info(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n",
                cudaGetErrorString(err));
        return;
    }

    printf("GPU Device %d: %s\n", device_id, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
#if CUDART_VERSION < 12000
    // memoryClockRate deprecated in CUDA 12, removed in CUDA 13
    printf("  Memory Clock Rate: %.2f GHz\n",
           prop.memoryClockRate * 1e-6);
#endif
    printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
}

/**
 * kraken_gpu_get_compute_capability - Get SM compute capability as integer
 *
 * Technique: Returns major*10 + minor (e.g., 87 for sm_87, 100 for sm_100).
 * Returns 0 on error.
 */
extern "C"
int kraken_gpu_get_compute_capability(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        return 0;
    }

    return prop.major * 10 + prop.minor;
}
