/**
 * GPU Runtime Library Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "gpu_runtime.h"
#include "gpu_common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global backend setting
static KrakenBackend g_backend = KRAKEN_BACKEND_AUTO;
static int g_gpu_initialized = 0;
static int g_active_device = 0;

/**
 * Get number of CUDA devices
 */
extern "C"
int kraken_gpu_device_count(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        // Likely no CUDA driver or no GPUs
        return 0;
    }

    return count;
}

/**
 * Check if GPU is available
 */
extern "C"
int kraken_gpu_is_available(void) {
    int count = kraken_gpu_device_count();

    if (count == 0) {
        return 0;
    }

    // Check if at least one device is usable
    cudaDeviceProp prop;
    for (int i = 0; i < count; i++) {
        cudaError_t err = cudaGetDeviceProperties(&prop, i);
        if (err == cudaSuccess && prop.major >= 7) {
            // Require compute capability >= 7.0 (Volta or newer)
            return 1;
        }
    }

    return 0;
}

/**
 * Get device information
 */
extern "C"
void kraken_gpu_get_device_info(int device_id, char* name_out, int* compute_capability_out) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        if (name_out) {
            strcpy(name_out, "Unknown");
        }
        if (compute_capability_out) {
            *compute_capability_out = 0;
        }
        return;
    }

    if (name_out) {
        strncpy(name_out, prop.name, 255);
        name_out[255] = '\0';
    }

    if (compute_capability_out) {
        *compute_capability_out = prop.major * 10 + prop.minor;
    }
}

/**
 * Initialize GPU runtime
 */
extern "C"
int kraken_gpu_init(int device_id) {
    if (g_gpu_initialized) {
        // Already initialized
        return 0;
    }

    int count = kraken_gpu_device_count();
    if (count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    if (device_id < 0 || device_id >= count) {
        fprintf(stderr, "Invalid device ID: %d (available: %d)\n", device_id, count);
        return -1;
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return -1;
    }

    g_active_device = device_id;
    g_gpu_initialized = 1;

    // Print device info
    fprintf(stderr, "KRAKEN GPU Runtime initialized:\n");
    kraken_gpu_print_device_info(device_id);

    return 0;
}

/**
 * Cleanup GPU runtime
 */
extern "C"
void kraken_gpu_cleanup(void) {
    if (!g_gpu_initialized) {
        return;
    }

    cudaDeviceReset();
    g_gpu_initialized = 0;
}

/**
 * Set global backend
 */
extern "C"
void kraken_set_global_backend(KrakenBackend backend) {
    g_backend = backend;

    const char* backend_str;
    switch (backend) {
        case KRAKEN_BACKEND_GPU:
            backend_str = "GPU (forced)";
            break;
        case KRAKEN_BACKEND_CPU:
            backend_str = "CPU (forced)";
            break;
        case KRAKEN_BACKEND_AUTO:
        default:
            backend_str = "AUTO (GPU if available)";
            break;
    }

    fprintf(stderr, "KRAKEN backend set to: %s\n", backend_str);
}

/**
 * Get active backend
 */
extern "C"
KrakenBackend kraken_get_active_backend(void) {
    return g_backend;
}

/**
 * Check if should use GPU (considering environment variable and backend setting)
 */
extern "C"
int kraken_should_use_gpu(void) {
    // Check environment variable first (highest priority)
    const char* env_backend = getenv("KRAKEN_GPU_BACKEND");
    if (env_backend != NULL) {
        if (strcmp(env_backend, "gpu") == 0) {
            return 1;  // Force GPU
        } else if (strcmp(env_backend, "cpu") == 0) {
            return 0;  // Force CPU
        }
        // "auto" or invalid: fall through to backend setting
    }

    // Check global backend setting
    switch (g_backend) {
        case KRAKEN_BACKEND_GPU:
            return 1;

        case KRAKEN_BACKEND_CPU:
            return 0;

        case KRAKEN_BACKEND_AUTO:
        default:
            // Auto-detect: use GPU if available
            return kraken_gpu_is_available();
    }
}
