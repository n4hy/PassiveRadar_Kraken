/**
 * GPU Runtime Library Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * CUDA Version Support:
 * - CUDA 11.8+: Core functionality
 * - CUDA 12.0+: Hopper architecture (sm_90)
 * - CUDA 13.0+: Blackwell architecture (sm_100/101/103)
 */

#include "gpu_runtime.h"
#include "gpu_common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** kraken_gpu_cuda_version - Return CUDA runtime version (e.g., 13000 for CUDA 13.0) */
extern "C"
int kraken_gpu_cuda_version(void) {
    return CUDART_VERSION;
}

/** kraken_gpu_driver_version - Return installed CUDA driver version */
extern "C"
int kraken_gpu_driver_version(void) {
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    return driver_version;
}

// Global backend setting
static KrakenBackend g_backend = KRAKEN_BACKEND_AUTO;
static int g_gpu_initialized = 0;
static int g_active_device = 0;  // Used to track which device is active

/**
 * kraken_gpu_device_count - Return number of available CUDA devices
 *
 * Technique: Calls cudaGetDeviceCount; returns 0 if no driver or no GPUs.
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
 * kraken_gpu_is_available - Check if a usable GPU (Volta+) is present
 *
 * Technique: Enumerates devices and checks compute capability >= 7.0
 * (Volta or newer) to ensure adequate feature support.
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
 * kraken_gpu_get_device_info - Query device name and compute capability
 *
 * Technique: Reads cudaDeviceProp for the specified device and copies
 * name string and compute capability (major*10 + minor) to output pointers.
 */
extern "C"
void kraken_gpu_get_device_info(int device_id, char* name_out, int* compute_capability_out) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        if (name_out) {
            strncpy(name_out, "Unknown", 255);
            name_out[255] = '\0';
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
 * kraken_gpu_init - Initialize CUDA runtime on specified device
 *
 * Technique: Validates device_id, calls cudaSetDevice, and prints device
 * info. Idempotent - returns 0 if already initialized. Sets global state
 * for active device tracking.
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
 * kraken_gpu_cleanup - Reset CUDA device and release all resources
 *
 * Technique: Calls cudaDeviceReset to destroy all allocations, streams,
 * and contexts on the active device. Clears initialized flag.
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
 * kraken_set_global_backend - Set processing backend (GPU/CPU/AUTO)
 *
 * Technique: Stores backend preference in global state. Logs the
 * selected mode to stderr for diagnostics.
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

/** kraken_get_active_backend - Return current backend setting (GPU/CPU/AUTO) */
extern "C"
KrakenBackend kraken_get_active_backend(void) {
    return g_backend;
}

/**
 * kraken_should_use_gpu - Determine whether GPU should be used for processing
 *
 * Technique: Priority-based decision: (1) KRAKEN_GPU_BACKEND environment
 * variable ("gpu"/"cpu") takes highest priority, (2) global backend
 * setting (GPU/CPU/AUTO), (3) in AUTO mode, checks hardware availability
 * via kraken_gpu_is_available().
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
