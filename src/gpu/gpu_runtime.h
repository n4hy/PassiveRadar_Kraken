/**
 * GPU Runtime Library for PassiveRadar_Kraken
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Device detection, backend selection, and runtime management.
 */

#ifndef KRAKEN_GPU_RUNTIME_H
#define KRAKEN_GPU_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Backend Selection Types
 */
typedef enum {
    KRAKEN_BACKEND_AUTO = 0,  // Auto-detect (prefer GPU if available)
    KRAKEN_BACKEND_GPU = 1,   // Force GPU (fail if unavailable)
    KRAKEN_BACKEND_CPU = 2    // Force CPU (even if GPU present)
} KrakenBackend;

/**
 * Device Query Functions
 */

// Get number of CUDA devices
int kraken_gpu_device_count(void);

// Check if GPU is available and ready
int kraken_gpu_is_available(void);

// Get device information (name, compute capability)
void kraken_gpu_get_device_info(int device_id, char* name_out, int* compute_capability_out);

// Initialize GPU runtime (call once at startup)
int kraken_gpu_init(int device_id);

// Cleanup GPU runtime (call once at shutdown)
void kraken_gpu_cleanup(void);

/**
 * Backend Selection Functions
 */

// Set global backend preference
void kraken_set_global_backend(KrakenBackend backend);

// Get currently active backend
KrakenBackend kraken_get_active_backend(void);

// Check if backend should use GPU (considering environment variable)
int kraken_should_use_gpu(void);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_GPU_RUNTIME_H
