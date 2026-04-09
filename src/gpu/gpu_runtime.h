/**
 * GPU Runtime Library for PassiveRadar_Kraken
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Device detection, backend selection, and runtime management.
 *
 * CUDA Version Support:
 * - CUDA 11.8+: Core functionality (Turing through Ada Lovelace)
 * - CUDA 12.0+: Hopper architecture (sm_90), C++20 device code
 * - CUDA 13.0+: Blackwell architecture (sm_100/101/103)
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

/**
 * kraken_gpu_device_count - Return the number of available CUDA devices
 */
int kraken_gpu_device_count(void);

/**
 * kraken_gpu_is_available - Check if at least one GPU is available and initialized
 */
int kraken_gpu_is_available(void);

/**
 * kraken_gpu_get_device_info - Query device name and compute capability for a given device ID
 */
void kraken_gpu_get_device_info(int device_id, char* name_out, int* compute_capability_out);

/**
 * kraken_gpu_init - Initialize the GPU runtime on the specified device (call once at startup)
 */
int kraken_gpu_init(int device_id);

/**
 * kraken_gpu_cleanup - Release GPU runtime resources (call once at shutdown)
 */
void kraken_gpu_cleanup(void);

/**
 * Version Information Functions
 */

/**
 * kraken_gpu_cuda_version - Return the CUDA runtime version as an integer (e.g., 13000 for CUDA 13.0)
 */
int kraken_gpu_cuda_version(void);

/**
 * kraken_gpu_driver_version - Return the CUDA driver version as an integer
 */
int kraken_gpu_driver_version(void);

/**
 * Backend Selection Functions
 */

/**
 * kraken_set_global_backend - Set the global compute backend preference (AUTO, GPU, or CPU)
 */
void kraken_set_global_backend(KrakenBackend backend);

/**
 * kraken_get_active_backend - Return the currently active compute backend
 */
KrakenBackend kraken_get_active_backend(void);

/**
 * kraken_should_use_gpu - Check if GPU should be used, considering backend setting and environment
 */
int kraken_should_use_gpu(void);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_GPU_RUNTIME_H
