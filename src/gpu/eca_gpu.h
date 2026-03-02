/**
 * GPU-Accelerated ECA-B Clutter Canceller Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Implements adaptive clutter rejection using least-squares optimal FIR filtering
 * with CUDA acceleration via cuBLAS and cuSOLVER.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create GPU ECA-B processor
 *
 * @param num_taps Number of FIR filter taps
 * @param max_delay Maximum delay samples between reference and surveillance
 * @return Opaque handle to processor, or nullptr on failure
 */
void* eca_gpu_create(int num_taps, int max_delay);

/**
 * Destroy GPU ECA-B processor
 *
 * @param handle Processor handle
 */
void eca_gpu_destroy(void* handle);

/**
 * Process samples through GPU ECA-B filter
 *
 * @param handle Processor handle
 * @param ref_in Reference signal (interleaved I/Q floats)
 * @param surv_in Surveillance signal (interleaved I/Q floats)
 * @param out_err Output error signal (surv - filtered_ref)
 * @param n_samples Number of complex samples
 */
void eca_gpu_process(void* handle, const float* ref_in, const float* surv_in,
                     float* out_err, int n_samples);

/**
 * Set delay between reference and surveillance signals
 *
 * @param handle Processor handle
 * @param delay_samples Delay in samples (0 to max_delay)
 */
void eca_gpu_set_delay(void* handle, int delay_samples);

/**
 * Get current delay setting
 *
 * @param handle Processor handle
 * @return Current delay in samples
 */
int eca_gpu_get_delay(void* handle);

#ifdef __cplusplus
}
#endif
