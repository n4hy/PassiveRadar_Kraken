/**
 * GPU-Accelerated Cross-Ambiguity Function (CAF) Processing
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * High-performance GPU implementation of CAF using batched cuFFT.
 * API matches CPU version exactly for drop-in replacement.
 */

#ifndef KRAKEN_CAF_GPU_H
#define KRAKEN_CAF_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create CAF processor instance (GPU version)
 *
 * API matches caf_create_full() from CPU version exactly.
 *
 * @param n_samples Number of samples per CPI
 * @param n_doppler Number of Doppler bins
 * @param n_range Number of range bins
 * @param doppler_start Start frequency for Doppler (Hz)
 * @param doppler_step Doppler bin spacing (Hz)
 * @param sample_rate Sample rate (Hz)
 * @return Opaque handle to GPU CAF processor, or NULL on error
 */
void* caf_gpu_create_full(int n_samples, int n_doppler, int n_range,
                          float doppler_start, float doppler_step,
                          float sample_rate);

/**
 * Destroy CAF processor instance (GPU version)
 *
 * @param handle Handle returned by caf_gpu_create_full()
 */
void caf_gpu_destroy(void* handle);

/**
 * Process one CPI through GPU CAF pipeline
 *
 * API matches caf_process_full() from CPU version exactly.
 *
 * @param handle Handle returned by caf_gpu_create_full()
 * @param ref Reference channel samples (interleaved float I/Q, length 2*n_samples)
 * @param surv Surveillance channel samples (interleaved float I/Q, length 2*n_samples)
 * @param output Output CAF surface (float, length n_range * n_doppler)
 */
void caf_gpu_process_full(void* handle, const float* ref, const float* surv,
                          float* output);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_CAF_GPU_H
