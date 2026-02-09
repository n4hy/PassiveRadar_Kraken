/**
 * GPU-Accelerated Doppler Processing
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * High-performance GPU implementation using batched 2D cuFFT.
 * API matches CPU version exactly for drop-in replacement.
 */

#ifndef KRAKEN_DOPPLER_GPU_H
#define KRAKEN_DOPPLER_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create Doppler processor instance (GPU version)
 *
 * API matches doppler_create() from CPU version exactly.
 *
 * @param fft_len Number of range bins (fast-time, columns)
 * @param doppler_len Number of Doppler bins (slow-time, rows)
 * @return Opaque handle to GPU Doppler processor, or NULL on error
 */
void* doppler_gpu_create(int fft_len, int doppler_len);

/**
 * Destroy Doppler processor instance (GPU version)
 *
 * @param handle Handle returned by doppler_gpu_create()
 */
void doppler_gpu_destroy(void* handle);

/**
 * Process Doppler FFT with log magnitude output (GPU version)
 *
 * API matches doppler_process() from CPU version exactly.
 *
 * Input: Complex array (interleaved float I/Q), size 2 * doppler_len * fft_len
 * Output: Float array (log magnitude in dB), size doppler_len * fft_len
 *
 * @param handle Handle returned by doppler_gpu_create()
 * @param input Input complex data (interleaved float I/Q)
 * @param output Output log magnitude in dB
 */
void doppler_gpu_process(void* handle, const float* input, float* output);

/**
 * Process Doppler FFT with complex output (GPU version)
 *
 * API matches doppler_process_complex() from CPU version exactly.
 *
 * Input: Complex array (interleaved float I/Q), size 2 * doppler_len * fft_len
 * Output: Complex array (interleaved float I/Q), size 2 * doppler_len * fft_len
 *
 * @param handle Handle returned by doppler_gpu_create()
 * @param input Input complex data (interleaved float I/Q)
 * @param output Output complex data (interleaved float I/Q, FFT-shifted)
 */
void doppler_gpu_process_complex(void* handle, const float* input, float* output);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_DOPPLER_GPU_H
