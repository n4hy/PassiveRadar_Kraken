/**
 * GPU-Accelerated CFAR Detection
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * High-performance GPU implementation of 2D CA-CFAR.
 * API matches CPU version exactly for drop-in replacement.
 */

#ifndef KRAKEN_CFAR_GPU_H
#define KRAKEN_CFAR_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 2D CA-CFAR detection (GPU version)
 *
 * API matches cfar_2d() from CPU version exactly.
 *
 * Performs Cell-Averaging CFAR on 2D input (typically Range-Doppler map).
 * For each cell, computes average of surrounding training cells (excluding
 * guard cells) and declares detection if cell > noise_average + threshold.
 *
 * Input is assumed to be log magnitude in dB.
 *
 * @param input Input data (log magnitude in dB), size rows * cols
 * @param output Detection mask (1.0 if detection, 0.0 otherwise), size rows * cols
 * @param rows Number of rows (typically Doppler bins)
 * @param cols Number of columns (typically Range bins)
 * @param guard Number of guard cells in each dimension
 * @param train Number of training cells beyond guard in each dimension
 * @param threshold Detection threshold in dB above noise average
 */
void cfar_gpu_2d(const float* input, float* output, int rows, int cols,
                 int guard, int train, float threshold);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_CFAR_GPU_H
