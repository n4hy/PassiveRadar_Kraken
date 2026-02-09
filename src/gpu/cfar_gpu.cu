/**
 * GPU-Accelerated CFAR Detection Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "cfar_gpu.h"
#include "gpu_common.h"
#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

/**
 * CUDA Kernel: 2D CA-CFAR Detection
 *
 * Each thread processes one cell (row, col).
 * Computes average of training cells and compares to threshold.
 *
 * Grid: (cols, rows) with 2D thread blocks
 */
__global__ void cfar_2d_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int rows, int cols,
                                int guard, int train,
                                float threshold) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (row < rows && col < cols) {
        // Initialize output to 0 (no detection)
        output[row * cols + col] = 0.0f;

        // Check if we have enough cells for CFAR window
        int margin = guard + train;
        if (row < margin || row >= rows - margin ||
            col < margin || col >= cols - margin) {
            // Too close to edge, no detection
            return;
        }

        // Compute noise average from training cells
        float sum = 0.0f;
        int count = 0;

        // Iterate over CFAR window
        for (int dr = -margin; dr <= margin; dr++) {
            for (int dc = -margin; dc <= margin; dc++) {
                // Skip guard cells (including center cell)
                if (abs(dr) <= guard && abs(dc) <= guard) {
                    continue;
                }

                // Accumulate training cell
                int r = row + dr;
                int c = col + dc;
                sum += input[r * cols + c];
                count++;
            }
        }

        // Compute noise average
        if (count == 0) {
            return;  // No training cells (shouldn't happen)
        }
        float noise_avg = sum / count;

        // Get cell value
        float cell_value = input[row * cols + col];

        // Detection test: cell > noise + threshold (dB scale)
        if (cell_value > noise_avg + threshold) {
            output[row * cols + col] = 1.0f;
        }
    }
}

/**
 * 2D CA-CFAR Detection (GPU version)
 */
extern "C"
void cfar_gpu_2d(const float* input, float* output, int rows, int cols,
                 int guard, int train, float threshold) {
    if (!input || !output || rows <= 0 || cols <= 0) {
        return;
    }

    // Allocate device memory
    size_t data_size = rows * cols * sizeof(float);
    float* d_input = (float*)kraken_gpu_alloc(data_size);
    float* d_output = (float*)kraken_gpu_alloc(data_size);

    if (!d_input || !d_output) {
        fprintf(stderr, "Failed to allocate GPU memory for CFAR\n");
        kraken_gpu_free(d_input);
        kraken_gpu_free(d_output);
        return;
    }

    // Transfer input to GPU
    if (kraken_gpu_memcpy_h2d(d_input, input, data_size) != 0) {
        fprintf(stderr, "Failed to transfer input to GPU\n");
        kraken_gpu_free(d_input);
        kraken_gpu_free(d_output);
        return;
    }

    // Launch CFAR kernel
    dim3 threads(16, 16);  // 256 threads per block
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);

    cfar_2d_kernel<<<blocks, threads>>>(d_input, d_output, rows, cols,
                                        guard, train, threshold);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CFAR kernel failed: %s\n", cudaGetErrorString(err));
        kraken_gpu_free(d_input);
        kraken_gpu_free(d_output);
        return;
    }

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Transfer result back to host
    if (kraken_gpu_memcpy_d2h(output, d_output, data_size) != 0) {
        fprintf(stderr, "Failed to transfer output from GPU\n");
    }

    // Free device memory
    kraken_gpu_free(d_input);
    kraken_gpu_free(d_output);
}
