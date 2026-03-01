/**
 * GPU-Accelerated Doppler Processing Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "doppler_gpu.h"
#include "gpu_common.h"
#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

/**
 * Doppler Processor State (opaque to caller)
 */
struct DopplerProcessorGPU {
    // Parameters
    int fft_len;      // Range bins (columns)
    int doppler_len;  // Doppler bins (rows)

    // cuFFT plan (batched 1D FFTs, one per range bin)
    cufftHandle fft_plan;

    // Device memory
    cufftComplex* d_input;        // Input data (after windowing)
    cufftComplex* d_output;       // FFT output (before shift)
    float* d_window;              // Hamming window coefficients
    float* d_magnitude_output;    // Log magnitude output
    float* d_complex_output;      // Complex output

    // Host pinned memory for transfers
    float* h_input_pinned;
    float* h_output_pinned;

    // CUDA stream for async operations
    cudaStream_t stream;
};

/**
 * CUDA Kernel: Convert interleaved float to complex and apply Hamming window
 *
 * Processes column-wise (each thread handles one element).
 * Grid: (fft_len, doppler_len)
 */
__global__ void apply_window_kernel(const float* __restrict__ input,
                                     const float* __restrict__ window,
                                     cufftComplex* __restrict__ output,
                                     int fft_len, int doppler_len) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Range bin
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Doppler bin

    if (col < fft_len && row < doppler_len) {
        int input_idx = row * fft_len + col;

        // Read interleaved I/Q
        float i = input[2 * input_idx];
        float q = input[2 * input_idx + 1];

        // Apply window
        float w = window[row];
        output[input_idx].x = i * w;
        output[input_idx].y = q * w;
    }
}

/**
 * CUDA Kernel: FFT shift and extract log magnitude
 *
 * Performs FFT shift (swap halves) and computes log magnitude in dB.
 * Grid: (fft_len, doppler_len)
 */
__global__ void fftshift_magnitude_kernel(const cufftComplex* __restrict__ input,
                                           float* __restrict__ output,
                                           int fft_len, int doppler_len) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Range bin
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Doppler bin

    if (col < fft_len && row < doppler_len) {
        // FFT shift: swap first and second halves
        // Destination index remains the same
        // Source index is shifted by (doppler_len + 1) / 2
        int src_row = (row + (doppler_len + 1) / 2) % doppler_len;
        int src_idx = src_row * fft_len + col;
        int dst_idx = row * fft_len + col;

        // Read complex value
        cufftComplex val = input[src_idx];

        // Compute magnitude squared
        float mag_sq = val.x * val.x + val.y * val.y;

        // Convert to dB (with small epsilon to avoid log(0))
        float val_db = 10.0f * log10f(mag_sq + 1e-12f);

        output[dst_idx] = val_db;
    }
}

/**
 * CUDA Kernel: FFT shift for complex output
 *
 * Performs FFT shift and writes complex values in interleaved format.
 * Grid: (fft_len, doppler_len)
 */
__global__ void fftshift_complex_kernel(const cufftComplex* __restrict__ input,
                                         float* __restrict__ output,
                                         int fft_len, int doppler_len) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Range bin
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Doppler bin

    if (col < fft_len && row < doppler_len) {
        // FFT shift source index
        int src_row = (row + (doppler_len + 1) / 2) % doppler_len;
        int src_idx = src_row * fft_len + col;
        int dst_idx = row * fft_len + col;

        // Read complex value
        cufftComplex val = input[src_idx];

        // Write as interleaved I/Q
        output[2 * dst_idx] = val.x;
        output[2 * dst_idx + 1] = val.y;
    }
}

/**
 * Precompute Hamming window on GPU
 */
static int precompute_window(DopplerProcessorGPU* proc) {
    // Allocate host buffer
    float* h_window = (float*)malloc(proc->doppler_len * sizeof(float));
    if (!h_window) {
        fprintf(stderr, "Failed to allocate host memory for window\n");
        return -1;
    }

    // Compute Hamming window on CPU
    if (proc->doppler_len > 1) {
        const float PI = 3.14159265358979323846f;
        for (int i = 0; i < proc->doppler_len; i++) {
            h_window[i] = 0.54f - 0.46f * cosf(2.0f * PI * i / (proc->doppler_len - 1));
        }
    } else {
        h_window[0] = 1.0f;
    }

    // Allocate GPU memory and transfer
    proc->d_window = (float*)kraken_gpu_alloc(proc->doppler_len * sizeof(float));
    if (!proc->d_window) {
        free(h_window);
        return -1;
    }

    if (kraken_gpu_memcpy_h2d(proc->d_window, h_window,
                               proc->doppler_len * sizeof(float)) != 0) {
        free(h_window);
        return -1;
    }

    free(h_window);
    return 0;
}

/**
 * Create Doppler processor instance
 */
extern "C"
void* doppler_gpu_create(int fft_len, int doppler_len) {
    if (fft_len <= 0 || doppler_len <= 0) {
        return nullptr;
    }

    DopplerProcessorGPU* proc = new DopplerProcessorGPU;
    if (!proc) {
        return nullptr;
    }

    // Store parameters
    proc->fft_len = fft_len;
    proc->doppler_len = doppler_len;

    // Create CUDA stream
    if (cudaStreamCreate(&proc->stream) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream\n");
        delete proc;
        return nullptr;
    }

    // Allocate device memory
    size_t data_size = doppler_len * fft_len * sizeof(cufftComplex);
    size_t float_size = doppler_len * fft_len * sizeof(float);
    size_t complex_float_size = 2 * doppler_len * fft_len * sizeof(float);

    proc->d_input = (cufftComplex*)kraken_gpu_alloc(data_size);
    proc->d_output = (cufftComplex*)kraken_gpu_alloc(data_size);
    proc->d_magnitude_output = (float*)kraken_gpu_alloc(float_size);
    proc->d_complex_output = (float*)kraken_gpu_alloc(complex_float_size);

    if (!proc->d_input || !proc->d_output ||
        !proc->d_magnitude_output || !proc->d_complex_output) {
        fprintf(stderr, "Failed to allocate GPU memory\n");
        doppler_gpu_destroy(proc);
        return nullptr;
    }

    // Precompute Hamming window
    if (precompute_window(proc) != 0) {
        fprintf(stderr, "Failed to precompute window\n");
        doppler_gpu_destroy(proc);
        return nullptr;
    }

    // Allocate pinned host memory
    proc->h_input_pinned = (float*)kraken_gpu_alloc_host(complex_float_size);
    proc->h_output_pinned = (float*)kraken_gpu_alloc_host(
        complex_float_size);  // Max of magnitude and complex output

    if (!proc->h_input_pinned || !proc->h_output_pinned) {
        fprintf(stderr, "Failed to allocate pinned host memory\n");
        doppler_gpu_destroy(proc);
        return nullptr;
    }

    // Create cuFFT plan (batched 1D FFTs along rows)
    // Each column gets its own FFT of length doppler_len
    int n[] = {doppler_len};           // FFT size
    int inembed[] = {doppler_len};     // Input stride
    int onembed[] = {doppler_len};     // Output stride
    int istride = fft_len;             // Distance between elements in same FFT
    int ostride = fft_len;             // Distance between elements in output
    int idist = 1;                     // Distance between FFTs (columns)
    int odist = 1;                     // Distance between output FFTs
    int batch = fft_len;               // Number of FFTs (one per column)

    if (cufftPlanMany(&proc->fft_plan, 1, n,
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to create cuFFT plan\n");
        doppler_gpu_destroy(proc);
        return nullptr;
    }

    // Set cuFFT stream
    cufftSetStream(proc->fft_plan, proc->stream);

    return (void*)proc;
}

/**
 * Destroy Doppler processor instance
 */
extern "C"
void doppler_gpu_destroy(void* handle) {
    if (!handle) {
        return;
    }

    DopplerProcessorGPU* proc = (DopplerProcessorGPU*)handle;

    // Destroy cuFFT plan
    if (proc->fft_plan) {
        cufftDestroy(proc->fft_plan);
    }

    // Free device memory
    kraken_gpu_free(proc->d_input);
    kraken_gpu_free(proc->d_output);
    kraken_gpu_free(proc->d_magnitude_output);
    kraken_gpu_free(proc->d_complex_output);
    kraken_gpu_free(proc->d_window);

    // Free pinned host memory
    kraken_gpu_free_host(proc->h_input_pinned);
    kraken_gpu_free_host(proc->h_output_pinned);

    // Destroy stream
    if (proc->stream) {
        cudaStreamDestroy(proc->stream);
    }

    delete proc;
}

/**
 * Process Doppler FFT with log magnitude output
 */
extern "C"
void doppler_gpu_process(void* handle, const float* input, float* output) {
    if (!handle || !input || !output) {
        return;
    }

    DopplerProcessorGPU* proc = (DopplerProcessorGPU*)handle;

    // ========================================================================
    // Step 1: Transfer input to GPU
    // ========================================================================
    size_t input_size = 2 * proc->doppler_len * proc->fft_len * sizeof(float);
    memcpy(proc->h_input_pinned, input, input_size);
    kraken_gpu_memcpy_h2d_async(proc->d_input, proc->h_input_pinned,
                                input_size, proc->stream);

    // ========================================================================
    // Step 2: Apply Hamming window
    // ========================================================================
    dim3 threads(16, 16);
    dim3 blocks((proc->fft_len + threads.x - 1) / threads.x,
                (proc->doppler_len + threads.y - 1) / threads.y);

    apply_window_kernel<<<blocks, threads, 0, proc->stream>>>(
        (float*)proc->d_input, proc->d_window, proc->d_input,
        proc->fft_len, proc->doppler_len
    );

    // ========================================================================
    // Step 3: Batched FFT (all columns in parallel)
    // ========================================================================
    cufftResult fft_err = cufftExecC2C(proc->fft_plan, proc->d_input, proc->d_output, CUFFT_FORWARD);
    if (fft_err != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT exec failed in doppler_gpu: %d\n", (int)fft_err);
        return;
    }

    // ========================================================================
    // Step 4: FFT shift and extract log magnitude
    // ========================================================================
    fftshift_magnitude_kernel<<<blocks, threads, 0, proc->stream>>>(
        proc->d_output, proc->d_magnitude_output,
        proc->fft_len, proc->doppler_len
    );

    // ========================================================================
    // Step 5: Transfer result to host
    // ========================================================================
    size_t output_size = proc->doppler_len * proc->fft_len * sizeof(float);
    kraken_gpu_memcpy_d2h_async(proc->h_output_pinned, proc->d_magnitude_output,
                                output_size, proc->stream);

    // Wait for completion
    cudaStreamSynchronize(proc->stream);

    // Copy to user output
    memcpy(output, proc->h_output_pinned, output_size);
}

/**
 * Process Doppler FFT with complex output
 */
extern "C"
void doppler_gpu_process_complex(void* handle, const float* input, float* output) {
    if (!handle || !input || !output) {
        return;
    }

    DopplerProcessorGPU* proc = (DopplerProcessorGPU*)handle;

    // ========================================================================
    // Step 1: Transfer input to GPU
    // ========================================================================
    size_t input_size = 2 * proc->doppler_len * proc->fft_len * sizeof(float);
    memcpy(proc->h_input_pinned, input, input_size);
    kraken_gpu_memcpy_h2d_async(proc->d_input, proc->h_input_pinned,
                                input_size, proc->stream);

    // ========================================================================
    // Step 2: Apply Hamming window
    // ========================================================================
    dim3 threads(16, 16);
    dim3 blocks((proc->fft_len + threads.x - 1) / threads.x,
                (proc->doppler_len + threads.y - 1) / threads.y);

    apply_window_kernel<<<blocks, threads, 0, proc->stream>>>(
        (float*)proc->d_input, proc->d_window, proc->d_input,
        proc->fft_len, proc->doppler_len
    );

    // ========================================================================
    // Step 3: Batched FFT (all columns in parallel)
    // ========================================================================
    cufftExecC2C(proc->fft_plan, proc->d_input, proc->d_output, CUFFT_FORWARD);

    // ========================================================================
    // Step 4: FFT shift for complex output
    // ========================================================================
    fftshift_complex_kernel<<<blocks, threads, 0, proc->stream>>>(
        proc->d_output, proc->d_complex_output,
        proc->fft_len, proc->doppler_len
    );

    // ========================================================================
    // Step 5: Transfer result to host
    // ========================================================================
    size_t output_size = 2 * proc->doppler_len * proc->fft_len * sizeof(float);
    kraken_gpu_memcpy_d2h_async(proc->h_output_pinned, proc->d_complex_output,
                                output_size, proc->stream);

    // Wait for completion
    cudaStreamSynchronize(proc->stream);

    // Copy to user output
    memcpy(output, proc->h_output_pinned, output_size);
}
