/**
 * GPU-Accelerated Cross-Ambiguity Function (CAF) Processing Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "caf_gpu.h"
#include "gpu_common.h"
#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

/**
 * CAF Processor State (opaque to caller)
 */
struct CAFProcessorGPU {
    // Parameters
    int n_samples;
    int n_doppler;
    int n_range;
    float doppler_start;
    float doppler_step;
    float sample_rate;

    // cuFFT plans
    cufftHandle fft_plan_ref;        // Forward FFT for reference
    cufftHandle fft_plan_surv;       // Forward FFT for surveillance (batched)
    cufftHandle ifft_plan;           // Inverse FFT (batched)

    // Device memory
    cufftComplex* d_ref;             // Reference samples (complex)
    cufftComplex* d_surv;            // Surveillance samples (complex)
    cufftComplex* d_surv_doppler;    // Doppler-shifted surveillance (n_doppler copies)
    cufftComplex* d_ref_fft;         // FFT of reference
    cufftComplex* d_surv_fft;        // FFT of Doppler-shifted surveillance (batched)
    cufftComplex* d_xcorr;           // Cross-correlation (batched)
    float* d_output;                 // Output magnitude

    // Precomputed Doppler phasors (on GPU)
    cufftComplex* d_doppler_phasors; // [n_doppler * n_samples]

    // Host pinned memory for transfers
    float* h_input_pinned;           // Pinned buffer for input
    float* h_output_pinned;          // Pinned buffer for output

    // CUDA stream for async operations
    cudaStream_t stream;
};

/**
 * CUDA Kernel: Interleaved float to complex conversion
 */
__global__ void interleaved_to_complex_kernel(const float* __restrict__ input,
                                               cufftComplex* __restrict__ output,
                                               int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        output[idx].x = input[2 * idx];       // I
        output[idx].y = input[2 * idx + 1];   // Q
    }
}

/**
 * CUDA Kernel: Apply Doppler shift (batched across all Doppler bins)
 *
 * Each thread processes one sample for one Doppler bin.
 * Grid: (n_doppler, n_samples/threads_per_block)
 */
__global__ void apply_doppler_shift_kernel(const cufftComplex* __restrict__ surv,
                                            const cufftComplex* __restrict__ phasors,
                                            cufftComplex* __restrict__ surv_doppler,
                                            int n_samples, int n_doppler) {
    int doppler_idx = blockIdx.x;
    int sample_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (doppler_idx < n_doppler && sample_idx < n_samples) {
        int phasor_idx = doppler_idx * n_samples + sample_idx;
        int output_idx = doppler_idx * n_samples + sample_idx;

        // Complex multiply: surv[sample_idx] * phasor[doppler_idx, sample_idx]
        cufftComplex s = surv[sample_idx];
        cufftComplex p = phasors[phasor_idx];

        surv_doppler[output_idx].x = s.x * p.x - s.y * p.y;  // Real
        surv_doppler[output_idx].y = s.x * p.y + s.y * p.x;  // Imag
    }
}

/**
 * CUDA Kernel: Complex conjugate multiply (element-wise)
 * ref_fft[i] * conj(surv_fft[doppler, i])
 */
__global__ void complex_conj_multiply_kernel(const cufftComplex* __restrict__ ref_fft,
                                              const cufftComplex* __restrict__ surv_fft,
                                              cufftComplex* __restrict__ xcorr,
                                              int n_samples, int n_doppler) {
    int doppler_idx = blockIdx.x;
    int sample_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (doppler_idx < n_doppler && sample_idx < n_samples) {
        int idx = doppler_idx * n_samples + sample_idx;

        cufftComplex r = ref_fft[sample_idx];
        cufftComplex s = surv_fft[idx];

        // Conjugate multiply: r * conj(s) = (r.x + i*r.y) * (s.x - i*s.y)
        xcorr[idx].x = r.x * s.x + r.y * s.y;  // Real
        xcorr[idx].y = r.y * s.x - r.x * s.y;  // Imag
    }
}

/**
 * CUDA Kernel: Extract magnitude from complex IFFT result
 * NOTE: cuFFT doesn't normalize IFFT, so we must divide by n_samples
 */
__global__ void extract_magnitude_kernel(const cufftComplex* __restrict__ xcorr,
                                          float* __restrict__ output,
                                          int n_samples, int n_doppler) {
    int doppler_idx = blockIdx.x;
    int range_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (doppler_idx < n_doppler && range_idx < n_samples) {
        int input_idx = doppler_idx * n_samples + range_idx;
        // Output layout: [n_doppler x n_range] row-major (matches CPU)
        int output_idx = doppler_idx * n_samples + range_idx;

        cufftComplex c = xcorr[input_idx];

        // Normalize by n_samples (cuFFT doesn't normalize IFFT)
        float scale = 1.0f / n_samples;
        float real_norm = c.x * scale;
        float imag_norm = c.y * scale;

        output[output_idx] = sqrtf(real_norm * real_norm + imag_norm * imag_norm);
    }
}

/**
 * Precompute Doppler phasors on GPU
 */
static int precompute_doppler_phasors(CAFProcessorGPU* proc) {
    // Allocate host buffer
    size_t phasor_size = proc->n_doppler * proc->n_samples * sizeof(cufftComplex);
    cufftComplex* h_phasors = (cufftComplex*)malloc(phasor_size);
    if (!h_phasors) {
        fprintf(stderr, "Failed to allocate host memory for phasors\n");
        return -1;
    }

    // Compute phasors on CPU (one-time cost)
    const float two_pi = 2.0f * M_PI;
    for (int d = 0; d < proc->n_doppler; d++) {
        float doppler_freq = proc->doppler_start + d * proc->doppler_step;
        for (int n = 0; n < proc->n_samples; n++) {
            float phase = two_pi * doppler_freq * n / proc->sample_rate;
            int idx = d * proc->n_samples + n;
            h_phasors[idx].x = cosf(-phase);  // Real (negative for Doppler shift)
            h_phasors[idx].y = sinf(-phase);  // Imag
        }
    }

    // Allocate GPU memory and transfer
    proc->d_doppler_phasors = (cufftComplex*)kraken_gpu_alloc(phasor_size);
    if (!proc->d_doppler_phasors) {
        free(h_phasors);
        return -1;
    }

    if (kraken_gpu_memcpy_h2d(proc->d_doppler_phasors, h_phasors, phasor_size) != 0) {
        free(h_phasors);
        return -1;
    }

    free(h_phasors);
    return 0;
}

/**
 * Create CAF processor instance
 */
extern "C"
void* caf_gpu_create_full(int n_samples, int n_doppler, int n_range,
                          float doppler_start, float doppler_step,
                          float sample_rate) {
    CAFProcessorGPU* proc = new CAFProcessorGPU;
    if (!proc) {
        return nullptr;
    }

    // Store parameters
    proc->n_samples = n_samples;
    proc->n_doppler = n_doppler;
    proc->n_range = n_range;
    proc->doppler_start = doppler_start;
    proc->doppler_step = doppler_step;
    proc->sample_rate = sample_rate;

    // Create CUDA stream
    if (cudaStreamCreate(&proc->stream) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream\n");
        delete proc;
        return nullptr;
    }

    // Allocate device memory
    size_t sample_size = n_samples * sizeof(cufftComplex);
    size_t doppler_batch_size = n_doppler * n_samples * sizeof(cufftComplex);
    size_t output_size = n_range * n_doppler * sizeof(float);

    proc->d_ref = (cufftComplex*)kraken_gpu_alloc(sample_size);
    proc->d_surv = (cufftComplex*)kraken_gpu_alloc(sample_size);
    proc->d_surv_doppler = (cufftComplex*)kraken_gpu_alloc(doppler_batch_size);
    proc->d_ref_fft = (cufftComplex*)kraken_gpu_alloc(sample_size);
    proc->d_surv_fft = (cufftComplex*)kraken_gpu_alloc(doppler_batch_size);
    proc->d_xcorr = (cufftComplex*)kraken_gpu_alloc(doppler_batch_size);
    proc->d_output = (float*)kraken_gpu_alloc(output_size);

    if (!proc->d_ref || !proc->d_surv || !proc->d_surv_doppler ||
        !proc->d_ref_fft || !proc->d_surv_fft || !proc->d_xcorr || !proc->d_output) {
        fprintf(stderr, "Failed to allocate GPU memory\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Allocate pinned host memory for fast transfers
    proc->h_input_pinned = (float*)kraken_gpu_alloc_host(2 * n_samples * sizeof(float));
    proc->h_output_pinned = (float*)kraken_gpu_alloc_host(output_size);

    if (!proc->h_input_pinned || !proc->h_output_pinned) {
        fprintf(stderr, "Failed to allocate pinned host memory\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Precompute Doppler phasors
    if (precompute_doppler_phasors(proc) != 0) {
        fprintf(stderr, "Failed to precompute Doppler phasors\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Create cuFFT plans
    // Reference: single 1D FFT
    if (cufftPlan1d(&proc->fft_plan_ref, n_samples, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to create reference FFT plan\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Surveillance: batched 1D FFT (n_doppler FFTs of length n_samples)
    if (cufftPlan1d(&proc->fft_plan_surv, n_samples, CUFFT_C2C, n_doppler) != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to create surveillance FFT plan\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // IFFT: batched (n_doppler IFFTs of length n_samples)
    if (cufftPlan1d(&proc->ifft_plan, n_samples, CUFFT_C2C, n_doppler) != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to create IFFT plan\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Set cuFFT stream for async execution
    cufftSetStream(proc->fft_plan_ref, proc->stream);
    cufftSetStream(proc->fft_plan_surv, proc->stream);
    cufftSetStream(proc->ifft_plan, proc->stream);

    return (void*)proc;
}

/**
 * Destroy CAF processor instance
 */
extern "C"
void caf_gpu_destroy(void* handle) {
    if (!handle) {
        return;
    }

    CAFProcessorGPU* proc = (CAFProcessorGPU*)handle;

    // Destroy cuFFT plans
    if (proc->fft_plan_ref) cufftDestroy(proc->fft_plan_ref);
    if (proc->fft_plan_surv) cufftDestroy(proc->fft_plan_surv);
    if (proc->ifft_plan) cufftDestroy(proc->ifft_plan);

    // Free device memory
    kraken_gpu_free(proc->d_ref);
    kraken_gpu_free(proc->d_surv);
    kraken_gpu_free(proc->d_surv_doppler);
    kraken_gpu_free(proc->d_ref_fft);
    kraken_gpu_free(proc->d_surv_fft);
    kraken_gpu_free(proc->d_xcorr);
    kraken_gpu_free(proc->d_output);
    kraken_gpu_free(proc->d_doppler_phasors);

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
 * Process one CPI through GPU CAF pipeline
 */
extern "C"
void caf_gpu_process_full(void* handle, const float* ref, const float* surv,
                          float* output) {
    if (!handle || !ref || !surv || !output) {
        return;
    }

    CAFProcessorGPU* proc = (CAFProcessorGPU*)handle;

    // ========================================================================
    // Step 1: Convert interleaved float to complex and transfer to GPU
    // ========================================================================

    // Copy reference to pinned buffer and transfer
    memcpy(proc->h_input_pinned, ref, 2 * proc->n_samples * sizeof(float));
    kraken_gpu_memcpy_h2d_async(proc->d_ref, proc->h_input_pinned,
                                2 * proc->n_samples * sizeof(float),
                                proc->stream);

    // Convert reference from interleaved float to complex
    int threads = 256;
    int blocks = (proc->n_samples + threads - 1) / threads;
    interleaved_to_complex_kernel<<<blocks, threads, 0, proc->stream>>>(
        (float*)proc->d_ref, proc->d_ref, proc->n_samples
    );

    // Copy surveillance to pinned buffer and transfer
    memcpy(proc->h_input_pinned, surv, 2 * proc->n_samples * sizeof(float));
    kraken_gpu_memcpy_h2d_async(proc->d_surv, proc->h_input_pinned,
                                2 * proc->n_samples * sizeof(float),
                                proc->stream);

    // Convert surveillance from interleaved float to complex
    interleaved_to_complex_kernel<<<blocks, threads, 0, proc->stream>>>(
        (float*)proc->d_surv, proc->d_surv, proc->n_samples
    );

    // ========================================================================
    // Step 2: Apply Doppler shifts to REFERENCE (to match CPU implementation)
    // ========================================================================

    dim3 doppler_blocks(proc->n_doppler, (proc->n_samples + threads - 1) / threads);
    apply_doppler_shift_kernel<<<doppler_blocks, threads, 0, proc->stream>>>(
        proc->d_ref, proc->d_doppler_phasors, proc->d_surv_doppler,
        proc->n_samples, proc->n_doppler
    );

    // ========================================================================
    // Step 3: Batched FFT (Doppler-shifted reference + surveillance)
    // ========================================================================

    // FFT surveillance (single, unshifted)
    cufftExecC2C(proc->fft_plan_ref, proc->d_surv, proc->d_ref_fft, CUFFT_FORWARD);

    // FFT reference (batched - all Doppler bins in one call)
    cufftExecC2C(proc->fft_plan_surv, proc->d_surv_doppler, proc->d_surv_fft, CUFFT_FORWARD);

    // ========================================================================
    // Step 4: Complex conjugate multiply (cross-correlation in frequency domain)
    // CPU does: shifted_ref_fft * conj(surv_fft)
    // So we pass surv_fft as first arg (gets conjugated), shifted_ref_fft as second
    // ========================================================================

    dim3 xcorr_blocks(proc->n_doppler, (proc->n_samples + threads - 1) / threads);
    complex_conj_multiply_kernel<<<xcorr_blocks, threads, 0, proc->stream>>>(
        proc->d_surv_fft, proc->d_ref_fft, proc->d_xcorr,
        proc->n_samples, proc->n_doppler
    );

    // ========================================================================
    // Step 5: Batched IFFT (cross-correlation back to time domain)
    // ========================================================================

    cufftExecC2C(proc->ifft_plan, proc->d_xcorr, proc->d_xcorr, CUFFT_INVERSE);

    // ========================================================================
    // Step 6: Extract magnitude and transfer to host
    // ========================================================================

    dim3 mag_blocks(proc->n_doppler, (proc->n_samples + threads - 1) / threads);
    extract_magnitude_kernel<<<mag_blocks, threads, 0, proc->stream>>>(
        proc->d_xcorr, proc->d_output, proc->n_samples, proc->n_doppler
    );

    // Transfer result to host (async to pinned buffer)
    size_t output_size = proc->n_range * proc->n_doppler * sizeof(float);
    kraken_gpu_memcpy_d2h_async(proc->h_output_pinned, proc->d_output,
                                output_size, proc->stream);

    // Wait for all operations to complete
    cudaStreamSynchronize(proc->stream);

    // Copy from pinned buffer to user output
    memcpy(output, proc->h_output_pinned, output_size);
}
