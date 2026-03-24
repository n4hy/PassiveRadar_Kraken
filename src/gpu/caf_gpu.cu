/**
 * GPU-Accelerated CAF Processing - CORRECTED IMPLEMENTATION
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Cross-Ambiguity Function (CAF) processing using NVIDIA CUDA.
 * Matches CPU algorithm exactly: applies Doppler shift to REFERENCE signal.
 */

#include "caf_gpu.h"
#include "gpu_common.h"
#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/**
 * CAF Processor State (opaque to caller)
 */
struct CAFProcessorGPU {
    // Parameters
    int n_samples;          // Input signal length
    int fft_len;            // FFT length = next_power_of_2(2 * n_samples) for linear correlation
    int n_doppler;
    int n_range;
    float doppler_start_hz;
    float doppler_step_hz;
    float sample_rate_hz;

    // cuFFT plans
    cufftHandle fft_plan_single;    // Single FFT for surveillance
    cufftHandle fft_plan_batch;     // Batched FFT for Doppler-shifted reference
    cufftHandle ifft_plan_batch;    // Batched IFFT for cross-correlation

    // Device memory (CLEAR NAMES - match what they actually contain)
    cufftComplex* d_reference;              // Reference signal (zero-padded to fft_len)
    cufftComplex* d_surveillance;           // Surveillance signal (zero-padded to fft_len)
    cufftComplex* d_ref_doppler_shifted;    // Doppler-shifted reference (n_doppler × fft_len)
    cufftComplex* d_reference_fft_batch;    // FFT of Doppler-shifted reference (n_doppler × fft_len)
    cufftComplex* d_surveillance_fft;       // FFT of surveillance (fft_len)
    cufftComplex* d_cross_corr;             // Cross-correlation result (n_doppler × fft_len)
    float*        d_output;                 // CAF magnitude output (n_doppler × n_range)

    // Precomputed Doppler phasors (n_doppler × n_samples)
    cufftComplex* d_doppler_phasors;

    // Pinned host memory for fast transfers
    float* h_input_pinned;
    float* h_output_pinned;

    // CUDA stream for async operations
    cudaStream_t stream;
};

/**
 * CUDA Kernel: Convert interleaved float (I,Q,I,Q,...) to cufftComplex with zero-padding
 * Copies n_samples from input, zero-pads the rest to fft_len
 */
__global__ void interleaved_to_complex_kernel(const float* __restrict__ input,
                                               cufftComplex* __restrict__ output,
                                               int n_samples,
                                               int fft_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < fft_len) {
        if (idx < n_samples) {
            output[idx].x = input[2 * idx];       // Real (I)
            output[idx].y = input[2 * idx + 1];   // Imag (Q)
        } else {
            output[idx].x = 0.0f;  // Zero-pad
            output[idx].y = 0.0f;
        }
    }
}

/**
 * CUDA Kernel: Apply Doppler shift to REFERENCE signal (batched for all Doppler bins)
 *
 * Each Doppler bin gets: reference[t] * phasor[doppler, t] for t < n_samples, zero-padded to fft_len
 * Grid: (n_doppler, ceil(fft_len / threads))
 *
 * NOTE: This applies shift to REFERENCE, not surveillance (to match CPU)
 */
__global__ void apply_doppler_shift_to_reference_kernel(
    const cufftComplex* __restrict__ reference,     // Input: reference signal (zero-padded to fft_len)
    const cufftComplex* __restrict__ phasors,       // Input: phasors (n_doppler × n_samples)
    cufftComplex* __restrict__ ref_doppler_shifted, // Output: shifted ref (n_doppler × fft_len)
    int n_samples,
    int fft_len,
    int n_doppler
) {
    int doppler_idx = blockIdx.x;
    int fft_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (doppler_idx < n_doppler && fft_idx < fft_len) {
        int output_idx = doppler_idx * fft_len + fft_idx;

        if (fft_idx < n_samples) {
            // Apply Doppler shift: reference[fft_idx] * phasor[doppler_idx, fft_idx]
            int phasor_idx = doppler_idx * n_samples + fft_idx;
            cufftComplex ref = reference[fft_idx];
            cufftComplex phasor = phasors[phasor_idx];

            ref_doppler_shifted[output_idx].x = ref.x * phasor.x - ref.y * phasor.y;  // Real
            ref_doppler_shifted[output_idx].y = ref.x * phasor.y + ref.y * phasor.x;  // Imag
        } else {
            // Zero-pad beyond n_samples
            ref_doppler_shifted[output_idx].x = 0.0f;
            ref_doppler_shifted[output_idx].y = 0.0f;
        }
    }
}

/**
 * CUDA Kernel: Complex conjugate multiply for cross-correlation
 *
 * Computes: surveillance_fft[i] * conj(reference_fft[doppler, i])
 * This matches CPU: Surv_FFT * conj(Ref_FFT)
 *
 * Grid: (n_doppler, ceil(n_samples / threads))
 */
__global__ void cross_correlation_multiply_kernel(
    const cufftComplex* __restrict__ surv_fft,     // Input: surveillance FFT (fft_len)
    const cufftComplex* __restrict__ ref_fft_batch,// Input: reference FFT batch (n_doppler × fft_len)
    cufftComplex* __restrict__ xcorr,              // Output: cross-correlation (n_doppler × fft_len)
    int fft_len,
    int n_doppler
) {
    int doppler_idx = blockIdx.x;
    int fft_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (doppler_idx < n_doppler && fft_idx < fft_len) {
        int batch_idx = doppler_idx * fft_len + fft_idx;

        cufftComplex surv = surv_fft[fft_idx];                   // Surveillance (same for all Doppler)
        cufftComplex ref = ref_fft_batch[batch_idx];            // Reference (different per Doppler)

        // Normalize (matches CPU: applies 1/fft_len before IFFT)
        float scale = 1.0f / fft_len;

        // Compute: surv * conj(ref) * scale = (surv.x + i*surv.y) * (ref.x - i*ref.y) * scale
        xcorr[batch_idx].x = (surv.x * ref.x + surv.y * ref.y) * scale;  // Real
        xcorr[batch_idx].y = (surv.y * ref.x - surv.x * ref.y) * scale;  // Imag
    }
}

/**
 * CUDA Kernel: Extract magnitude from complex IFFT result
 *
 * NOTE: Normalization already applied before IFFT (in cross_correlation_multiply_kernel)
 * Extracts only first n_range samples from fft_len-length IFFT output
 */
__global__ void extract_magnitude_kernel(
    const cufftComplex* __restrict__ xcorr,  // Input: IFFT result (n_doppler × fft_len)
    float* __restrict__ output,              // Output: magnitude (n_doppler × n_range)
    int fft_len,
    int n_range,
    int n_doppler
) {
    int doppler_idx = blockIdx.x;
    int range_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (doppler_idx < n_doppler && range_idx < n_range) {
        // Input has fft_len samples per Doppler bin
        int input_idx = doppler_idx * fft_len + range_idx;

        cufftComplex c = xcorr[input_idx];

        // Output layout: [n_doppler × n_range] row-major (matches CPU)
        int output_idx = doppler_idx * n_range + range_idx;
        output[output_idx] = sqrtf(c.x * c.x + c.y * c.y);
    }
}

/**
 * Precompute Doppler phasors on GPU
 * Phasor[doppler, t] = exp(-j * 2 * pi * fd * t / sample_rate)
 */
static int precompute_doppler_phasors(CAFProcessorGPU* proc) {
    size_t phasor_size = proc->n_doppler * proc->n_samples * sizeof(cufftComplex);
    cufftComplex* h_phasors = (cufftComplex*)malloc(phasor_size);
    if (!h_phasors) {
        fprintf(stderr, "Failed to allocate host memory for phasors\n");
        return -1;
    }

    // Compute phasors on CPU
    const float PI = 3.14159265358979323846f;
    for (int d = 0; d < proc->n_doppler; d++) {
        float fd = proc->doppler_start_hz + d * proc->doppler_step_hz;
        for (int t = 0; t < proc->n_samples; t++) {
            float phase = -2.0f * PI * fd * t / proc->sample_rate_hz;
            int idx = d * proc->n_samples + t;
            h_phasors[idx].x = cosf(phase);  // Real
            h_phasors[idx].y = sinf(phase);  // Imag
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
                           float doppler_start_hz, float doppler_step_hz,
                           float sample_rate_hz) {
    if (n_samples <= 0 || n_doppler <= 0 || n_range <= 0) {
        return nullptr;
    }

    CAFProcessorGPU* proc = new CAFProcessorGPU;
    if (!proc) {
        return nullptr;
    }

    // Store parameters
    proc->n_samples = n_samples;
    proc->n_doppler = n_doppler;
    proc->n_range = n_range;
    proc->doppler_start_hz = doppler_start_hz;
    proc->doppler_step_hz = doppler_step_hz;
    proc->sample_rate_hz = sample_rate_hz;

    // Compute FFT length: next power of 2 >= 2 * n_samples (for linear correlation)
    proc->fft_len = 1;
    while (proc->fft_len < 2 * n_samples) {
        proc->fft_len <<= 1;
    }
    printf("CAF GPU: n_samples=%d, fft_len=%d (for linear correlation)\n", n_samples, proc->fft_len);

    // Create CUDA stream
    if (cudaStreamCreate(&proc->stream) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream\n");
        delete proc;
        return nullptr;
    }

    // Allocate device memory (using fft_len for zero-padded signals)
    size_t fft_size = proc->fft_len * sizeof(cufftComplex);
    size_t batch_size = n_doppler * proc->fft_len * sizeof(cufftComplex);
    size_t output_size = n_range * n_doppler * sizeof(float);

    proc->d_reference = (cufftComplex*)kraken_gpu_alloc(fft_size);
    proc->d_surveillance = (cufftComplex*)kraken_gpu_alloc(fft_size);
    proc->d_ref_doppler_shifted = (cufftComplex*)kraken_gpu_alloc(batch_size);
    proc->d_reference_fft_batch = (cufftComplex*)kraken_gpu_alloc(batch_size);
    proc->d_surveillance_fft = (cufftComplex*)kraken_gpu_alloc(fft_size);
    proc->d_cross_corr = (cufftComplex*)kraken_gpu_alloc(batch_size);
    proc->d_output = (float*)kraken_gpu_alloc(output_size);

    if (!proc->d_reference || !proc->d_surveillance || !proc->d_ref_doppler_shifted ||
        !proc->d_reference_fft_batch || !proc->d_surveillance_fft ||
        !proc->d_cross_corr || !proc->d_output) {
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

    // Create cuFFT plans (using fft_len for linear correlation)
    // Plan 1: Single 1D FFT for surveillance (zero-padded to fft_len)
    if (cufftPlan1d(&proc->fft_plan_single, proc->fft_len, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to create surveillance FFT plan\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Plan 2: Batched 1D FFT for Doppler-shifted reference (n_doppler FFTs of length fft_len)
    if (cufftPlan1d(&proc->fft_plan_batch, proc->fft_len, CUFFT_C2C, n_doppler) != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to create batched reference FFT plan\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Plan 3: Batched IFFT for cross-correlation (n_doppler IFFTs of length fft_len)
    if (cufftPlan1d(&proc->ifft_plan_batch, proc->fft_len, CUFFT_C2C, n_doppler) != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to create batched IFFT plan\n");
        caf_gpu_destroy(proc);
        return nullptr;
    }

    // Set cuFFT stream for async execution
    cufftSetStream(proc->fft_plan_single, proc->stream);
    cufftSetStream(proc->fft_plan_batch, proc->stream);
    cufftSetStream(proc->ifft_plan_batch, proc->stream);

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
    if (proc->fft_plan_single) cufftDestroy(proc->fft_plan_single);
    if (proc->fft_plan_batch) cufftDestroy(proc->fft_plan_batch);
    if (proc->ifft_plan_batch) cufftDestroy(proc->ifft_plan_batch);

    // Free device memory
    kraken_gpu_free(proc->d_reference);
    kraken_gpu_free(proc->d_surveillance);
    kraken_gpu_free(proc->d_ref_doppler_shifted);
    kraken_gpu_free(proc->d_reference_fft_batch);
    kraken_gpu_free(proc->d_surveillance_fft);
    kraken_gpu_free(proc->d_cross_corr);
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
 *
 * Algorithm (matches CPU exactly):
 * 1. Apply Doppler shift to REFERENCE signal (batched for all Doppler bins)
 * 2. FFT Doppler-shifted reference (batched)
 * 3. FFT surveillance signal (single)
 * 4. Cross-correlation multiply: Surv_FFT * conj(Ref_FFT) (batched)
 * 5. IFFT (batched)
 * 6. Extract magnitude with normalization
 */
extern "C"
void caf_gpu_process_full(void* handle, const float* ref, const float* surv,
                          float* output) {
    if (!handle || !ref || !surv || !output) {
        return;
    }

    CAFProcessorGPU* proc = (CAFProcessorGPU*)handle;

    int threads = 256;
    int blocks_fft = (proc->fft_len + threads - 1) / threads;

    // ========================================================================
    // Step 1: Transfer reference and surveillance to GPU, convert to complex with zero-padding
    // ========================================================================

    // Reference signal (copy n_samples, zero-pad to fft_len)
    memcpy(proc->h_input_pinned, ref, 2 * proc->n_samples * sizeof(float));
    kraken_gpu_memcpy_h2d_async(proc->d_reference, proc->h_input_pinned,
                                2 * proc->n_samples * sizeof(float),
                                proc->stream);
    interleaved_to_complex_kernel<<<blocks_fft, threads, 0, proc->stream>>>(
        (float*)proc->d_reference, proc->d_reference, proc->n_samples, proc->fft_len
    );

    // Wait for reference transfer to complete before reusing pinned buffer
    cudaStreamSynchronize(proc->stream);

    // Surveillance signal (copy n_samples, zero-pad to fft_len)
    memcpy(proc->h_input_pinned, surv, 2 * proc->n_samples * sizeof(float));
    kraken_gpu_memcpy_h2d_async(proc->d_surveillance, proc->h_input_pinned,
                                2 * proc->n_samples * sizeof(float),
                                proc->stream);
    interleaved_to_complex_kernel<<<blocks_fft, threads, 0, proc->stream>>>(
        (float*)proc->d_surveillance, proc->d_surveillance, proc->n_samples, proc->fft_len
    );

    // ========================================================================
    // Step 2: Apply Doppler shifts to REFERENCE (batched for all Doppler bins)
    // ========================================================================

    dim3 doppler_grid(proc->n_doppler, (proc->fft_len + threads - 1) / threads);
    apply_doppler_shift_to_reference_kernel<<<doppler_grid, threads, 0, proc->stream>>>(
        proc->d_reference,
        proc->d_doppler_phasors,
        proc->d_ref_doppler_shifted,
        proc->n_samples,
        proc->fft_len,
        proc->n_doppler
    );

    // ========================================================================
    // Step 3: FFT surveillance (single, unshifted)
    // ========================================================================

    {
        cufftResult fft_result = cufftExecC2C(proc->fft_plan_single, proc->d_surveillance,
                                              proc->d_surveillance_fft, CUFFT_FORWARD);
        if (fft_result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT Error in %s:%d: surveillance FFT failed (code %d)\n",
                    __FILE__, __LINE__, fft_result);
        }
    }

    // ========================================================================
    // Step 4: FFT Doppler-shifted reference (batched - all Doppler bins)
    // ========================================================================

    {
        cufftResult fft_result = cufftExecC2C(proc->fft_plan_batch, proc->d_ref_doppler_shifted,
                                              proc->d_reference_fft_batch, CUFFT_FORWARD);
        if (fft_result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT Error in %s:%d: reference FFT failed (code %d)\n",
                    __FILE__, __LINE__, fft_result);
        }
    }

    // ========================================================================
    // Step 5: Cross-correlation multiply: Surv_FFT * conj(Ref_FFT)
    // ========================================================================

    cross_correlation_multiply_kernel<<<doppler_grid, threads, 0, proc->stream>>>(
        proc->d_surveillance_fft,
        proc->d_reference_fft_batch,
        proc->d_cross_corr,
        proc->fft_len,
        proc->n_doppler
    );

    // ========================================================================
    // Step 6: IFFT (batched - all Doppler bins)
    // ========================================================================

    {
        cufftResult fft_result = cufftExecC2C(proc->ifft_plan_batch, proc->d_cross_corr,
                                              proc->d_cross_corr, CUFFT_INVERSE);
        if (fft_result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT Error in %s:%d: IFFT failed (code %d)\n",
                    __FILE__, __LINE__, fft_result);
        }
    }

    // ========================================================================
    // Step 7: Extract magnitude (only first n_range samples)
    // ========================================================================

    dim3 range_grid(proc->n_doppler, (proc->n_range + threads - 1) / threads);
    extract_magnitude_kernel<<<range_grid, threads, 0, proc->stream>>>(
        proc->d_cross_corr,
        proc->d_output,
        proc->fft_len,
        proc->n_range,
        proc->n_doppler
    );

    // ========================================================================
    // Step 8: Transfer result to host
    // ========================================================================

    size_t output_size = proc->n_range * proc->n_doppler * sizeof(float);
    kraken_gpu_memcpy_d2h_async(proc->h_output_pinned, proc->d_output,
                                output_size, proc->stream);

    // Wait for all operations to complete
    cudaStreamSynchronize(proc->stream);

    // Copy from pinned buffer to user output
    memcpy(output, proc->h_output_pinned, output_size);
}
