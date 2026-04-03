/**
 * GPU-Accelerated ECA-B Clutter Canceller Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * ECA-B (Eigenspace-based Clutter Canceller - Batched) using NVIDIA CUDA.
 * Uses cuSOLVER for Cholesky decomposition when available.
 *
 * CUDA Version Compatibility:
 * - CUDA 11.8+: Core functionality with cuSOLVER Cholesky
 * - CUDA 12.0+: Enhanced cuSOLVER performance
 * - CUDA 13.0+: Blackwell optimizations
 */

#include "eca_gpu.h"
#include "gpu_common.h"
#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>

// cuSOLVER available in CUDA 7.0+
#define HAVE_CUSOLVER 1

/**
 * ECA-B GPU Processor State
 */
struct ECAProcessorGPU {
    // Parameters
    int num_taps;
    int max_delay;
    int history_len;
    std::atomic<int> delay_samples;
    float diagonal_loading;

    // Device memory
    cuFloatComplex* d_ref_history;    // Reference signal history
    cuFloatComplex* d_full_ref;       // Full reference (history + new)
    cuFloatComplex* d_surv;           // Surveillance signal
    cuFloatComplex* d_output;         // Output error signal

    // Workspace for matrix operations
    cuFloatComplex* d_R;              // Autocorrelation matrix
    cuFloatComplex* d_p;              // Cross-correlation vector
    cuFloatComplex* d_w;              // Filter weights
    float* d_temp;                    // Temporary workspace

    // cuSOLVER workspace for Cholesky
#if HAVE_CUSOLVER
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
    cuFloatComplex* d_chol_work;     // cuSOLVER workspace
    int* d_info;                      // cuSOLVER info
    int chol_work_size;               // Workspace size
#endif

    // Pinned host memory
    float* h_input_pinned;
    float* h_output_pinned;

    // CUDA stream
    cudaStream_t stream;
};

/**
 * CUDA Kernel: Convert interleaved float to cuFloatComplex
 */
__global__ void interleaved_to_complex_kernel(const float* __restrict__ input,
                                                cuFloatComplex* __restrict__ output,
                                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = make_cuFloatComplex(input[2*idx], input[2*idx + 1]);
    }
}

/**
 * CUDA Kernel: Convert cuFloatComplex to interleaved float
 */
__global__ void complex_to_interleaved_kernel(const cuFloatComplex* __restrict__ input,
                                                float* __restrict__ output,
                                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[2*idx] = cuCrealf(input[idx]);
        output[2*idx + 1] = cuCimagf(input[idx]);
    }
}

/**
 * CUDA Kernel: Compute dot product of two complex vectors
 * Result stored in shared memory and reduced
 */
__global__ void complex_dot_kernel(const cuFloatComplex* __restrict__ a,
                                    const cuFloatComplex* __restrict__ b,
                                    cuFloatComplex* __restrict__ result,
                                    int n) {
    __shared__ float sum_re[256];
    __shared__ float sum_im[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_re = 0.0f;
    float local_im = 0.0f;

    // Each thread computes partial sum
    if (idx < n) {
        cuFloatComplex a_val = a[idx];
        cuFloatComplex b_conj = cuConjf(b[idx]);
        cuFloatComplex prod = cuCmulf(a_val, b_conj);
        local_re = cuCrealf(prod);
        local_im = cuCimagf(prod);
    }

    sum_re[tid] = local_re;
    sum_im[tid] = local_im;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_re[tid] += sum_re[tid + s];
            sum_im[tid] += sum_im[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(&result[0].x, sum_re[0]);
        atomicAdd(&result[0].y, sum_im[0]);
    }
}

/**
 * CUDA Kernel: Add diagonal loading to matrix
 */
__global__ void add_diagonal_kernel(cuFloatComplex* __restrict__ matrix,
                                     int n, float loading) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cuFloatComplex val = matrix[idx * n + idx];
        matrix[idx * n + idx] = make_cuFloatComplex(
            cuCrealf(val) + loading, cuCimagf(val)
        );
    }
}

/**
 * CUDA Kernel: Apply FIR filter (convolution)
 * Matches correlation indexing: output[n] = sum_j(w[j] * ref[base - j + n])
 * With start_idx = base - (num_taps - 1), use reversed tap order.
 */
__global__ void apply_fir_kernel(const cuFloatComplex* __restrict__ w,
                                  const cuFloatComplex* __restrict__ ref,
                                  cuFloatComplex* __restrict__ output,
                                  int num_taps, int start_idx, int n_samples) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < n_samples) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

        for (int j = 0; j < num_taps; j++) {
            // Reverse tap order to match correlation indexing
            int ref_idx = start_idx + n + (num_taps - 1 - j);
            sum = cuCaddf(sum, cuCmulf(w[j], ref[ref_idx]));
        }

        output[n] = sum;
    }
}

/**
 * CUDA Kernel: Compute error (surveillance - filtered)
 */
__global__ void compute_error_kernel(const cuFloatComplex* __restrict__ surv,
                                      const cuFloatComplex* __restrict__ filtered,
                                      cuFloatComplex* __restrict__ error,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        error[idx] = cuCsubf(surv[idx], filtered[idx]);
    }
}

/**
 * Host function: Solve linear system R*w = p using Cholesky decomposition (CPU fallback)
 * Used when cuSOLVER is not available or for small matrices
 */
void solve_linear_system_cpu(cuFloatComplex* h_R, cuFloatComplex* h_p,
                              cuFloatComplex* h_w, int n) {
    // Cholesky decomposition: R = L * L^H
    // Then solve L * y = p (forward substitution)
    // Then solve L^H * w = y (backward substitution)

    // Work buffers
    cuFloatComplex* L = new cuFloatComplex[n * n];
    cuFloatComplex* y = new cuFloatComplex[n];

    // Initialize L to zero
    for (int i = 0; i < n * n; i++) {
        L[i] = make_cuFloatComplex(0.0f, 0.0f);
    }

    // Cholesky decomposition (column-major lower triangular)
    for (int j = 0; j < n; j++) {
        // Compute diagonal element L[j,j]
        float sum_diag = cuCrealf(h_R[j * n + j]);
        for (int k = 0; k < j; k++) {
            cuFloatComplex Ljk = L[j * n + k];
            sum_diag -= cuCrealf(Ljk) * cuCrealf(Ljk) + cuCimagf(Ljk) * cuCimagf(Ljk);
        }

        if (sum_diag <= 0.0f) {
            // Matrix not positive definite, fall back to diagonal
            L[j * n + j] = make_cuFloatComplex(sqrtf(fabsf(cuCrealf(h_R[j * n + j])) + 1e-6f), 0.0f);
        } else {
            L[j * n + j] = make_cuFloatComplex(sqrtf(sum_diag), 0.0f);
        }

        float L_jj = cuCrealf(L[j * n + j]);

        // Compute off-diagonal elements L[i,j] for i > j
        for (int i = j + 1; i < n; i++) {
            cuFloatComplex sum = h_R[i * n + j];
            for (int k = 0; k < j; k++) {
                cuFloatComplex Lik = L[i * n + k];
                cuFloatComplex Ljk_conj = cuConjf(L[j * n + k]);
                sum = cuCsubf(sum, cuCmulf(Lik, Ljk_conj));
            }
            L[i * n + j] = make_cuFloatComplex(cuCrealf(sum) / L_jj, cuCimagf(sum) / L_jj);
        }
    }

    // Forward substitution: L * y = p
    for (int i = 0; i < n; i++) {
        cuFloatComplex sum = h_p[i];
        for (int j = 0; j < i; j++) {
            sum = cuCsubf(sum, cuCmulf(L[i * n + j], y[j]));
        }
        float L_ii = cuCrealf(L[i * n + i]);
        y[i] = make_cuFloatComplex(cuCrealf(sum) / L_ii, cuCimagf(sum) / L_ii);
    }

    // Backward substitution: L^H * w = y
    for (int i = n - 1; i >= 0; i--) {
        cuFloatComplex sum = y[i];
        for (int j = i + 1; j < n; j++) {
            cuFloatComplex L_ji_conj = cuConjf(L[j * n + i]);
            sum = cuCsubf(sum, cuCmulf(L_ji_conj, h_w[j]));
        }
        float L_ii = cuCrealf(L[i * n + i]);
        h_w[i] = make_cuFloatComplex(cuCrealf(sum) / L_ii, cuCimagf(sum) / L_ii);
    }

    delete[] L;
    delete[] y;
}

#if HAVE_CUSOLVER
/**
 * GPU function: Solve linear system R*w = p using cuSOLVER Cholesky
 */
int solve_linear_system_gpu(ECAProcessorGPU* proc, int n) {
    cusolverStatus_t status;

    // Cholesky factorization: R = L * L^H (lower triangular)
    status = cusolverDnCpotrf(
        proc->cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        proc->d_R,
        n,
        proc->d_chol_work,
        proc->chol_work_size,
        proc->d_info
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "ECA GPU: cusolverDnCpotrf failed: %d\n", status);
        return -1;
    }

    // Solve R * w = p using the Cholesky factorization
    // First copy p to w (cuSOLVER solves in-place)
    cudaMemcpyAsync(proc->d_w, proc->d_p, n * sizeof(cuFloatComplex),
                    cudaMemcpyDeviceToDevice, proc->stream);

    status = cusolverDnCpotrs(
        proc->cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        1,       // nrhs
        proc->d_R,
        n,
        proc->d_w,
        n,
        proc->d_info
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "ECA GPU: cusolverDnCpotrs failed: %d\n", status);
        return -1;
    }

    return 0;
}
#endif

/**
 * Create GPU ECA-B processor
 */
void* eca_gpu_create(int num_taps, int max_delay) {
    if (num_taps <= 0) {
        fprintf(stderr, "ECA GPU: Invalid num_taps %d\n", num_taps);
        return nullptr;
    }

    ECAProcessorGPU* proc = new ECAProcessorGPU();
    proc->num_taps = num_taps;
    proc->max_delay = (max_delay > 0) ? max_delay : 4096;
    proc->history_len = num_taps + proc->max_delay;
    proc->delay_samples.store(0, std::memory_order_release);
    proc->diagonal_loading = 1e-6f;

    // Create CUDA stream
    if (cudaStreamCreate(&proc->stream) != cudaSuccess) {
        fprintf(stderr, "ECA GPU: Failed to create CUDA stream\n");
        delete proc;
        return nullptr;
    }

#if HAVE_CUSOLVER
    // Initialize cuSOLVER
    if (cusolverDnCreate(&proc->cusolver_handle) != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "ECA GPU: Failed to create cuSOLVER handle\n");
        cudaStreamDestroy(proc->stream);
        delete proc;
        return nullptr;
    }
    cusolverDnSetStream(proc->cusolver_handle, proc->stream);

    // Initialize cuBLAS
    if (cublasCreate(&proc->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ECA GPU: Failed to create cuBLAS handle\n");
        cusolverDnDestroy(proc->cusolver_handle);
        cudaStreamDestroy(proc->stream);
        delete proc;
        return nullptr;
    }
    cublasSetStream(proc->cublas_handle, proc->stream);

    // Query cuSOLVER workspace size for Cholesky
    int work_size = 0;
    cusolverDnCpotrf_bufferSize(
        proc->cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        num_taps,
        nullptr,  // A (not needed for size query)
        num_taps,
        &work_size
    );
    proc->chol_work_size = work_size;
    proc->d_chol_work = (cuFloatComplex*)kraken_gpu_alloc(work_size * sizeof(cuFloatComplex));
    proc->d_info = (int*)kraken_gpu_alloc(sizeof(int));

    if (!proc->d_chol_work || !proc->d_info) {
        fprintf(stderr, "ECA GPU: Failed to allocate cuSOLVER workspace\n");
        cusolverDnDestroy(proc->cusolver_handle);
        cublasDestroy(proc->cublas_handle);
        cudaStreamDestroy(proc->stream);
        delete proc;
        return nullptr;
    }
#endif

    // Allocate device memory
    size_t max_samples = 65536;
    proc->d_ref_history = (cuFloatComplex*)kraken_gpu_alloc(proc->history_len * sizeof(cuFloatComplex));
    proc->d_full_ref = (cuFloatComplex*)kraken_gpu_alloc((proc->history_len + max_samples + proc->max_delay) * sizeof(cuFloatComplex));
    proc->d_surv = (cuFloatComplex*)kraken_gpu_alloc(max_samples * sizeof(cuFloatComplex));
    proc->d_output = (cuFloatComplex*)kraken_gpu_alloc(max_samples * sizeof(cuFloatComplex));
    proc->d_R = (cuFloatComplex*)kraken_gpu_alloc(num_taps * num_taps * sizeof(cuFloatComplex));
    proc->d_p = (cuFloatComplex*)kraken_gpu_alloc(num_taps * sizeof(cuFloatComplex));
    proc->d_w = (cuFloatComplex*)kraken_gpu_alloc(num_taps * sizeof(cuFloatComplex));
    proc->d_temp = (float*)kraken_gpu_alloc(max_samples * sizeof(float));

    if (!proc->d_ref_history || !proc->d_full_ref || !proc->d_surv ||
        !proc->d_output || !proc->d_R || !proc->d_p || !proc->d_w || !proc->d_temp) {
        fprintf(stderr, "ECA GPU: Failed to allocate device memory\n");
        eca_gpu_destroy(proc);
        return nullptr;
    }

    // Initialize history to zero
    if (cudaMemset(proc->d_ref_history, 0, proc->history_len * sizeof(cuFloatComplex)) != cudaSuccess) {
        fprintf(stderr, "ECA GPU: Failed to initialize history\n");
        eca_gpu_destroy(proc);
        return nullptr;
    }

    // Allocate pinned host memory
    proc->h_input_pinned = (float*)kraken_gpu_alloc_host(2 * max_samples * sizeof(float));
    proc->h_output_pinned = (float*)kraken_gpu_alloc_host(2 * max_samples * sizeof(float));

    if (!proc->h_input_pinned || !proc->h_output_pinned) {
        fprintf(stderr, "ECA GPU: Failed to allocate pinned memory\n");
        eca_gpu_destroy(proc);
        return nullptr;
    }

    printf("ECA GPU: Created processor with %d taps, cuSOLVER Cholesky enabled\n", num_taps);
    return proc;
}

/**
 * Destroy GPU ECA-B processor
 */
void eca_gpu_destroy(void* handle) {
    if (!handle) return;

    ECAProcessorGPU* proc = static_cast<ECAProcessorGPU*>(handle);

#if HAVE_CUSOLVER
    if (proc->cusolver_handle) cusolverDnDestroy(proc->cusolver_handle);
    if (proc->cublas_handle) cublasDestroy(proc->cublas_handle);
    if (proc->d_chol_work) kraken_gpu_free(proc->d_chol_work);
    if (proc->d_info) kraken_gpu_free(proc->d_info);
#endif

    if (proc->d_ref_history) kraken_gpu_free(proc->d_ref_history);
    if (proc->d_full_ref) kraken_gpu_free(proc->d_full_ref);
    if (proc->d_surv) kraken_gpu_free(proc->d_surv);
    if (proc->d_output) kraken_gpu_free(proc->d_output);
    if (proc->d_R) kraken_gpu_free(proc->d_R);
    if (proc->d_p) kraken_gpu_free(proc->d_p);
    if (proc->d_w) kraken_gpu_free(proc->d_w);
    if (proc->d_temp) kraken_gpu_free(proc->d_temp);

    if (proc->h_input_pinned) kraken_gpu_free_host(proc->h_input_pinned);
    if (proc->h_output_pinned) kraken_gpu_free_host(proc->h_output_pinned);

    if (proc->stream) cudaStreamDestroy(proc->stream);

    delete proc;
}

/**
 * Process samples through GPU ECA-B filter
 */
void eca_gpu_process(void* handle, const float* ref_in, const float* surv_in,
                     float* out_err, int n_samples) {
    if (!handle || !ref_in || !surv_in || !out_err || n_samples <= 0) return;

    ECAProcessorGPU* proc = static_cast<ECAProcessorGPU*>(handle);
    const int current_delay = proc->delay_samples.load(std::memory_order_acquire);
    const int base = proc->history_len + current_delay;
    const int full_ref_len = proc->history_len + current_delay + n_samples;
    const int num_taps = proc->num_taps;

    dim3 block(256);
    dim3 grid_samples((n_samples + block.x - 1) / block.x);
    dim3 grid_taps((num_taps + block.x - 1) / block.x);

    // Copy history to full_ref
    cudaError_t err;
    err = cudaMemcpyAsync(proc->d_full_ref, proc->d_ref_history,
                          proc->history_len * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToDevice, proc->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ECA GPU: cudaMemcpyAsync failed\n");
        return;
    }

    // Convert and copy input data
    memcpy(proc->h_input_pinned, ref_in, 2 * (current_delay + n_samples) * sizeof(float));
    interleaved_to_complex_kernel<<<dim3((current_delay + n_samples + 255) / 256), block, 0, proc->stream>>>(
        proc->h_input_pinned,
        proc->d_full_ref + proc->history_len,
        current_delay + n_samples
    );

    // Sync before reusing pinned buffer to avoid race condition
    cudaStreamSynchronize(proc->stream);

    memcpy(proc->h_input_pinned, surv_in, 2 * n_samples * sizeof(float));
    interleaved_to_complex_kernel<<<grid_samples, block, 0, proc->stream>>>(
        proc->h_input_pinned, proc->d_surv, n_samples
    );

    // Build autocorrelation matrix R on GPU
    // R[j,k] = sum_n(ref[base-j+n] * conj(ref[base-k+n]))
    cudaMemsetAsync(proc->d_R, 0, num_taps * num_taps * sizeof(cuFloatComplex), proc->stream);

    for (int j = 0; j < num_taps; j++) {
        for (int k = 0; k < num_taps; k++) {
            cuFloatComplex* d_result = proc->d_R + (j * num_taps + k);
            cudaMemsetAsync(d_result, 0, sizeof(cuFloatComplex), proc->stream);

            int grid_size = (n_samples + 255) / 256;
            complex_dot_kernel<<<grid_size, block, 0, proc->stream>>>(
                proc->d_full_ref + (base - j),
                proc->d_full_ref + (base - k),
                d_result,
                n_samples
            );
        }
    }

    // Add diagonal loading
    add_diagonal_kernel<<<grid_taps, block, 0, proc->stream>>>(
        proc->d_R, num_taps, proc->diagonal_loading
    );

    // Build cross-correlation vector p
    // p[j] = sum_n(ref[base-j+n] * conj(surv[n]))
    cudaMemsetAsync(proc->d_p, 0, num_taps * sizeof(cuFloatComplex), proc->stream);

    for (int j = 0; j < num_taps; j++) {
        cuFloatComplex* d_result = proc->d_p + j;
        cudaMemsetAsync(d_result, 0, sizeof(cuFloatComplex), proc->stream);

        int grid_size = (n_samples + 255) / 256;
        complex_dot_kernel<<<grid_size, block, 0, proc->stream>>>(
            proc->d_full_ref + (base - j),
            proc->d_surv,
            d_result,
            n_samples
        );
    }

    // Synchronize before solver
    cudaStreamSynchronize(proc->stream);

#if HAVE_CUSOLVER
    // Solve R*w=p using cuSOLVER Cholesky (GPU-accelerated)
    int result = solve_linear_system_gpu(proc, num_taps);
    if (result != 0) {
        // Fall back to CPU solver on failure
        cuFloatComplex* h_R = new cuFloatComplex[num_taps * num_taps];
        cuFloatComplex* h_p = new cuFloatComplex[num_taps];
        cuFloatComplex* h_w = new cuFloatComplex[num_taps];

        cudaMemcpy(h_R, proc->d_R, num_taps * num_taps * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p, proc->d_p, num_taps * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        solve_linear_system_cpu(h_R, h_p, h_w, num_taps);

        cudaMemcpy(proc->d_w, h_w, num_taps * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        delete[] h_R;
        delete[] h_p;
        delete[] h_w;
    }
#else
    // CPU-only fallback
    cuFloatComplex* h_R = new cuFloatComplex[num_taps * num_taps];
    cuFloatComplex* h_p = new cuFloatComplex[num_taps];
    cuFloatComplex* h_w = new cuFloatComplex[num_taps];

    cudaMemcpy(h_R, proc->d_R, num_taps * num_taps * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p, proc->d_p, num_taps * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    solve_linear_system_cpu(h_R, h_p, h_w, num_taps);

    cudaMemcpy(proc->d_w, h_w, num_taps * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    delete[] h_R;
    delete[] h_p;
    delete[] h_w;
#endif

    // Apply FIR filter on GPU
    int start_idx = base - (num_taps - 1);
    apply_fir_kernel<<<grid_samples, block, 0, proc->stream>>>(
        proc->d_w, proc->d_full_ref, proc->d_output, num_taps, start_idx, n_samples
    );

    // Compute error signal
    compute_error_kernel<<<grid_samples, block, 0, proc->stream>>>(
        proc->d_surv, proc->d_output, proc->d_output, n_samples
    );

    // Convert back to interleaved and transfer to host
    complex_to_interleaved_kernel<<<grid_samples, block, 0, proc->stream>>>(
        proc->d_output, proc->h_output_pinned, n_samples
    );

    // Synchronize before host-to-host copy (streams don't support H2H)
    cudaStreamSynchronize(proc->stream);
    memcpy(out_err, proc->h_output_pinned, 2 * n_samples * sizeof(float));

    // Update history
    cudaMemcpyAsync(proc->d_ref_history,
                    proc->d_full_ref + (full_ref_len - proc->history_len),
                    proc->history_len * sizeof(cuFloatComplex),
                    cudaMemcpyDeviceToDevice, proc->stream);

    cudaStreamSynchronize(proc->stream);
}

/**
 * Set delay
 */
void eca_gpu_set_delay(void* handle, int delay_samples) {
    if (!handle) return;
    ECAProcessorGPU* proc = static_cast<ECAProcessorGPU*>(handle);

    if (delay_samples < 0) delay_samples = 0;
    if (delay_samples > proc->max_delay) delay_samples = proc->max_delay;

    proc->delay_samples.store(delay_samples, std::memory_order_release);
}

/**
 * Get delay
 */
int eca_gpu_get_delay(void* handle) {
    if (!handle) return 0;
    ECAProcessorGPU* proc = static_cast<ECAProcessorGPU*>(handle);
    return proc->delay_samples.load(std::memory_order_acquire);
}
