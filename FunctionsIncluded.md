# OptMathKernels API Reference

**Complete Function Reference for OptimizedKernelsForRaspberryPi5**

This document provides a comprehensive reference for all functions available in the OptMathKernels library, which provides hardware-accelerated mathematical operations for:
- **NEON SIMD** (ARM Cortex-A76 on Raspberry Pi 5)
- **CUDA** (NVIDIA GPUs: RTX 2000/3000/4000/5000 series)
- **Vulkan Compute** (Cross-platform GPU acceleration)

---

## Table of Contents

1. [NEON Backend](#neon-backend)
   - [Vector Operations](#neon-vector-operations)
   - [Complex Number Operations](#neon-complex-operations)
   - [Matrix Operations](#neon-matrix-operations)
   - [Transcendental Functions](#neon-transcendental-functions)
   - [Radar Signal Processing](#neon-radar-operations)
2. [CUDA Backend](#cuda-backend)
   - [Device Management](#cuda-device-management)
   - [Memory Management](#cuda-memory-management)
   - [Vector Operations](#cuda-vector-operations)
   - [Matrix Operations](#cuda-matrix-operations)
   - [FFT Operations](#cuda-fft-operations)
   - [Linear Algebra](#cuda-linear-algebra)
   - [Radar Signal Processing](#cuda-radar-operations)
3. [Vulkan Backend](#vulkan-backend)
   - [Vector Operations](#vulkan-vector-operations)
   - [Matrix Operations](#vulkan-matrix-operations)
   - [DSP Operations](#vulkan-dsp-operations)
4. [Radar Kernels](#radar-kernels)
   - [Cross-Ambiguity Function](#caf-functions)
   - [CFAR Detection](#cfar-functions)
   - [Clutter Filtering](#clutter-filtering-functions)
   - [Beamforming](#beamforming-functions)

---

## NEON Backend

**Header:** `#include <optmath/neon_kernels.hpp>`
**Namespace:** `optmath::neon`

The NEON backend provides SIMD-accelerated operations optimized for ARM Cortex-A76 (Raspberry Pi 5). All functions use 128-bit NEON registers for 4-way float32 parallelism.

### NEON Vector Operations

#### Low-Level Functions (Pointer-Based)

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `neon_add_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Element-wise vector addition |
| `neon_sub_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Element-wise vector subtraction |
| `neon_mul_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Element-wise vector multiplication |
| `neon_div_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Element-wise vector division |
| `neon_scale_f32` | `void` | `float* out, const float* a, float scalar, size_t n` | Scalar multiplication |
| `neon_fma_f32` | `void` | `float* out, const float* a, const float* b, const float* c, size_t n` | Fused multiply-add: `a * b + c` |
| `neon_dot_f32` | `float` | `const float* a, const float* b, size_t n` | Dot product |
| `neon_sum_f32` | `float` | `const float* a, size_t n` | Sum of all elements |
| `neon_max_f32` | `float` | `const float* a, size_t n` | Maximum element |
| `neon_min_f32` | `float` | `const float* a, size_t n` | Minimum element |
| `neon_abs_f32` | `void` | `float* out, const float* a, size_t n` | Absolute value |
| `neon_sqrt_f32` | `void` | `float* out, const float* a, size_t n` | Square root |
| `neon_rsqrt_f32` | `void` | `float* out, const float* a, size_t n` | Reciprocal square root |
| `neon_clamp_f32` | `void` | `float* out, const float* a, float lo, float hi, size_t n` | Clamp to range [lo, hi] |
| `neon_threshold_f32` | `void` | `float* out, const float* a, float thresh, size_t n` | Set to 0 if below threshold |

#### Eigen Wrapper Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `neon_add` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector addition |
| `neon_sub` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector subtraction |
| `neon_mul` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise multiplication |
| `neon_div` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise division |
| `neon_scale` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, float scalar` | Scalar multiplication |
| `neon_fma` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b, const Eigen::VectorXf& c` | Fused multiply-add |
| `neon_dot` | `float` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Dot product |
| `neon_sum` | `float` | `const Eigen::VectorXf& a` | Sum reduction |
| `neon_max` | `float` | `const Eigen::VectorXf& a` | Max reduction |
| `neon_min` | `float` | `const Eigen::VectorXf& a` | Min reduction |
| `neon_norm` | `float` | `const Eigen::VectorXf& a` | L2 norm |
| `neon_normalize` | `Eigen::VectorXf` | `const Eigen::VectorXf& a` | Normalize to unit length |

---

### NEON Complex Operations

**Header:** `#include <optmath/neon_complex.hpp>`

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `neon_complex_mul_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, size_t n` | Complex multiplication (split format) |
| `neon_complex_conj_mul_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, size_t n` | Complex conjugate multiplication: `a * conj(b)` |
| `neon_complex_mul_interleaved_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Complex multiplication (interleaved IQ format) |
| `neon_complex_conj_mul_interleaved_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Complex conjugate multiplication (interleaved) |
| `neon_complex_magnitude_f32` | `void` | `float* out, const float* re, const float* im, size_t n` | Compute `|z| = sqrt(re² + im²)` |
| `neon_complex_magnitude_squared_f32` | `void` | `float* out, const float* re, const float* im, size_t n` | Compute `|z|² = re² + im²` |
| `neon_complex_phase_f32` | `void` | `float* out, const float* re, const float* im, size_t n` | Compute `arg(z) = atan2(im, re)` |
| `neon_complex_exp_f32` | `void` | `float* out_re, float* out_im, const float* phase, size_t n` | Compute `e^(j*phase)` |
| `neon_complex_dot_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, size_t n` | Complex dot product |
| `neon_complex_accumulate_f32` | `void` | `float* acc_re, float* acc_im, const float* a_re, const float* a_im, size_t n` | Accumulate complex sum |

#### Eigen Wrappers for Complex Operations

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `neon_complex_mul` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& a, const Eigen::VectorXcf& b` | Complex vector multiplication |
| `neon_complex_conj_mul` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& a, const Eigen::VectorXcf& b` | Complex conjugate multiplication |
| `neon_complex_dot` | `std::complex<float>` | `const Eigen::VectorXcf& a, const Eigen::VectorXcf& b` | Complex dot product |
| `neon_complex_magnitude` | `Eigen::VectorXf` | `const Eigen::VectorXcf& a` | Magnitude of each element |
| `neon_complex_phase` | `Eigen::VectorXf` | `const Eigen::VectorXcf& a` | Phase of each element |

---

### NEON Matrix Operations

**Header:** `#include <optmath/neon_gemm.hpp>`

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `neon_gemm_f32` | `void` | `float* C, const float* A, const float* B, int M, int N, int K, float alpha, float beta` | General matrix multiply: `C = alpha*A*B + beta*C` |
| `neon_gemv_f32` | `void` | `float* y, const float* A, const float* x, int M, int N, float alpha, float beta` | Matrix-vector multiply: `y = alpha*A*x + beta*y` |
| `neon_transpose_f32` | `void` | `float* out, const float* A, int M, int N` | Matrix transpose |
| `neon_mat_add_f32` | `void` | `float* C, const float* A, const float* B, int M, int N` | Matrix addition |
| `neon_mat_scale_f32` | `void` | `float* out, const float* A, float scalar, int M, int N` | Matrix scalar multiplication |

#### Eigen Wrappers

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `neon_gemm` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A, const Eigen::MatrixXf& B` | Matrix multiplication |
| `neon_gemv` | `Eigen::VectorXf` | `const Eigen::MatrixXf& A, const Eigen::VectorXf& x` | Matrix-vector multiplication |
| `neon_transpose` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Matrix transpose |

---

### NEON Transcendental Functions

**Header:** `#include <optmath/neon_kernels.hpp>`

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `neon_exp_f32` | `void` | `float* out, const float* a, size_t n` | Exponential (polynomial approximation) |
| `neon_log_f32` | `void` | `float* out, const float* a, size_t n` | Natural logarithm |
| `neon_sin_f32` | `void` | `float* out, const float* a, size_t n` | Sine |
| `neon_cos_f32` | `void` | `float* out, const float* a, size_t n` | Cosine |
| `neon_sincos_f32` | `void` | `float* sin_out, float* cos_out, const float* a, size_t n` | Simultaneous sine and cosine |
| `neon_tan_f32` | `void` | `float* out, const float* a, size_t n` | Tangent |
| `neon_atan2_f32` | `void` | `float* out, const float* y, const float* x, size_t n` | Two-argument arctangent |
| `neon_pow_f32` | `void` | `float* out, const float* base, const float* exp, size_t n` | Power function |
| `neon_sigmoid_f32` | `void` | `float* out, const float* a, size_t n` | Sigmoid activation |
| `neon_tanh_f32` | `void` | `float* out, const float* a, size_t n` | Hyperbolic tangent |
| `neon_relu_f32` | `void` | `float* out, const float* a, size_t n` | ReLU activation |
| `neon_leaky_relu_f32` | `void` | `float* out, const float* a, float alpha, size_t n` | Leaky ReLU |

---

### NEON Radar Operations

**Header:** `#include <optmath/radar_kernels.hpp>`

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `caf_f32` | `void` | `float* out_mag, const float* ref_re, const float* ref_im, const float* surv_re, const float* surv_im, size_t n_samples, size_t n_doppler_bins, float doppler_start, float doppler_step, float sample_rate, size_t n_range_bins` | Cross-Ambiguity Function |
| `cfar_ca_f32` | `void` | `uint8_t* detections, float* threshold, const float* input, size_t n, size_t guard_cells, size_t ref_cells, float pfa_factor` | Cell-Averaging CFAR |
| `cfar_2d_f32` | `void` | `uint8_t* detections, const float* input, size_t n_doppler, size_t n_range, size_t guard_range, size_t guard_doppler, size_t ref_range, size_t ref_doppler, float pfa_factor` | 2D CFAR detector |
| `nlms_filter_f32` | `void` | `float* output, float* weights, const float* input, const float* reference, size_t n, size_t filter_length, float mu, float eps` | NLMS adaptive filter |
| `steering_vector_ula_f32` | `void` | `float* steering_re, float* steering_im, size_t n_elements, float d_lambda, float theta_rad` | ULA steering vector |

---

## CUDA Backend

**Header:** `#include <optmath/cuda_backend.hpp>`
**Namespace:** `optmath::cuda`

The CUDA backend provides GPU-accelerated operations using cuBLAS, cuFFT, and cuSOLVER. Supports RTX 2000 (Turing), 3000 (Ampere), 4000 (Ada Lovelace), and 5000 (Blackwell) series GPUs.

### CUDA Device Management

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `is_available` | `bool` | `void` | Check if CUDA is available |
| `init` | `void` | `int device_id = 0` | Initialize CUDA context |
| `cleanup` | `void` | `void` | Release CUDA resources |
| `synchronize` | `void` | `void` | Wait for all operations to complete |
| `get_device_count` | `int` | `void` | Number of CUDA devices |
| `get_device_info` | `DeviceInfo` | `int device_id = 0` | Get device capabilities |
| `set_device` | `void` | `int device_id` | Set active device |
| `get_device` | `int` | `void` | Get current device |

#### DeviceInfo Structure

```cpp
struct DeviceInfo {
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
    bool tensor_cores;      // Volta+ (SM 7.0+)
    bool tf32_support;      // Ampere+ (SM 8.0+)
    bool fp16_support;      // All modern GPUs
    bool fp8_support;       // Blackwell (SM 10.0+)
    bool blackwell;         // Blackwell architecture

    bool is_blackwell_or_newer() const;
};
```

---

### CUDA Memory Management

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_malloc` | `void*` | `size_t bytes` | Allocate device memory |
| `cuda_free` | `void` | `void* ptr` | Free device memory |
| `cuda_memcpy_h2d` | `void` | `void* dst, const void* src, size_t bytes` | Copy host to device |
| `cuda_memcpy_d2h` | `void` | `void* dst, const void* src, size_t bytes` | Copy device to host |
| `cuda_memcpy_d2d` | `void` | `void* dst, const void* src, size_t bytes` | Copy device to device |
| `cuda_memset` | `void` | `void* ptr, int value, size_t bytes` | Set device memory |

---

### CUDA Vector Operations

#### Low-Level Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_add_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Vector addition |
| `cuda_sub_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Vector subtraction |
| `cuda_mul_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Element-wise multiplication |
| `cuda_div_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Element-wise division |
| `cuda_scale_f32` | `void` | `float* out, const float* a, float scalar, size_t n` | Scalar multiplication |
| `cuda_axpy_f32` | `void` | `float* y, float alpha, const float* x, size_t n` | y = alpha*x + y (cuBLAS) |
| `cuda_dot_f32` | `float` | `const float* a, const float* b, size_t n` | Dot product (cuBLAS) |
| `cuda_nrm2_f32` | `float` | `const float* a, size_t n` | L2 norm (cuBLAS) |
| `cuda_asum_f32` | `float` | `const float* a, size_t n` | Sum of absolute values |
| `cuda_reduce_sum_f32` | `float` | `const float* a, size_t n` | Parallel sum reduction |
| `cuda_reduce_max_f32` | `float` | `const float* a, size_t n` | Parallel max reduction |
| `cuda_reduce_min_f32` | `float` | `const float* a, size_t n` | Parallel min reduction |

#### Transcendental Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_exp_f32` | `void` | `float* out, const float* in, size_t n` | Exponential |
| `cuda_log_f32` | `void` | `float* out, const float* in, size_t n` | Natural logarithm |
| `cuda_sin_f32` | `void` | `float* out, const float* in, size_t n` | Sine |
| `cuda_cos_f32` | `void` | `float* out, const float* in, size_t n` | Cosine |
| `cuda_sincos_f32` | `void` | `float* sin_out, float* cos_out, const float* in, size_t n` | Simultaneous sin/cos |
| `cuda_sqrt_f32` | `void` | `float* out, const float* in, size_t n` | Square root |
| `cuda_rsqrt_f32` | `void` | `float* out, const float* in, size_t n` | Reciprocal square root |
| `cuda_pow_f32` | `void` | `float* out, const float* base, const float* exp, size_t n` | Power |
| `cuda_atan2_f32` | `void` | `float* out, const float* y, const float* x, size_t n` | Two-argument arctangent |

#### Activation Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_sigmoid_f32` | `void` | `float* out, const float* in, size_t n` | Sigmoid: `1/(1+exp(-x))` |
| `cuda_tanh_f32` | `void` | `float* out, const float* in, size_t n` | Hyperbolic tangent |
| `cuda_relu_f32` | `void` | `float* out, const float* in, size_t n` | ReLU: `max(0, x)` |
| `cuda_leaky_relu_f32` | `void` | `float* out, const float* in, float alpha, size_t n` | Leaky ReLU |
| `cuda_gelu_f32` | `void` | `float* out, const float* in, size_t n` | GELU activation |
| `cuda_softmax_f32` | `void` | `float* out, const float* in, size_t n` | Softmax |

#### Eigen Wrappers

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_add` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector addition |
| `cuda_mul` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise multiply |
| `cuda_scale` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, float s` | Scalar multiply |
| `cuda_dot` | `float` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Dot product |
| `cuda_sum` | `float` | `const Eigen::VectorXf& a` | Sum |
| `cuda_max` | `float` | `const Eigen::VectorXf& a` | Maximum |
| `cuda_min` | `float` | `const Eigen::VectorXf& a` | Minimum |
| `cuda_sqrt` | `Eigen::VectorXf` | `const Eigen::VectorXf& a` | Square root |
| `cuda_exp` | `Eigen::VectorXf` | `const Eigen::VectorXf& x` | Exponential |
| `cuda_log` | `Eigen::VectorXf` | `const Eigen::VectorXf& x` | Logarithm |
| `cuda_sin` | `Eigen::VectorXf` | `const Eigen::VectorXf& x` | Sine |
| `cuda_cos` | `Eigen::VectorXf` | `const Eigen::VectorXf& x` | Cosine |
| `cuda_sigmoid` | `Eigen::VectorXf` | `const Eigen::VectorXf& x` | Sigmoid |
| `cuda_relu` | `Eigen::VectorXf` | `const Eigen::VectorXf& x` | ReLU |

---

### CUDA Matrix Operations

#### Low-Level Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_mat_mul_f32` | `void` | `float* C, const float* A, const float* B, int M, int N, int K, bool transA, bool transB` | GEMM via cuBLAS |
| `cuda_mat_add_f32` | `void` | `float* C, const float* A, const float* B, int M, int N` | Matrix addition |
| `cuda_mat_scale_f32` | `void` | `float* out, const float* A, float scalar, int M, int N` | Scalar multiply |
| `cuda_mat_transpose_f32` | `void` | `float* out, const float* A, int M, int N` | Transpose |
| `cuda_mat_vec_mul_f32` | `void` | `float* out, const float* A, const float* x, int M, int N` | GEMV via cuBLAS |
| `cuda_batched_mat_mul_f32` | `void` | `float** C, float** A, float** B, int M, int N, int K, int batch` | Batched GEMM |

#### Tensor Core Functions (Ampere+)

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_mat_mul_tensorcore_f32` | `void` | `float* C, const float* A, const float* B, int M, int N, int K` | TF32 Tensor Core GEMM |
| `cuda_mat_mul_tensorcore_fp16` | `void` | `void* C, const void* A, const void* B, int M, int N, int K` | FP16 Tensor Core GEMM |

#### Eigen Wrappers

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_gemm` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A, const Eigen::MatrixXf& B` | Matrix multiplication |
| `cuda_gemv` | `Eigen::VectorXf` | `const Eigen::MatrixXf& A, const Eigen::VectorXf& x` | Matrix-vector multiply |
| `cuda_transpose` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Transpose |
| `cuda_mat_add` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A, const Eigen::MatrixXf& B` | Matrix addition |
| `cuda_mat_scale` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A, float scalar` | Scalar multiply |

---

### CUDA FFT Operations

#### CudaFFTPlan Class

```cpp
class CudaFFTPlan {
public:
    bool create_1d(size_t n, bool inverse = false);
    bool create_1d_batch(size_t n, size_t batch, bool inverse = false);
    bool create_2d(size_t nx, size_t ny, bool inverse = false);
    void execute(float* inout);
    void execute(const float* in, float* out);
    void destroy();
};
```

#### One-Shot Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_fft_1d_f32` | `void` | `float* inout, size_t n, bool inverse` | In-place 1D FFT |
| `cuda_fft_1d_batch_f32` | `void` | `float* inout, size_t n, size_t batch, bool inverse` | Batched 1D FFT |
| `cuda_fft_2d_f32` | `void` | `float* inout, size_t nx, size_t ny, bool inverse` | 2D FFT |

#### Eigen Wrappers

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_fft` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& x` | Forward FFT |
| `cuda_ifft` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& x` | Inverse FFT |
| `cuda_fft2` | `Eigen::MatrixXcf` | `const Eigen::MatrixXcf& x` | 2D FFT |
| `cuda_ifft2` | `Eigen::MatrixXcf` | `const Eigen::MatrixXcf& x` | 2D inverse FFT |
| `cuda_rfft` | `Eigen::VectorXcf` | `const Eigen::VectorXf& x` | Real-to-complex FFT |
| `cuda_irfft` | `Eigen::VectorXf` | `const Eigen::VectorXcf& x, size_t n` | Complex-to-real IFFT |

---

### CUDA Linear Algebra

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_cholesky` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Cholesky decomposition (cuSOLVER) |
| `cuda_lu` | `pair<MatrixXf, VectorXi>` | `const Eigen::MatrixXf& A` | LU decomposition |
| `cuda_qr` | `pair<MatrixXf, MatrixXf>` | `const Eigen::MatrixXf& A` | QR decomposition |
| `cuda_svd` | `SVDResult` | `const Eigen::MatrixXf& A` | Singular Value Decomposition |
| `cuda_eig` | `pair<VectorXf, MatrixXf>` | `const Eigen::MatrixXf& A` | Eigendecomposition |
| `cuda_solve` | `Eigen::VectorXf` | `const Eigen::MatrixXf& A, const Eigen::VectorXf& b` | Solve Ax = b |
| `cuda_inverse` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Matrix inverse |

---

### CUDA Radar Operations

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_caf` | `Eigen::MatrixXf` | `const Eigen::VectorXcf& ref, const Eigen::VectorXcf& surv, size_t n_doppler, float doppler_start, float doppler_step, float sample_rate, size_t n_range` | Cross-Ambiguity Function |
| `cuda_cfar_2d` | `Eigen::MatrixXi` | `const Eigen::MatrixXf& power_map, int guard_r, int guard_d, int ref_r, int ref_d, float pfa` | 2D CFAR detector |
| `cuda_cfar_ca` | `Eigen::VectorXi` | `const Eigen::VectorXf& power, int guard, int ref, float pfa` | 1D CA-CFAR |
| `cuda_doppler_process` | `Eigen::MatrixXcf` | `const Eigen::MatrixXcf& data, size_t fft_size, int window` | Doppler processing |
| `cuda_bartlett_spectrum` | `Eigen::VectorXf` | `const Eigen::VectorXcf& array_data, float d_lambda, int n_angles` | Bartlett beamformer |
| `cuda_steering_vectors_ula` | `Eigen::MatrixXcf` | `int n_elements, float d_lambda, const Eigen::VectorXf& angles` | ULA steering vectors |
| `cuda_nlms_filter` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& surv, const Eigen::VectorXcf& ref, int len, float mu, float eps` | NLMS adaptive filter |
| `cuda_projection_clutter` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& surv, const Eigen::MatrixXcf& subspace` | Projection clutter cancellation |

#### Window Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cuda_generate_window` | `Eigen::VectorXf` | `size_t n, WindowType type, float param` | Generate window function |
| `cuda_apply_window` | `void` | `Eigen::VectorXf& data, const Eigen::VectorXf& window` | Apply window (in-place) |
| `cuda_apply_window` | `void` | `Eigen::VectorXcf& data, const Eigen::VectorXf& window` | Apply window to complex |

**WindowType enum:** `RECTANGULAR`, `HAMMING`, `HANNING`, `BLACKMAN`, `BLACKMAN_HARRIS`, `KAISER`, `GAUSSIAN`, `TUKEY`

---

## Vulkan Backend

**Header:** `#include <optmath/vulkan_backend.hpp>`
**Namespace:** `optmath::vulkan`

The Vulkan backend provides GPU-accelerated operations using Vulkan compute shaders. Works on any GPU with Vulkan 1.2 support.

### Vulkan Availability

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `is_available` | `bool` | `void` | Check if Vulkan is available |

### Vulkan Vector Operations

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `vulkan_vec_add` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector addition |
| `vulkan_vec_sub` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector subtraction |
| `vulkan_vec_mul` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise multiply |
| `vulkan_vec_div` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise divide |
| `vulkan_vec_dot` | `float` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Dot product |
| `vulkan_vec_norm` | `float` | `const Eigen::VectorXf& a` | L2 norm |
| `vulkan_reduce_sum` | `float` | `const Eigen::VectorXf& a` | Sum reduction |
| `vulkan_reduce_max` | `float` | `const Eigen::VectorXf& a` | Max reduction |
| `vulkan_reduce_min` | `float` | `const Eigen::VectorXf& a` | Min reduction |
| `vulkan_scan_prefix_sum` | `Eigen::VectorXf` | `const Eigen::VectorXf& a` | Parallel prefix sum |

### Vulkan Matrix Operations

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `vulkan_mat_add` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Matrix addition |
| `vulkan_mat_sub` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Matrix subtraction |
| `vulkan_mat_mul` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Matrix multiplication |
| `vulkan_mat_transpose` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a` | Transpose |
| `vulkan_mat_scale` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, float scalar` | Scalar multiply |
| `vulkan_mat_vec_mul` | `Eigen::VectorXf` | `const Eigen::MatrixXf& a, const Eigen::VectorXf& v` | Matrix-vector multiply |
| `vulkan_mat_outer_product` | `Eigen::MatrixXf` | `const Eigen::VectorXf& u, const Eigen::VectorXf& v` | Outer product |
| `vulkan_mat_elementwise_mul` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Hadamard product |

### Vulkan DSP Operations

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `vulkan_convolution_1d` | `Eigen::VectorXf` | `const Eigen::VectorXf& x, const Eigen::VectorXf& k` | 1D convolution |
| `vulkan_convolution_2d` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& x, const Eigen::MatrixXf& k` | 2D convolution |
| `vulkan_correlation_1d` | `Eigen::VectorXf` | `const Eigen::VectorXf& x, const Eigen::VectorXf& k` | 1D correlation |
| `vulkan_correlation_2d` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& x, const Eigen::MatrixXf& k` | 2D correlation |
| `vulkan_fft_radix2` | `void` | `Eigen::VectorXf& data, bool inverse` | Radix-2 FFT |
| `vulkan_fft_radix4` | `void` | `Eigen::VectorXf& data, bool inverse` | Radix-4 FFT |

---

## Radar Kernels

**Header:** `#include <optmath/radar_kernels.hpp>`
**Namespace:** `optmath::radar`

Specialized functions for passive radar signal processing.

### CAF Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `caf_f32` | `void` | `float* out, const float* ref_re, const float* ref_im, const float* surv_re, const float* surv_im, size_t n_samples, size_t n_doppler, float doppler_start, float doppler_step, float fs, size_t n_range` | Cross-Ambiguity Function (range-Doppler map) |
| `caf_fft_f32` | `void` | Same as above | CAF using FFT (faster for large arrays) |
| `caf` | `Eigen::MatrixXf` | `const Eigen::VectorXcf& ref, const Eigen::VectorXcf& surv, size_t n_doppler, float doppler_start, float doppler_step, float fs, size_t n_range` | Eigen wrapper |

### CFAR Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `cfar_ca_f32` | `void` | `uint8_t* det, float* thresh, const float* in, size_t n, size_t guard, size_t ref, float pfa` | 1D Cell-Averaging CFAR |
| `cfar_os_f32` | `void` | `uint8_t* det, float* thresh, const float* in, size_t n, size_t guard, size_t ref, size_t k, float pfa` | 1D Ordered-Statistic CFAR |
| `cfar_2d_f32` | `void` | `uint8_t* det, const float* in, size_t nd, size_t nr, size_t gr, size_t gd, size_t rr, size_t rd, float pfa` | 2D CFAR |
| `cfar_ca` | `Eigen::Matrix<uint8_t,...>` | `const Eigen::VectorXf& in, size_t guard, size_t ref, float pfa` | Eigen wrapper |
| `cfar_2d` | `Eigen::Matrix<uint8_t,...>` | `const Eigen::MatrixXf& in, size_t gr, size_t gd, size_t rr, size_t rd, float pfa` | Eigen wrapper |

### Clutter Filtering Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `nlms_filter_f32` | `void` | `float* out, float* w, const float* in, const float* ref, size_t n, size_t len, float mu, float eps` | NLMS adaptive filter for clutter cancellation |
| `projection_clutter_f32` | `void` | `float* out, const float* in, const float* subspace, size_t n, size_t dim` | Projection-based clutter cancellation |
| `nlms_filter` | `Eigen::VectorXf` | `const Eigen::VectorXf& in, const Eigen::VectorXf& ref, size_t len, float mu, float eps` | Eigen wrapper |
| `projection_clutter` | `Eigen::VectorXf` | `const Eigen::VectorXf& in, const Eigen::MatrixXf& subspace` | Eigen wrapper |

### Beamforming Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `steering_vector_ula_f32` | `void` | `float* re, float* im, size_t n_elem, float d_lambda, float theta` | Generate ULA steering vector |
| `beamform_delay_sum_f32` | `void` | `float* out, const float* ins, const int* delays, const float* weights, size_t n_ch, size_t n_samp` | Delay-and-sum beamformer |
| `beamform_phase_f32` | `void` | `float* out_re, float* out_im, const float* ins_re, const float* ins_im, const float* phases, const float* weights, size_t n_ch, size_t n_samp` | Phase-shift beamformer |
| `steering_vector_ula` | `Eigen::VectorXcf` | `size_t n_elem, float d_lambda, float theta` | Eigen wrapper |
| `beamform_delay_sum` | `Eigen::VectorXf` | `const Eigen::MatrixXf& ins, const Eigen::VectorXi& delays, const Eigen::VectorXf& weights` | Eigen wrapper |
| `beamform_phase` | `Eigen::VectorXcf` | `const Eigen::MatrixXcf& ins, const Eigen::VectorXf& phases, const Eigen::VectorXf& weights` | Eigen wrapper |

### Window Functions

| Function | Return | Parameters | Description |
|----------|--------|------------|-------------|
| `generate_window_f32` | `void` | `float* w, size_t n, WindowType type, float beta` | Generate window |
| `apply_window_f32` | `void` | `float* data, const float* w, size_t n` | Apply window (real) |
| `apply_window_complex_f32` | `void` | `float* re, float* im, const float* w, size_t n` | Apply window (complex) |
| `generate_window` | `Eigen::VectorXf` | `size_t n, WindowType type, float beta` | Eigen wrapper |
| `apply_window` | `void` | `Eigen::VectorXf& data, const Eigen::VectorXf& w` | Eigen wrapper |
| `apply_window` | `void` | `Eigen::VectorXcf& data, const Eigen::VectorXf& w` | Eigen wrapper |

---

## Summary

| Backend | Functions | Primary Use |
|---------|-----------|-------------|
| **NEON** | 83 | ARM SIMD (Raspberry Pi 5) |
| **CUDA** | 242 | NVIDIA GPUs (RTX 2000-5000) |
| **Vulkan** | 23 | Cross-platform GPU |
| **Radar** | 48 | Passive radar signal processing |
| **Total** | **396** | |

---

## See Also

- [README.md](README.md) - Main project documentation
- [OptimizedKernelsForRaspberryPi5](https://github.com/n4hy/OptimizedKernelsForRaspberryPi5) - Source repository
