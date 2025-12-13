#pragma once

#include <vector>
#include <cstddef>

namespace optmath {
namespace neon {

    /**
     * @brief Checks if NEON acceleration was compiled in.
     */
    bool is_available();

    // --- Core Intrinsics Wrappers ---

    float neon_dot_f32(const float* a, const float* b, std::size_t n);

    void neon_add_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_sub_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_mul_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_div_f32(float* out, const float* a, const float* b, std::size_t n);

    // Reductions
    float neon_norm_f32(const float* a, std::size_t n);
    float neon_reduce_sum_f32(const float* a, std::size_t n);
    float neon_reduce_max_f32(const float* a, std::size_t n);
    float neon_reduce_min_f32(const float* a, std::size_t n);

    // Matrix
    // C += A * B (4x4 block)
    void neon_gemm_4x4_f32(float* C, const float* A, std::size_t lda, const float* B, std::size_t ldb, std::size_t ldc);

    void neon_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y);

    void neon_relu_f32(float* data, std::size_t n);
    void neon_sigmoid_f32(float* data, std::size_t n);
    void neon_tanh_f32(float* data, std::size_t n);

}
}
