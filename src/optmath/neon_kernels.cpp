#include "neon_kernels.hpp"
#include <cmath>
#include <algorithm>

// Check for ARM architecture to enable NEON
#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define OPTMATH_USE_NEON
#endif

namespace optmath {
namespace neon {

bool is_available() {
#ifdef OPTMATH_USE_NEON
    return true;
#else
    return false;
#endif
}

// =========================================================================
// Core Intrinsics Implementations
// =========================================================================

float neon_dot_f32(const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Unrolled loop 4x
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, a0, b0);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        vsum = vmlaq_f32(vsum, a1, b1);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        vsum = vmlaq_f32(vsum, a2, b2);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        vsum = vmlaq_f32(vsum, a3, b3);
    }

    // Residual blocks of 4
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }

    float sum = vaddvq_f32(vsum);

    // Scalar tail
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

void neon_add_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
#endif
}

void neon_sub_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
#endif
}

void neon_mul_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

void neon_div_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vdivq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
#endif
}

float neon_norm_f32(const float* a, std::size_t n) {
    float dot = neon_dot_f32(a, a, n);
    return std::sqrt(dot);
}

float neon_reduce_sum_f32(const float* a, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        vsum = vaddq_f32(vsum, vld1q_f32(a + i));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 4));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 8));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 12));
    }
    for (; i + 3 < n; i += 4) {
        vsum = vaddq_f32(vsum, vld1q_f32(a + i));
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; ++i) sum += a[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i=0; i<n; ++i) sum += a[i];
    return sum;
#endif
}

float neon_reduce_max_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_NEON
    float32x4_t vmax = vdupq_n_f32(-3.402823466e+38f);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(a + i));
    }
    float max_val = vmaxvq_f32(vmax);
    for (; i < n; ++i) if(a[i] > max_val) max_val = a[i];
    return max_val;
#else
    float m = a[0];
    for(size_t i=1; i<n; ++i) if(a[i] > m) m = a[i];
    return m;
#endif
}

float neon_reduce_min_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_NEON
    float32x4_t vmin = vdupq_n_f32(3.402823466e+38f);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vmin = vminq_f32(vmin, vld1q_f32(a + i));
    }
    float min_val = vminvq_f32(vmin);
    for (; i < n; ++i) if(a[i] < min_val) min_val = a[i];
    return min_val;
#else
    float m = a[0];
    for(size_t i=1; i<n; ++i) if(a[i] < m) m = a[i];
    return m;
#endif
}

void neon_gemm_4x4_f32(float* C, const float* A, std::size_t lda, const float* B, std::size_t ldb, std::size_t ldc) {
#ifdef OPTMATH_USE_NEON
    // Load C columns
    float32x4_t c0 = vld1q_f32(C);
    float32x4_t c1 = vld1q_f32(C + ldc);
    float32x4_t c2 = vld1q_f32(C + 2*ldc);
    float32x4_t c3 = vld1q_f32(C + 3*ldc);

    // Load A columns
    float32x4_t a0 = vld1q_f32(A);
    float32x4_t a1 = vld1q_f32(A + lda);
    float32x4_t a2 = vld1q_f32(A + 2*lda);
    float32x4_t a3 = vld1q_f32(A + 3*lda);

    // B is 4x4 block.
    // Helper to accumulate one column of B into C
    auto accumulate_col = [&](float32x4_t& c_col, const float* b_col_ptr) {
        c_col = vmlaq_n_f32(c_col, a0, b_col_ptr[0]);
        c_col = vmlaq_n_f32(c_col, a1, b_col_ptr[1]);
        c_col = vmlaq_n_f32(c_col, a2, b_col_ptr[2]);
        c_col = vmlaq_n_f32(c_col, a3, b_col_ptr[3]);
    };

    accumulate_col(c0, B);
    accumulate_col(c1, B + ldb);
    accumulate_col(c2, B + 2*ldb);
    accumulate_col(c3, B + 3*ldb);

    vst1q_f32(C, c0);
    vst1q_f32(C + ldc, c1);
    vst1q_f32(C + 2*ldc, c2);
    vst1q_f32(C + 3*ldc, c3);
#else
    // Fallback scalar
    for(int j=0; j<4; ++j) {
        for(int i=0; i<4; ++i) {
            float sum = 0.0f;
            for(int k=0; k<4; ++k) {
                sum += A[i + k*lda] * B[k + j*ldb];
            }
            C[i + j*ldc] += sum;
        }
    }
#endif
}

void neon_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y) {
    size_t n_y = (n_x >= n_h) ? (n_x - n_h + 1) : 0;
    for (size_t i = 0; i < n_y; ++i) {
        y[i] = neon_dot_f32(x + i, h, n_h);
    }
}

void neon_relu_f32(float* data, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for(; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vmaxq_f32(v, vzero));
    }
    for(; i < n; ++i) {
        if(data[i] < 0.0f) data[i] = 0.0f;
    }
#else
    for(size_t i=0; i<n; ++i) if(data[i] < 0.0f) data[i] = 0.0f;
#endif
}

void neon_sigmoid_f32(float* data, std::size_t n) {
    for(size_t i=0; i<n; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

void neon_tanh_f32(float* data, std::size_t n) {
    for(size_t i=0; i<n; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

}
}
