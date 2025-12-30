#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "optmath/neon_kernels.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

using Complex = std::complex<float>;

// Helper for solving Ax = b using Cholesky Decomposition
// A must be Hermitian and Positive Definite
class LinearSolver {
public:
    static bool solve_cholesky(const std::vector<Complex>& A, const std::vector<Complex>& b, std::vector<Complex>& x, int N) {
        // 1. Cholesky Decomposition: A = L * L^H
        // L is lower triangular
        std::vector<Complex> L(N * N, 0.0f);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                Complex sum = 0.0f;
                for (int k = 0; k < j; ++k) {
                    sum += L[i * N + k] * std::conj(L[j * N + k]);
                }

                if (i == j) {
                    // Diagonal element
                    float val = std::real(A[i * N + i] - sum);
                    if (val <= 0.0f) return false; // Not positive definite
                    L[i * N + j] = std::sqrt(val);
                } else {
                    // Off-diagonal
                    L[i * N + j] = (A[i * N + j] - sum) / L[j * N + j]; // L[j,j] is real
                }
            }
        }

        // 2. Forward Substitution: L * y = b
        std::vector<Complex> y(N);
        for (int i = 0; i < N; ++i) {
            Complex sum = 0.0f;
            for (int j = 0; j < i; ++j) {
                sum += L[i * N + j] * y[j];
            }
            y[i] = (b[i] - sum) / L[i * N + i];
        }

        // 3. Backward Substitution: L^H * x = y
        // L^H is upper triangular
        for (int i = N - 1; i >= 0; --i) {
            Complex sum = 0.0f;
            for (int j = i + 1; j < N; ++j) {
                // (L^H)_{ij} = conj(L_{ji})
                sum += std::conj(L[j * N + i]) * x[j];
            }
            x[i] = (y[i] - sum) / std::conj(L[i * N + i]); // Diagonal of L is real, so conj is same
        }

        return true;
    }
};

class ECABCanceller {
private:
    int num_taps;
    float diagonal_loading;

    // History buffer
    std::vector<Complex> ref_history;

public:
    ECABCanceller(int taps) : num_taps(taps), diagonal_loading(1e-6f) {
        // Enable Flush-to-Zero and Denormals-Are-Zero on x86 to prevent performance cliff
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        #endif

        // Pre-fill history with zeros
        ref_history.resize(num_taps - 1, Complex(0.0f, 0.0f));
    }

    // Process a block
    // ref_in: Reference signal (Regressor)
    // surv_in: Surveillance signal (Desired)
    // out_err: Output (Error signal)
    // n_samples: Block size
    void process(const Complex* ref_in, const Complex* surv_in, Complex* out_err, int n_samples) {
        if (n_samples == 0) return;

        // 1. Prepare Reference Data with History
        std::vector<Complex> full_ref;
        full_ref.reserve(ref_history.size() + n_samples);
        full_ref.insert(full_ref.end(), ref_history.begin(), ref_history.end());
        full_ref.insert(full_ref.end(), ref_in, ref_in + n_samples);

        // Convert full_ref and surv_in to SoA for NEON operations
        int full_ref_len = full_ref.size();
        std::vector<float> ref_re(full_ref_len);
        std::vector<float> ref_im(full_ref_len);

        for(int i=0; i<full_ref_len; ++i) {
            ref_re[i] = full_ref[i].real();
            ref_im[i] = full_ref[i].imag();
        }

        std::vector<float> surv_re(n_samples);
        std::vector<float> surv_im(n_samples);
        for(int i=0; i<n_samples; ++i) {
            surv_re[i] = surv_in[i].real();
            surv_im[i] = surv_in[i].imag();
        }

        // 2. Compute Autocorrelation Matrix R (num_taps x num_taps)
        std::vector<Complex> R(num_taps * num_taps, 0.0f);

        for (int j = 0; j < num_taps; ++j) {
            int offset_j = num_taps - 1 - j;
            const float* r_re_j = ref_re.data() + offset_j;
            const float* r_im_j = ref_im.data() + offset_j;

            for (int k = j; k < num_taps; ++k) { // Upper triangle
                int offset_k = num_taps - 1 - k;
                const float* r_re_k = ref_re.data() + offset_k;
                const float* r_im_k = ref_im.data() + offset_k;

                // 4 dot products
                float rr = optmath::neon::neon_dot_f32(r_re_j, r_re_k, n_samples);
                float ii = optmath::neon::neon_dot_f32(r_im_j, r_im_k, n_samples);
                float ri = optmath::neon::neon_dot_f32(r_re_j, r_im_k, n_samples);
                float ir = optmath::neon::neon_dot_f32(r_im_j, r_re_k, n_samples);

                float real_part = rr + ii;
                float imag_part = ri - ir;

                R[j * num_taps + k] = Complex(real_part, imag_part);

                if (j != k) {
                    R[k * num_taps + j] = Complex(real_part, -imag_part);
                }
            }
            // Diagonal Loading
            R[j * num_taps + j] += diagonal_loading;
        }

        // 3. Compute Cross-correlation vector p (num_taps x 1)
        std::vector<Complex> p(num_taps, 0.0f);

        const float* s_re = surv_re.data();
        const float* s_im = surv_im.data();

        for (int j = 0; j < num_taps; ++j) {
            int offset_j = num_taps - 1 - j;
            const float* r_re_j = ref_re.data() + offset_j;
            const float* r_im_j = ref_im.data() + offset_j;

            // dot(conj(ref), surv)
            float rr = optmath::neon::neon_dot_f32(r_re_j, s_re, n_samples);
            float ii = optmath::neon::neon_dot_f32(r_im_j, s_im, n_samples);
            float ri = optmath::neon::neon_dot_f32(r_re_j, s_im, n_samples);
            float ir = optmath::neon::neon_dot_f32(r_im_j, s_re, n_samples);

            p[j] = Complex(rr + ii, ri - ir);
        }

        // 4. Solve R * w = p
        std::vector<Complex> w(num_taps);
        if (!LinearSolver::solve_cholesky(R, p, w, num_taps)) {
            std::fill(w.begin(), w.end(), 0.0f);
        }

        // 5. Compute Output
        // e[n] = surv[n] - y[n]
        // y[n] = sum_{k=0}^{P-1} w_k * ref[n-k]

        std::vector<float> w_re(num_taps);
        std::vector<float> w_im(num_taps);
        for(int k=0; k<num_taps; ++k) {
            w_re[k] = w[k].real();
            w_im[k] = w[k].imag();
        }

        // Reverse weights for dot product (convolution)
        std::reverse(w_re.begin(), w_re.end());
        std::reverse(w_im.begin(), w_im.end());

        for (int n = 0; n < n_samples; ++n) {
            const float* r_ptr_re = ref_re.data() + n;
            const float* r_ptr_im = ref_im.data() + n;

            float dot_wr_rr = optmath::neon::neon_dot_f32(w_re.data(), r_ptr_re, num_taps);
            float dot_wi_ri = optmath::neon::neon_dot_f32(w_im.data(), r_ptr_im, num_taps);
            float dot_wr_ri = optmath::neon::neon_dot_f32(w_re.data(), r_ptr_im, num_taps);
            float dot_wi_rr = optmath::neon::neon_dot_f32(w_im.data(), r_ptr_re, num_taps);

            float y_re = dot_wr_rr - dot_wi_ri;
            float y_im = dot_wr_ri + dot_wi_rr;

            out_err[n] = surv_in[n] - Complex(y_re, y_im);
        }

        // 6. Update History
        int start_copy = full_ref.size() - (num_taps - 1);
        for (int i = 0; i < num_taps - 1; ++i) {
            ref_history[i] = full_ref[start_copy + i];
        }
    }
};

extern "C" {
    void* eca_b_create(int taps) {
        return new ECABCanceller(taps);
    }

    void eca_b_destroy(void* ptr) {
        if (ptr) delete static_cast<ECABCanceller*>(ptr);
    }

    void eca_b_process(void* ptr, const float* ref_in, const float* surv_in, float* out_err, int n_samples) {
        if (!ptr) return;
        ECABCanceller* obj = static_cast<ECABCanceller*>(ptr);
        const Complex* c_ref = reinterpret_cast<const Complex*>(ref_in);
        const Complex* c_surv = reinterpret_cast<const Complex*>(surv_in);
        Complex* c_out = reinterpret_cast<Complex*>(out_err);
        obj->process(c_ref, c_surv, c_out, n_samples);
    }
}
