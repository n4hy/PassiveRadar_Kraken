#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

#if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
#endif

using Complex = std::complex<float>;

// Helper for solving Ax = b using Cholesky Decomposition
class LinearSolver {
public:
    static bool solve_cholesky(const std::vector<Complex>& A, const std::vector<Complex>& b, std::vector<Complex>& x, int N, std::vector<Complex>& L, std::vector<Complex>& y) {
        // Reuse L and y vectors passed from caller to avoid allocation
        L.assign(N * N, 0.0f);
        y.resize(N);

        // 1. Cholesky Decomposition: A = L * L^H
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                Complex sum = 0.0f;
                for (int k = 0; k < j; ++k) {
                    sum += L[i * N + k] * std::conj(L[j * N + k]);
                }

                if (i == j) {
                    float val = std::real(A[i * N + i] - sum);
                    if (val <= 0.0f) return false;
                    L[i * N + j] = std::sqrt(val);
                } else {
                    L[i * N + j] = (A[i * N + j] - sum) / L[j * N + j];
                }
            }
        }

        // 2. Forward Substitution: L * y = b
        for (int i = 0; i < N; ++i) {
            Complex sum = 0.0f;
            for (int j = 0; j < i; ++j) {
                sum += L[i * N + j] * y[j];
            }
            y[i] = (b[i] - sum) / L[i * N + i];
        }

        // 3. Backward Substitution: L^H * x = y
        for (int i = N - 1; i >= 0; --i) {
            Complex sum = 0.0f;
            for (int j = i + 1; j < N; ++j) {
                sum += std::conj(L[j * N + i]) * x[j];
            }
            x[i] = (y[i] - sum) / std::conj(L[i * N + i]);
        }

        return true;
    }
};

class ECABCanceller {
private:
    int num_taps;
    float diagonal_loading;

    // Persistent buffers to avoid re-allocation churn
    std::vector<Complex> ref_history;

    // Scratch buffers for process()
    std::vector<Complex> full_ref;
    std::vector<float> ref_re;
    std::vector<float> ref_im;
    std::vector<float> surv_re;
    std::vector<float> surv_im;
    std::vector<Complex> R;
    std::vector<Complex> p;
    std::vector<Complex> w;
    std::vector<float> w_re;
    std::vector<float> w_im;

    // Scratch buffers for LinearSolver
    std::vector<Complex> solver_L;
    std::vector<Complex> solver_y;

    // Inline Dot Product for max compiler optimization
    static FORCE_INLINE float dot_prod(const float* a, const float* b, int n) {
        float sum = 0.0f;
        // The compiler auto-vectorizer does a great job here with -O3 -ffast-math
        // We unroll slightly to encourage it
        int i = 0;
        for (; i <= n - 8; i += 8) {
            sum += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3] +
                   a[i+4] * b[i+4] + a[i+5] * b[i+5] + a[i+6] * b[i+6] + a[i+7] * b[i+7];
        }
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

public:
    ECABCanceller(int taps) : num_taps(taps), diagonal_loading(1e-6f) {
        // Enable FTZ/DAZ on x86
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        #endif

        ref_history.resize(num_taps - 1, Complex(0.0f, 0.0f));

        // Pre-allocate fixed size buffers
        R.resize(num_taps * num_taps);
        p.resize(num_taps);
        w.resize(num_taps);
        w_re.resize(num_taps);
        w_im.resize(num_taps);
        solver_L.resize(num_taps * num_taps);
        solver_y.resize(num_taps);
        full_ref.reserve(65536); // Reserve capacity
    }

    void process(const Complex* ref_in, const Complex* surv_in, Complex* out_err, int n_samples) {
        if (n_samples == 0) return;

        // 1. Prepare Reference Data
        // Resize scratch buffers to match current n_samples if needed
        int full_ref_len = ref_history.size() + n_samples;

        // Use assign/resize/copy to reuse memory
        full_ref.resize(full_ref_len);

        // Copy history
        std::copy(ref_history.begin(), ref_history.end(), full_ref.begin());
        // Copy new input
        std::copy(ref_in, ref_in + n_samples, full_ref.begin() + ref_history.size());

        ref_re.resize(full_ref_len);
        ref_im.resize(full_ref_len);
        surv_re.resize(n_samples);
        surv_im.resize(n_samples);

        // SoA conversion loop
        for(int i=0; i<full_ref_len; ++i) {
            ref_re[i] = full_ref[i].real();
            ref_im[i] = full_ref[i].imag();
        }
        for(int i=0; i<n_samples; ++i) {
            surv_re[i] = surv_in[i].real();
            surv_im[i] = surv_in[i].imag();
        }

        // 2. Compute Autocorrelation Matrix R
        // Optimized O(M*N + M^2) implementation exploiting Toeplitz-like structure

        // Step 2a: Compute first row (j=0, k=0..M-1)
        {
            int j = 0;
            int offset_j = num_taps - 1 - j;
            const float* r_re_j = ref_re.data() + offset_j;
            const float* r_im_j = ref_im.data() + offset_j;

            for (int k = 0; k < num_taps; ++k) {
                int offset_k = num_taps - 1 - k;
                const float* r_re_k = ref_re.data() + offset_k;
                const float* r_im_k = ref_im.data() + offset_k;

                float rr = dot_prod(r_re_j, r_re_k, n_samples);
                float ii = dot_prod(r_im_j, r_im_k, n_samples);
                float ri = dot_prod(r_re_j, r_im_k, n_samples);
                float ir = dot_prod(r_im_j, r_re_k, n_samples);

                // R[0, k] = dot(r_0, r_k)
                float real_part = rr + ii;
                float imag_part = ri - ir;
                R[k] = Complex(real_part, imag_part); // R[0*M + k]
            }
        }

        // Step 2b: Update subsequent diagonals
        // R[j+1, k+1] = R[j, k] + u[-1-j]*conj(u[-1-k]) - u[N-1-j]*conj(u[N-1-k])
        // Iterate for each diagonal starting at Row 0
        for (int k_start = 0; k_start < num_taps; ++k_start) {
            // Propagate along diagonal starting at R[0, k_start]
            // We need to compute R[1, k_start+1], R[2, k_start+2], ...
            // Max index is num_taps-1

            // Loop runs as long as j+1 < num_taps and k+1 < num_taps
            // Since we start at j=0, k=k_start, loop limit is determined by k_start
            int max_steps = num_taps - 1 - k_start;

            for (int i = 0; i < max_steps; ++i) {
                int j = i;       // Current row
                int k = k_start + i; // Current col

                // Current Value R[j, k]
                Complex curr = R[j * num_taps + k];

                // Calculate update terms
                // u[-1-j] -> full_ref[num_taps - 1 - 1 - j]
                // u[N-1-j] -> full_ref[num_taps - 1 + n_samples - 1 - j]

                int idx_prev_j = num_taps - 2 - j;
                int idx_prev_k = num_taps - 2 - k;
                int idx_last_j = num_taps + n_samples - 2 - j;
                int idx_last_k = num_taps + n_samples - 2 - k;

                // These indices are guaranteed to be >= 0 because j < num_taps-1
                // And we have sufficient history (num_taps-1) in full_ref

                Complex u_prev_j = full_ref[idx_prev_j];
                Complex u_prev_k = full_ref[idx_prev_k];
                Complex u_last_j = full_ref[idx_last_j];
                Complex u_last_k = full_ref[idx_last_k];

                Complex next_val = curr + (u_prev_j * std::conj(u_prev_k)) - (u_last_j * std::conj(u_last_k));

                // Store R[j+1, k+1]
                R[(j + 1) * num_taps + (k + 1)] = next_val;
            }
        }

        // Step 2c: Fill lower triangle using Hermitian symmetry
        for (int j = 0; j < num_taps; ++j) {
            for (int k = 0; k < j; ++k) {
                R[j * num_taps + k] = std::conj(R[k * num_taps + j]);
            }
            // Add diagonal loading
            R[j * num_taps + j] += diagonal_loading;
        }

        // 3. Compute Cross-correlation p
        const float* s_re = surv_re.data();
        const float* s_im = surv_im.data();

        for (int j = 0; j < num_taps; ++j) {
            int offset_j = num_taps - 1 - j;
            const float* r_re_j = ref_re.data() + offset_j;
            const float* r_im_j = ref_im.data() + offset_j;

            float rr = dot_prod(r_re_j, s_re, n_samples);
            float ii = dot_prod(r_im_j, s_im, n_samples);
            float ri = dot_prod(r_re_j, s_im, n_samples);
            float ir = dot_prod(r_im_j, s_re, n_samples);

            p[j] = Complex(rr + ii, ri - ir);
        }

        // 4. Solve R * w = p
        if (!LinearSolver::solve_cholesky(R, p, w, num_taps, solver_L, solver_y)) {
            std::fill(w.begin(), w.end(), 0.0f);
        }

        // 5. Compute Output
        for(int k=0; k<num_taps; ++k) {
            w_re[k] = w[k].real();
            w_im[k] = w[k].imag();
        }
        std::reverse(w_re.begin(), w_re.end());
        std::reverse(w_im.begin(), w_im.end());

        for (int n = 0; n < n_samples; ++n) {
            const float* r_ptr_re = ref_re.data() + n;
            const float* r_ptr_im = ref_im.data() + n;

            float dot_wr_rr = dot_prod(w_re.data(), r_ptr_re, num_taps);
            float dot_wi_ri = dot_prod(w_im.data(), r_ptr_im, num_taps);
            float dot_wr_ri = dot_prod(w_re.data(), r_ptr_im, num_taps);
            float dot_wi_rr = dot_prod(w_im.data(), r_ptr_re, num_taps);

            float y_re = dot_wr_rr - dot_wi_ri;
            float y_im = dot_wr_ri + dot_wi_rr;

            out_err[n] = surv_in[n] - Complex(y_re, y_im);
        }

        // 6. Update History
        int start_copy = full_ref.size() - (num_taps - 1);
        // Use copy instead of manual loop for speed
        std::copy(full_ref.begin() + start_copy, full_ref.end(), ref_history.begin());
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
