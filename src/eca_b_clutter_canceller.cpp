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
    static bool solve_cholesky(const std::vector<Complex>& A,
                               const std::vector<Complex>& b,
                               std::vector<Complex>& x,
                               int N,
                               std::vector<Complex>& L,
                               std::vector<Complex>& y) {
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
    int   num_taps;
    float diagonal_loading;

    // NEW: allow cancellation at a non-zero lag between ref and surv.
    // We keep a long reference history (ring-buffer equivalent) so that
    // the FIR still has num_taps taps but is applied at ref index (n - delay).
    //
    // Interpretation:
    //   If surv[n] contains ref delayed by D samples (plus channel), then set delay=D.
    //
    int delay_samples;

    // We keep history_len = (num_taps - 1) + max_delay samples of ref history.
    // That gives us enough "past" to align the FIR at delay_samples without
    // increasing the FIR tap count.
    int max_delay;
    int history_len;

    // Persistent buffers to avoid re-allocation churn
    std::vector<Complex> ref_history;

    // Scratch buffers for process()
    std::vector<Complex> full_ref;
    std::vector<float>   ref_re;
    std::vector<float>   ref_im;
    std::vector<float>   surv_re;
    std::vector<float>   surv_im;
    std::vector<Complex> R;
    std::vector<Complex> p;
    std::vector<Complex> w;
    std::vector<float>   w_re;
    std::vector<float>   w_im;

    // Scratch buffers for LinearSolver
    std::vector<Complex> solver_L;
    std::vector<Complex> solver_y;

    // Inline Dot Product for max compiler optimization
    static FORCE_INLINE float dot_prod(const float* a, const float* b, int n) {
        float sum = 0.0f;
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
    explicit ECABCanceller(int taps, int max_delay_samples = 4096)
        : num_taps(taps),
          diagonal_loading(1e-6f),
          delay_samples(0),
          max_delay(std::max(0, max_delay_samples)),
          history_len((taps) + std::max(0, max_delay_samples)) { // history_len >= num_taps to avoid index -1

        // Enable FTZ/DAZ on x86
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        #endif

        // Long history = (num_taps + max_delay)
        if (history_len < num_taps) history_len = num_taps;
        ref_history.assign(history_len, Complex(0.0f, 0.0f));

        // Pre-allocate fixed size buffers
        R.resize(num_taps * num_taps);
        p.resize(num_taps);
        w.resize(num_taps);
        w_re.resize(num_taps);
        w_im.resize(num_taps);
        solver_L.resize(num_taps * num_taps);
        solver_y.resize(num_taps);

        // Reserve to reduce reallocs; will resize per-call.
        full_ref.reserve(65536);
    }

    void set_delay(int d) {
        if (d < 0) d = 0;
        if (d > max_delay) d = max_delay;
        delay_samples = d;
    }

    int get_delay() const { return delay_samples; }

    void process(const Complex* ref_in, const Complex* surv_in, Complex* out_err, int n_samples) {
        if (n_samples == 0) return;

        // Base index where surv[0] aligns in full_ref:
        //   base = history_len + delay_samples
        // In the original code, base=(num_taps-1) which was == history_len.
        // We updated history_len to be larger, so we must use history_len as the reference point for ref_in[0].
        const int base = history_len + delay_samples;

        // 1. Prepare Reference Data
        const int full_ref_len = history_len + n_samples;

        full_ref.resize(full_ref_len);

        // Copy history (long)
        std::copy(ref_history.begin(), ref_history.end(), full_ref.begin());

        // Copy new input after history
        std::copy(ref_in, ref_in + n_samples, full_ref.begin() + history_len);

        ref_re.resize(full_ref_len);
        ref_im.resize(full_ref_len);
        surv_re.resize(n_samples);
        surv_im.resize(n_samples);

        // SoA conversion loop
        for (int i = 0; i < full_ref_len; ++i) {
            ref_re[i] = full_ref[i].real();
            ref_im[i] = full_ref[i].imag();
        }
        for (int i = 0; i < n_samples; ++i) {
            surv_re[i] = surv_in[i].real();
            surv_im[i] = surv_in[i].imag();
        }

        // 2. Compute Autocorrelation Matrix R
        // We compute R over the aligned reference windows of length n_samples.
        // The j-th tap uses ref_re/im starting at (base - j), length n_samples.
        {
            int j = 0;
            int offset_j = base - j;
            const float* r_re_j = ref_re.data() + offset_j;
            const float* r_im_j = ref_im.data() + offset_j;

            for (int k = 0; k < num_taps; ++k) {
                int offset_k = base - k;
                const float* r_re_k = ref_re.data() + offset_k;
                const float* r_im_k = ref_im.data() + offset_k;

                float rr = dot_prod(r_re_j, r_re_k, n_samples);
                float ii = dot_prod(r_im_j, r_im_k, n_samples);
                float ri = dot_prod(r_re_j, r_im_k, n_samples);
                float ir = dot_prod(r_im_j, r_re_k, n_samples);

                float real_part = rr + ii;
                float imag_part = ri - ir;
                R[k] = Complex(real_part, imag_part); // R[0*M + k]
            }
        }

        // Step 2b: Update subsequent diagonals
        // R[j+1, k+1] = R[j, k] + u[-1-j]*conj(u[-1-k]) - u[N-1-j]*conj(u[N-1-k])
        // In our aligned indexing, the "window" spans indices:
        //   start = (base - (num_taps-1))  ...  start + (n_samples + num_taps - 2)
        // where start = delay_samples.
        for (int k_start = 0; k_start < num_taps; ++k_start) {
            int max_steps = num_taps - 1 - k_start;

            for (int i = 0; i < max_steps; ++i) {
                int j = i;
                int k = k_start + i;

                Complex curr = R[j * num_taps + k];

                // Indices in full_ref for the diagonal update.
                // The original code derived these indices assuming base=(num_taps-1).
                // Here, the aligned "window start" is start = (base - (num_taps-1)) = delay_samples.
                const int start = delay_samples;

                // u_prev_* corresponds to the sample just BEFORE the current window for that tap.
                // u_last_* corresponds to the sample just AFTER the current window for that tap.
                //
                // For tap j, its sequence begins at (base - j) = (start + (num_taps-1) - j).
                // The element before the window is at (base - j - 1) = (start + (num_taps-2) - j).
                // The last element in the window is at (base - j + (n_samples - 1)).
                // The element after the window is at (base - j + n_samples).
                int idx_prev_j = (base - 1) - j;          // base - 1 - j
                int idx_prev_k = (base - 1) - k;
                int idx_last_j = (base + n_samples - 1) - j;  // Remove term at n=N-1 (which was at end of previous window)
                int idx_last_k = (base + n_samples - 1) - k;

                // Bounds should be safe given history_len >= base and full_ref_len = history_len + n_samples.
                Complex u_prev_j = full_ref[idx_prev_j];
                Complex u_prev_k = full_ref[idx_prev_k];
                Complex u_last_j = full_ref[idx_last_j];
                Complex u_last_k = full_ref[idx_last_k];

                Complex next_val = curr + (u_prev_j * std::conj(u_prev_k)) - (u_last_j * std::conj(u_last_k));
                R[(j + 1) * num_taps + (k + 1)] = next_val;
            }
        }

        // Step 2c: Fill lower triangle using Hermitian symmetry + diagonal loading
        for (int j = 0; j < num_taps; ++j) {
            for (int k = 0; k < j; ++k) {
                R[j * num_taps + k] = std::conj(R[k * num_taps + j]);
            }
            R[j * num_taps + j] += diagonal_loading;
        }

        // 3. Compute Cross-correlation p
        const float* s_re = surv_re.data();
        const float* s_im = surv_im.data();

        for (int j = 0; j < num_taps; ++j) {
            int offset_j = base - j;
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
            // std::cerr << "ECA-B: Cholesky failed. R matrix might be non-positive-definite." << std::endl;
            std::fill(w.begin(), w.end(), 0.0f);
        }

        // 5. Compute Output
        for (int k = 0; k < num_taps; ++k) {
            w_re[k] = w[k].real();
            w_im[k] = w[k].imag();
        }
        std::reverse(w_re.begin(), w_re.end());
        std::reverse(w_im.begin(), w_im.end());

        // The correct input pointer shift for the FIR (to keep the "last sample" aligned at base+n):
        // We need r[n] (which is at ref_re[base + n]) to be at the END of the dot_prod buffer.
        // The buffer length is num_taps.
        // So buffer start should be (base + n) - (num_taps - 1).
        const int start = base - (num_taps - 1);

        for (int n = 0; n < n_samples; ++n) {
            const float* r_ptr_re = ref_re.data() + (start + n);
            const float* r_ptr_im = ref_im.data() + (start + n);

            float dot_wr_rr = dot_prod(w_re.data(), r_ptr_re, num_taps);
            float dot_wi_ri = dot_prod(w_im.data(), r_ptr_im, num_taps);
            float dot_wr_ri = dot_prod(w_re.data(), r_ptr_im, num_taps);
            float dot_wi_rr = dot_prod(w_im.data(), r_ptr_re, num_taps);

            float y_re = dot_wr_rr - dot_wi_ri;
            float y_im = dot_wr_ri + dot_wi_rr;

            out_err[n] = surv_in[n] - Complex(y_re, y_im);
        }

        // 6. Update History (keep last history_len samples of full_ref)
        std::copy(full_ref.end() - history_len, full_ref.end(), ref_history.begin());
    }
};

extern "C" {

void* eca_b_create(int taps) {
    // Keep ABI stable: default max_delay=4096.
    return new ECABCanceller(taps, 4096);
}

void eca_b_destroy(void* ptr) {
    if (ptr) delete static_cast<ECABCanceller*>(ptr);
}

void eca_b_process(void* ptr, const float* ref_in, const float* surv_in, float* out_err, int n_samples) {
    if (!ptr) return;
    ECABCanceller* obj = static_cast<ECABCanceller*>(ptr);
    const Complex* c_ref  = reinterpret_cast<const Complex*>(ref_in);
    const Complex* c_surv = reinterpret_cast<const Complex*>(surv_in);
    Complex*       c_out  = reinterpret_cast<Complex*>(out_err);
    obj->process(c_ref, c_surv, c_out, n_samples);
}

// NEW: optional setter for delay (does not break existing users of the C ABI).
void eca_b_set_delay(void* ptr, int delay_samples) {
    if (!ptr) return;
    ECABCanceller* obj = static_cast<ECABCanceller*>(ptr);
    obj->set_delay(delay_samples);
}

int eca_b_get_delay(void* ptr) {
    if (!ptr) return 0;
    ECABCanceller* obj = static_cast<ECABCanceller*>(ptr);
    return obj->get_delay();
}

} // extern "C"
