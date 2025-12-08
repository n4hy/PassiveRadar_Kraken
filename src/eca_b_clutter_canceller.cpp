#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>

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

    // Buffers to hold a "batch" of history if needed,
    // but typically ECA-B works on the current block + overlap.
    // For simplicity in streaming, we will construct the matrix from the available n_samples.
    // However, to properly cancel the beginning of the block, we need history of the Reference signal.

    std::vector<Complex> ref_history;

public:
    ECABCanceller(int taps) : num_taps(taps), diagonal_loading(1e-6f) {
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
        // We need extended reference to form the columns of X.
        // X has dimensions (n_samples) x (num_taps).
        // Row i of X: [r[i], r[i-1], ..., r[i-taps+1]]
        // So we need history of length `num_taps - 1`.

        std::vector<Complex> full_ref;
        full_ref.reserve(ref_history.size() + n_samples);
        full_ref.insert(full_ref.end(), ref_history.begin(), ref_history.end());
        full_ref.insert(full_ref.end(), ref_in, ref_in + n_samples);

        // 2. Compute Autocorrelation Matrix R (num_taps x num_taps)
        // R = X^H * X
        // R_jk = sum_{n=0}^{N-1} conj(X_{nj}) * X_{nk}
        //      = sum_{n=0}^{N-1} conj(full_ref[n + (taps-1) - j]) * full_ref[n + (taps-1) - k]
        // Let's index full_ref such that `current input n` corresponds to index `n + num_taps - 1`.
        // x[n] vector is `full_ref[n ... n+taps-1]` reversed?
        // Standard definition: clutter[n] = sum_{k=0}^{P-1} w_k * ref[n-k].
        // So row n of X contains: ref[n], ref[n-1], ... ref[n-P+1].
        // In `full_ref`, ref[n] is at index `n + num_taps - 1`.
        // So ref[n-k] is at index `n + num_taps - 1 - k`.

        // R is symmetric (Hermitian). We only calculate upper triangle.
        std::vector<Complex> R(num_taps * num_taps, 0.0f);

        // Optimization: This is O(N * P^2). For N=4096, P=64, ~16M ops. Feasible.
        for (int j = 0; j < num_taps; ++j) {
            for (int k = j; k < num_taps; ++k) { // Upper triangle
                Complex sum = 0.0f;
                for (int n = 0; n < n_samples; ++n) {
                    // X_{nj} = ref[n-j] -> full_ref[n + num_taps - 1 - j]
                    Complex val_j = full_ref[n + num_taps - 1 - j];
                    Complex val_k = full_ref[n + num_taps - 1 - k];

                    // R_jk = (X^H X)_jk = sum_n conj(X_nj) * X_nk ?
                    // Actually X^H X = sum_n x_n * x_n^H where x_n is column vector (row of X transposed).
                    // Element (j,k) of R = sum_n conj(X_{nj}) * X_{nk}
                    sum += std::conj(val_j) * val_k;
                }
                R[j * num_taps + k] = sum;
                if (j != k) {
                    R[k * num_taps + j] = std::conj(sum);
                }
            }
            // Diagonal Loading
            R[j * num_taps + j] += diagonal_loading;
        }

        // 3. Compute Cross-correlation vector p (num_taps x 1)
        // p = X^H * s
        // p_j = sum_{n=0}^{N-1} conj(X_{nj}) * s[n]
        std::vector<Complex> p(num_taps, 0.0f);
        for (int j = 0; j < num_taps; ++j) {
            Complex sum = 0.0f;
            for (int n = 0; n < n_samples; ++n) {
                Complex val_j = full_ref[n + num_taps - 1 - j];
                sum += std::conj(val_j) * surv_in[n];
            }
            p[j] = sum;
        }

        // 4. Solve R * w = p
        std::vector<Complex> w(num_taps);
        if (!LinearSolver::solve_cholesky(R, p, w, num_taps)) {
            // Fallback: If failed (e.g. singular), just zero weights or use previous?
            // For now, zero output (pass through signal?)
            // Actually if w=0, clutter estimate is 0, so e = surv.
            // This is "fail-safe" (no cancellation).
            std::fill(w.begin(), w.end(), 0.0f);
        }

        // 5. Compute Output
        // e[n] = surv[n] - y[n]
        // y[n] = sum_{k=0}^{P-1} w_k * ref[n-k]
        for (int n = 0; n < n_samples; ++n) {
            Complex clutter_est = 0.0f;
            for (int k = 0; k < num_taps; ++k) {
                clutter_est += w[k] * full_ref[n + num_taps - 1 - k];
            }
            out_err[n] = surv_in[n] - clutter_est;
        }

        // 6. Update History
        // Save the last `num_taps - 1` samples of full_ref
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
