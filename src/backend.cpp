#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cstdint>

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#include <optmath/radar_kernels.hpp>
#else
#define HAVE_OPTMATHKERNELS 0
#endif

// Backend processing: CFAR, Fusion

using Complex = std::complex<float>;

/**
 * Backend - CFAR detection and non-coherent fusion for passive radar
 *
 * Technique: Provides 2D Cell-Averaging CFAR (CA-CFAR) detection using
 * prefix-sum acceleration and non-coherent integration (power summation)
 * across multiple surveillance channels.
 */
class Backend {
public:
    /**
     * cfar_1d - 2D Cell-Averaging CFAR detector on range-Doppler map
     *
     * Technique: Operates in dB domain with additive threshold. Uses a 2D
     * prefix sum (integral image) for O(1) rectangle queries, enabling
     * efficient noise estimation from the training cells surrounding each
     * cell-under-test. Guard cells prevent target self-masking. A cell is
     * declared a detection when it exceeds the local noise estimate plus
     * the threshold (in dB). Output is a binary detection mask.
     */
    static void cfar_1d(const float* input, float* output, int rows, int cols, int guard, int train, float threshold) {
        // Input is Log Mag (dB), threshold is additive dB.
        // cfar_2d_f32 uses multiplicative threshold on linear data, so we use
        // a 2D sliding-window approach optimized with row/column prefix sums.
        for (int i = 0; i < rows * cols; ++i) output[i] = 0.0f;

        const int window = guard + train;
        const int outer = 2 * window + 1;
        const int inner = 2 * guard + 1;
        const int n_train = outer * outer - inner * inner;
        if (n_train <= 0) return;

        // Compute 2D prefix sum for O(1) rectangle queries
        const int R = rows, C = cols;
        std::vector<double> psum((R + 1) * (C + 1), 0.0);
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                psum[(r + 1) * (C + 1) + (c + 1)] =
                    input[r * C + c]
                    + psum[r * (C + 1) + (c + 1)]
                    + psum[(r + 1) * (C + 1) + c]
                    - psum[r * (C + 1) + c];
            }
        }

        // Lambda for rectangle sum [r1..r2, c1..c2] inclusive
        auto rect_sum = [&](int r1, int c1, int r2, int c2) -> double {
            return psum[(r2 + 1) * (C + 1) + (c2 + 1)]
                 - psum[r1 * (C + 1) + (c2 + 1)]
                 - psum[(r2 + 1) * (C + 1) + c1]
                 + psum[r1 * (C + 1) + c1];
        };

        for (int r = window; r < R - window; ++r) {
            for (int c = window; c < C - window; ++c) {
                double outer_sum = rect_sum(r - window, c - window, r + window, c + window);
                double inner_sum = rect_sum(r - guard, c - guard, r + guard, c + guard);
                float noise = static_cast<float>((outer_sum - inner_sum) / n_train);
                float cell = input[r * C + c];

                // dB domain: additive threshold
                if (cell > noise + threshold) {
                    output[r * C + c] = 1.0f;
                }
            }
        }
    }

    /**
     * fusion - Non-coherent integration across multiple surveillance channels
     *
     * Technique: Converts dB-domain inputs to linear power via exp(dB * ln10/10),
     * sums linear power across all channels, then converts back to dB.
     * Optionally uses NEON-optimized batch exponential (neon_fast_exp_f32)
     * via OptMathKernels for ARM acceleration. This implements standard
     * non-coherent integration (power summation) for improved detection SNR.
     */
    static void fusion(const float* const* inputs, int num_inputs, float* output, int size) {
        std::vector<float> sum_linear(size, 0.0f);

        // log(10)/10 precomputed for dB-to-linear conversion: 10^(db/10) = exp(db * ln10/10)
        constexpr float db_to_ln = 0.23025850929940458f; // ln(10)/10

#if HAVE_OPTMATHKERNELS
        std::vector<float> scaled(size);
        std::vector<float> exp_out(size);
        for (int k = 0; k < num_inputs; ++k) {
            const float* in = inputs[k];
            // Scale input: in[i] * db_to_ln
            for (int i = 0; i < size; ++i) {
                scaled[i] = in[i] * db_to_ln;
            }
            // Batch exp using NEON approximation
            optmath::neon::neon_fast_exp_f32(exp_out.data(), scaled.data(), size);
            for (int i = 0; i < size; ++i) {
                sum_linear[i] += exp_out[i];
            }
        }
#else
        for (int k = 0; k < num_inputs; ++k) {
            const float* in = inputs[k];
            for (int i = 0; i < size; ++i) {
                // dB to Linear Power using expf (much faster than pow on ARM)
                sum_linear[i] += expf(in[i] * db_to_ln);
            }
        }
#endif

        for (int i = 0; i < size; ++i) {
            float val = 10.0f * std::log10(sum_linear[i] + 1e-12f);
            output[i] = val;
        }
    }
};

extern "C" {
    /**
     * cfar_2d - C API wrapper for 2D CFAR detection
     *
     * Technique: Delegates to Backend::cfar_1d which implements prefix-sum
     * accelerated 2D CA-CFAR on a range-Doppler map.
     */
    void cfar_2d(const float* input, float* output, int rows, int cols, int guard, int train, float threshold) {
        if (!input || !output || rows <= 0 || cols <= 0) return;
        Backend::cfar_1d(input, output, rows, cols, guard, train, threshold);
    }

    /**
     * fusion_process - C API wrapper for non-coherent integration
     *
     * Technique: Delegates to Backend::fusion for multi-channel power summation.
     */
    void fusion_process(const float** inputs, int num_inputs, float* output, int size) {
        if (!inputs || !output || num_inputs <= 0 || size <= 0) return;
        Backend::fusion(inputs, num_inputs, output, size);
    }
}
