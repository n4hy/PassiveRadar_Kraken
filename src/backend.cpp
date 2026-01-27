#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

// Backend processing: CFAR, Fusion
// Simplified implementation

using Complex = std::complex<float>;

class Backend {
public:
    // Simple 1D CA-CFAR
    // input: log magnitude buffer (rows x cols)
    // output: detection mask (1.0 or 0.0)
    static void cfar_1d(const float* input, float* output, int rows, int cols, int guard, int train, float threshold) {
        // Iterate over each row (Range bins often, or Doppler bins)
        // Let's assume input is [Doppler x Range] or [Range x Doppler].
        // Usually CFAR is done in both dimensions or 2D.
        // Let's do 2D CFAR.

        for (int i = 0; i < rows * cols; ++i) output[i] = 0.0f;

        for (int r = guard + train; r < rows - (guard + train); ++r) {
            for (int c = guard + train; c < cols - (guard + train); ++c) {
                float sum = 0.0f;
                int count = 0;

                // Training cells
                for (int dr = -(train + guard); dr <= (train + guard); ++dr) {
                    for (int dc = -(train + guard); dc <= (train + guard); ++dc) {
                        if (std::abs(dr) <= guard && std::abs(dc) <= guard) continue; // Guard
                        sum += input[(r + dr) * cols + (c + dc)];
                        count++;
                    }
                }

                if (count == 0) continue; // Avoid division by zero
                float noise = sum / count;
                float cell = input[r * cols + c];

                // Input is Log Mag (dB). Threshold is dB.
                // If linear: cell > threshold * noise
                // If dB: cell > noise + threshold
                if (cell > noise + threshold) {
                    output[r * cols + c] = 1.0f;
                }
            }
        }
    }

    // Non-coherent Integration
    // Sums magnitude squared (or log mag?)
    // If inputs are log mag: convert to linear, sum, convert back?
    // Or just average dB (video integration).
    // Let's assume inputs are Log Mag (dB).
    // Summing dB is equivalent to multiplying in linear. Product?
    // Non-coherent integration usually sums power (Linear).
    static void fusion(const float* const* inputs, int num_inputs, float* output, int size) {
        std::vector<float> sum_linear(size, 0.0f);

        for (int k = 0; k < num_inputs; ++k) {
            const float* in = inputs[k];
            for (int i = 0; i < size; ++i) {
                // dB to Linear Power
                float db = in[i];
                float pwr = std::pow(10.0f, db / 10.0f);
                sum_linear[i] += pwr;
            }
        }

        for (int i = 0; i < size; ++i) {
            float val = 10.0f * std::log10(sum_linear[i] + 1e-12f);
            output[i] = val;
        }
    }
};

extern "C" {
    void cfar_2d(const float* input, float* output, int rows, int cols, int guard, int train, float threshold) {
        if (!input || !output || rows <= 0 || cols <= 0) return;
        Backend::cfar_1d(input, output, rows, cols, guard, train, threshold);
    }

    void fusion_process(const float** inputs, int num_inputs, float* output, int size) {
        if (!inputs || !output || num_inputs <= 0 || size <= 0) return;
        Backend::fusion(inputs, num_inputs, output, size);
    }
}
