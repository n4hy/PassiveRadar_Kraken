#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>

// Constants
const float PI = 3.14159265358979323846f;

using Complex = std::complex<float>;

// Simple in-place Cooley-Tukey FFT (Radix-2, Decimation in Time)
// n must be a power of 2
void fft_radix2(std::vector<Complex>& a, bool invert) {
    size_t n = a.size();
    if (n <= 1) return;

    // Bit-reversal permutation
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    // Butterfly updates
    for (size_t len = 2; len <= n; len <<= 1) {
        float angle = 2 * PI / len * (invert ? -1 : 1);
        Complex wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                Complex u = a[i + k];
                Complex v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (Complex& x : a) {
            x /= static_cast<float>(n);
        }
    }
}

class DopplerProcessor {
private:
    int fft_len;      // Range bins (Fast-time) - Columns
    int doppler_len;  // Doppler bins (Slow-time) - Rows

    std::vector<float> window;
    // We process column by column?
    // Input is usually (doppler_len, fft_len) i.e. (Slow, Fast)
    // We want to FFT along Slow axis (axis 0 in numpy terms)
    // So we treat each range bin (column) as an independent signal of length doppler_len.

public:
    DopplerProcessor(int n_fft, int n_doppler) : fft_len(n_fft), doppler_len(n_doppler) {
        // Pre-calculate Hamming window
        window.resize(doppler_len);
        if (doppler_len > 1) {
            for (int i = 0; i < doppler_len; ++i) {
                window[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (doppler_len - 1));
            }
        } else {
            window[0] = 1.0f;
        }
    }

    // Input: flatten array of complex floats. Size: doppler_len * fft_len
    // Layout: Row-major: [Slow0_Fast0, Slow0_Fast1, ... Slow1_Fast0...]
    // Effectively: matrix[doppler_index][range_index]
    //
    // Output: flatten array of floats (Mag Log). Size: doppler_len * fft_len
    // Output Layout: [Doppler0_Range0, Doppler0_Range1, ...]
    void process(const Complex* input, float* output) {
        // We need to iterate over each Range Bin (column)
        // Extract the column, window it, FFT it, Shift it, MagLog it, Write back.

        // Since input is Row-Major (Slow-Time is rows), a column is strided by fft_len.

        // Optimize: To avoid tons of allocations, use a thread-local or member buffer?
        // Since we are likely single-threaded here, a member buffer is fine, but re-entrant is better.
        // Let's allocate a temporary buffer for one column.
        std::vector<Complex> column(doppler_len);

        for (int r = 0; r < fft_len; ++r) {
            // 1. Extract and Window Column 'r'
            for (int d = 0; d < doppler_len; ++d) {
                // Input index: d * fft_len + r
                column[d] = input[d * fft_len + r] * window[d];
            }

            // 2. FFT
            fft_radix2(column, false);

            // 3. FFT Shift, MagSq, Log, Write to Output
            // FFT Shift for 1D array of even length swaps first half and second half.
            // If doppler_len is 128: 0..63 goes to 64..127, 64..127 goes to 0..63.

            int half = doppler_len / 2;

            // Output layout is also (Doppler, Range) -> Row Major
            // We write column 'r' into the output matrix.

            // Write second half of FFT to first half of Output
            for (int d = 0; d < half; ++d) {
                int src_idx = d + half;
                int dst_row = d;

                float mag_sq = std::norm(column[src_idx]);
                float val_db = 10.0f * std::log10(mag_sq + 1e-12f);

                output[dst_row * fft_len + r] = val_db;
            }

            // Write first half of FFT to second half of Output
            for (int d = 0; d < half; ++d) {
                int src_idx = d;
                int dst_row = d + half;

                float mag_sq = std::norm(column[src_idx]);
                float val_db = 10.0f * std::log10(mag_sq + 1e-12f);

                output[dst_row * fft_len + r] = val_db;
            }

            // Handle odd length if necessary (unlikely for doppler_len usually power of 2)
            if (doppler_len % 2 != 0) {
                // If odd, fftshift usually shifts (N-1)/2.
                // Numpy: indices [0, 1, 2] -> [2, 0, 1] for N=3?
                // np.fft.fftshift([0,1,2]) -> [2, 0, 1].
                // Let's assume even for now (128).
            }
        }
    }
};

extern "C" {
    void* doppler_create(int fft_len, int doppler_len) {
        return new DopplerProcessor(fft_len, doppler_len);
    }

    void doppler_destroy(void* ptr) {
        if (ptr) delete static_cast<DopplerProcessor*>(ptr);
    }

    void doppler_process(void* ptr, const float* input, float* output) {
        if (!ptr) return;
        DopplerProcessor* obj = static_cast<DopplerProcessor*>(ptr);
        // Cast input to Complex* (interleaved float I/Q)
        const Complex* c_input = reinterpret_cast<const Complex*>(input);
        obj->process(c_input, output);
    }
}
