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

    // Process logic shared by output modes
    // returns windowed, FFT'd, Shifted column
    void process_column(const Complex* input, int r, std::vector<Complex>& result) {
        result.resize(doppler_len);

        // 1. Extract and Window
        for (int d = 0; d < doppler_len; ++d) {
            result[d] = input[d * fft_len + r] * window[d];
        }

        // 2. FFT
        fft_radix2(result, false);

        // 3. FFT Shift (in place swap)
        // result buffer is modified to be shifted
        // Using rotate or manual swap
        int half = doppler_len / 2;
        // Simple manual swap for first half vs second half
        std::vector<Complex> temp = result;
        for (int d = 0; d < half; ++d) {
            result[d] = temp[d + half];
            result[d + half] = temp[d];
        }
    }

    // Input: flatten array of complex floats. Size: doppler_len * fft_len
    // Output: flatten array of floats (Mag Log). Size: doppler_len * fft_len
    void process(const Complex* input, float* output) {
        std::vector<Complex> shifted_col;
        shifted_col.reserve(doppler_len);

        for (int r = 0; r < fft_len; ++r) {
            process_column(input, r, shifted_col);

            // Log Mag
            for (int d = 0; d < doppler_len; ++d) {
                float mag_sq = std::norm(shifted_col[d]);
                float val_db = 10.0f * std::log10(mag_sq + 1e-12f);
                output[d * fft_len + r] = val_db;
            }
        }
    }

    // New: Output raw Complex data (for AoA)
    // Output must be pre-allocated: doppler_len * fft_len * sizeof(complex)
    void process_complex(const Complex* input, Complex* output) {
        std::vector<Complex> shifted_col;
        shifted_col.reserve(doppler_len);

        for (int r = 0; r < fft_len; ++r) {
            process_column(input, r, shifted_col);

            // Copy complex
            for (int d = 0; d < doppler_len; ++d) {
                output[d * fft_len + r] = shifted_col[d];
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
        const Complex* c_input = reinterpret_cast<const Complex*>(input);
        obj->process(c_input, output);
    }

    void doppler_process_complex(void* ptr, const float* input, float* output) {
        if (!ptr) return;
        DopplerProcessor* obj = static_cast<DopplerProcessor*>(ptr);
        const Complex* c_input = reinterpret_cast<const Complex*>(input);
        Complex* c_output = reinterpret_cast<Complex*>(output);
        obj->process_complex(c_input, c_output);
    }
}
