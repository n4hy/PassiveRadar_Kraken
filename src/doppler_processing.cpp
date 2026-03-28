#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fftw3.h>
#include <mutex>
#include <stdexcept>

// Centralized FFTW init (shared across all .so files in this project)
#include "fftw_init.h"

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#include <optmath/radar_kernels.hpp>
#else
#define HAVE_OPTMATHKERNELS 0
#endif

// Use standard M_PI (POSIX) or define it
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static constexpr float PI = static_cast<float>(M_PI);

using Complex = std::complex<float>;

class DopplerProcessor {
private:
    int fft_len;      // Range bins (Fast-time) - Columns
    int doppler_len;  // Doppler bins (Slow-time) - Rows

    std::vector<float> window;

    // FFTW Resources
    fftwf_plan plan;
    fftwf_complex* in_buf;
    fftwf_complex* out_buf;

public:
    // Non-copyable, non-movable (owns FFTW resources)
    DopplerProcessor(const DopplerProcessor&) = delete;
    DopplerProcessor& operator=(const DopplerProcessor&) = delete;
    DopplerProcessor(DopplerProcessor&&) = delete;
    DopplerProcessor& operator=(DopplerProcessor&&) = delete;

    DopplerProcessor(int n_fft, int n_doppler) : fft_len(n_fft), doppler_len(n_doppler) {
        // Initialize FFTW thread support (centralized, safe to call multiple times)
        kraken_fftw_init();

        // Pre-calculate Hamming window
        window.resize(doppler_len);
#if HAVE_OPTMATHKERNELS
        if (doppler_len > 1) {
            optmath::radar::generate_window_f32(window.data(), doppler_len,
                                                 optmath::radar::WindowType::HAMMING);
        } else {
            window[0] = 1.0f;
        }
#else
        if (doppler_len > 1) {
            for (int i = 0; i < doppler_len; ++i) {
                window[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (doppler_len - 1));
            }
        } else {
            window[0] = 1.0f;
        }
#endif

        // Initialize FFTW
        // We will process one column at a time
        in_buf = fftwf_alloc_complex(doppler_len);
        out_buf = fftwf_alloc_complex(doppler_len);
        if (!in_buf || !out_buf) {
            if (in_buf) fftwf_free(in_buf);
            if (out_buf) fftwf_free(out_buf);
            throw std::runtime_error("DopplerProcessor: FFTW buffer allocation failed");
        }

        // Create plan
        // FFTW_ESTIMATE is faster to plan, FFTW_MEASURE provides faster execution but slower startup
        plan = fftwf_plan_dft_1d(doppler_len, in_buf, out_buf, FFTW_FORWARD, FFTW_ESTIMATE);
        if (!plan) {
            fftwf_free(in_buf);
            fftwf_free(out_buf);
            throw std::runtime_error("DopplerProcessor: FFTW plan creation failed");
        }
    }

    ~DopplerProcessor() {
        fftwf_destroy_plan(plan);
        fftwf_free(in_buf);
        fftwf_free(out_buf);
    }

    // Process logic shared by output modes
    // Fills 'result' with FFT'd and Shifted data for a column
    void process_column(const Complex* input, int r, std::vector<Complex>& result) {
        result.resize(doppler_len);

        // 1. Extract column and apply window -> Copy to FFTW input
        for (int d = 0; d < doppler_len; ++d) {
            in_buf[d][0] = input[d * fft_len + r].real();
            in_buf[d][1] = input[d * fft_len + r].imag();
        }
        // Apply window function to extracted column
#if HAVE_OPTMATHKERNELS
        {
            // in_buf is fftwf_complex (interleaved float[2]), treat as separate re/im
            // Extract re/im, apply window, put back
            std::vector<float> col_re(doppler_len), col_im(doppler_len);
            for (int d = 0; d < doppler_len; ++d) {
                col_re[d] = in_buf[d][0];
                col_im[d] = in_buf[d][1];
            }
            optmath::radar::apply_window_complex_f32(col_re.data(), col_im.data(), window.data(), doppler_len);
            for (int d = 0; d < doppler_len; ++d) {
                in_buf[d][0] = col_re[d];
                in_buf[d][1] = col_im[d];
            }
        }
#else
        for (int d = 0; d < doppler_len; ++d) {
            in_buf[d][0] *= window[d];
            in_buf[d][1] *= window[d];
        }
#endif

        // 2. FFT
        fftwf_execute(plan);

        // 3. FFT Shift: swap halves to center DC

        // If even: 0..half-1 -> half..end; half..end -> 0..half-1
        // If odd (N=3): half=1. 0(1) -> 2; 1(2) -> 0; 2(3) -> 1?
        // Standard fftshift: center is at floor(N/2).
        // 0..ceil(N/2)-1 goes to end.
        // For N=3: [0, 1, 2] -> [2, 0, 1]. Element 0 goes to 1. Element 1 goes to 2. Element 2 goes to 0.
        // Let's stick to even/simple logic or use a helper.
        // Using (i + half) % N for destination index works for even N.

        // FFT shift: swap halves to center DC (NumPy fftshift convention)
        // For N elements, second half starts at index (N+1)/2
        const int shift = (doppler_len + 1) / 2;
        // First half of output: read from [shift..doppler_len-1]
        for (int i = 0; i < doppler_len - shift; ++i) {
            result[i] = Complex(out_buf[shift + i][0], out_buf[shift + i][1]);
        }
        // Second half of output: read from [0..shift-1]
        for (int i = 0; i < shift; ++i) {
            result[doppler_len - shift + i] = Complex(out_buf[i][0], out_buf[i][1]);
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
        if (fft_len <= 0 || doppler_len <= 0) return nullptr;
        return new DopplerProcessor(fft_len, doppler_len);
    }

    void doppler_destroy(void* ptr) {
        if (ptr) delete static_cast<DopplerProcessor*>(ptr);
    }

    void doppler_process(void* ptr, const float* input, float* output) {
        if (!ptr || !input || !output) return;
        DopplerProcessor* obj = static_cast<DopplerProcessor*>(ptr);
        const Complex* c_input = reinterpret_cast<const Complex*>(input);
        obj->process(c_input, output);
    }

    void doppler_process_complex(void* ptr, const float* input, float* output) {
        if (!ptr || !input || !output) return;
        DopplerProcessor* obj = static_cast<DopplerProcessor*>(ptr);
        const Complex* c_input = reinterpret_cast<const Complex*>(input);
        Complex* c_output = reinterpret_cast<Complex*>(output);
        obj->process_complex(c_input, c_output);
    }
}
