#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fftw3.h>
#include <mutex>

// FFTW thread initialization (call once per process)
namespace {
    std::once_flag fftw_init_flag;
    void init_fftw_threads() {
        std::call_once(fftw_init_flag, []() {
            fftwf_init_threads();
            // Use 1 thread per plan by default; can be increased for large FFTs
            fftwf_plan_with_nthreads(1);
        });
    }
}

// Constants
const float PI = 3.14159265358979323846f;

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
    DopplerProcessor(int n_fft, int n_doppler) : fft_len(n_fft), doppler_len(n_doppler) {
        // Initialize FFTW thread support (safe to call multiple times)
        init_fftw_threads();

        // Pre-calculate Hamming window
        window.resize(doppler_len);
        if (doppler_len > 1) {
            for (int i = 0; i < doppler_len; ++i) {
                window[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (doppler_len - 1));
            }
        } else {
            window[0] = 1.0f;
        }

        // Initialize FFTW
        // We will process one column at a time
        in_buf = fftwf_alloc_complex(doppler_len);
        out_buf = fftwf_alloc_complex(doppler_len);

        // Create plan
        // FFTW_ESTIMATE is faster to plan, FFTW_MEASURE provides faster execution but slower startup
        plan = fftwf_plan_dft_1d(doppler_len, in_buf, out_buf, FFTW_FORWARD, FFTW_ESTIMATE);
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

        // 1. Extract and Window -> Copy to FFTW input
        for (int d = 0; d < doppler_len; ++d) {
            Complex val = input[d * fft_len + r] * window[d];
            in_buf[d][0] = val.real();
            in_buf[d][1] = val.imag();
        }

        // 2. FFT
        fftwf_execute(plan);

        // 3. FFT Shift and Copy to result
        // Shift: Swap first half and second half
        int half = doppler_len / 2;
        int odd = doppler_len % 2; // Usually 0 for powers of 2, but FFTW handles any size

        // If even: 0..half-1 -> half..end; half..end -> 0..half-1
        // If odd (N=3): half=1. 0(1) -> 2; 1(2) -> 0; 2(3) -> 1?
        // Standard fftshift: center is at floor(N/2).
        // 0..ceil(N/2)-1 goes to end.
        // For N=3: [0, 1, 2] -> [2, 0, 1]. Element 0 goes to 1. Element 1 goes to 2. Element 2 goes to 0.
        // Let's stick to even/simple logic or use a helper.
        // Using (i + half) % N for destination index works for even N.

        // Copy out_buf to result with shift
        for (int i = 0; i < doppler_len; ++i) {
            int dest_idx = (i + (doppler_len + 1) / 2) % doppler_len; // General shift formula?
            // NumPy: [0, 1, 2] -> [2, 0, 1].
            // i=0 -> dest=2. (0 + 2)%3 = 2.
            // i=1 -> dest=0. (1 + 2)%3 = 0.
            // i=2 -> dest=1. (2 + 2)%3 = 1.
            // Correct.

            // Wait, usually we read FROM shifted index?
            // We want result[0] to be the most negative frequency.
            // That corresponds to index (N+1)/2 in the FFT output.

            int src_idx = (i + (doppler_len + 1) / 2) % doppler_len;
            result[i] = Complex(out_buf[src_idx][0], out_buf[src_idx][1]);
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
