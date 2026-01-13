#include <complex>
#include <vector>
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <cstring>

using Complex = std::complex<float>;

// Helper function to get index of max magnitude
struct MaxResult {
    int index;
    float val;
    Complex c_val;
};

// Reuses logic similar to CAF but optimized for seeking peak
class TimeAligner {
    int n_samples;
    int fft_len;
    fftwf_plan fwd_ref, fwd_surv, inv_out;
    fftwf_complex *buf_ref, *buf_surv, *buf_prod;

public:
    TimeAligner(int samples) : n_samples(samples) {
        fft_len = 1;
        while (fft_len < 2 * n_samples) fft_len <<= 1;

        buf_ref = fftwf_alloc_complex(fft_len);
        buf_surv = fftwf_alloc_complex(fft_len);
        buf_prod = fftwf_alloc_complex(fft_len);

        fwd_ref = fftwf_plan_dft_1d(fft_len, buf_ref, buf_ref, FFTW_FORWARD, FFTW_ESTIMATE);
        fwd_surv = fftwf_plan_dft_1d(fft_len, buf_surv, buf_surv, FFTW_FORWARD, FFTW_ESTIMATE);
        inv_out = fftwf_plan_dft_1d(fft_len, buf_prod, buf_prod, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    ~TimeAligner() {
        fftwf_destroy_plan(fwd_ref);
        fftwf_destroy_plan(fwd_surv);
        fftwf_destroy_plan(inv_out);
        fftwf_free(buf_ref);
        fftwf_free(buf_surv);
        fftwf_free(buf_prod);
    }

    // Returns delay (samples) and phase (radians) of peak correlation
    void compute_offset(const Complex* ref, const Complex* surv, int* out_delay, float* out_phase) {
        std::memset(buf_ref, 0, fft_len * sizeof(fftwf_complex));
        std::memset(buf_surv, 0, fft_len * sizeof(fftwf_complex));

        for(int i=0; i<n_samples; ++i) {
            buf_ref[i][0] = ref[i].real();
            buf_ref[i][1] = ref[i].imag();
            buf_surv[i][0] = surv[i].real();
            buf_surv[i][1] = surv[i].imag();
        }

        fftwf_execute(fwd_ref);
        fftwf_execute(fwd_surv);

        float scale = 1.0f / fft_len;
        for(int i=0; i<fft_len; ++i) {
            float sr = buf_surv[i][0];
            float si = buf_surv[i][1];
            float rr = buf_ref[i][0];
            float ri = -buf_ref[i][1]; // conj(Ref)
            buf_prod[i][0] = (sr*rr - si*ri) * scale;
            buf_prod[i][1] = (sr*ri + si*rr) * scale;
        }

        fftwf_execute(inv_out);

        // Find Peak
        int best_idx = 0;
        float max_mag = -1.0f;
        Complex best_val = 0.0f;

        // Search full range?
        // Correlation lag 0 is at 0.
        // Lag 1 is at 1.
        // Lag -1 is at fft_len - 1.
        // We assume delay is small relative to n_samples?
        // Let's search all.
        for(int i=0; i<fft_len; ++i) {
            float re = buf_prod[i][0];
            float im = buf_prod[i][1];
            float mag = re*re + im*im;
            if (mag > max_mag) {
                max_mag = mag;
                best_idx = i;
                best_val = Complex(re, im);
            }
        }

        // Convert index to signed delay
        if (best_idx > fft_len/2) {
            best_idx -= fft_len;
        }

        if (out_delay) *out_delay = best_idx;
        if (out_phase) *out_phase = std::arg(best_val);
    }
};

extern "C" {
    void* align_create(int n) { return new TimeAligner(n); }
    void align_destroy(void* p) { delete static_cast<TimeAligner*>(p); }
    void align_compute(void* p, const float* ref, const float* surv, int* delay, float* phase) {
        static_cast<TimeAligner*>(p)->compute_offset(
            reinterpret_cast<const Complex*>(ref),
            reinterpret_cast<const Complex*>(surv),
            delay, phase
        );
    }
}
