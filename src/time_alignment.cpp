#include <complex>
#include <vector>
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>

// Centralized FFTW init (shared across all .so files in this project)
#include "fftw_init.h"

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#else
#define HAVE_OPTMATHKERNELS 0
#endif

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
    std::vector<float> interleaved_prod;

public:
    // Non-copyable, non-movable (owns FFTW resources)
    TimeAligner(const TimeAligner&) = delete;
    TimeAligner& operator=(const TimeAligner&) = delete;
    TimeAligner(TimeAligner&&) = delete;
    TimeAligner& operator=(TimeAligner&&) = delete;

    TimeAligner(int samples) : n_samples(samples) {
        // Initialize FFTW thread support (centralized, safe to call multiple times)
        kraken_fftw_init();

        fft_len = 1;
        while (fft_len < 2 * n_samples && fft_len > 0) fft_len <<= 1;
        if (fft_len <= 0) fft_len = 1 << 20; // Cap at 1M on overflow

        buf_ref = fftwf_alloc_complex(fft_len);
        buf_surv = fftwf_alloc_complex(fft_len);
        buf_prod = fftwf_alloc_complex(fft_len);
        if (!buf_ref || !buf_surv || !buf_prod) {
            if (buf_ref) fftwf_free(buf_ref);
            if (buf_surv) fftwf_free(buf_surv);
            if (buf_prod) fftwf_free(buf_prod);
            throw std::runtime_error("TimeAligner: FFTW buffer allocation failed");
        }

        fwd_ref = fftwf_plan_dft_1d(fft_len, buf_ref, buf_ref, FFTW_FORWARD, FFTW_ESTIMATE);
        fwd_surv = fftwf_plan_dft_1d(fft_len, buf_surv, buf_surv, FFTW_FORWARD, FFTW_ESTIMATE);
        inv_out = fftwf_plan_dft_1d(fft_len, buf_prod, buf_prod, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!fwd_ref || !fwd_surv || !inv_out) {
            if (fwd_ref) fftwf_destroy_plan(fwd_ref);
            if (fwd_surv) fftwf_destroy_plan(fwd_surv);
            if (inv_out) fftwf_destroy_plan(inv_out);
            fftwf_free(buf_ref);
            fftwf_free(buf_surv);
            fftwf_free(buf_prod);
            throw std::runtime_error("TimeAligner: FFTW plan creation failed");
        }
        interleaved_prod.resize(2 * fft_len);
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
#if HAVE_OPTMATHKERNELS
        // Interleaved conjugate multiply: surv * conj(ref)
        // fftwf_complex is float[2], contiguous interleaved layout
        optmath::neon::neon_complex_conj_mul_interleaved_f32(
            interleaved_prod.data(),
            reinterpret_cast<const float*>(buf_surv),
            reinterpret_cast<const float*>(buf_ref),
            fft_len
        );
        // Scale and copy to buf_prod
        for (int i = 0; i < fft_len; i++) {
            buf_prod[i][0] = interleaved_prod[2*i] * scale;
            buf_prod[i][1] = interleaved_prod[2*i+1] * scale;
        }
#else
        for(int i=0; i<fft_len; ++i) {
            float sr = buf_surv[i][0];
            float si = buf_surv[i][1];
            float rr = buf_ref[i][0];
            float ri = -buf_ref[i][1]; // conj(Ref)
            buf_prod[i][0] = (sr*rr - si*ri) * scale;
            buf_prod[i][1] = (sr*ri + si*rr) * scale;
        }
#endif

        fftwf_execute(inv_out);

        // Find Peak
        int best_idx = 0;
        float max_mag = 0.0f;
        Complex best_val = Complex(buf_prod[0][0], buf_prod[0][1]);

#if HAVE_OPTMATHKERNELS
        // Batch magnitude-squared computation, then find argmax
        std::vector<float> mag_sq(fft_len);
        const float* prod_ptr = reinterpret_cast<const float*>(buf_prod);
        // Deinterleave for magnitude_squared (buf_prod is interleaved re/im)
        std::vector<float> prod_re(fft_len), prod_im(fft_len);
        for (int i = 0; i < fft_len; ++i) {
            prod_re[i] = buf_prod[i][0];
            prod_im[i] = buf_prod[i][1];
        }
        optmath::neon::neon_complex_magnitude_squared_f32(mag_sq.data(), prod_re.data(), prod_im.data(), fft_len);
        // Find argmax (no NEON argmax available, use scalar scan over magnitudes)
        for (int i = 0; i < fft_len; ++i) {
            if (mag_sq[i] > max_mag) {
                max_mag = mag_sq[i];
                best_idx = i;
            }
        }
        best_val = Complex(buf_prod[best_idx][0], buf_prod[best_idx][1]);
#else
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
#endif

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
    void align_destroy(void* p) { if (p) delete static_cast<TimeAligner*>(p); }
    void align_compute(void* p, const float* ref, const float* surv, int* delay, float* phase) {
        if (!p || !ref || !surv) return;
        static_cast<TimeAligner*>(p)->compute_offset(
            reinterpret_cast<const Complex*>(ref),
            reinterpret_cast<const Complex*>(surv),
            delay, phase
        );
    }
}
