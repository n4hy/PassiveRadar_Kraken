#include <complex>
#include <vector>
#include <fftw3.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>

using Complex = std::complex<float>;

class CafProcessor {
private:
    int n_samples;
    int fft_len;

    fftwf_plan fwd_ref;
    fftwf_plan fwd_surv;
    fftwf_plan inv_out;

    fftwf_complex* buf_ref;
    fftwf_complex* buf_surv;
    fftwf_complex* buf_prod;

public:
    CafProcessor(int samples) : n_samples(samples) {
        // Find next power of 2 >= 2 * n_samples for linear correlation
        // This ensures no circular wrap-around artifacts interfere with the main correlation peak
        // within the range of interest (0..N).
        fft_len = 1;
        while (fft_len < 2 * n_samples) fft_len <<= 1;

        buf_ref = fftwf_alloc_complex(fft_len);
        buf_surv = fftwf_alloc_complex(fft_len);
        buf_prod = fftwf_alloc_complex(fft_len);

        // Planning (ESTIMATE for speed of creation)
        fwd_ref = fftwf_plan_dft_1d(fft_len, buf_ref, buf_ref, FFTW_FORWARD, FFTW_ESTIMATE);
        fwd_surv = fftwf_plan_dft_1d(fft_len, buf_surv, buf_surv, FFTW_FORWARD, FFTW_ESTIMATE);
        inv_out = fftwf_plan_dft_1d(fft_len, buf_prod, buf_prod, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    ~CafProcessor() {
        fftwf_destroy_plan(fwd_ref);
        fftwf_destroy_plan(fwd_surv);
        fftwf_destroy_plan(inv_out);
        fftwf_free(buf_ref);
        fftwf_free(buf_surv);
        fftwf_free(buf_prod);
    }

    // Computes Range Profile: IFFT( FFT(Surv) * conj(FFT(Ref)) )
    // Result corresponds to cross-correlation: R[k] = sum Surv[n] * conj(Ref[n-k])
    // The peak index 'k' corresponds to the delay of Surv relative to Ref.
    void process(const Complex* ref, const Complex* surv, Complex* out) {
        // 1. Zero Pads
        std::memset(buf_ref, 0, fft_len * sizeof(fftwf_complex));
        std::memset(buf_surv, 0, fft_len * sizeof(fftwf_complex));

        // 2. Load Data
        for(int i=0; i<n_samples; ++i) {
            buf_ref[i][0] = ref[i].real();
            buf_ref[i][1] = ref[i].imag();
            buf_surv[i][0] = surv[i].real();
            buf_surv[i][1] = surv[i].imag();
        }

        // 3. FFT
        fftwf_execute(fwd_ref);
        fftwf_execute(fwd_surv);

        // 4. Multiply: Surv * conj(Ref)
        // Note: For standard Cross-Corr (Ref, Surv), we usually do Ref * conj(Surv) -> peak at neg lag?
        // Let's stick to standard PBR: Surv is delayed Ref.
        // S(t) = R(t - tau).
        // Want peak at tau.
        // Match Filter: S * conj(R).
        // F(S) * conj(F(R)).
        // F(S) ~ F(R) * exp(-j w tau).
        // Product ~ |F(R)|^2 * exp(-j w tau).
        // IFFT -> Delta(t - tau).
        // Correct.
        float scale = 1.0f / fft_len;
        for(int i=0; i<fft_len; ++i) {
            float sr = buf_surv[i][0];
            float si = buf_surv[i][1];
            float rr = buf_ref[i][0];
            float ri = -buf_ref[i][1]; // Conjugate Ref

            // (sr + j si) * (rr + j ri)
            buf_prod[i][0] = (sr*rr - si*ri) * scale;
            buf_prod[i][1] = (sr*ri + si*rr) * scale;
        }

        // 5. IFFT
        fftwf_execute(inv_out);

        // 6. Output first n_samples
        // Delay 0 is at index 0. Delay 1 is at index 1.
        for(int i=0; i<n_samples; ++i) {
            out[i] = Complex(buf_prod[i][0], buf_prod[i][1]);
        }
    }
};

extern "C" {
    void* caf_create(int n_samples) {
        return new CafProcessor(n_samples);
    }

    void caf_destroy(void* ptr) {
        if (ptr) delete static_cast<CafProcessor*>(ptr);
    }

    void caf_process(void* ptr, const float* ref, const float* surv, float* out) {
        if (!ptr) return;
        CafProcessor* obj = static_cast<CafProcessor*>(ptr);
        obj->process(reinterpret_cast<const Complex*>(ref),
                     reinterpret_cast<const Complex*>(surv),
                     reinterpret_cast<Complex*>(out));
    }
}
