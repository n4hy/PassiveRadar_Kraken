/**
 * CAF Processing with OptMathKernels Optimization
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Cross-Ambiguity Function (CAF) processing for passive radar.
 * Uses OptMathKernels NEON-optimized functions when available,
 * with FFTW3 fallback for complex operations.
 */

#include <complex>
#include <vector>
#include <fftw3.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>
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

// Check for OptMathKernels availability
#if __has_include(<optmath/radar_kernels.hpp>)
#define HAVE_OPTMATHKERNELS 1
#include <optmath/radar_kernels.hpp>
#include <optmath/neon_kernels.hpp>
#else
#define HAVE_OPTMATHKERNELS 0
#endif

using Complex = std::complex<float>;

/**
 * @brief CAF Processor with OptMathKernels optimization
 *
 * Computes Cross-Ambiguity Function for passive radar:
 * CAF(tau, fd) = integral{ surv(t) * conj(ref(t-tau)) * exp(-j*2*pi*fd*t) } dt
 *
 * For each Doppler bin, applies frequency shift to reference, then
 * cross-correlates with surveillance signal.
 */
class CafProcessor {
private:
    int n_samples;
    int fft_len;
    int n_doppler_bins;
    int n_range_bins;

    float doppler_start;
    float doppler_step;
    float sample_rate;

    // FFTW plans and buffers
    fftwf_plan fwd_ref;
    fftwf_plan fwd_surv;
    fftwf_plan inv_out;

    fftwf_complex* buf_ref;
    fftwf_complex* buf_surv;
    fftwf_complex* buf_prod;

    // Deinterleaved buffers for NEON operations
    std::vector<float> ref_re, ref_im;
    std::vector<float> surv_re, surv_im;
    std::vector<float> shifted_re, shifted_im;
    std::vector<float> caf_out;

    // Cached buffers for compute_range_profile (avoid allocation in hot path)
    std::vector<float> fft_surv_re, fft_surv_im;
    std::vector<float> fft_ref_re, fft_ref_im;
    std::vector<float> prod_re, prod_im;

    // Precomputed Doppler shift phasors
    std::vector<std::vector<float>> doppler_phasor_re;
    std::vector<std::vector<float>> doppler_phasor_im;

    bool use_optmath;

    /**
     * @brief Precompute Doppler shift phasors for efficiency
     */
    void precompute_doppler_phasors() {
        doppler_phasor_re.resize(n_doppler_bins);
        doppler_phasor_im.resize(n_doppler_bins);

        for (int d = 0; d < n_doppler_bins; d++) {
            float fd = doppler_start + d * doppler_step;
            doppler_phasor_re[d].resize(n_samples);
            doppler_phasor_im[d].resize(n_samples);

            for (int t = 0; t < n_samples; t++) {
                float phase = -2.0f * M_PI * fd * t / sample_rate;
                doppler_phasor_re[d][t] = std::cos(phase);
                doppler_phasor_im[d][t] = std::sin(phase);
            }
        }
    }

public:
    /**
     * @brief Construct CAF processor
     * @param samples Number of samples per CPI
     * @param n_doppler Number of Doppler bins
     * @param n_range Number of range bins
     * @param dop_start Starting Doppler frequency (Hz)
     * @param dop_step Doppler step size (Hz)
     * @param fs Sample rate (Hz)
     */
    CafProcessor(int samples, int n_doppler = 64, int n_range = 256,
                 float dop_start = -125.0f, float dop_step = 3.9f,
                 float fs = 250000.0f)
        : n_samples(samples),
          n_doppler_bins(n_doppler),
          n_range_bins(n_range),
          doppler_start(dop_start),
          doppler_step(dop_step),
          sample_rate(fs)
    {
        // Initialize FFTW thread support (safe to call multiple times)
        init_fftw_threads();

        // FFT length: next power of 2 >= 2 * n_samples for linear correlation
        fft_len = 1;
        while (fft_len < 2 * n_samples) fft_len <<= 1;

        // FFTW buffers
        buf_ref = fftwf_alloc_complex(fft_len);
        buf_surv = fftwf_alloc_complex(fft_len);
        buf_prod = fftwf_alloc_complex(fft_len);

        // FFTW plans
        fwd_ref = fftwf_plan_dft_1d(fft_len, buf_ref, buf_ref, FFTW_FORWARD, FFTW_ESTIMATE);
        fwd_surv = fftwf_plan_dft_1d(fft_len, buf_surv, buf_surv, FFTW_FORWARD, FFTW_ESTIMATE);
        inv_out = fftwf_plan_dft_1d(fft_len, buf_prod, buf_prod, FFTW_BACKWARD, FFTW_ESTIMATE);

        // Deinterleaved buffers
        ref_re.resize(n_samples);
        ref_im.resize(n_samples);
        surv_re.resize(n_samples);
        surv_im.resize(n_samples);
        shifted_re.resize(n_samples);
        shifted_im.resize(n_samples);
        caf_out.resize(n_doppler * n_range);

        // Pre-allocate buffers for compute_range_profile
        fft_surv_re.resize(fft_len);
        fft_surv_im.resize(fft_len);
        fft_ref_re.resize(fft_len);
        fft_ref_im.resize(fft_len);
        prod_re.resize(fft_len);
        prod_im.resize(fft_len);

        // Precompute Doppler phasors
        precompute_doppler_phasors();

        // Check for OptMathKernels
#if HAVE_OPTMATHKERNELS
        use_optmath = true;
        // Note: OptMathKernels NEON optimization enabled
#else
        use_optmath = false;
        // Note: Using scalar implementation
#endif
    }

    ~CafProcessor() {
        fftwf_destroy_plan(fwd_ref);
        fftwf_destroy_plan(fwd_surv);
        fftwf_destroy_plan(inv_out);
        fftwf_free(buf_ref);
        fftwf_free(buf_surv);
        fftwf_free(buf_prod);
    }

    /**
     * @brief Compute single-Doppler range profile using FFT
     *
     * Computes: IFFT( FFT(surv) * conj(FFT(shifted_ref)) )
     *
     * @param shifted_ref_re, shifted_ref_im Doppler-shifted reference (n_samples)
     * @param surv_re, surv_im Surveillance signal (n_samples)
     * @param out Range profile output (n_range_bins magnitudes)
     */
    void compute_range_profile(const float* sref_re, const float* sref_im,
                               const float* surv_re_in, const float* surv_im_in,
                               float* out) {
        // Zero-pad and load reference
        std::memset(buf_ref, 0, fft_len * sizeof(fftwf_complex));
        for (int i = 0; i < n_samples; i++) {
            buf_ref[i][0] = sref_re[i];
            buf_ref[i][1] = sref_im[i];
        }

        // Zero-pad and load surveillance
        std::memset(buf_surv, 0, fft_len * sizeof(fftwf_complex));
        for (int i = 0; i < n_samples; i++) {
            buf_surv[i][0] = surv_re_in[i];
            buf_surv[i][1] = surv_im_in[i];
        }

        // FFT both signals
        fftwf_execute(fwd_ref);
        fftwf_execute(fwd_surv);

        // Multiply: Surv * conj(Ref) with normalization
        float scale = 1.0f / fft_len;

#if HAVE_OPTMATHKERNELS
        // Use NEON-optimized complex conjugate multiply
        // Deinterleave FFT outputs for NEON processing (use pre-allocated buffers)
        for (int i = 0; i < fft_len; i++) {
            fft_surv_re[i] = buf_surv[i][0];
            fft_surv_im[i] = buf_surv[i][1];
            fft_ref_re[i] = buf_ref[i][0];
            fft_ref_im[i] = -buf_ref[i][1];  // Conjugate
        }

        // Complex multiply using NEON
        optmath::neon::neon_complex_mul_f32(
            prod_re.data(), prod_im.data(),
            fft_surv_re.data(), fft_surv_im.data(),
            fft_ref_re.data(), fft_ref_im.data(),
            fft_len
        );

        // Scale and reinterleave
        for (int i = 0; i < fft_len; i++) {
            buf_prod[i][0] = prod_re[i] * scale;
            buf_prod[i][1] = prod_im[i] * scale;
        }
#else
        // Scalar fallback
        for (int i = 0; i < fft_len; i++) {
            float sr = buf_surv[i][0];
            float si = buf_surv[i][1];
            float rr = buf_ref[i][0];
            float ri = -buf_ref[i][1];  // Conjugate

            buf_prod[i][0] = (sr * rr - si * ri) * scale;
            buf_prod[i][1] = (sr * ri + si * rr) * scale;
        }
#endif

        // IFFT
        fftwf_execute(inv_out);

        // Extract magnitude for first n_range_bins
        for (int i = 0; i < n_range_bins && i < n_samples; i++) {
            float re = buf_prod[i][0];
            float im = buf_prod[i][1];
            out[i] = std::sqrt(re * re + im * im);
        }
    }

    /**
     * @brief Process full CAF: all Doppler bins x range bins
     *
     * @param ref Input reference signal (interleaved complex, n_samples)
     * @param surv Input surveillance signal (interleaved complex, n_samples)
     * @param caf_out Output CAF magnitude [n_doppler_bins x n_range_bins] (row-major)
     */
    void process_caf(const Complex* ref, const Complex* surv, float* caf_output) {
        // Deinterleave inputs
        for (int i = 0; i < n_samples; i++) {
            ref_re[i] = ref[i].real();
            ref_im[i] = ref[i].imag();
            surv_re[i] = surv[i].real();
            surv_im[i] = surv[i].imag();
        }

#if HAVE_OPTMATHKERNELS
        // Use OptMathKernels CAF function if it provides full CAF
        // For now, we still use per-Doppler processing with NEON complex ops
#endif

        // Process each Doppler bin
        for (int d = 0; d < n_doppler_bins; d++) {
            // Apply Doppler shift to reference: ref * exp(-j*2*pi*fd*t)
#if HAVE_OPTMATHKERNELS
            optmath::neon::neon_complex_mul_f32(
                shifted_re.data(), shifted_im.data(),
                ref_re.data(), ref_im.data(),
                doppler_phasor_re[d].data(), doppler_phasor_im[d].data(),
                n_samples
            );
#else
            for (int t = 0; t < n_samples; t++) {
                float pr = doppler_phasor_re[d][t];
                float pi = doppler_phasor_im[d][t];
                shifted_re[t] = ref_re[t] * pr - ref_im[t] * pi;
                shifted_im[t] = ref_re[t] * pi + ref_im[t] * pr;
            }
#endif

            // Compute range profile for this Doppler
            compute_range_profile(
                shifted_re.data(), shifted_im.data(),
                surv_re.data(), surv_im.data(),
                caf_output + d * n_range_bins
            );
        }
    }

    /**
     * @brief Simple single-row CAF (for compatibility with original API)
     */
    void process(const Complex* ref, const Complex* surv, Complex* out) {
        // Use zero Doppler shift
        for (int i = 0; i < n_samples; i++) {
            ref_re[i] = ref[i].real();
            ref_im[i] = ref[i].imag();
            surv_re[i] = surv[i].real();
            surv_im[i] = surv[i].imag();
        }

        // Zero-pad and load
        std::memset(buf_ref, 0, fft_len * sizeof(fftwf_complex));
        std::memset(buf_surv, 0, fft_len * sizeof(fftwf_complex));

        for (int i = 0; i < n_samples; i++) {
            buf_ref[i][0] = ref_re[i];
            buf_ref[i][1] = ref_im[i];
            buf_surv[i][0] = surv_re[i];
            buf_surv[i][1] = surv_im[i];
        }

        // FFT
        fftwf_execute(fwd_ref);
        fftwf_execute(fwd_surv);

        // Multiply
        float scale = 1.0f / fft_len;
        for (int i = 0; i < fft_len; i++) {
            float sr = buf_surv[i][0];
            float si = buf_surv[i][1];
            float rr = buf_ref[i][0];
            float ri = -buf_ref[i][1];

            buf_prod[i][0] = (sr * rr - si * ri) * scale;
            buf_prod[i][1] = (sr * ri + si * rr) * scale;
        }

        // IFFT
        fftwf_execute(inv_out);

        // Output
        for (int i = 0; i < n_samples; i++) {
            out[i] = Complex(buf_prod[i][0], buf_prod[i][1]);
        }
    }

    int get_n_doppler() const { return n_doppler_bins; }
    int get_n_range() const { return n_range_bins; }
    bool using_optmath() const { return use_optmath; }
};

// C API for Python ctypes binding
extern "C" {
    void* caf_create(int n_samples) {
        return new CafProcessor(n_samples);
    }

    void* caf_create_full(int n_samples, int n_doppler, int n_range,
                          float doppler_start, float doppler_step, float sample_rate) {
        return new CafProcessor(n_samples, n_doppler, n_range,
                                doppler_start, doppler_step, sample_rate);
    }

    void caf_destroy(void* ptr) {
        if (ptr) delete static_cast<CafProcessor*>(ptr);
    }

    void caf_process(void* ptr, const float* ref, const float* surv, float* out) {
        if (!ptr || !ref || !surv || !out) return;
        CafProcessor* obj = static_cast<CafProcessor*>(ptr);
        obj->process(reinterpret_cast<const Complex*>(ref),
                     reinterpret_cast<const Complex*>(surv),
                     reinterpret_cast<Complex*>(out));
    }

    void caf_process_full(void* ptr, const float* ref, const float* surv, float* out) {
        if (!ptr || !ref || !surv || !out) return;
        CafProcessor* obj = static_cast<CafProcessor*>(ptr);
        obj->process_caf(reinterpret_cast<const Complex*>(ref),
                         reinterpret_cast<const Complex*>(surv),
                         out);
    }

    int caf_get_n_doppler(void* ptr) {
        if (!ptr) return 0;
        return static_cast<CafProcessor*>(ptr)->get_n_doppler();
    }

    int caf_get_n_range(void* ptr) {
        if (!ptr) return 0;
        return static_cast<CafProcessor*>(ptr)->get_n_range();
    }

    int caf_using_optmath(void* ptr) {
        if (!ptr) return 0;
        return static_cast<CafProcessor*>(ptr)->using_optmath() ? 1 : 0;
    }
}
