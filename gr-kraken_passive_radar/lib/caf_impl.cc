/*
 * CAF (Cross-Ambiguity Function) — C++ GNU Radio Block
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * FFT cross-correlation: IFFT( FFT(surv) * conj(FFT(ref)) )
 * Replaces Python ctypes wrapper to eliminate GIL from hot path.
 */

#include "caf_impl.h"
#include <gnuradio/io_signature.h>
#include <cstring>
#include <stdexcept>

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#endif

namespace gr {
namespace kraken_passive_radar {

caf::sptr caf::make(int n_samples)
{
    return gnuradio::make_block_sptr<caf_impl>(n_samples);
}

caf_impl::caf_impl(int n_samples)
    : gr::sync_decimator("caf",
                         // 2 inputs: ref + surv, scalar complex
                         gr::io_signature::make(2, 2, sizeof(gr_complex)),
                         // 1 output: range profile vector
                         gr::io_signature::make(1, 1, sizeof(gr_complex) * n_samples),
                         n_samples),  // decimation factor
      d_n_samples(n_samples)
{
    // FFT length: next power of 2 >= 2*n_samples for linear correlation
    d_fft_len = 1;
    while (d_fft_len < 2 * n_samples) d_fft_len <<= 1;

    d_buf_ref  = fftwf_alloc_complex(d_fft_len);
    d_buf_surv = fftwf_alloc_complex(d_fft_len);
    d_buf_prod = fftwf_alloc_complex(d_fft_len);
    if (!d_buf_ref || !d_buf_surv || !d_buf_prod) {
        if (d_buf_ref)  fftwf_free(d_buf_ref);
        if (d_buf_surv) fftwf_free(d_buf_surv);
        if (d_buf_prod) fftwf_free(d_buf_prod);
        throw std::runtime_error("caf: FFTW buffer allocation failed");
    }

    d_fwd_ref  = fftwf_plan_dft_1d(d_fft_len, d_buf_ref,  d_buf_ref,  FFTW_FORWARD,  FFTW_ESTIMATE);
    d_fwd_surv = fftwf_plan_dft_1d(d_fft_len, d_buf_surv, d_buf_surv, FFTW_FORWARD,  FFTW_ESTIMATE);
    d_inv      = fftwf_plan_dft_1d(d_fft_len, d_buf_prod, d_buf_prod, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!d_fwd_ref || !d_fwd_surv || !d_inv) {
        if (d_fwd_ref)  fftwf_destroy_plan(d_fwd_ref);
        if (d_fwd_surv) fftwf_destroy_plan(d_fwd_surv);
        if (d_inv)      fftwf_destroy_plan(d_inv);
        fftwf_free(d_buf_ref);
        fftwf_free(d_buf_surv);
        fftwf_free(d_buf_prod);
        throw std::runtime_error("caf: FFTW plan creation failed");
    }
}

caf_impl::~caf_impl()
{
    fftwf_destroy_plan(d_fwd_ref);
    fftwf_destroy_plan(d_fwd_surv);
    fftwf_destroy_plan(d_inv);
    fftwf_free(d_buf_ref);
    fftwf_free(d_buf_surv);
    fftwf_free(d_buf_prod);
}

int caf_impl::work(int noutput_items,
                   gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items)
{
    const gr_complex* ref  = static_cast<const gr_complex*>(input_items[0]);
    const gr_complex* surv = static_cast<const gr_complex*>(input_items[1]);
    gr_complex* out = static_cast<gr_complex*>(output_items[0]);

    const int N = d_n_samples;
    const int F = d_fft_len;
    const float scale = 1.0f / F;

    for (int frame = 0; frame < noutput_items; frame++) {
        const gr_complex* r = ref  + frame * N;
        const gr_complex* s = surv + frame * N;
        gr_complex* o = out + frame * N;  // output vector for this frame

        // Load ref into FFTW buffer, zero-pad
        std::memset(d_buf_ref, 0, F * sizeof(fftwf_complex));
        std::memcpy(d_buf_ref, r, N * sizeof(fftwf_complex));

        // Load surv into FFTW buffer, zero-pad
        std::memset(d_buf_surv, 0, F * sizeof(fftwf_complex));
        std::memcpy(d_buf_surv, s, N * sizeof(fftwf_complex));

        // Forward FFTs (in-place)
        fftwf_execute(d_fwd_ref);
        fftwf_execute(d_fwd_surv);

        // Spectral multiply: surv * conj(ref), with 1/N scaling
#ifdef HAVE_OPTMATHKERNELS
        // NEON path: deinterleave → conj_mul → scale → reinterleave
        // Operating directly on fftwf_complex (float[2]) arrays
        // fftwf_complex is float[2], same layout as interleaved re/im
        optmath::neon::neon_complex_conj_mul_interleaved_f32(
            reinterpret_cast<float*>(d_buf_prod),
            reinterpret_cast<const float*>(d_buf_surv),
            reinterpret_cast<const float*>(d_buf_ref),
            F);
        // Scale
        float* prod_f = reinterpret_cast<float*>(d_buf_prod);
        for (int i = 0; i < 2 * F; i++) {
            prod_f[i] *= scale;
        }
#else
        for (int i = 0; i < F; i++) {
            float sr = d_buf_surv[i][0];
            float si = d_buf_surv[i][1];
            float rr = d_buf_ref[i][0];
            float ri = d_buf_ref[i][1];
            // surv * conj(ref) = (sr*rr + si*ri) + j*(si*rr - sr*ri)
            d_buf_prod[i][0] = (sr * rr + si * ri) * scale;
            d_buf_prod[i][1] = (si * rr - sr * ri) * scale;
        }
#endif

        // Inverse FFT
        fftwf_execute(d_inv);

        // Copy first N samples to output
        std::memcpy(o, d_buf_prod, N * sizeof(gr_complex));
    }

    return noutput_items;
}

} // namespace kraken_passive_radar
} // namespace gr
