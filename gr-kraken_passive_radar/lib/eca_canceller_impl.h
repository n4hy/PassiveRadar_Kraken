/*
 * ECA Canceller Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Batch Toeplitz least-squares with FFT-based cross-correlation and FIR.
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H

#include <gnuradio/kraken_passive_radar/eca_canceller.h>
#include <gnuradio/thread/thread.h>
#include <fftw3.h>
#include <vector>
#include <complex>

namespace gr {
namespace kraken_passive_radar {

/**
 * eca_canceller_impl - Implementation of the ECA-B clutter canceller block
 *
 * Technique: Batch Toeplitz least-squares with Cholesky decomposition for
 * adaptive FIR weight computation, and FFT-based cross-correlation and
 * FIR filtering for efficient clutter subtraction.
 */
class eca_canceller_impl : public eca_canceller
{
private:
    int d_num_taps;
    float d_diag_loading;
    int d_num_surv;

    // Toeplitz solver buffers
    std::vector<std::complex<float>> d_R;   // L x L autocorrelation
    std::vector<std::complex<float>> d_L;   // L x L Cholesky factor
    std::vector<std::complex<float>> d_p;   // L cross-correlation
    std::vector<std::complex<float>> d_w;   // L filter weights
    std::vector<std::complex<float>> d_y;   // L Cholesky intermediate

    // SoA deinterleave buffers (for autocorrelation dot products)
    std::vector<float> d_ref_re, d_ref_im;
    std::vector<float> d_surv_re, d_surv_im;

    // FFT-based cross-correlation and FIR application
    int d_fir_fft_len;              // Current FFT length (0 = not initialized)
    fftwf_plan d_fir_fwd;           // Forward FFT plan
    fftwf_plan d_fir_inv;           // Inverse FFT plan
    fftwf_complex* d_fir_ref_fft;   // Cached FFT(ref) — reused across channels
    fftwf_complex* d_fir_a;         // Working buffer for FFT(surv) or FFT(w)
    fftwf_complex* d_fir_b;         // Product → IFFT buffer

    gr::thread::mutex d_setlock;

    /** dot_f32 - Compute dot product of two float arrays */
    static inline float dot_f32(const float* a, const float* b, int n);
    /** cholesky_decompose - Decompose the Toeplitz autocorrelation matrix using Cholesky factorization */
    bool cholesky_decompose();
    /** cholesky_solve - Solve for optimal FIR filter weights via Cholesky back-substitution */
    void cholesky_solve();
    /** setup_fir_fft - Allocate FFTW plans and buffers for FFT-based FIR filtering */
    void setup_fir_fft(int n_output);
    /** cleanup_fir_fft - Destroy FFTW plans and free FFT buffers */
    void cleanup_fir_fft();

public:
    /**
     * eca_canceller_impl - Construct ECA-B canceller with tap count, regularization, and channel count
     */
    eca_canceller_impl(int num_taps, float reg_factor, int num_surv);
    ~eca_canceller_impl();

    /**
     * work - Compute adaptive FIR weights and subtract clutter from all surveillance channels
     */
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    /** set_num_taps - Update the number of FIR filter taps at runtime */
    void set_num_taps(int num_taps) override;
    /** set_reg_factor - Update the diagonal loading regularization factor */
    void set_reg_factor(float reg_factor) override;
    /** set_mu - Set the step size for adaptive weight updates */
    void set_mu(float mu);
};

} // namespace kraken_passive_radar
} // namespace gr

#endif
