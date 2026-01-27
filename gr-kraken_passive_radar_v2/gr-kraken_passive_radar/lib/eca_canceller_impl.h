/*
 * ECA Canceller Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H

#include <gnuradio/kraken_passive_radar/eca_canceller.h>
#include <gnuradio/thread/thread.h>
#include <vector>
#include <complex>

namespace gr {
namespace kraken_passive_radar {

class eca_canceller_impl : public eca_canceller
{
private:
    int d_num_taps;
    float d_reg_factor;
    int d_num_surv;
    float d_mu;   // NLMS step size
    float d_eps;  // NLMS regularization

    // Adaptive filter weights: num_surv x num_taps (separate real/imag)
    std::vector<std::vector<float>> d_weights_re;
    std::vector<std::vector<float>> d_weights_im;

    // History buffer for reference signal
    std::vector<std::complex<float>> d_ref_history;

    // Temporary buffers for deinterleaving
    std::vector<float> d_ref_re, d_ref_im;
    std::vector<float> d_surv_re, d_surv_im;
    std::vector<float> d_out_re, d_out_im;

    // Thread safety mutex
    gr::thread::mutex d_setlock;

    // NLMS filter for complex signals
    void nlms_filter_complex(
        float* out_re, float* out_im,
        const float* surv_re, const float* surv_im,
        const float* ref_re, const float* ref_im,
        float* weights_re, float* weights_im,
        int n_samples);

public:
    eca_canceller_impl(int num_taps, float reg_factor, int num_surv);
    ~eca_canceller_impl();

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    void set_num_taps(int num_taps) override;
    void set_reg_factor(float reg_factor) override;
    void set_mu(float mu);
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H */
