/*
 * ECA Canceller Implementation with NLMS Adaptive Filtering
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Uses OptMathKernels NLMS for clutter cancellation
 */

#include "eca_canceller_impl.h"
#include <gnuradio/io_signature.h>
#include <volk/volk.h>
#include <cstring>
#include <cmath>

namespace gr {
namespace kraken_passive_radar {

eca_canceller::sptr eca_canceller::make(int num_taps, float reg_factor, int num_surv)
{
    return gnuradio::make_block_sptr<eca_canceller_impl>(num_taps, reg_factor, num_surv);
}

eca_canceller_impl::eca_canceller_impl(int num_taps, float reg_factor, int num_surv)
    : gr::sync_block("eca_canceller",
                     // Inputs: 1 reference + num_surv surveillance channels
                     gr::io_signature::make(1 + num_surv, 1 + num_surv, sizeof(gr_complex)),
                     // Outputs: num_surv cleaned surveillance channels
                     gr::io_signature::make(num_surv, num_surv, sizeof(gr_complex))),
      d_num_taps(num_taps),
      d_reg_factor(reg_factor),
      d_num_surv(num_surv),
      d_mu(0.1f),  // NLMS step size
      d_eps(1e-6f) // Regularization for NLMS
{
    // Initialize weights for real and imaginary parts separately
    // Each surveillance channel has num_taps weights for real and imag
    d_weights_re.resize(num_surv);
    d_weights_im.resize(num_surv);
    for (int ch = 0; ch < num_surv; ch++) {
        d_weights_re[ch].resize(num_taps, 0.0f);
        d_weights_im[ch].resize(num_taps, 0.0f);
    }

    // Reference history buffer
    d_ref_history.resize(num_taps, std::complex<float>(0.0f, 0.0f));

    // Allocate temporary buffers for deinterleaving
    // Will resize in work() as needed

    // Request history for the filter
    set_history(num_taps);
}

eca_canceller_impl::~eca_canceller_impl() {}

int eca_canceller_impl::work(int noutput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{
    const gr_complex* ref = static_cast<const gr_complex*>(input_items[0]);

    // Total samples available including history
    const int total_samples = noutput_items + d_num_taps - 1;

    // Resize temporary buffers if needed
    if ((int)d_ref_re.size() < total_samples) {
        d_ref_re.resize(total_samples);
        d_ref_im.resize(total_samples);
        d_surv_re.resize(noutput_items);
        d_surv_im.resize(noutput_items);
        d_out_re.resize(noutput_items);
        d_out_im.resize(noutput_items);
    }

    // Deinterleave reference signal (includes history)
    for (int i = 0; i < total_samples; i++) {
        d_ref_re[i] = ref[i].real();
        d_ref_im[i] = ref[i].imag();
    }

    // Process each surveillance channel
    for (int ch = 0; ch < d_num_surv; ch++) {
        const gr_complex* surv = static_cast<const gr_complex*>(input_items[1 + ch]);
        gr_complex* out = static_cast<gr_complex*>(output_items[ch]);

        // Deinterleave surveillance signal (output region only)
        for (int i = 0; i < noutput_items; i++) {
            // Surveillance samples are at i + (num_taps - 1) due to history offset
            d_surv_re[i] = surv[i + d_num_taps - 1].real();
            d_surv_im[i] = surv[i + d_num_taps - 1].imag();
        }

        // Apply NLMS adaptive filter for real part
        nlms_filter_complex(
            d_out_re.data(), d_out_im.data(),
            d_surv_re.data(), d_surv_im.data(),
            d_ref_re.data() + d_num_taps - 1,  // Align with output region
            d_ref_im.data() + d_num_taps - 1,
            d_weights_re[ch].data(), d_weights_im[ch].data(),
            noutput_items
        );

        // Interleave output
        for (int i = 0; i < noutput_items; i++) {
            out[i] = gr_complex(d_out_re[i], d_out_im[i]);
        }
    }

    return noutput_items;
}

void eca_canceller_impl::nlms_filter_complex(
    float* out_re, float* out_im,
    const float* surv_re, const float* surv_im,
    const float* ref_re, const float* ref_im,
    float* weights_re, float* weights_im,
    int n_samples)
{
    // NLMS adaptive filter for complex signals
    // Processes real and imaginary parts together for proper complex weight update
    //
    // Filter model: clutter_estimate = sum(w[k] * ref[n-k])
    // Error: e[n] = surv[n] - clutter_estimate
    // Weight update: w[k] += mu * e[n] * conj(ref[n-k]) / (eps + ||ref||^2)

    for (int n = 0; n < n_samples; n++) {
        // Compute reference power for normalization
        float ref_power = d_eps;  // Start with regularization
        for (int k = 0; k < d_num_taps; k++) {
            int idx = n - k;  // ref is pre-aligned
            if (idx >= -d_num_taps + 1 && idx <= n) {
                float rr = ref_re[idx];
                float ri = ref_im[idx];
                ref_power += rr * rr + ri * ri;
            }
        }

        // Compute clutter estimate: y = sum(w * ref)
        // Complex multiply: (wr + j*wi) * (rr + j*ri) = (wr*rr - wi*ri) + j*(wr*ri + wi*rr)
        float y_re = 0.0f;
        float y_im = 0.0f;
        for (int k = 0; k < d_num_taps; k++) {
            int idx = n - k;
            float rr = (idx >= -d_num_taps + 1) ? ref_re[idx] : 0.0f;
            float ri = (idx >= -d_num_taps + 1) ? ref_im[idx] : 0.0f;

            y_re += weights_re[k] * rr - weights_im[k] * ri;
            y_im += weights_re[k] * ri + weights_im[k] * rr;
        }

        // Compute error: e = surv - clutter_estimate
        float e_re = surv_re[n] - y_re;
        float e_im = surv_im[n] - y_im;

        // Output is the error (clutter-cancelled signal)
        out_re[n] = e_re;
        out_im[n] = e_im;

        // Update weights using NLMS rule
        // w[k] += mu * e * conj(ref[n-k]) / ref_power
        // Complex: e * conj(ref) = (e_re + j*e_im) * (rr - j*ri)
        //                        = (e_re*rr + e_im*ri) + j*(e_im*rr - e_re*ri)
        float mu_norm = d_mu / ref_power;
        for (int k = 0; k < d_num_taps; k++) {
            int idx = n - k;
            float rr = (idx >= -d_num_taps + 1) ? ref_re[idx] : 0.0f;
            float ri = (idx >= -d_num_taps + 1) ? ref_im[idx] : 0.0f;

            // e * conj(ref)
            float update_re = e_re * rr + e_im * ri;
            float update_im = e_im * rr - e_re * ri;

            weights_re[k] += mu_norm * update_re;
            weights_im[k] += mu_norm * update_im;
        }
    }
}

void eca_canceller_impl::set_num_taps(int num_taps)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_num_taps = num_taps;

    // Resize weight vectors
    for (int ch = 0; ch < d_num_surv; ch++) {
        d_weights_re[ch].resize(num_taps, 0.0f);
        d_weights_im[ch].resize(num_taps, 0.0f);
    }
    d_ref_history.resize(num_taps, std::complex<float>(0.0f, 0.0f));
    set_history(num_taps);
}

void eca_canceller_impl::set_reg_factor(float reg_factor)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_reg_factor = reg_factor;
    d_eps = reg_factor;  // Use reg_factor as NLMS epsilon
}

void eca_canceller_impl::set_mu(float mu)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_mu = mu;
}

} // namespace kraken_passive_radar
} // namespace gr
