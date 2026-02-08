#include "eca_canceller_impl.h"
#include <gnuradio/io_signature.h>
#include <volk/volk.h>
#include <cstring>

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
      d_mu(0.01f),  // NLMS step size (conservative default)
      d_eps(reg_factor)  // Use reg_factor as epsilon
{
    // Initialize weights to zero
    d_weights.resize(num_surv);
    for (int ch = 0; ch < num_surv; ch++) {
        d_weights[ch].resize(num_taps, std::complex<float>(0.0f, 0.0f));
    }

    // Reference history buffer
    d_ref_history.resize(num_taps, std::complex<float>(0.0f, 0.0f));

    // Request history for the filter
    set_history(num_taps);
}

eca_canceller_impl::~eca_canceller_impl() {}

int eca_canceller_impl::work(int noutput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{
    const gr_complex* ref = static_cast<const gr_complex*>(input_items[0]);

    // Process each surveillance channel
    for (int ch = 0; ch < d_num_surv; ch++) {
        const gr_complex* surv = static_cast<const gr_complex*>(input_items[1 + ch]);
        gr_complex* out = static_cast<gr_complex*>(output_items[ch]);

        // NLMS adaptive clutter cancellation
        // Algorithm: w[k] += mu * e[n] * conj(ref[n-k]) / (eps + ||ref||^2)
        for (int i = 0; i < noutput_items; i++) {
            // Compute reference signal power for normalization
            float ref_power = d_eps;  // Start with regularization
            for (int l = 0; l < d_num_taps; l++) {
                gr_complex r = ref[i + (d_num_taps - 1) - l];
                ref_power += std::norm(r);  // |r|^2
            }

            // Compute clutter estimate: y[n] = sum(w[l] * ref[n-l])
            gr_complex clutter_estimate(0.0f, 0.0f);
            for (int l = 0; l < d_num_taps; l++) {
                clutter_estimate += d_weights[ch][l] * ref[i + (d_num_taps - 1) - l];
            }

            // Compute error: e[n] = surv[n] - clutter_estimate
            gr_complex error = surv[i + (d_num_taps - 1)] - clutter_estimate;

            // Output is the error (clutter-cancelled signal)
            out[i] = error;

            // Update weights using NLMS rule
            // w[l] += (mu / ref_power) * e[n] * conj(ref[n-l])
            float mu_norm = d_mu / ref_power;
            for (int l = 0; l < d_num_taps; l++) {
                gr_complex ref_sample = ref[i + (d_num_taps - 1) - l];
                d_weights[ch][l] += mu_norm * error * std::conj(ref_sample);
            }
        }
    }

    return noutput_items;
}

void eca_canceller_impl::set_num_taps(int num_taps)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_num_taps = num_taps;
    
    // Resize weight vectors
    for (int ch = 0; ch < d_num_surv; ch++) {
        d_weights[ch].resize(num_taps, std::complex<float>(0.0f, 0.0f));
    }
    d_ref_history.resize(num_taps, std::complex<float>(0.0f, 0.0f));
    set_history(num_taps);
}

void eca_canceller_impl::set_reg_factor(float reg_factor)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_reg_factor = reg_factor;
    d_eps = reg_factor;  // Update NLMS epsilon
}

} // namespace kraken_passive_radar
} // namespace gr
