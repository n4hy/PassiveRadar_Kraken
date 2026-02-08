/*
 * Coherence Monitor Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "coherence_monitor_impl.h"
#include <gnuradio/io_signature.h>
#include <volk/volk.h>
#include <cmath>
#include <numeric>

namespace gr {
namespace kraken_passive_radar {

coherence_monitor::sptr
coherence_monitor::make(int num_channels,
                        float sample_rate,
                        float measure_interval_ms,
                        float measure_duration_ms,
                        float corr_threshold,
                        float phase_threshold_deg)
{
    return gnuradio::make_block_sptr<coherence_monitor_impl>(
        num_channels, sample_rate, measure_interval_ms, measure_duration_ms,
        corr_threshold, phase_threshold_deg);
}

coherence_monitor_impl::coherence_monitor_impl(int num_channels,
                                               float sample_rate,
                                               float measure_interval_ms,
                                               float measure_duration_ms,
                                               float corr_threshold,
                                               float phase_threshold_deg)
    : gr::sync_block("coherence_monitor",
                     gr::io_signature::make(num_channels, num_channels, sizeof(gr_complex)),
                     gr::io_signature::make(num_channels, num_channels, sizeof(gr_complex))),
      d_num_channels(num_channels),
      d_sample_rate(sample_rate),
      d_measure_interval_samples(static_cast<uint64_t>(measure_interval_ms * sample_rate / 1000.0)),
      d_measure_duration_samples(static_cast<int>(measure_duration_ms * sample_rate / 1000.0)),
      d_corr_threshold(corr_threshold),
      d_phase_threshold_rad(phase_threshold_deg * M_PI / 180.0),
      d_calibration_needed(false),
      d_calibration_in_progress(false),
      d_sample_count(0),
      d_measure_count(0),
      d_consecutive_failures(0)
{
    // Initialize per-channel statistics
    d_correlation.resize(num_channels, 1.0f);
    d_phase_offset.resize(num_channels, 0.0f);
    d_phase_variance.resize(num_channels, 0.0f);
    
    // Measurement buffers
    d_ref_buffer.resize(d_measure_duration_samples);
    d_surv_buffer.resize(d_measure_duration_samples);
    d_phase_history.resize(num_channels);
    for (int i = 0; i < num_channels; i++) {
        d_phase_history[i].resize(16, 0.0f);  // Keep last 16 phase measurements
    }
    
    // Register message ports
    message_port_register_out(pmt::mp("cal_request"));
    message_port_register_in(pmt::mp("cal_complete"));
    set_msg_handler(pmt::mp("cal_complete"),
        [this](const pmt::pmt_t& msg) { this->handle_cal_complete(msg); });
}

coherence_monitor_impl::~coherence_monitor_impl()
{
}

void coherence_monitor_impl::handle_cal_complete(const pmt::pmt_t& msg)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_calibration_in_progress = false;
    // Force immediate measurement to verify calibration
    d_sample_count = d_measure_interval_samples;
}

bool coherence_monitor_impl::is_calibration_needed() const
{
    return d_calibration_needed;
}

float coherence_monitor_impl::get_correlation(int channel) const
{
    if (channel >= 0 && channel < d_num_channels) {
        return d_correlation[channel];
    }
    return 0.0f;
}

float coherence_monitor_impl::get_phase_offset(int channel) const
{
    if (channel >= 0 && channel < d_num_channels) {
        return d_phase_offset[channel];
    }
    return 0.0f;
}

float coherence_monitor_impl::get_phase_variance(int channel) const
{
    if (channel >= 0 && channel < d_num_channels) {
        return d_phase_variance[channel];
    }
    return 0.0f;
}

void coherence_monitor_impl::set_measure_interval(float interval_ms)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_measure_interval_samples = static_cast<uint64_t>(interval_ms * d_sample_rate / 1000.0);
}

void coherence_monitor_impl::set_corr_threshold(float threshold)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_corr_threshold = threshold;
}

void coherence_monitor_impl::set_phase_threshold(float threshold_deg)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_phase_threshold_rad = threshold_deg * M_PI / 180.0;
}

void coherence_monitor_impl::request_calibration()
{
    gr::thread::scoped_lock lock(d_mutex);
    d_calibration_needed = true;
}

void coherence_monitor_impl::acknowledge_calibration()
{
    gr::thread::scoped_lock lock(d_mutex);
    d_calibration_in_progress = false;
    d_calibration_needed = false;
    d_consecutive_failures = 0;
}

float coherence_monitor_impl::compute_correlation(const gr_complex* ref,
                                                   const gr_complex* surv,
                                                   int length,
                                                   float& phase_out)
{
    // Compute cross-correlation coefficient:
    // rho = |sum(ref * conj(surv))| / sqrt(sum(|ref|²) * sum(|surv|²))
    
    gr_complex cross_sum(0.0f, 0.0f);
    float ref_power = 0.0f;
    float surv_power = 0.0f;
    
    for (int i = 0; i < length; i++) {
        cross_sum += ref[i] * std::conj(surv[i]);
        ref_power += std::norm(ref[i]);
        surv_power += std::norm(surv[i]);
    }
    
    float denom = std::sqrt(ref_power * surv_power);
    if (denom < 1e-10f) {
        phase_out = 0.0f;
        return 0.0f;
    }
    
    // Phase of cross-correlation gives relative phase offset
    phase_out = std::arg(cross_sum);
    
    // Magnitude of normalized cross-correlation is the coefficient
    return std::abs(cross_sum) / denom;
}

float coherence_monitor_impl::compute_phase_variance(const std::vector<float>& phases)
{
    if (phases.empty()) return 0.0f;
    
    // Circular variance for phase data
    float sum_cos = 0.0f;
    float sum_sin = 0.0f;
    
    for (float p : phases) {
        sum_cos += std::cos(p);
        sum_sin += std::sin(p);
    }
    
    float mean_cos = sum_cos / phases.size();
    float mean_sin = sum_sin / phases.size();
    float R = std::sqrt(mean_cos * mean_cos + mean_sin * mean_sin);
    
    // Circular variance = 1 - R, convert to standard deviation approx
    // For small variance: std ≈ sqrt(2(1-R))
    return std::sqrt(2.0f * (1.0f - R));
}

void coherence_monitor_impl::perform_measurement(const gr_complex* const* inputs, int offset)
{
    bool any_failure = false;
    
    // Reference channel is input 0
    const gr_complex* ref = inputs[0] + offset;
    
    // Check each surveillance channel against reference
    for (int ch = 1; ch < d_num_channels; ch++) {
        const gr_complex* surv = inputs[ch] + offset;
        
        // Compute correlation and phase
        float phase;
        float corr = compute_correlation(ref, surv, d_measure_duration_samples, phase);
        
        d_correlation[ch] = corr;
        d_phase_offset[ch] = phase;
        
        // Update phase history (circular buffer)
        int hist_idx = d_measure_count % d_phase_history[ch].size();
        d_phase_history[ch][hist_idx] = phase;
        
        // Compute phase variance from history
        d_phase_variance[ch] = compute_phase_variance(d_phase_history[ch]);
        
        // Check thresholds
        if (corr < d_corr_threshold) {
            GR_LOG_WARN(d_logger, 
                boost::format("Channel %d correlation %.3f below threshold %.3f") 
                % ch % corr % d_corr_threshold);
            any_failure = true;
        }
        
        if (d_phase_variance[ch] > d_phase_threshold_rad) {
            GR_LOG_WARN(d_logger,
                boost::format("Channel %d phase variance %.2f° exceeds threshold %.2f°")
                % ch % (d_phase_variance[ch] * 180.0 / M_PI) 
                % (d_phase_threshold_rad * 180.0 / M_PI));
            any_failure = true;
        }
    }
    
    // Reference channel (ch 0) always has perfect self-correlation
    d_correlation[0] = 1.0f;
    d_phase_offset[0] = 0.0f;
    d_phase_variance[0] = 0.0f;
    
    d_measure_count++;
    
    // Handle failures with hysteresis
    if (any_failure) {
        d_consecutive_failures++;
        
        // Require 3 consecutive failures to request calibration
        // Prevents spurious triggers from transient interference
        if (d_consecutive_failures >= 3 && !d_calibration_in_progress) {
            d_calibration_needed = true;
            d_calibration_in_progress = true;
            
            // Send calibration request message
            pmt::pmt_t msg = pmt::make_dict();
            msg = pmt::dict_add(msg, pmt::mp("type"), pmt::mp("cal_request"));
            msg = pmt::dict_add(msg, pmt::mp("reason"), pmt::mp("coherence_degraded"));
            msg = pmt::dict_add(msg, pmt::mp("measure_count"), pmt::mp(d_measure_count));
            message_port_pub(pmt::mp("cal_request"), msg);
            
            GR_LOG_INFO(d_logger, "Calibration requested due to coherence degradation");
        }
    } else {
        d_consecutive_failures = 0;
        if (d_calibration_in_progress) {
            // Calibration verified successful
            d_calibration_needed = false;
            d_calibration_in_progress = false;
            GR_LOG_INFO(d_logger, "Calibration verified successful");
        }
    }
}

int coherence_monitor_impl::work(int noutput_items,
                                  gr_vector_const_void_star &input_items,
                                  gr_vector_void_star &output_items)
{
    gr::thread::scoped_lock lock(d_mutex);
    
    // Build array of input pointers
    std::vector<const gr_complex*> inputs(d_num_channels);
    for (int i = 0; i < d_num_channels; i++) {
        inputs[i] = static_cast<const gr_complex*>(input_items[i]);
    }
    
    // Pass-through: copy inputs to outputs
    for (int ch = 0; ch < d_num_channels; ch++) {
        gr_complex* out = static_cast<gr_complex*>(output_items[ch]);
        memcpy(out, inputs[ch], noutput_items * sizeof(gr_complex));
    }
    
    // Check if it's time to measure
    int samples_processed = 0;
    while (samples_processed < noutput_items) {
        int samples_to_interval = d_measure_interval_samples - 
                                  (d_sample_count % d_measure_interval_samples);
        int samples_this_chunk = std::min(samples_to_interval, 
                                          noutput_items - samples_processed);
        
        d_sample_count += samples_this_chunk;
        samples_processed += samples_this_chunk;
        
        // Time to measure?
        if ((d_sample_count % d_measure_interval_samples) == 0) {
            // Check if we have enough samples for measurement
            int measure_offset = samples_processed - d_measure_duration_samples;
            if (measure_offset >= 0 && 
                measure_offset + d_measure_duration_samples <= noutput_items) {
                perform_measurement(inputs.data(), measure_offset);
            }
        }
    }
    
    return noutput_items;
}

} /* namespace kraken_passive_radar */
} /* namespace gr */
