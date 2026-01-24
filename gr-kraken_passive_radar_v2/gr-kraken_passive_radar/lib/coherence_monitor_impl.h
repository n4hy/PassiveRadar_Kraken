/*
 * Coherence Monitor Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_COHERENCE_MONITOR_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_COHERENCE_MONITOR_IMPL_H

#include <gnuradio/kraken_passive_radar/coherence_monitor.h>
#include <gnuradio/thread/thread.h>
#include <vector>

namespace gr {
namespace kraken_passive_radar {

class coherence_monitor_impl : public coherence_monitor
{
private:
    int d_num_channels;
    float d_sample_rate;
    uint64_t d_measure_interval_samples;
    int d_measure_duration_samples;
    float d_corr_threshold;
    float d_phase_threshold_rad;
    
    bool d_calibration_needed;
    bool d_calibration_in_progress;
    uint64_t d_sample_count;
    uint64_t d_measure_count;
    int d_consecutive_failures;
    
    // Per-channel statistics
    std::vector<float> d_correlation;
    std::vector<float> d_phase_offset;
    std::vector<float> d_phase_variance;
    std::vector<std::vector<float>> d_phase_history;
    
    // Measurement buffers
    std::vector<gr_complex> d_ref_buffer;
    std::vector<gr_complex> d_surv_buffer;
    
    gr::thread::mutex d_mutex;
    
    float compute_correlation(const gr_complex* ref, const gr_complex* surv, 
                              int length, float& phase_out);
    float compute_phase_variance(const std::vector<float>& phases);
    void perform_measurement(const gr_complex* const* inputs, int offset);
    void handle_cal_complete(const pmt::pmt_t& msg);

public:
    coherence_monitor_impl(int num_channels,
                           float sample_rate,
                           float measure_interval_ms,
                           float measure_duration_ms,
                           float corr_threshold,
                           float phase_threshold_deg);
    ~coherence_monitor_impl();

    bool is_calibration_needed() const override;
    float get_correlation(int channel) const override;
    float get_phase_offset(int channel) const override;
    float get_phase_variance(int channel) const override;
    
    void set_measure_interval(float interval_ms) override;
    void set_corr_threshold(float threshold) override;
    void set_phase_threshold(float threshold_deg) override;
    
    void request_calibration() override;
    void acknowledge_calibration() override;

    int work(int noutput_items,
             gr_vector_const_void_star &input_items,
             gr_vector_void_star &output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_COHERENCE_MONITOR_IMPL_H */
