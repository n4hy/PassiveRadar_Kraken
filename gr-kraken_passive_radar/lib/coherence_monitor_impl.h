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

/**
 * coherence_monitor_impl - Implementation of the phase coherence monitor block
 *
 * Technique: Periodic cross-correlation and phase variance measurement
 * between reference and surveillance channels with threshold-based
 * calibration triggering via GNU Radio message ports.
 */
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
    
    mutable gr::thread::mutex d_mutex;
    
    /**
     * compute_correlation - Compute normalized cross-correlation coefficient and phase between two signals
     */
    float compute_correlation(const gr_complex* ref, const gr_complex* surv,
                              int length, float& phase_out);

    /**
     * compute_phase_variance - Compute variance from a history of phase measurements
     */
    float compute_phase_variance(const std::vector<float>& phases);

    /**
     * perform_measurement - Execute one coherence measurement across all channels at a given offset
     */
    void perform_measurement(const gr_complex* const* inputs, int offset);

    /**
     * handle_cal_complete - Message handler invoked when calibration completion is signaled
     */
    void handle_cal_complete(const pmt::pmt_t& msg);

public:
    /**
     * coherence_monitor_impl - Construct coherence monitor with channel count and measurement parameters
     */
    coherence_monitor_impl(int num_channels,
                           float sample_rate,
                           float measure_interval_ms,
                           float measure_duration_ms,
                           float corr_threshold,
                           float phase_threshold_deg);
    ~coherence_monitor_impl();

    /** is_calibration_needed - Check if coherence has dropped below acceptable thresholds */
    bool is_calibration_needed() const override;
    /** get_correlation - Return current correlation coefficient for a channel */
    float get_correlation(int channel) const override;
    /** get_phase_offset - Return current phase offset for a channel */
    float get_phase_offset(int channel) const override;
    /** get_phase_variance - Return current phase variance for a channel */
    float get_phase_variance(int channel) const override;

    /** set_measure_interval - Update measurement interval in milliseconds */
    void set_measure_interval(float interval_ms) override;
    /** set_corr_threshold - Update minimum acceptable correlation coefficient */
    void set_corr_threshold(float threshold) override;
    /** set_phase_threshold - Update maximum acceptable phase std dev in degrees */
    void set_phase_threshold(float threshold_deg) override;

    /** request_calibration - Manually trigger a calibration request */
    void request_calibration() override;
    /** acknowledge_calibration - Signal that external calibration has completed */
    void acknowledge_calibration() override;

    /**
     * work - Process input samples, perform periodic coherence measurements, and output pass-through data
     */
    int work(int noutput_items,
             gr_vector_const_void_star &input_items,
             gr_vector_void_star &output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_COHERENCE_MONITOR_IMPL_H */
