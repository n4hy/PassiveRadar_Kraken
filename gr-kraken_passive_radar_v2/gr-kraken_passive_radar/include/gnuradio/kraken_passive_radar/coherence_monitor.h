/*
 * Coherence Monitor Block for KrakenSDR Passive Radar
 * 
 * Monitors phase coherence between channels and triggers recalibration
 * when coherence degrades. Uses periodic measurement to minimize overhead.
 *
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_COHERENCE_MONITOR_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_COHERENCE_MONITOR_H

#include <gnuradio/sync_block.h>
#include <gnuradio/kraken_passive_radar/api.h>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief Coherence monitor for KrakenSDR phase calibration
 * \ingroup kraken_passive_radar
 *
 * Monitors inter-channel coherence by computing:
 * 1. Cross-correlation coefficient between reference and each surveillance channel
 * 2. Phase variance across measurement windows
 * 
 * When coherence drops below threshold OR phase variance exceeds limit:
 * - Sets calibration_needed flag
 * - Outputs message on "cal_request" port
 * - After receiving "cal_complete" message, verifies calibration success
 *
 * Measurement is done periodically (not every sample) to minimize CPU load.
 * Default: measure every 1 second using 10ms of samples.
 *
 * Calibration criteria:
 * - Correlation coefficient > corr_threshold (default 0.95)
 * - Phase standard deviation < phase_threshold (default 5 degrees)
 */
class KRAKEN_PASSIVE_RADAR_API coherence_monitor : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<coherence_monitor> sptr;

    /*!
     * \brief Create coherence monitor
     * \param num_channels Number of input channels (typically 5)
     * \param sample_rate Sample rate in Hz
     * \param measure_interval_ms Interval between measurements in milliseconds
     * \param measure_duration_ms Duration of each measurement in milliseconds  
     * \param corr_threshold Minimum acceptable correlation coefficient
     * \param phase_threshold_deg Maximum acceptable phase std dev in degrees
     */
    static sptr make(int num_channels = 5,
                     float sample_rate = 2.4e6,
                     float measure_interval_ms = 1000.0,
                     float measure_duration_ms = 10.0,
                     float corr_threshold = 0.95,
                     float phase_threshold_deg = 5.0);

    // Getters for current coherence state
    virtual bool is_calibration_needed() const = 0;
    virtual float get_correlation(int channel) const = 0;
    virtual float get_phase_offset(int channel) const = 0;
    virtual float get_phase_variance(int channel) const = 0;
    
    // Runtime parameter adjustment
    virtual void set_measure_interval(float interval_ms) = 0;
    virtual void set_corr_threshold(float threshold) = 0;
    virtual void set_phase_threshold(float threshold_deg) = 0;
    
    // Manual calibration trigger/acknowledge
    virtual void request_calibration() = 0;
    virtual void acknowledge_calibration() = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_COHERENCE_MONITOR_H */
