/*
 * AoA Estimator Block for KrakenSDR Passive Radar
 *
 * Angle-of-Arrival estimation using Bartlett beamformer
 * on 4-element surveillance array.
 *
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_AOA_ESTIMATOR_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_AOA_ESTIMATOR_H

#include <gnuradio/sync_block.h>
#include <gnuradio/kraken_passive_radar/api.h>
#include <vector>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief Array type enumeration
 */
enum class array_type_t {
    ULA = 0,    // Uniform Linear Array
    UCA = 1     // Uniform Circular Array
};

/*!
 * \brief AoA estimation result
 */
struct aoa_result_t {
    int detection_id;
    float aoa_deg;              // Angle of arrival in degrees
    float aoa_confidence;       // Confidence/quality metric (0-1)
    float spectrum_peak;        // Peak value of angular spectrum
    float spectrum_width_deg;   // 3dB width of peak
};

/*!
 * \brief Angle-of-Arrival Estimator for passive radar
 * \ingroup kraken_passive_radar
 *
 * Computes AoA for each detection using Bartlett beamforming
 * on the 4 surveillance channels of KrakenSDR.
 *
 * The KrakenSDR has 5 channels:
 *   - Channel 0: Reference (connected to illuminator)
 *   - Channels 1-4: Surveillance array (4 elements)
 *
 * For each detection, extracts complex samples at the detection
 * range/Doppler bin from each surveillance channel's CAF output,
 * then applies Bartlett beamforming to estimate AoA.
 *
 * Bartlett spectrum: P(theta) = |a(theta)^H * x|^2
 * where a(theta) is the steering vector and x is the array response.
 *
 * Supports both ULA and UCA array configurations.
 */
class KRAKEN_PASSIVE_RADAR_API aoa_estimator : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<aoa_estimator> sptr;

    /*!
     * \brief Create AoA estimator block
     * \param num_elements Number of array elements (typically 4)
     * \param d_lambda Element spacing in wavelengths (default 0.5)
     * \param n_angles Number of angles in scan (resolution)
     * \param min_angle_deg Minimum scan angle (degrees)
     * \param max_angle_deg Maximum scan angle (degrees)
     * \param array_type ULA (0) or UCA (1)
     * \param num_range_bins Range bins in CAF
     * \param num_doppler_bins Doppler bins in CAF
     * \param max_detections Maximum detections per frame
     */
    static sptr make(int num_elements = 4,
                     float d_lambda = 0.5f,
                     int n_angles = 181,
                     float min_angle_deg = -90.0f,
                     float max_angle_deg = 90.0f,
                     int array_type = 0,
                     int num_range_bins = 256,
                     int num_doppler_bins = 64,
                     int max_detections = 100);

    virtual void set_d_lambda(float d_lambda) = 0;
    virtual void set_scan_range(float min_deg, float max_deg) = 0;
    virtual void set_array_type(int type) = 0;

    // Get AoA results from last frame
    virtual std::vector<aoa_result_t> get_aoa_results() const = 0;

    // Get full angular spectrum for debugging
    virtual std::vector<float> get_spectrum() const = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_AOA_ESTIMATOR_H */
