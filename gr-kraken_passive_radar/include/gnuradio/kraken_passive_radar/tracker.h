/*
 * Multi-Target Tracker Block for KrakenSDR Passive Radar
 *
 * Kalman filter tracker with Global Nearest Neighbor (GNN)
 * data association for tracking multiple targets.
 *
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_H

#include <gnuradio/sync_block.h>
#include <gnuradio/kraken_passive_radar/api.h>
#include <vector>
#include <array>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief Track status enumeration
 */
enum class track_status_t {
    TENTATIVE = 0,   // New track, needs confirmation
    CONFIRMED = 1,   // Confirmed track with sufficient hits
    COASTING = 2     // No measurement, predicting forward
};

/*!
 * \brief Single track state structure
 */
struct track_t {
    int id;                         // Unique track ID
    track_status_t status;          // Track status

    // State vector: [range_m, doppler_hz, range_rate, doppler_rate]
    std::array<float, 4> state;

    // State covariance (4x4, stored row-major)
    std::array<float, 16> covariance;

    // Track quality metrics
    int hits;                       // Total measurement updates
    int misses;                     // Consecutive prediction-only cycles
    int age;                        // Total frames since creation
    float score;                    // Track quality score

    // History for display (last N positions)
    static constexpr int MAX_HISTORY = 50;
    std::vector<std::array<float, 2>> history;  // [(range, doppler), ...]
};

/*!
 * \brief Multi-target tracker for passive radar
 * \ingroup kraken_passive_radar
 *
 * Implements Extended Kalman Filter (linear motion model) with
 * Global Nearest Neighbor (GNN) data association.
 *
 * State model: constant velocity in range and Doppler
 *   x = [range, doppler, range_rate, doppler_rate]^T
 *   x[k+1] = F * x[k] + w, where w ~ N(0, Q)
 *
 * Measurement model:
 *   z = [range, doppler]^T = H * x + v, where v ~ N(0, R)
 *
 * Track lifecycle:
 *   TENTATIVE -> CONFIRMED after confirm_hits consecutive updates
 *   CONFIRMED -> COASTING after one miss
 *   COASTING -> deleted after delete_misses consecutive misses
 *
 * Input: Detection list from clustering block
 * Output: Track list as packed float vector
 */
class KRAKEN_PASSIVE_RADAR_API tracker : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<tracker> sptr;

    /*!
     * \brief Create multi-target tracker
     * \param dt Frame period in seconds
     * \param process_noise_range Process noise std dev for range (m)
     * \param process_noise_doppler Process noise std dev for Doppler (Hz)
     * \param meas_noise_range Measurement noise std dev for range (m)
     * \param meas_noise_doppler Measurement noise std dev for Doppler (Hz)
     * \param gate_threshold Chi-squared gate threshold for association
     * \param confirm_hits Hits required to confirm tentative track
     * \param delete_misses Misses before track deletion
     * \param max_tracks Maximum simultaneous tracks
     * \param max_detections Maximum detections per frame (input size)
     */
    static sptr make(float dt = 0.1f,
                     float process_noise_range = 50.0f,
                     float process_noise_doppler = 5.0f,
                     float meas_noise_range = 100.0f,
                     float meas_noise_doppler = 2.0f,
                     float gate_threshold = 9.21f,  // chi2(2) @ 99%
                     int confirm_hits = 3,
                     int delete_misses = 5,
                     int max_tracks = 50,
                     int max_detections = 100);

    virtual void set_process_noise(float range, float doppler) = 0;
    virtual void set_measurement_noise(float range, float doppler) = 0;
    virtual void set_gate_threshold(float threshold) = 0;
    virtual void set_confirm_hits(int hits) = 0;
    virtual void set_delete_misses(int misses) = 0;

    // Get tracks
    virtual std::vector<track_t> get_tracks() const = 0;
    virtual std::vector<track_t> get_confirmed_tracks() const = 0;
    virtual int get_num_tracks() const = 0;
    virtual int get_num_confirmed_tracks() const = 0;

    // Reset tracker state
    virtual void reset() = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_H */
