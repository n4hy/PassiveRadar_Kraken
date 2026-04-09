/*
 * SRUKF Multi-Target Tracker Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Square Root Unscented Kalman Filter with coordinated-turn model.
 * State: [range, doppler, range_rate, doppler_rate, turn_rate] (5D)
 * Measurement: [range, doppler, aoa_deg] (3D)
 * All Eigen matrices pre-allocated — zero heap allocation in work().
 *
 * April 2026 Enhancements:
 * - GPU acceleration hook for batch UKF operations (CUDA 11.8+)
 * - RTS smoother capability for post-processing
 * - Adaptive process noise based on maneuver detection
 * - Joseph-form covariance update for enhanced numerical stability
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_IMPL_H

#include <gnuradio/kraken_passive_radar/tracker.h>
#include <gnuradio/thread/thread.h>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <deque>

namespace gr {
namespace kraken_passive_radar {

static constexpr int NSIGMA = 2 * NX + 1;  // 11 sigma points

using StateVec = Eigen::Matrix<float, NX, 1>;
using StateMat = Eigen::Matrix<float, NX, NX>;
using MeasVec  = Eigen::Matrix<float, NY, 1>;
using MeasMat  = Eigen::Matrix<float, NY, NY>;
using GainMat  = Eigen::Matrix<float, NX, NY>;

/**
 * srukf_track_t - Internal track state for the Square Root UKF tracker
 *
 * Stores the state estimate, sqrt covariance factor, track lifecycle
 * metadata, RTS smoother history, and adaptive noise parameters.
 */
struct srukf_track_t {
    int id;
    track_status_t status;
    StateVec x;         // state estimate
    StateMat S;         // lower-triangular sqrt of P (P = S*S^T)
    float r_aoa;
    int hits, misses, age;
    float score;
    std::vector<std::array<float, 2>> history;
    static constexpr int MAX_HISTORY = 50;

    // RTS smoother history (for post-processing refinement)
    static constexpr int MAX_RTS_HISTORY = 20;
    std::deque<StateVec> rts_x_history;    // State history for smoothing
    std::deque<StateMat> rts_S_history;    // Covariance sqrt history

    // Adaptive noise parameters
    float maneuver_indicator;              // Innovation-based maneuver detection
    float adaptive_q_scale;                // Process noise scaling factor
};

/**
 * tracker_impl - Implementation of the multi-target SRUKF tracker block
 *
 * Technique: Square Root Unscented Kalman Filter with coordinated-turn
 * dynamics model, Global Nearest Neighbor data association, Cholesky
 * rank-1 updates, RTS smoother for post-processing, and adaptive
 * process noise based on innovation-driven maneuver detection.
 */
class tracker_impl : public tracker
{
private:
    float d_dt;
    float d_meas_noise_range, d_meas_noise_doppler, d_meas_noise_aoa;
    float d_R_range, d_R_doppler;
    float d_gate_threshold;
    int d_confirm_hits, d_delete_misses, d_max_tracks, d_max_detections;
    int d_next_track_id, d_frame_count;

    // Process noise
    StateMat d_Q;
    StateMat d_sqrt_Q;  // Cholesky of Q

    // UKF weights (α=1, β=2, κ=0)
    float d_gamma;  // sqrt(NX + lambda)
    float d_wm0, d_wc0, d_wmi, d_wci;

    // Tracks
    std::vector<srukf_track_t> d_tracks;
    std::vector<bool> d_det_used;
    std::vector<std::pair<int,int>> d_associations;

    // Pre-allocated SRUKF scratch (shared across all tracks, used under mutex)
    Eigen::Matrix<float, NX, NSIGMA> d_chi;       // sigma points
    Eigen::Matrix<float, NX, NSIGMA> d_chi_pred;  // propagated sigma points
    Eigen::Matrix<float, NY, NSIGMA> d_zeta;      // measurement sigma points
    // QR compound matrix: max rows = 2*NX + NX = 15 (predict) or 2*NX + NY = 13 (update)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> d_qr_buf;

    mutable gr::thread::mutex d_mutex;

    /** compute_weights - Compute UKF sigma point weights from alpha, beta, kappa parameters */
    void compute_weights();
    /** build_process_noise - Construct and Cholesky-factor the process noise covariance matrix Q */
    void build_process_noise(float q_range, float q_doppler, float q_turn);

    /** f - Coordinated-turn state transition function for the process model */
    StateVec f(const StateVec& x) const;
    /** h - Direct measurement function mapping state to observation space */
    MeasVec h(const StateVec& x) const;

    /**
     * cholupdate - Cholesky rank-1 update or downdate in-place on lower-triangular factor S
     *
     * Technique: Modifies S such that S*S^T += sign * v*v^T
     */
    template<int N>
    static void cholupdate(Eigen::Matrix<float,N,N>& S,
                           Eigen::Matrix<float,N,1> v, float sign);

    /** predict_track - Propagate track state forward one time step using SRUKF predict */
    void predict_track(srukf_track_t& track);
    /** update_track - Incorporate a measurement into the track using SRUKF update */
    void update_track(srukf_track_t& track,
                      float range_m, float doppler_hz,
                      float aoa_deg, float aoa_confidence);
    /** compute_distance - Compute statistical distance between a track prediction and a measurement */
    float compute_distance(const srukf_track_t& track,
                          float range_m, float doppler_hz) const;
    /** associate_detections - Assign detections to tracks using Global Nearest Neighbor */
    void associate_detections(const float* detections, int num_dets);
    /** create_track - Initialize a new tentative track from an unassociated detection */
    void create_track(float range_m, float doppler_hz,
                     float aoa_deg, float aoa_confidence);
    /** delete_stale_tracks - Remove tracks that have exceeded the maximum consecutive miss count */
    void delete_stale_tracks();
    /** update_track_status - Transition track lifecycle state based on hit/miss outcome */
    void update_track_status(srukf_track_t& track, bool updated);
    /** to_public_track - Convert internal srukf_track_t to the public track_t structure */
    track_t to_public_track(const srukf_track_t& t) const;

    /** save_rts_history - Store current state and covariance for RTS smoother backpass */
    void save_rts_history(srukf_track_t& track);
    /** run_rts_smoother - Apply Rauch-Tung-Striebel backward smoothing pass over stored history */
    void run_rts_smoother(srukf_track_t& track);

    /** update_maneuver_indicator - Detect target maneuvers from innovation magnitude */
    void update_maneuver_indicator(srukf_track_t& track, const MeasVec& innovation);
    /** get_adaptive_sqrt_Q - Return scaled process noise sqrt factor based on maneuver indicator */
    StateMat get_adaptive_sqrt_Q(const srukf_track_t& track) const;

public:
    /**
     * tracker_impl - Construct SRUKF tracker with timing, noise, gating, and lifecycle parameters
     */
    tracker_impl(float dt, float process_noise_range, float process_noise_doppler,
                 float meas_noise_range, float meas_noise_doppler,
                 float gate_threshold, int confirm_hits, int delete_misses,
                 int max_tracks, int max_detections);
    ~tracker_impl();

    /**
     * work - Run full tracking cycle: predict, associate, update, create, and prune tracks
     */
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    /** set_process_noise - Update process noise standard deviations and rebuild Q matrix */
    void set_process_noise(float range, float doppler) override;
    /** set_measurement_noise - Update measurement noise standard deviations */
    void set_measurement_noise(float range, float doppler) override;
    /** set_gate_threshold - Update chi-squared gate threshold for association */
    void set_gate_threshold(float threshold) override;
    /** set_confirm_hits - Update hits required to confirm a tentative track */
    void set_confirm_hits(int hits) override;
    /** set_delete_misses - Update consecutive misses before track deletion */
    void set_delete_misses(int misses) override;

    /** get_tracks - Return all active tracks converted to public format */
    std::vector<track_t> get_tracks() const override;
    /** get_confirmed_tracks - Return only confirmed tracks in public format */
    std::vector<track_t> get_confirmed_tracks() const override;
    /** get_num_tracks - Return total number of active tracks */
    int get_num_tracks() const override;
    /** get_num_confirmed_tracks - Return number of confirmed tracks */
    int get_num_confirmed_tracks() const override;
    /** reset - Clear all tracks and reset tracker to initial state */
    void reset() override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif
