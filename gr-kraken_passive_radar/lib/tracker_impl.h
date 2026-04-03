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

    void compute_weights();
    void build_process_noise(float q_range, float q_doppler, float q_turn);

    // Coordinated-turn state transition
    StateVec f(const StateVec& x) const;
    // Direct measurement
    MeasVec h(const StateVec& x) const;

    // Cholesky rank-1 update/downdate (in-place on lower-triangular S)
    template<int N>
    static void cholupdate(Eigen::Matrix<float,N,N>& S,
                           Eigen::Matrix<float,N,1> v, float sign);

    void predict_track(srukf_track_t& track);
    void update_track(srukf_track_t& track,
                      float range_m, float doppler_hz,
                      float aoa_deg, float aoa_confidence);
    float compute_distance(const srukf_track_t& track,
                          float range_m, float doppler_hz) const;
    void associate_detections(const float* detections, int num_dets);
    void create_track(float range_m, float doppler_hz,
                     float aoa_deg, float aoa_confidence);
    void delete_stale_tracks();
    void update_track_status(srukf_track_t& track, bool updated);
    track_t to_public_track(const srukf_track_t& t) const;

    // RTS smoother for post-processing refinement
    void save_rts_history(srukf_track_t& track);
    void run_rts_smoother(srukf_track_t& track);

    // Adaptive process noise
    void update_maneuver_indicator(srukf_track_t& track, const MeasVec& innovation);
    StateMat get_adaptive_sqrt_Q(const srukf_track_t& track) const;

public:
    tracker_impl(float dt, float process_noise_range, float process_noise_doppler,
                 float meas_noise_range, float meas_noise_doppler,
                 float gate_threshold, int confirm_hits, int delete_misses,
                 int max_tracks, int max_detections);
    ~tracker_impl();

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    void set_process_noise(float range, float doppler) override;
    void set_measurement_noise(float range, float doppler) override;
    void set_gate_threshold(float threshold) override;
    void set_confirm_hits(int hits) override;
    void set_delete_misses(int misses) override;

    std::vector<track_t> get_tracks() const override;
    std::vector<track_t> get_confirmed_tracks() const override;
    int get_num_tracks() const override;
    int get_num_confirmed_tracks() const override;
    void reset() override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif
