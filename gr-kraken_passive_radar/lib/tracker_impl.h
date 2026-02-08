/*
 * Multi-Target Tracker Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_IMPL_H

#include <gnuradio/kraken_passive_radar/tracker.h>
#include <gnuradio/thread/thread.h>
#include <vector>
#include <array>

namespace gr {
namespace kraken_passive_radar {

class tracker_impl : public tracker
{
private:
    // Parameters
    float d_dt;
    float d_process_noise_range;
    float d_process_noise_doppler;
    float d_meas_noise_range;
    float d_meas_noise_doppler;
    float d_gate_threshold;
    int d_confirm_hits;
    int d_delete_misses;
    int d_max_tracks;
    int d_max_detections;

    // Track management
    std::vector<track_t> d_tracks;
    int d_next_track_id;

    // Kalman filter matrices (constant)
    std::array<float, 16> d_F;  // State transition (4x4)
    std::array<float, 8> d_H;   // Measurement (2x4)
    std::array<float, 16> d_Q;  // Process noise covariance (4x4)
    std::array<float, 4> d_R;   // Measurement noise covariance (2x2)

    // Working buffers
    std::vector<bool> d_det_used;
    std::vector<std::pair<int, int>> d_associations;  // (track_idx, det_idx)

    // Thread safety
    mutable gr::thread::mutex d_mutex;

    // Matrix operations (inline for speed)
    void mat4x4_mult_vec4(const float* M, const float* v, float* out);
    void mat4x4_mult_mat4x4(const float* A, const float* B, float* out);
    void mat4x4_transpose(const float* M, float* out);
    void mat4x4_add(const float* A, const float* B, float* out);
    void mat2x2_inverse(const float* M, float* out);
    void mat2x4_mult_mat4x4(const float* A, const float* B, float* out);
    void mat4x2_mult_mat2x2(const float* A, const float* B, float* out);
    void mat4x2_mult_mat2x4(const float* A, const float* B, float* out);

    // Kalman filter operations
    void predict_track(track_t& track);
    void update_track(track_t& track, float range_m, float doppler_hz);
    float compute_distance(const track_t& track, float range_m, float doppler_hz);

    // Data association
    void associate_detections(const float* detections, int num_dets);

    // Track lifecycle
    void create_track(float range_m, float doppler_hz);
    void delete_stale_tracks();
    void update_track_status(track_t& track, bool updated);

    // Initialize matrices
    void init_matrices();

public:
    tracker_impl(float dt,
                 float process_noise_range,
                 float process_noise_doppler,
                 float meas_noise_range,
                 float meas_noise_doppler,
                 float gate_threshold,
                 int confirm_hits,
                 int delete_misses,
                 int max_tracks,
                 int max_detections);
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

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_IMPL_H */
