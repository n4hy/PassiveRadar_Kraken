/*
 * Multi-Target Tracker Implementation Header (SRUKF)
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_TRACKER_IMPL_H

#include <gnuradio/kraken_passive_radar/tracker.h>
#include <gnuradio/thread/thread.h>
#include "radar_target_model.h"
#include "SRUKF.h"
#include <vector>
#include <memory>

namespace gr {
namespace kraken_passive_radar {

// Per-track SRUKF filter state
struct srukf_track_t {
    int id;
    track_status_t status;

    // The SRUKF filter for this track
    std::unique_ptr<RadarTargetModel> model;
    std::unique_ptr<UKFCore::SRUKF<RTM_NX, RTM_NY>> filter;

    // Track quality metrics
    int hits;
    int misses;
    int age;
    float score;

    // History for display (last N positions)
    static constexpr int MAX_HISTORY = 50;
    std::vector<std::array<float, 2>> history;  // [(range, doppler), ...]
};

class tracker_impl : public tracker
{
private:
    // Parameters
    float d_dt;
    float d_process_noise_range;
    float d_process_noise_doppler;
    float d_process_noise_turn;
    float d_meas_noise_range;
    float d_meas_noise_doppler;
    float d_meas_noise_aoa;
    float d_gate_threshold;
    int d_confirm_hits;
    int d_delete_misses;
    int d_max_tracks;
    int d_max_detections;

    // Track management
    std::vector<srukf_track_t> d_tracks;
    int d_next_track_id;
    int d_frame_count;

    // Working buffers
    std::vector<bool> d_det_used;
    std::vector<std::pair<int, int>> d_associations;  // (track_idx, det_idx)

    // Thread safety
    mutable gr::thread::mutex d_mutex;

    // SRUKF operations
    void predict_track(srukf_track_t& track);
    void update_track(srukf_track_t& track,
                      float range_m, float doppler_hz,
                      float aoa_deg, float aoa_confidence);
    float compute_distance(const srukf_track_t& track,
                          float range_m, float doppler_hz);

    // Data association
    void associate_detections(const float* detections, int num_dets);

    // Track lifecycle
    void create_track(float range_m, float doppler_hz,
                      float aoa_deg, float aoa_confidence);
    void delete_stale_tracks();
    void update_track_status(srukf_track_t& track, bool updated);

    // Convert srukf_track_t to public track_t
    track_t to_public_track(const srukf_track_t& t) const;

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
