/*
 * Multi-Target Tracker Implementation (EKF)
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Extended Kalman Filter tracker with
 * Global Nearest Neighbor (GNN) association.
 *
 * State vector: [range, doppler, range_rate, doppler_rate, turn_rate]
 * Measurement:  [range, doppler, aoa_deg]
 *
 * Constant-velocity state transition with decoupled turn_rate.
 * Linear measurement model (range and doppler observed directly).
 * AoA handled adaptively: gated on confidence, high noise when unreliable.
 */

#include "tracker_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace gr {
namespace kraken_passive_radar {

// Pad float count to next multiple of 1024 (= 4096 bytes) for buffer alignment
static inline int pad4k(int n) { return ((n + 1023) & ~1023); }

// Detection stride: 12 floats per AoA-augmented detection
static constexpr int DET_STRIDE = 12;

// AoA confidence threshold: below this, AoA measurement is not used
static constexpr float AOA_CONFIDENCE_THRESHOLD = 0.3f;

// Very large AoA noise variance when AoA is unreliable
static constexpr float AOA_UNINFORMATIVE_VAR = 1e6f;

tracker::sptr tracker::make(float dt,
                            float process_noise_range,
                            float process_noise_doppler,
                            float meas_noise_range,
                            float meas_noise_doppler,
                            float gate_threshold,
                            int confirm_hits,
                            int delete_misses,
                            int max_tracks,
                            int max_detections)
{
    return gnuradio::make_block_sptr<tracker_impl>(
        dt, process_noise_range, process_noise_doppler,
        meas_noise_range, meas_noise_doppler,
        gate_threshold, confirm_hits, delete_misses,
        max_tracks, max_detections);
}

tracker_impl::tracker_impl(float dt,
                           float process_noise_range,
                           float process_noise_doppler,
                           float meas_noise_range,
                           float meas_noise_doppler,
                           float gate_threshold,
                           int confirm_hits,
                           int delete_misses,
                           int max_tracks,
                           int max_detections)
    : gr::sync_block("tracker",
                     gr::io_signature::make(1, 1, pad4k(max_detections * DET_STRIDE) * sizeof(float)),
                     gr::io_signature::make(1, 1, pad4k(max_tracks * 20) * sizeof(float))),
      d_dt(dt),
      d_meas_noise_range(meas_noise_range),
      d_meas_noise_doppler(meas_noise_doppler),
      d_meas_noise_aoa(10.0f),
      d_gate_threshold(gate_threshold),
      d_confirm_hits(confirm_hits),
      d_delete_misses(delete_misses),
      d_max_tracks(max_tracks),
      d_max_detections(max_detections),
      d_next_track_id(1),
      d_frame_count(0)
{
    d_tracks.reserve(max_tracks);
    d_det_used.resize(max_detections, false);
    d_associations.reserve(max_tracks);

    // Precompute constant state transition matrix F (CV model)
    // x[k+1] = F * x[k]
    d_F = Eigen::Matrix<float, NX, NX>::Identity();
    d_F(0, 2) = dt;  // range += range_rate * dt
    d_F(1, 3) = dt;  // doppler += doppler_rate * dt

    // Measurement matrix H: observe range and doppler directly
    // AoA row is zero (handled via adaptive noise)
    d_H = Eigen::Matrix<float, NY, NX>::Zero();
    d_H(0, 0) = 1.0f;  // range
    d_H(1, 1) = 1.0f;  // doppler
    // H(2,:) = 0: AoA not a function of state

    // Measurement noise variances
    d_R_range = meas_noise_range * meas_noise_range;
    d_R_doppler = meas_noise_doppler * meas_noise_doppler;

    // Process noise
    build_process_noise(process_noise_range, process_noise_doppler, 0.01f);
}

tracker_impl::~tracker_impl() {}

void tracker_impl::build_process_noise(float q_range, float q_doppler, float q_turn)
{
    // Discrete white noise acceleration model
    float dt = d_dt;
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;

    float qr = q_range * q_range;
    float qd = q_doppler * q_doppler;
    float qt = q_turn * q_turn;

    d_Q = Eigen::Matrix<float, NX, NX>::Zero();

    // Range block (states 0, 2)
    d_Q(0, 0) = dt4 / 4.0f * qr;
    d_Q(0, 2) = dt3 / 2.0f * qr;
    d_Q(2, 0) = dt3 / 2.0f * qr;
    d_Q(2, 2) = dt2 * qr;

    // Doppler block (states 1, 3)
    d_Q(1, 1) = dt4 / 4.0f * qd;
    d_Q(1, 3) = dt3 / 2.0f * qd;
    d_Q(3, 1) = dt3 / 2.0f * qd;
    d_Q(3, 3) = dt2 * qd;

    // Turn rate (state 4): random walk
    d_Q(4, 4) = qt * dt;
}

void tracker_impl::predict_track(ekf_track_t& track)
{
    // EKF predict: x = F*x, P = F*P*F' + Q
    track.x = d_F * track.x;
    track.P = d_F * track.P * d_F.transpose() + d_Q;
    track.age++;
}

void tracker_impl::update_track(ekf_track_t& track,
                                float range_m, float doppler_hz,
                                float aoa_deg, float aoa_confidence)
{
    // Build measurement vector
    Eigen::Matrix<float, NY, 1> z;
    z(0) = range_m;
    z(1) = doppler_hz;
    z(2) = aoa_deg;

    // Build R (measurement noise covariance) with adaptive AoA
    Eigen::Matrix<float, NY, NY> R = Eigen::Matrix<float, NY, NY>::Zero();
    R(0, 0) = d_R_range;
    R(1, 1) = d_R_doppler;
    if (aoa_confidence < AOA_CONFIDENCE_THRESHOLD) {
        R(2, 2) = AOA_UNINFORMATIVE_VAR;
    } else {
        float aoa_sigma = d_meas_noise_aoa / std::max(aoa_confidence, 0.1f);
        R(2, 2) = aoa_sigma * aoa_sigma;
    }

    // Innovation
    Eigen::Matrix<float, NY, 1> y = z - d_H * track.x;

    // Innovation covariance: S = H*P*H' + R
    Eigen::Matrix<float, NY, NY> S = d_H * track.P * d_H.transpose() + R;

    // Kalman gain: K = P*H' * S^{-1}
    Eigen::Matrix<float, NX, NY> K = track.P * d_H.transpose() * S.inverse();

    // State update
    track.x = track.x + K * y;

    // Covariance update: P = (I - K*H)*P
    Eigen::Matrix<float, NX, NX> I_KH =
        Eigen::Matrix<float, NX, NX>::Identity() - K * d_H;
    track.P = I_KH * track.P;

    // Ensure symmetry
    track.P = (track.P + track.P.transpose()) * 0.5f;

    // Update history
    if (static_cast<int>(track.history.size()) >= ekf_track_t::MAX_HISTORY) {
        track.history.erase(track.history.begin());
    }
    track.history.push_back({track.x(0), track.x(1)});

    track.hits++;
    track.misses = 0;
}

float tracker_impl::compute_distance(const ekf_track_t& track,
                                     float range_m, float doppler_hz) const
{
    // Mahalanobis distance in range-Doppler (2D gating)
    float y0 = range_m - track.x(0);
    float y1 = doppler_hz - track.x(1);

    // S = P[0:2,0:2] + R[0:2,0:2]
    float S00 = track.P(0, 0) + d_R_range;
    float S01 = track.P(0, 1);
    float S10 = track.P(1, 0);
    float S11 = track.P(1, 1) + d_R_doppler;

    // 2x2 inverse
    float det = S00 * S11 - S01 * S10;
    if (std::abs(det) < 1e-10f) {
        det = (det >= 0) ? 1e-10f : -1e-10f;
    }
    float inv_det = 1.0f / det;

    float Sy0 = (S11 * y0 - S01 * y1) * inv_det;
    float Sy1 = (-S10 * y0 + S00 * y1) * inv_det;

    return y0 * Sy0 + y1 * Sy1;
}

void tracker_impl::associate_detections(const float* detections, int num_dets)
{
    d_associations.clear();
    std::fill(d_det_used.begin(), d_det_used.begin() + num_dets, false);

    // Greedy GNN: for each track, find nearest valid detection
    for (size_t t = 0; t < d_tracks.size(); t++) {
        float min_dist = std::numeric_limits<float>::max();
        int best_det = -1;

        for (int d = 0; d < num_dets; d++) {
            if (d_det_used[d]) continue;

            float range_m = detections[d * DET_STRIDE + 3];
            float doppler_hz = detections[d * DET_STRIDE + 4];

            if (range_m <= 0.0f) continue;

            float dist = compute_distance(d_tracks[t], range_m, doppler_hz);

            if (dist < d_gate_threshold && dist < min_dist) {
                min_dist = dist;
                best_det = d;
            }
        }

        if (best_det >= 0) {
            d_associations.emplace_back(t, best_det);
            d_det_used[best_det] = true;
        }
    }
}

void tracker_impl::create_track(float range_m, float doppler_hz,
                                float aoa_deg, float aoa_confidence)
{
    if (d_tracks.size() >= static_cast<size_t>(d_max_tracks)) {
        return;
    }

    ekf_track_t track;
    track.id = d_next_track_id++;
    track.status = track_status_t::TENTATIVE;

    // Initialize state
    track.x = Eigen::Matrix<float, NX, 1>::Zero();
    track.x(0) = range_m;
    track.x(1) = doppler_hz;
    // range_rate, doppler_rate, turn_rate = 0

    // Initial covariance
    track.P = Eigen::Matrix<float, NX, NX>::Zero();
    track.P(0, 0) = d_R_range;       // known from measurement
    track.P(1, 1) = d_R_doppler;     // known from measurement
    track.P(2, 2) = 1000.0f;         // large range rate uncertainty
    track.P(3, 3) = 100.0f;          // large doppler rate uncertainty
    track.P(4, 4) = 0.1f;            // small turn rate uncertainty

    track.r_aoa = (aoa_confidence >= AOA_CONFIDENCE_THRESHOLD)
        ? d_meas_noise_aoa * d_meas_noise_aoa
        : AOA_UNINFORMATIVE_VAR;

    track.hits = 1;
    track.misses = 0;
    track.age = 0;
    track.score = 1.0f;
    track.history.push_back({range_m, doppler_hz});

    d_tracks.push_back(std::move(track));
}

void tracker_impl::delete_stale_tracks()
{
    d_tracks.erase(
        std::remove_if(d_tracks.begin(), d_tracks.end(),
            [this](const ekf_track_t& t) {
                return t.misses >= d_delete_misses;
            }),
        d_tracks.end());
}

void tracker_impl::update_track_status(ekf_track_t& track, bool updated)
{
    if (updated) {
        if (track.status == track_status_t::TENTATIVE &&
            track.hits >= d_confirm_hits) {
            track.status = track_status_t::CONFIRMED;
        } else if (track.status == track_status_t::COASTING) {
            track.status = track_status_t::CONFIRMED;
        }
        track.score = std::min(track.score + 0.1f, 1.0f);
    } else {
        track.misses++;
        if (track.status == track_status_t::CONFIRMED) {
            track.status = track_status_t::COASTING;
        }
        track.score = std::max(track.score - 0.2f, 0.0f);
    }
}

track_t tracker_impl::to_public_track(const ekf_track_t& t) const
{
    track_t pub;
    pub.id = t.id;
    pub.status = t.status;

    for (int i = 0; i < NX; i++) {
        pub.state[i] = t.x(i);
    }

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NX; j++) {
            pub.covariance[i * NX + j] = t.P(i, j);
        }
    }

    pub.hits = t.hits;
    pub.misses = t.misses;
    pub.age = t.age;
    pub.score = t.score;
    pub.history = t.history;

    return pub;
}

int tracker_impl::work(int noutput_items,
                       gr_vector_const_void_star& input_items,
                       gr_vector_void_star& output_items)
{
    const float* detections = static_cast<const float*>(input_items[0]);
    float* tracks_out = static_cast<float*>(output_items[0]);

    const int det_size = pad4k(d_max_detections * DET_STRIDE);
    const int track_size = pad4k(d_max_tracks * 20);

    for (int frame = 0; frame < noutput_items; frame++) {
        const float* frame_dets = detections + frame * det_size;
        float* frame_tracks = tracks_out + frame * track_size;

        // Clear output
        std::fill(frame_tracks, frame_tracks + track_size, 0.0f);

        // Count valid detections
        int num_dets = 0;
        for (int d = 0; d < d_max_detections; d++) {
            if (frame_dets[d * DET_STRIDE + 3] > 0.0f) {
                num_dets++;
            } else {
                break;
            }
        }

        {
            gr::thread::scoped_lock lock(d_mutex);

            // 1. Predict all tracks
            for (auto& track : d_tracks) {
                predict_track(track);
            }

            // 2. Associate detections to tracks
            associate_detections(frame_dets, num_dets);

            // 3. Update associated tracks
            std::vector<bool> track_updated(d_tracks.size(), false);
            for (const auto& assoc : d_associations) {
                int t_idx = assoc.first;
                int d_idx = assoc.second;

                float range_m = frame_dets[d_idx * DET_STRIDE + 3];
                float doppler_hz = frame_dets[d_idx * DET_STRIDE + 4];
                float aoa_deg = frame_dets[d_idx * DET_STRIDE + 10];
                float aoa_conf = frame_dets[d_idx * DET_STRIDE + 11];

                update_track(d_tracks[t_idx], range_m, doppler_hz,
                            aoa_deg, aoa_conf);
                track_updated[t_idx] = true;
            }

            // 4. Update track status
            for (size_t t = 0; t < d_tracks.size(); t++) {
                update_track_status(d_tracks[t], track_updated[t]);
            }

            // 5. Create new tracks for unassociated detections
            for (int d = 0; d < num_dets; d++) {
                if (!d_det_used[d]) {
                    float range_m = frame_dets[d * DET_STRIDE + 3];
                    float doppler_hz = frame_dets[d * DET_STRIDE + 4];
                    float aoa_deg = frame_dets[d * DET_STRIDE + 10];
                    float aoa_conf = frame_dets[d * DET_STRIDE + 11];
                    if (range_m > 0.0f) {
                        create_track(range_m, doppler_hz, aoa_deg, aoa_conf);
                    }
                }
            }

            // 6. Delete stale tracks
            delete_stale_tracks();

            // 7. Pack output (20 floats/track)
            int out_idx = 0;
            for (const auto& track : d_tracks) {
                if (out_idx >= d_max_tracks) break;

                int base = out_idx * 20;
                frame_tracks[base + 0] = static_cast<float>(track.id);
                frame_tracks[base + 1] = static_cast<float>(track.status);
                frame_tracks[base + 2] = track.x(0);   // range
                frame_tracks[base + 3] = track.x(1);   // doppler
                frame_tracks[base + 4] = track.x(2);   // range_rate
                frame_tracks[base + 5] = track.x(3);   // doppler_rate
                frame_tracks[base + 6] = track.P(0, 0);
                frame_tracks[base + 7] = track.P(1, 1);
                frame_tracks[base + 8] = track.P(2, 2);
                frame_tracks[base + 9] = track.P(3, 3);
                frame_tracks[base + 10] = static_cast<float>(track.hits);
                frame_tracks[base + 11] = static_cast<float>(track.misses);
                frame_tracks[base + 12] = static_cast<float>(track.age);
                frame_tracks[base + 13] = track.score;

                int hist_len = std::min(static_cast<int>(track.history.size()), 2);
                frame_tracks[base + 14] = static_cast<float>(hist_len);
                for (int h = 0; h < hist_len; h++) {
                    int hist_idx = track.history.size() - hist_len + h;
                    frame_tracks[base + 15 + h * 2] = track.history[hist_idx][0];
                    frame_tracks[base + 16 + h * 2] = track.history[hist_idx][1];
                }

                frame_tracks[base + 19] = track.x(4);  // turn_rate

                out_idx++;
            }
        }
    }

    return noutput_items;
}

void tracker_impl::set_process_noise(float range, float doppler)
{
    gr::thread::scoped_lock lock(d_mutex);
    build_process_noise(range, doppler, 0.01f);
}

void tracker_impl::set_measurement_noise(float range, float doppler)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_meas_noise_range = range;
    d_meas_noise_doppler = doppler;
    d_R_range = range * range;
    d_R_doppler = doppler * doppler;
}

void tracker_impl::set_gate_threshold(float threshold)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_gate_threshold = threshold;
}

void tracker_impl::set_confirm_hits(int hits)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_confirm_hits = std::max(1, hits);
}

void tracker_impl::set_delete_misses(int misses)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_delete_misses = std::max(1, misses);
}

std::vector<track_t> tracker_impl::get_tracks() const
{
    gr::thread::scoped_lock lock(d_mutex);
    std::vector<track_t> result;
    result.reserve(d_tracks.size());
    for (const auto& t : d_tracks) {
        result.push_back(to_public_track(t));
    }
    return result;
}

std::vector<track_t> tracker_impl::get_confirmed_tracks() const
{
    gr::thread::scoped_lock lock(d_mutex);
    std::vector<track_t> result;
    for (const auto& t : d_tracks) {
        if (t.status == track_status_t::CONFIRMED) {
            result.push_back(to_public_track(t));
        }
    }
    return result;
}

int tracker_impl::get_num_tracks() const
{
    gr::thread::scoped_lock lock(d_mutex);
    return static_cast<int>(d_tracks.size());
}

int tracker_impl::get_num_confirmed_tracks() const
{
    gr::thread::scoped_lock lock(d_mutex);
    int count = 0;
    for (const auto& t : d_tracks) {
        if (t.status == track_status_t::CONFIRMED) count++;
    }
    return count;
}

void tracker_impl::reset()
{
    gr::thread::scoped_lock lock(d_mutex);
    d_tracks.clear();
    d_next_track_id = 1;
}

} // namespace kraken_passive_radar
} // namespace gr
