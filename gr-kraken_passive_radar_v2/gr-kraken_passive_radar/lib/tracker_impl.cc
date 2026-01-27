/*
 * Multi-Target Tracker Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Kalman filter tracker with Global Nearest Neighbor (GNN) association.
 * State vector: [range, doppler, range_rate, doppler_rate]
 */

#include "tracker_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace gr {
namespace kraken_passive_radar {

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
                     // Input: detection list (10 floats per detection)
                     gr::io_signature::make(1, 1, max_detections * 10 * sizeof(float)),
                     // Output: track list (20 floats per track)
                     gr::io_signature::make(1, 1, max_tracks * 20 * sizeof(float))),
      d_dt(dt),
      d_process_noise_range(process_noise_range),
      d_process_noise_doppler(process_noise_doppler),
      d_meas_noise_range(meas_noise_range),
      d_meas_noise_doppler(meas_noise_doppler),
      d_gate_threshold(gate_threshold),
      d_confirm_hits(confirm_hits),
      d_delete_misses(delete_misses),
      d_max_tracks(max_tracks),
      d_max_detections(max_detections),
      d_next_track_id(1)
{
    d_tracks.reserve(max_tracks);
    d_det_used.resize(max_detections, false);
    d_associations.reserve(max_tracks);

    init_matrices();
}

tracker_impl::~tracker_impl() {}

void tracker_impl::init_matrices()
{
    // State transition matrix F (constant velocity model)
    // x[k+1] = F * x[k]
    // [r]     [1 0 dt 0 ] [r]
    // [d]   = [0 1 0  dt] [d]
    // [r']    [0 0 1  0 ] [r']
    // [d']    [0 0 0  1 ] [d']
    d_F = {
        1.0f, 0.0f, d_dt, 0.0f,
        0.0f, 1.0f, 0.0f, d_dt,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    // Measurement matrix H
    // z = H * x
    // [r_m]   [1 0 0 0] [r]
    // [d_m] = [0 1 0 0] [d]
    //                   [r']
    //                   [d']
    d_H = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    };

    // Process noise covariance Q
    // Discrete white noise acceleration model
    float dt2 = d_dt * d_dt;
    float dt3 = dt2 * d_dt;
    float dt4 = dt3 * d_dt;

    float q_r = d_process_noise_range * d_process_noise_range;
    float q_d = d_process_noise_doppler * d_process_noise_doppler;

    d_Q = {
        dt4/4*q_r, 0.0f,       dt3/2*q_r, 0.0f,
        0.0f,      dt4/4*q_d,  0.0f,      dt3/2*q_d,
        dt3/2*q_r, 0.0f,       dt2*q_r,   0.0f,
        0.0f,      dt3/2*q_d,  0.0f,      dt2*q_d
    };

    // Measurement noise covariance R
    d_R = {
        d_meas_noise_range * d_meas_noise_range, 0.0f,
        0.0f, d_meas_noise_doppler * d_meas_noise_doppler
    };
}

// Matrix operations
void tracker_impl::mat4x4_mult_vec4(const float* M, const float* v, float* out)
{
    for (int i = 0; i < 4; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < 4; j++) {
            out[i] += M[i*4 + j] * v[j];
        }
    }
}

void tracker_impl::mat4x4_mult_mat4x4(const float* A, const float* B, float* out)
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i*4 + j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                out[i*4 + j] += A[i*4 + k] * B[k*4 + j];
            }
        }
    }
}

void tracker_impl::mat4x4_transpose(const float* M, float* out)
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[j*4 + i] = M[i*4 + j];
        }
    }
}

void tracker_impl::mat4x4_add(const float* A, const float* B, float* out)
{
    for (int i = 0; i < 16; i++) {
        out[i] = A[i] + B[i];
    }
}

void tracker_impl::mat2x2_inverse(const float* M, float* out)
{
    float det = M[0] * M[3] - M[1] * M[2];
    if (std::abs(det) < 1e-10f) {
        // Near-singular, use pseudoinverse
        det = 1e-10f;
    }
    float inv_det = 1.0f / det;
    out[0] = M[3] * inv_det;
    out[1] = -M[1] * inv_det;
    out[2] = -M[2] * inv_det;
    out[3] = M[0] * inv_det;
}

void tracker_impl::mat2x4_mult_mat4x4(const float* A, const float* B, float* out)
{
    // A is 2x4, B is 4x4, result is 2x4
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            out[i*4 + j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                out[i*4 + j] += A[i*4 + k] * B[k*4 + j];
            }
        }
    }
}

void tracker_impl::mat4x2_mult_mat2x2(const float* A, const float* B, float* out)
{
    // A is 4x2, B is 2x2, result is 4x2
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            out[i*2 + j] = 0.0f;
            for (int k = 0; k < 2; k++) {
                out[i*2 + j] += A[i*2 + k] * B[k*2 + j];
            }
        }
    }
}

void tracker_impl::mat4x2_mult_mat2x4(const float* A, const float* B, float* out)
{
    // A is 4x2, B is 2x4, result is 4x4
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i*4 + j] = 0.0f;
            for (int k = 0; k < 2; k++) {
                out[i*4 + j] += A[i*2 + k] * B[k*4 + j];
            }
        }
    }
}

void tracker_impl::predict_track(track_t& track)
{
    // Predict state: x_pred = F * x
    std::array<float, 4> x_pred;
    mat4x4_mult_vec4(d_F.data(), track.state.data(), x_pred.data());
    track.state = x_pred;

    // Predict covariance: P_pred = F * P * F^T + Q
    std::array<float, 16> FP, FPFt;
    mat4x4_mult_mat4x4(d_F.data(), track.covariance.data(), FP.data());

    std::array<float, 16> Ft;
    mat4x4_transpose(d_F.data(), Ft.data());
    mat4x4_mult_mat4x4(FP.data(), Ft.data(), FPFt.data());

    mat4x4_add(FPFt.data(), d_Q.data(), track.covariance.data());

    track.age++;
}

void tracker_impl::update_track(track_t& track, float range_m, float doppler_hz)
{
    // Innovation: y = z - H * x
    float z[2] = {range_m, doppler_hz};
    float Hx[2] = {track.state[0], track.state[1]};  // H * x simplified
    float y[2] = {z[0] - Hx[0], z[1] - Hx[1]};

    // Innovation covariance: S = H * P * H^T + R
    // Since H = [I2 0], this simplifies to S = P[0:2,0:2] + R
    float S[4] = {
        track.covariance[0] + d_R[0], track.covariance[1] + d_R[1],
        track.covariance[4] + d_R[2], track.covariance[5] + d_R[3]
    };

    // Kalman gain: K = P * H^T * S^(-1)
    // H^T is 4x2: [[1,0],[0,1],[0,0],[0,0]]
    // P * H^T is 4x2: first two columns of P
    float PHt[8] = {
        track.covariance[0], track.covariance[1],
        track.covariance[4], track.covariance[5],
        track.covariance[8], track.covariance[9],
        track.covariance[12], track.covariance[13]
    };

    float S_inv[4];
    mat2x2_inverse(S, S_inv);

    float K[8];  // 4x2
    mat4x2_mult_mat2x2(PHt, S_inv, K);

    // Update state: x = x + K * y
    for (int i = 0; i < 4; i++) {
        track.state[i] += K[i*2 + 0] * y[0] + K[i*2 + 1] * y[1];
    }

    // Update covariance: P = (I - K * H) * P
    // K * H is 4x4
    float KH[16];
    // H is 2x4, K is 4x2
    mat4x2_mult_mat2x4(K, d_H.data(), KH);

    // I - K * H
    float ImKH[16];
    for (int i = 0; i < 16; i++) {
        ImKH[i] = -KH[i];
    }
    for (int i = 0; i < 4; i++) {
        ImKH[i*5] += 1.0f;  // Add identity
    }

    // P_new = (I - K*H) * P
    std::array<float, 16> P_new;
    mat4x4_mult_mat4x4(ImKH, track.covariance.data(), P_new.data());
    track.covariance = P_new;

    // Update track history
    if (track.history.size() >= track_t::MAX_HISTORY) {
        track.history.erase(track.history.begin());
    }
    track.history.push_back({track.state[0], track.state[1]});

    track.hits++;
    track.misses = 0;
}

float tracker_impl::compute_distance(const track_t& track, float range_m, float doppler_hz)
{
    // Mahalanobis distance using innovation covariance
    // d^2 = y^T * S^(-1) * y

    float y[2] = {range_m - track.state[0], doppler_hz - track.state[1]};

    // S = H * P * H^T + R (simplified as before)
    float S[4] = {
        track.covariance[0] + d_R[0], track.covariance[1] + d_R[1],
        track.covariance[4] + d_R[2], track.covariance[5] + d_R[3]
    };

    float S_inv[4];
    mat2x2_inverse(S, S_inv);

    // d^2 = y^T * S^(-1) * y
    float Sinv_y[2] = {
        S_inv[0] * y[0] + S_inv[1] * y[1],
        S_inv[2] * y[0] + S_inv[3] * y[1]
    };

    return y[0] * Sinv_y[0] + y[1] * Sinv_y[1];
}

void tracker_impl::associate_detections(const float* detections, int num_dets)
{
    d_associations.clear();
    std::fill(d_det_used.begin(), d_det_used.begin() + num_dets, false);

    // Greedy GNN: for each track, find nearest valid detection
    // More sophisticated methods (auction, Hungarian) possible but slower

    for (size_t t = 0; t < d_tracks.size(); t++) {
        float min_dist = std::numeric_limits<float>::max();
        int best_det = -1;

        for (int d = 0; d < num_dets; d++) {
            if (d_det_used[d]) continue;

            // Detection format: [id, range_bin, doppler_bin, range_m, doppler_hz, ...]
            float range_m = detections[d * 10 + 3];
            float doppler_hz = detections[d * 10 + 4];

            // Skip invalid detections (id < 0 means empty slot)
            if (range_m == 0.0f && doppler_hz == 0.0f) continue;

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

void tracker_impl::create_track(float range_m, float doppler_hz)
{
    if (d_tracks.size() >= static_cast<size_t>(d_max_tracks)) {
        return;
    }

    track_t track;
    track.id = d_next_track_id++;
    track.status = track_status_t::TENTATIVE;

    // Initialize state
    track.state = {range_m, doppler_hz, 0.0f, 0.0f};

    // Initialize covariance (large initial uncertainty for velocities)
    track.covariance = {
        d_meas_noise_range * d_meas_noise_range, 0.0f, 0.0f, 0.0f,
        0.0f, d_meas_noise_doppler * d_meas_noise_doppler, 0.0f, 0.0f,
        0.0f, 0.0f, 1000.0f, 0.0f,  // Large velocity uncertainty
        0.0f, 0.0f, 0.0f, 100.0f
    };

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
            [this](const track_t& t) {
                return t.misses >= d_delete_misses;
            }),
        d_tracks.end());
}

void tracker_impl::update_track_status(track_t& track, bool updated)
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

int tracker_impl::work(int noutput_items,
                       gr_vector_const_void_star& input_items,
                       gr_vector_void_star& output_items)
{
    const float* detections = static_cast<const float*>(input_items[0]);
    float* tracks_out = static_cast<float*>(output_items[0]);

    const int det_size = d_max_detections * 10;
    const int track_size = d_max_tracks * 20;

    for (int frame = 0; frame < noutput_items; frame++) {
        const float* frame_dets = detections + frame * det_size;
        float* frame_tracks = tracks_out + frame * track_size;

        // Clear output
        std::fill(frame_tracks, frame_tracks + track_size, 0.0f);

        // Count valid detections
        int num_dets = 0;
        for (int d = 0; d < d_max_detections; d++) {
            // Check if detection is valid (range_m > 0)
            if (frame_dets[d * 10 + 3] > 0.0f) {
                num_dets = d + 1;
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

                float range_m = frame_dets[d_idx * 10 + 3];
                float doppler_hz = frame_dets[d_idx * 10 + 4];

                update_track(d_tracks[t_idx], range_m, doppler_hz);
                track_updated[t_idx] = true;
            }

            // 4. Update track status for non-updated tracks
            for (size_t t = 0; t < d_tracks.size(); t++) {
                update_track_status(d_tracks[t], track_updated[t]);
            }

            // 5. Create new tracks for unassociated detections
            for (int d = 0; d < num_dets; d++) {
                if (!d_det_used[d]) {
                    float range_m = frame_dets[d * 10 + 3];
                    float doppler_hz = frame_dets[d * 10 + 4];
                    if (range_m > 0.0f) {
                        create_track(range_m, doppler_hz);
                    }
                }
            }

            // 6. Delete stale tracks
            delete_stale_tracks();

            // 7. Pack output
            // Output format per track (20 floats):
            // [id, status, range, doppler, range_rate, doppler_rate,
            //  cov[0], cov[5], cov[10], cov[15],  (diagonal)
            //  hits, misses, age, score,
            //  hist_len, hist[0].r, hist[0].d, hist[1].r, hist[1].d, hist[2].r]
            int out_idx = 0;
            for (const auto& track : d_tracks) {
                if (out_idx >= d_max_tracks) break;

                int base = out_idx * 20;
                frame_tracks[base + 0] = static_cast<float>(track.id);
                frame_tracks[base + 1] = static_cast<float>(track.status);
                frame_tracks[base + 2] = track.state[0];
                frame_tracks[base + 3] = track.state[1];
                frame_tracks[base + 4] = track.state[2];
                frame_tracks[base + 5] = track.state[3];
                frame_tracks[base + 6] = track.covariance[0];
                frame_tracks[base + 7] = track.covariance[5];
                frame_tracks[base + 8] = track.covariance[10];
                frame_tracks[base + 9] = track.covariance[15];
                frame_tracks[base + 10] = static_cast<float>(track.hits);
                frame_tracks[base + 11] = static_cast<float>(track.misses);
                frame_tracks[base + 12] = static_cast<float>(track.age);
                frame_tracks[base + 13] = track.score;

                // Recent history (up to 2 points to fit within 20 floats per track)
                // Fields 15-18 = 4 floats for 2 history points (range, doppler each)
                // Field 19 is unused padding
                int hist_len = std::min(static_cast<int>(track.history.size()), 2);
                frame_tracks[base + 14] = static_cast<float>(hist_len);
                for (int h = 0; h < hist_len && (base + 15 + h*2 + 1) < (out_idx + 1) * 20; h++) {
                    int hist_idx = track.history.size() - hist_len + h;
                    frame_tracks[base + 15 + h*2] = track.history[hist_idx][0];
                    frame_tracks[base + 16 + h*2] = track.history[hist_idx][1];
                }
                // Zero padding for remaining slots
                for (int h = hist_len; h < 2; h++) {
                    frame_tracks[base + 15 + h*2] = 0.0f;
                    frame_tracks[base + 16 + h*2] = 0.0f;
                }

                out_idx++;
            }
        }
    }

    return noutput_items;
}

void tracker_impl::set_process_noise(float range, float doppler)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_process_noise_range = range;
    d_process_noise_doppler = doppler;
    init_matrices();  // Recalculate Q
}

void tracker_impl::set_measurement_noise(float range, float doppler)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_meas_noise_range = range;
    d_meas_noise_doppler = doppler;
    d_R = {
        range * range, 0.0f,
        0.0f, doppler * doppler
    };
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
    return d_tracks;
}

std::vector<track_t> tracker_impl::get_confirmed_tracks() const
{
    gr::thread::scoped_lock lock(d_mutex);
    std::vector<track_t> confirmed;
    for (const auto& t : d_tracks) {
        if (t.status == track_status_t::CONFIRMED) {
            confirmed.push_back(t);
        }
    }
    return confirmed;
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
