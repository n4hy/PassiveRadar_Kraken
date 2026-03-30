/*
 * Multi-Target Tracker Implementation (SRUKF)
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Square Root Unscented Kalman Filter tracker with
 * Global Nearest Neighbor (GNN) association.
 *
 * State vector: [range, doppler, range_rate, doppler_rate, turn_rate]
 * Measurement:  [range, doppler, aoa_deg]
 *
 * The SRUKF propagates the Cholesky factor of the covariance,
 * guaranteeing positive definiteness and providing better numerical
 * stability than the standard UKF or linear Kalman filter.
 *
 * The coordinated-turn model allows tracking maneuvering targets
 * that change direction, unlike constant-velocity linear filters.
 */

#include "tracker_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>

namespace gr {
namespace kraken_passive_radar {

// Pad float count to next multiple of 1024 (= 4096 bytes) for buffer alignment
static inline int pad4k(int n) { return ((n + 1023) & ~1023); }

// Detection stride: 12 floats per AoA-augmented detection
// [id, range_bin, doppler_bin, range_m, doppler_hz, snr_db,
//  power_sum, cluster_size, peak_range, peak_doppler, aoa_deg, aoa_confidence]
static constexpr int DET_STRIDE = 12;

// AoA confidence threshold: below this, AoA measurement is not used
static constexpr float AOA_CONFIDENCE_THRESHOLD = 0.3f;

// Very large AoA noise variance when AoA is unreliable (effectively ignores it)
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
                     // Input: AoA-augmented detection list (padded to 4096-byte boundary)
                     gr::io_signature::make(1, 1, pad4k(max_detections * DET_STRIDE) * sizeof(float)),
                     // Output: track list (padded to 4096-byte boundary)
                     gr::io_signature::make(1, 1, pad4k(max_tracks * 20) * sizeof(float))),
      d_dt(dt),
      d_process_noise_range(process_noise_range),
      d_process_noise_doppler(process_noise_doppler),
      d_process_noise_turn(0.01f),       // Small turn rate process noise
      d_meas_noise_range(meas_noise_range),
      d_meas_noise_doppler(meas_noise_doppler),
      d_meas_noise_aoa(10.0f),           // 10 degrees default AoA noise
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
}

tracker_impl::~tracker_impl() {}

void tracker_impl::predict_track(srukf_track_t& track)
{
    // Zero control input
    Eigen::Matrix<float, RTM_NX, 1> u = Eigen::Matrix<float, RTM_NX, 1>::Zero();
    track.filter->predict(0.0f, u);
    track.age++;
}

void tracker_impl::update_track(srukf_track_t& track,
                                float range_m, float doppler_hz,
                                float aoa_deg, float aoa_confidence)
{
    // Build measurement vector
    Eigen::Matrix<float, RTM_NY, 1> z;
    z(0) = range_m;
    z(1) = doppler_hz;
    z(2) = aoa_deg;

    // If AoA confidence is low, inflate AoA noise to make it uninformative
    if (aoa_confidence < AOA_CONFIDENCE_THRESHOLD) {
        track.model->set_measurement_noise_aoa(std::sqrt(AOA_UNINFORMATIVE_VAR));
    } else {
        // Scale AoA noise inversely with confidence
        float aoa_sigma = d_meas_noise_aoa / std::max(aoa_confidence, 0.1f);
        track.model->set_measurement_noise_aoa(aoa_sigma);
    }

    track.filter->update(0.0f, z);

    // Update track history
    const auto& state = track.filter->getState();
    if (static_cast<int>(track.history.size()) >= srukf_track_t::MAX_HISTORY) {
        track.history.erase(track.history.begin());
    }
    track.history.push_back({state(0), state(1)});

    track.hits++;
    track.misses = 0;
}

float tracker_impl::compute_distance(const srukf_track_t& track,
                                     float range_m, float doppler_hz)
{
    // Mahalanobis distance using SRUKF predicted state and covariance
    const auto& x = track.filter->getState();
    auto P = track.filter->getCovariance();

    // Innovation in range-Doppler only (2D sub-problem for gating)
    float y0 = range_m - x(0);
    float y1 = doppler_hz - x(1);

    // S = P[0:2, 0:2] + R[0:2, 0:2]  (2x2 innovation covariance)
    float S00 = P(0, 0) + d_meas_noise_range * d_meas_noise_range;
    float S01 = P(0, 1);
    float S10 = P(1, 0);
    float S11 = P(1, 1) + d_meas_noise_doppler * d_meas_noise_doppler;

    // 2x2 inverse
    float det = S00 * S11 - S01 * S10;
    if (std::abs(det) < 1e-10f) {
        det = (det >= 0) ? 1e-10f : -1e-10f;
    }
    float inv_det = 1.0f / det;

    float Si00 = S11 * inv_det;
    float Si01 = -S01 * inv_det;
    float Si10 = -S10 * inv_det;
    float Si11 = S00 * inv_det;

    // d^2 = y^T * S^{-1} * y
    float Sy0 = Si00 * y0 + Si01 * y1;
    float Sy1 = Si10 * y0 + Si11 * y1;

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

            // Skip invalid detections
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

    srukf_track_t track;
    track.id = d_next_track_id++;
    track.status = track_status_t::TENTATIVE;

    // Create model and filter for this track
    track.model = std::make_unique<RadarTargetModel>(
        d_dt,
        d_process_noise_range,
        d_process_noise_doppler,
        d_process_noise_turn,
        d_meas_noise_range,
        d_meas_noise_doppler,
        d_meas_noise_aoa
    );

    track.filter = std::make_unique<UKFCore::SRUKF<RTM_NX, RTM_NY>>(*track.model);

    // Initialize state
    Eigen::Matrix<float, RTM_NX, 1> x0;
    x0(0) = range_m;
    x0(1) = doppler_hz;
    x0(2) = 0.0f;    // unknown range rate
    x0(3) = 0.0f;    // unknown doppler rate
    x0(4) = 0.0f;    // unknown turn rate (assume straight initially)

    // Initial covariance: known position, large velocity/turn uncertainty
    Eigen::Matrix<float, RTM_NX, RTM_NX> P0 = Eigen::Matrix<float, RTM_NX, RTM_NX>::Zero();
    P0(0, 0) = d_meas_noise_range * d_meas_noise_range;
    P0(1, 1) = d_meas_noise_doppler * d_meas_noise_doppler;
    P0(2, 2) = 1000.0f;   // Large range rate uncertainty
    P0(3, 3) = 100.0f;    // Large doppler rate uncertainty
    P0(4, 4) = 0.1f;      // Small turn rate uncertainty (expect ~straight)

    track.filter->initialize(x0, P0);

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
            [this](const srukf_track_t& t) {
                return t.misses >= d_delete_misses;
            }),
        d_tracks.end());
}

void tracker_impl::update_track_status(srukf_track_t& track, bool updated)
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

track_t tracker_impl::to_public_track(const srukf_track_t& t) const
{
    track_t pub;
    pub.id = t.id;
    pub.status = t.status;

    const auto& x = t.filter->getState();
    auto P = t.filter->getCovariance();

    // Map 5-state to public 5-state array
    for (int i = 0; i < NX; i++) {
        pub.state[i] = x(i);
    }

    // Map NX x NX covariance (row-major)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NX; j++) {
            pub.covariance[i * NX + j] = P(i, j);
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

        // Count valid detections: packed contiguously, stop at first empty slot
        int num_dets = 0;
        for (int d = 0; d < d_max_detections; d++) {
            if (frame_dets[d * DET_STRIDE + 3] > 0.0f) {
                num_dets++;
            } else {
                break;  // detections are packed, first gap means end
            }
        }

        {
            gr::thread::scoped_lock lock(d_mutex);

            // 1. Predict all tracks
            for (auto& track : d_tracks) {
                predict_track(track);
            }

            // 2. Associate detections to tracks (range-Doppler gating)
            associate_detections(frame_dets, num_dets);

            // Diagnostic: log every ~244 frames (~1 second at 244 fps)
            d_frame_count++;
            if (d_frame_count % 244 == 1 && num_dets > 0 && !d_tracks.empty()) {
                fprintf(stderr, "[TRACKER] frame=%d dets=%d tracks=%zu assoc=%zu\n",
                        d_frame_count, num_dets, d_tracks.size(), d_associations.size());
                // Show first track state and nearest detection distance
                const auto& t0 = d_tracks[0];
                const auto& x0 = t0.filter->getState();
                auto P0 = t0.filter->getCovariance();
                fprintf(stderr, "  Track[0] id=%d hits=%d miss=%d state=[%.1f, %.1f] P_diag=[%.1f, %.1f]\n",
                        t0.id, t0.hits, t0.misses, x0(0), x0(1), P0(0,0), P0(1,1));
                // Show first detection
                float dr = frame_dets[0 * DET_STRIDE + 3];
                float dd = frame_dets[0 * DET_STRIDE + 4];
                float dist = compute_distance(t0, dr, dd);
                fprintf(stderr, "  Det[0] range=%.1f doppler=%.1f mahal_dist=%.3f gate=%.2f\n",
                        dr, dd, dist, d_gate_threshold);
            }

            // 3. Update associated tracks (with AoA when confident)
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

            // 4. Update track status for non-updated tracks
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

            // 7. Pack output (backward-compatible 20 floats/track format)
            // [id, status, range, doppler, range_rate, doppler_rate,
            //  cov[0,0], cov[1,1], cov[2,2], cov[3,3],
            //  hits, misses, age, score,
            //  hist_len, hist[0].r, hist[0].d, hist[1].r, hist[1].d, turn_rate]
            int out_idx = 0;
            for (const auto& track : d_tracks) {
                if (out_idx >= d_max_tracks) break;

                const auto& x = track.filter->getState();
                auto P = track.filter->getCovariance();

                int base = out_idx * 20;
                frame_tracks[base + 0] = static_cast<float>(track.id);
                frame_tracks[base + 1] = static_cast<float>(track.status);
                frame_tracks[base + 2] = x(0);   // range
                frame_tracks[base + 3] = x(1);   // doppler
                frame_tracks[base + 4] = x(2);   // range_rate
                frame_tracks[base + 5] = x(3);   // doppler_rate
                frame_tracks[base + 6] = P(0, 0);  // range variance
                frame_tracks[base + 7] = P(1, 1);  // doppler variance
                frame_tracks[base + 8] = P(2, 2);  // range_rate variance
                frame_tracks[base + 9] = P(3, 3);  // doppler_rate variance
                frame_tracks[base + 10] = static_cast<float>(track.hits);
                frame_tracks[base + 11] = static_cast<float>(track.misses);
                frame_tracks[base + 12] = static_cast<float>(track.age);
                frame_tracks[base + 13] = track.score;

                // History (up to 2 points to fit in 20 floats)
                int hist_len = std::min(static_cast<int>(track.history.size()), 2);
                frame_tracks[base + 14] = static_cast<float>(hist_len);
                for (int h = 0; h < hist_len; h++) {
                    int hist_idx = track.history.size() - hist_len + h;
                    frame_tracks[base + 15 + h * 2] = track.history[hist_idx][0];
                    frame_tracks[base + 16 + h * 2] = track.history[hist_idx][1];
                }

                // Slot 19: turn rate (new field, was padding)
                frame_tracks[base + 19] = x(4);  // turn_rate

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
    for (auto& track : d_tracks) {
        track.model->set_process_noise(range, doppler);
    }
}

void tracker_impl::set_measurement_noise(float range, float doppler)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_meas_noise_range = range;
    d_meas_noise_doppler = doppler;
    for (auto& track : d_tracks) {
        track.model->set_measurement_noise(range, doppler);
    }
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
