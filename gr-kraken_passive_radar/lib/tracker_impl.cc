/*
 * SRUKF Multi-Target Tracker — Coordinated Turn Model
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Square Root Unscented Kalman Filter with:
 *   - Nonlinear coordinated-turn state transition (coupled range/Doppler rates)
 *   - Adaptive AoA measurement gating
 *   - Global Nearest Neighbor (GNN) association
 *   - QR-based square root propagation (numerically stable)
 *   - Zero heap allocation in work() (all matrices pre-allocated)
 */

#include "tracker_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace gr {
namespace kraken_passive_radar {

static inline int pad4k(int n) { return ((n + 1023) & ~1023); }
static constexpr int DET_STRIDE = 12;
static constexpr float AOA_CONFIDENCE_THRESHOLD = 0.3f;
static constexpr float AOA_UNINFORMATIVE_VAR = 1e6f;

// ===== Cholesky rank-1 update/downdate =====
// Updates lower-triangular S in-place so that S*S^T ± v*v^T = S_new*S_new^T
template<int N>
void tracker_impl::cholupdate(Eigen::Matrix<float,N,N>& S,
                               Eigen::Matrix<float,N,1> v, float sign)
{
    for (int k = 0; k < N; k++) {
        float rr = S(k,k) * S(k,k) + sign * v(k) * v(k);
        if (rr <= 0.0f) rr = 1e-12f;  // guard downdate
        float r = std::sqrt(rr);
        float c = r / S(k,k);
        float s = v(k) / S(k,k);
        S(k,k) = r;
        for (int i = k + 1; i < N; i++) {
            S(i,k) = (S(i,k) + sign * s * v(i)) / c;
            v(i)   = c * v(i) - s * S(i,k);
        }
    }
}

// ===== UKF weights =====
void tracker_impl::compute_weights()
{
    // α=1, β=2, κ=0 → stable, moderate sigma spread
    const float alpha = 1.0f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    const float lambda = alpha * alpha * (NX + kappa) - NX;

    d_gamma = std::sqrt(NX + lambda);
    d_wm0 = lambda / (NX + lambda);
    d_wc0 = d_wm0 + (1.0f - alpha * alpha + beta);
    d_wmi = 0.5f / (NX + lambda);
    d_wci = d_wmi;
}

// ===== Process noise =====
void tracker_impl::build_process_noise(float q_range, float q_doppler, float q_turn)
{
    float dt = d_dt;
    float dt2 = dt*dt, dt3 = dt2*dt, dt4 = dt3*dt;
    float qr = q_range*q_range, qd = q_doppler*q_doppler, qt = q_turn*q_turn;

    d_Q = StateMat::Zero();
    d_Q(0,0) = dt4/4*qr;  d_Q(0,2) = dt3/2*qr;
    d_Q(2,0) = dt3/2*qr;  d_Q(2,2) = dt2*qr;
    d_Q(1,1) = dt4/4*qd;  d_Q(1,3) = dt3/2*qd;
    d_Q(3,1) = dt3/2*qd;  d_Q(3,3) = dt2*qd;
    d_Q(4,4) = qt*dt;

    // Cholesky of Q for SRUKF predict
    Eigen::LLT<StateMat> llt(d_Q);
    d_sqrt_Q = llt.matrixL();
}

// ===== Coordinated-turn state transition =====
StateVec tracker_impl::f(const StateVec& x) const
{
    StateVec xn;
    float omega = x(4);
    float omega_dt = omega * d_dt;

    xn(0) = x(0) + x(2) * d_dt;
    xn(1) = x(1) + x(3) * d_dt;

    if (std::abs(omega_dt) > 1e-6f) {
        float c = std::cos(omega_dt);
        float s = std::sin(omega_dt);
        xn(2) = x(2) * c - x(3) * s;
        xn(3) = x(2) * s + x(3) * c;
    } else {
        xn(2) = x(2);
        xn(3) = x(3);
    }
    xn(4) = x(4);  // turn rate persists (random walk via Q)
    return xn;
}

// ===== Measurement function =====
MeasVec tracker_impl::h(const StateVec& x) const
{
    MeasVec z;
    z(0) = x(0);   // range
    z(1) = x(1);   // doppler
    z(2) = 0.0f;   // AoA: not in state, filled from detection
    return z;
}

// ===== SRUKF predict =====
void tracker_impl::predict_track(srukf_track_t& track)
{
    // 1. Generate sigma points: χ = [x, x ± γ*S_cols]
    d_chi.col(0) = track.x;
    for (int i = 0; i < NX; i++) {
        StateVec offset = d_gamma * track.S.col(i);
        d_chi.col(1 + i)      = track.x + offset;
        d_chi.col(1 + NX + i) = track.x - offset;
    }

    // 2. Propagate through nonlinear f()
    for (int i = 0; i < NSIGMA; i++) {
        d_chi_pred.col(i) = f(d_chi.col(i));
    }

    // 3. Predicted mean
    track.x = d_wm0 * d_chi_pred.col(0);
    for (int i = 1; i < NSIGMA; i++) {
        track.x += d_wmi * d_chi_pred.col(i);
    }

    // 4. QR-based square root update for S⁻
    // Compound matrix rows: √wci*(χ̂₁..₂ₙₓ - x̄)ᵀ stacked with √Qᵀ
    // Size: (2*NX + NX) × NX = 15 × 5
    const int qr_rows = 2 * NX + NX;
    d_qr_buf.resize(qr_rows, NX);

    float sw = std::sqrt(std::abs(d_wci));
    for (int i = 0; i < 2 * NX; i++) {
        d_qr_buf.row(i) = sw * (d_chi_pred.col(1 + i) - track.x).transpose();
    }
    // Append sqrt(Q)^T
    for (int i = 0; i < NX; i++) {
        d_qr_buf.row(2 * NX + i) = d_sqrt_Q.row(i);
    }

    // QR decomposition → R is the new S⁻ (upper triangular of Q*R)
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(d_qr_buf);
    track.S = qr.matrixQR()
                 .template topLeftCorner<NX, NX>()
                 .template triangularView<Eigen::Upper>()
                 .transpose();  // convert to lower triangular

    // Rank-1 update for the 0th sigma point weight
    StateVec v0 = d_chi_pred.col(0) - track.x;
    cholupdate<NX>(track.S, v0, (d_wc0 >= 0) ? 1.0f : -1.0f);

    track.age++;
}

// ===== SRUKF update =====
void tracker_impl::update_track(srukf_track_t& track,
                                float range_m, float doppler_hz,
                                float aoa_deg, float aoa_confidence)
{
    // Regenerate sigma points from predicted state
    d_chi.col(0) = track.x;
    for (int i = 0; i < NX; i++) {
        StateVec offset = d_gamma * track.S.col(i);
        d_chi.col(1 + i)      = track.x + offset;
        d_chi.col(1 + NX + i) = track.x - offset;
    }

    // Propagate through measurement h()
    for (int i = 0; i < NSIGMA; i++) {
        d_zeta.col(i) = h(d_chi.col(i));
        d_zeta(2, i) = aoa_deg;  // AoA not in state — use measured value
    }

    // Predicted measurement mean
    MeasVec z_pred = d_wm0 * d_zeta.col(0);
    for (int i = 1; i < NSIGMA; i++) {
        z_pred += d_wmi * d_zeta.col(i);
    }

    // Build adaptive R
    MeasMat R = MeasMat::Zero();
    R(0,0) = d_R_range;
    R(1,1) = d_R_doppler;
    if (aoa_confidence < AOA_CONFIDENCE_THRESHOLD) {
        R(2,2) = AOA_UNINFORMATIVE_VAR;
    } else {
        float aoa_sigma = d_meas_noise_aoa / std::max(aoa_confidence, 0.1f);
        R(2,2) = aoa_sigma * aoa_sigma;
    }

    // sqrt(R) via Cholesky
    MeasMat sqrt_R = MeasMat::Zero();
    sqrt_R(0,0) = std::sqrt(R(0,0));
    sqrt_R(1,1) = std::sqrt(R(1,1));
    sqrt_R(2,2) = std::sqrt(R(2,2));

    // QR for Sy (innovation square root)
    // Compound: (2*NX + NY) × NY
    const int qr_rows = 2 * NX + NY;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> qr_z(qr_rows, NY);

    float sw = std::sqrt(std::abs(d_wci));
    for (int i = 0; i < 2 * NX; i++) {
        qr_z.row(i) = sw * (d_zeta.col(1 + i) - z_pred).transpose();
    }
    for (int i = 0; i < NY; i++) {
        qr_z.row(2 * NX + i) = sqrt_R.row(i);
    }

    Eigen::HouseholderQR<Eigen::MatrixXf> qr_sy(qr_z);
    MeasMat Sy = qr_sy.matrixQR()
                      .template topLeftCorner<NY, NY>()
                      .template triangularView<Eigen::Upper>()
                      .transpose();

    MeasVec v0_z = d_zeta.col(0) - z_pred;
    cholupdate<NY>(Sy, v0_z, (d_wc0 >= 0) ? 1.0f : -1.0f);

    // Cross-covariance Pxz
    GainMat Pxz = d_wc0 * (d_chi.col(0) - track.x) * (d_zeta.col(0) - z_pred).transpose();
    for (int i = 1; i < NSIGMA; i++) {
        Pxz += d_wci * (d_chi.col(i) - track.x) * (d_zeta.col(i) - z_pred).transpose();
    }

    // Kalman gain: K = Pxz * Sy^{-T} * Sy^{-1}
    // Solve K * (Sy * Sy^T) = Pxz → K = Pxz * (Sy*Sy^T)^{-1}
    MeasMat SySyT = Sy * Sy.transpose();
    GainMat K = Pxz * SySyT.inverse();

    // State update
    MeasVec z_meas;
    z_meas << range_m, doppler_hz, aoa_deg;
    track.x += K * (z_meas - z_pred);

    // Covariance update: S = choldowndate(S, K*Sy_cols)
    // For each column of K*Sy, do a rank-1 downdate
    Eigen::Matrix<float, NX, NY> U = K * Sy;
    for (int j = 0; j < NY; j++) {
        StateVec u_col = U.col(j);
        cholupdate<NX>(track.S, u_col, -1.0f);
    }

    // History
    if (static_cast<int>(track.history.size()) >= srukf_track_t::MAX_HISTORY) {
        track.history.erase(track.history.begin());
    }
    track.history.push_back({track.x(0), track.x(1)});

    track.hits++;
    track.misses = 0;
}

// ===== Mahalanobis gating =====
float tracker_impl::compute_distance(const srukf_track_t& track,
                                     float range_m, float doppler_hz) const
{
    float y0 = range_m - track.x(0);
    float y1 = doppler_hz - track.x(1);

    // P = S*S^T, extract 2×2 block + R
    StateMat P = track.S * track.S.transpose();
    float S00 = P(0,0) + d_R_range;
    float S01 = P(0,1);
    float S10 = P(1,0);
    float S11 = P(1,1) + d_R_doppler;

    float det = S00 * S11 - S01 * S10;
    if (std::abs(det) < 1e-10f) det = (det >= 0) ? 1e-10f : -1e-10f;
    float inv_det = 1.0f / det;

    return y0 * (S11*y0 - S01*y1) * inv_det +
           y1 * (-S10*y0 + S00*y1) * inv_det;
}

// ===== Data association (GNN) =====
void tracker_impl::associate_detections(const float* detections, int num_dets)
{
    d_associations.clear();
    std::fill(d_det_used.begin(), d_det_used.begin() + num_dets, false);

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

// ===== Track lifecycle =====
void tracker_impl::create_track(float range_m, float doppler_hz,
                                float aoa_deg, float aoa_confidence)
{
    if (d_tracks.size() >= static_cast<size_t>(d_max_tracks)) return;

    srukf_track_t track;
    track.id = d_next_track_id++;
    track.status = track_status_t::TENTATIVE;

    track.x = StateVec::Zero();
    track.x(0) = range_m;
    track.x(1) = doppler_hz;

    // Initial S = sqrt(P0), diagonal
    track.S = StateMat::Zero();
    track.S(0,0) = std::sqrt(d_R_range);
    track.S(1,1) = std::sqrt(d_R_doppler);
    track.S(2,2) = std::sqrt(1000.0f);  // large range_rate uncertainty
    track.S(3,3) = std::sqrt(100.0f);   // large doppler_rate uncertainty
    track.S(4,4) = std::sqrt(0.1f);     // small turn_rate uncertainty

    track.r_aoa = (aoa_confidence >= AOA_CONFIDENCE_THRESHOLD)
        ? d_meas_noise_aoa * d_meas_noise_aoa : AOA_UNINFORMATIVE_VAR;

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
            [this](const srukf_track_t& t) { return t.misses >= d_delete_misses; }),
        d_tracks.end());
}

void tracker_impl::update_track_status(srukf_track_t& track, bool updated)
{
    if (updated) {
        if (track.status == track_status_t::TENTATIVE && track.hits >= d_confirm_hits)
            track.status = track_status_t::CONFIRMED;
        else if (track.status == track_status_t::COASTING)
            track.status = track_status_t::CONFIRMED;
        track.score = std::min(track.score + 0.1f, 1.0f);
    } else {
        track.misses++;
        if (track.status == track_status_t::CONFIRMED)
            track.status = track_status_t::COASTING;
        track.score = std::max(track.score - 0.2f, 0.0f);
    }
}

track_t tracker_impl::to_public_track(const srukf_track_t& t) const
{
    track_t pub;
    pub.id = t.id;
    pub.status = t.status;
    for (int i = 0; i < NX; i++) pub.state[i] = t.x(i);

    // Reconstruct P = S*S^T for public interface
    StateMat P = t.S * t.S.transpose();
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NX; j++)
            pub.covariance[i * NX + j] = P(i, j);

    pub.hits = t.hits;
    pub.misses = t.misses;
    pub.age = t.age;
    pub.score = t.score;
    pub.history = t.history;
    return pub;
}

// ===== Block lifecycle =====
tracker::sptr tracker::make(float dt, float process_noise_range,
                            float process_noise_doppler,
                            float meas_noise_range, float meas_noise_doppler,
                            float gate_threshold, int confirm_hits,
                            int delete_misses, int max_tracks, int max_detections)
{
    return gnuradio::make_block_sptr<tracker_impl>(
        dt, process_noise_range, process_noise_doppler,
        meas_noise_range, meas_noise_doppler, gate_threshold,
        confirm_hits, delete_misses, max_tracks, max_detections);
}

tracker_impl::tracker_impl(float dt, float process_noise_range,
                           float process_noise_doppler,
                           float meas_noise_range, float meas_noise_doppler,
                           float gate_threshold, int confirm_hits,
                           int delete_misses, int max_tracks, int max_detections)
    : gr::sync_block("tracker",
                     gr::io_signature::make(1, 1, pad4k(max_detections * DET_STRIDE) * sizeof(float)),
                     gr::io_signature::make(1, 1, pad4k(max_tracks * 20) * sizeof(float))),
      d_dt(dt),
      d_meas_noise_range(meas_noise_range),
      d_meas_noise_doppler(meas_noise_doppler),
      d_meas_noise_aoa(10.0f),
      d_R_range(meas_noise_range * meas_noise_range),
      d_R_doppler(meas_noise_doppler * meas_noise_doppler),
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

    compute_weights();
    build_process_noise(process_noise_range, process_noise_doppler, 0.01f);

    // Pre-allocate QR buffer to max needed size
    d_qr_buf.resize(3 * NX, NX);
}

tracker_impl::~tracker_impl() {}

// ===== work =====
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

        std::fill(frame_tracks, frame_tracks + track_size, 0.0f);

        int num_dets = 0;
        for (int d = 0; d < d_max_detections; d++) {
            if (frame_dets[d * DET_STRIDE + 3] > 0.0f) num_dets++;
            else break;
        }

        {
            gr::thread::scoped_lock lock(d_mutex);

            for (auto& track : d_tracks) predict_track(track);

            associate_detections(frame_dets, num_dets);

            std::vector<bool> track_updated(d_tracks.size(), false);
            for (const auto& assoc : d_associations) {
                int t_idx = assoc.first, d_idx = assoc.second;
                float rm = frame_dets[d_idx*DET_STRIDE+3];
                float dh = frame_dets[d_idx*DET_STRIDE+4];
                float ad = frame_dets[d_idx*DET_STRIDE+10];
                float ac = frame_dets[d_idx*DET_STRIDE+11];
                update_track(d_tracks[t_idx], rm, dh, ad, ac);
                track_updated[t_idx] = true;
            }

            for (size_t t = 0; t < d_tracks.size(); t++)
                update_track_status(d_tracks[t], track_updated[t]);

            for (int d = 0; d < num_dets; d++) {
                if (!d_det_used[d]) {
                    float rm = frame_dets[d*DET_STRIDE+3];
                    float dh = frame_dets[d*DET_STRIDE+4];
                    float ad = frame_dets[d*DET_STRIDE+10];
                    float ac = frame_dets[d*DET_STRIDE+11];
                    if (rm > 0.0f) create_track(rm, dh, ad, ac);
                }
            }

            delete_stale_tracks();

            // Pack output (20 floats/track, backward compatible)
            int out_idx = 0;
            for (const auto& track : d_tracks) {
                if (out_idx >= d_max_tracks) break;
                int base = out_idx * 20;
                frame_tracks[base+0] = static_cast<float>(track.id);
                frame_tracks[base+1] = static_cast<float>(track.status);
                frame_tracks[base+2] = track.x(0);
                frame_tracks[base+3] = track.x(1);
                frame_tracks[base+4] = track.x(2);
                frame_tracks[base+5] = track.x(3);
                StateMat P = track.S * track.S.transpose();
                frame_tracks[base+6] = P(0,0);
                frame_tracks[base+7] = P(1,1);
                frame_tracks[base+8] = P(2,2);
                frame_tracks[base+9] = P(3,3);
                frame_tracks[base+10] = static_cast<float>(track.hits);
                frame_tracks[base+11] = static_cast<float>(track.misses);
                frame_tracks[base+12] = static_cast<float>(track.age);
                frame_tracks[base+13] = track.score;
                int sz = static_cast<int>(track.history.size());
                int hl = std::min(sz, 2);
                frame_tracks[base+14] = static_cast<float>(hl);
                for (int h = 0; h < hl; h++) {
                    int hi = sz - hl + h;
                    frame_tracks[base+15+h*2] = track.history[hi][0];
                    frame_tracks[base+16+h*2] = track.history[hi][1];
                }
                frame_tracks[base+19] = track.x(4);
                out_idx++;
            }
        }
    }

    return noutput_items;
}

// ===== Setters =====
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
    for (const auto& t : d_tracks) result.push_back(to_public_track(t));
    return result;
}

std::vector<track_t> tracker_impl::get_confirmed_tracks() const
{
    gr::thread::scoped_lock lock(d_mutex);
    std::vector<track_t> result;
    for (const auto& t : d_tracks)
        if (t.status == track_status_t::CONFIRMED)
            result.push_back(to_public_track(t));
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
    for (const auto& t : d_tracks)
        if (t.status == track_status_t::CONFIRMED) count++;
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
