/*
 * Radar Target Motion and Measurement Model for SRUKF Tracker
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * State vector (5D):
 *   [0] range_m      - bistatic range in meters
 *   [1] doppler_hz   - Doppler shift in Hz
 *   [2] range_rate   - range rate of change (m/frame)
 *   [3] doppler_rate  - Doppler rate of change (Hz/frame)
 *   [4] turn_rate    - maneuver rate coupling range_rate <-> doppler_rate
 *
 * Measurement vector (3D):
 *   [0] range_m      - measured bistatic range
 *   [1] doppler_hz   - measured Doppler shift
 *   [2] aoa_deg      - measured angle of arrival (degrees)
 *
 * The turn_rate state allows the filter to track coordinated maneuvers
 * where range and Doppler rates change together (e.g., a turning aircraft
 * whose bistatic geometry is evolving). This is NOT a physical turn rate
 * in heading, but a generalized coupling term in range-Doppler space.
 */

#ifndef RADAR_TARGET_MODEL_H
#define RADAR_TARGET_MODEL_H

#include "StateSpaceModel.h"
#include <cmath>

namespace gr {
namespace kraken_passive_radar {

// State and measurement dimensions (match tracker.h NX/NY)
static constexpr int RTM_NX = 5;
static constexpr int RTM_NY = 3;

class RadarTargetModel : public UKFModel::StateSpaceModel<RTM_NX, RTM_NY>
{
public:
    RadarTargetModel(float dt,
                     float process_noise_range,
                     float process_noise_doppler,
                     float process_noise_turn,
                     float meas_noise_range,
                     float meas_noise_doppler,
                     float meas_noise_aoa)
        : dt_(dt),
          q_range_(process_noise_range * process_noise_range),
          q_doppler_(process_noise_doppler * process_noise_doppler),
          q_turn_(process_noise_turn * process_noise_turn),
          r_range_(meas_noise_range * meas_noise_range),
          r_doppler_(meas_noise_doppler * meas_noise_doppler),
          r_aoa_(meas_noise_aoa * meas_noise_aoa)
    {}

    /*
     * Nonlinear state transition: coordinated turn in range-Doppler space
     *
     * range[k+1]        = range[k] + range_rate[k] * dt
     * doppler[k+1]      = doppler[k] + doppler_rate[k] * dt
     * range_rate[k+1]   = range_rate[k] * cos(omega*dt) - doppler_rate[k] * sin(omega*dt)
     * doppler_rate[k+1] = range_rate[k] * sin(omega*dt) + doppler_rate[k] * cos(omega*dt)
     * turn_rate[k+1]    = turn_rate[k]   (random walk)
     *
     * When turn_rate ~0, this degenerates to constant velocity (linear).
     * When turn_rate != 0, range_rate and doppler_rate rotate, modeling
     * the coupled rate changes of a maneuvering target.
     */
    State f(const State& x, float /*t_k*/,
            const Eigen::Ref<const State>& /*u_k*/) const override
    {
        State x_new;
        float omega = x(4);  // turn rate
        float omega_dt = omega * dt_;

        x_new(0) = x(0) + x(2) * dt_;  // range
        x_new(1) = x(1) + x(3) * dt_;  // doppler

        if (std::abs(omega_dt) > 1e-6f) {
            float c = std::cos(omega_dt);
            float s = std::sin(omega_dt);
            x_new(2) = x(2) * c - x(3) * s;   // range_rate rotated
            x_new(3) = x(2) * s + x(3) * c;   // doppler_rate rotated
        } else {
            // Small angle: avoid numerical issues, degenerate to CV
            x_new(2) = x(2);
            x_new(3) = x(3);
        }

        x_new(4) = x(4);  // turn rate persists (random walk via Q)
        return x_new;
    }

    /*
     * Measurement function: direct observation of range, doppler, AoA
     */
    Observation h(const State& x, float /*t_k*/) const override
    {
        Observation z;
        z(0) = x(0);  // range
        z(1) = x(1);  // doppler
        z(2) = 0.0f;  // AoA: not directly in state, filled from detection
        // Note: AoA measurement is handled specially in the tracker
        // by setting z(2) and adjusting R when AoA confidence is low
        return z;
    }

    /*
     * Process noise covariance Q
     * Discrete white noise acceleration model for range/doppler
     * Random walk for turn rate
     */
    StateMat Q(float /*t_k*/) const override
    {
        StateMat Q = StateMat::Zero();
        float dt2 = dt_ * dt_;
        float dt3 = dt2 * dt_;
        float dt4 = dt3 * dt_;

        // Range block (states 0, 2)
        Q(0, 0) = dt4 / 4.0f * q_range_;
        Q(0, 2) = dt3 / 2.0f * q_range_;
        Q(2, 0) = dt3 / 2.0f * q_range_;
        Q(2, 2) = dt2 * q_range_;

        // Doppler block (states 1, 3)
        Q(1, 1) = dt4 / 4.0f * q_doppler_;
        Q(1, 3) = dt3 / 2.0f * q_doppler_;
        Q(3, 1) = dt3 / 2.0f * q_doppler_;
        Q(3, 3) = dt2 * q_doppler_;

        // Turn rate (state 4): random walk
        Q(4, 4) = q_turn_ * dt_;

        return Q;
    }

    /*
     * Measurement noise covariance R
     */
    ObsMat R(float /*t_k*/) const override
    {
        ObsMat R = ObsMat::Zero();
        R(0, 0) = r_range_;
        R(1, 1) = r_doppler_;
        R(2, 2) = r_aoa_;
        return R;
    }

    // No angular states in this model (AoA is a measurement, not state)
    bool isAngularState(int /*i*/) const override { return false; }

    // Update noise parameters at runtime
    void set_process_noise(float range, float doppler) {
        q_range_ = range * range;
        q_doppler_ = doppler * doppler;
    }

    void set_measurement_noise(float range, float doppler) {
        r_range_ = range * range;
        r_doppler_ = doppler * doppler;
    }

    void set_measurement_noise_aoa(float aoa_deg) {
        r_aoa_ = aoa_deg * aoa_deg;
    }

    float get_r_aoa() const { return r_aoa_; }

private:
    float dt_;
    float q_range_, q_doppler_, q_turn_;
    float r_range_, r_doppler_, r_aoa_;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif // RADAR_TARGET_MODEL_H
