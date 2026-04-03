/**
 * GPU-Accelerated UKF/SRUKF Kalman Filter Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Unscented Kalman Filter optimized for multi-target tracking.
 *
 * State: [range, doppler, range_rate, doppler_rate, turn_rate] (5D)
 * Measurement: [range, doppler, aoa_deg] (3D)
 */

#ifndef KRAKEN_UKF_GPU_H
#define KRAKEN_UKF_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create UKF GPU processor
 *
 * @param max_tracks Maximum number of concurrent tracks
 * @param alpha UKF spread parameter (typically 1e-3 to 1)
 * @param beta UKF prior knowledge parameter (2 for Gaussian)
 * @param kappa UKF secondary scaling parameter (typically 0 or 3-n)
 * @return Opaque handle to processor, or NULL on failure
 */
void* ukf_gpu_create(int max_tracks, float alpha, float beta, float kappa);

/**
 * Destroy UKF GPU processor
 *
 * @param handle Processor handle from ukf_gpu_create
 */
void ukf_gpu_destroy(void* handle);

/**
 * Set process noise covariance (sqrt form)
 *
 * @param handle Processor handle
 * @param sqrt_Q Process noise sqrt covariance [NX x NX], row-major
 */
void ukf_gpu_set_process_noise(void* handle, const float* sqrt_Q);

/**
 * Set measurement noise covariance (sqrt form)
 *
 * @param handle Processor handle
 * @param sqrt_R Measurement noise sqrt covariance [NY x NY], row-major
 */
void ukf_gpu_set_measurement_noise(void* handle, const float* sqrt_R);

/**
 * Batch predict step for all tracks
 *
 * @param handle Processor handle
 * @param n_tracks Number of active tracks
 * @param dt Time step in seconds
 */
void ukf_gpu_predict(void* handle, int n_tracks, float dt);

/**
 * Batch update step for all tracks
 *
 * @param handle Processor handle
 * @param measurements Measurement vectors [n_tracks x NY], row-major
 * @param n_tracks Number of active tracks
 */
void ukf_gpu_update(void* handle, const float* measurements, int n_tracks);

/**
 * Get current track states
 *
 * @param handle Processor handle
 * @param states_out Output buffer for states [n_tracks x NX], row-major
 * @param n_tracks Number of tracks to retrieve
 */
void ukf_gpu_get_states(void* handle, float* states_out, int n_tracks);

/**
 * Set track states
 *
 * @param handle Processor handle
 * @param states_in Input states [n_tracks x NX], row-major
 * @param n_tracks Number of tracks
 */
void ukf_gpu_set_states(void* handle, const float* states_in, int n_tracks);

/**
 * Set track covariance (sqrt form)
 *
 * @param handle Processor handle
 * @param sqrt_P_in Input sqrt covariance [n_tracks x NX x NX], row-major
 * @param n_tracks Number of tracks
 */
void ukf_gpu_set_covariance(void* handle, const float* sqrt_P_in, int n_tracks);

/**
 * Check if GPU UKF is available
 *
 * @return 1 if GPU available, 0 otherwise
 */
int ukf_gpu_is_available(void);

#ifdef __cplusplus
}
#endif

#endif /* KRAKEN_UKF_GPU_H */
