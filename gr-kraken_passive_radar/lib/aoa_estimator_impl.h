/*
 * AoA Estimator Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_AOA_ESTIMATOR_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_AOA_ESTIMATOR_IMPL_H

#include <gnuradio/kraken_passive_radar/aoa_estimator.h>
#include <gnuradio/thread/thread.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <complex>
#include <unordered_map>
#include <deque>

namespace gr {
namespace kraken_passive_radar {

/**
 * aoa_estimator_impl - Implementation of the AoA estimator block
 *
 * Technique: Bartlett and MUSIC beamforming using precomputed steering
 * vectors (ULA/UCA), Eigen for covariance decomposition, and per-bin
 * snapshot ring buffers for MUSIC covariance estimation.
 */
class aoa_estimator_impl : public aoa_estimator
{
private:
    int d_num_elements;
    float d_d_lambda;
    int d_n_angles;
    float d_min_angle_deg;
    float d_max_angle_deg;
    array_type_t d_array_type;
    int d_num_range_bins;
    int d_num_doppler_bins;
    int d_max_detections;

    // MUSIC parameters
    aoa_algorithm_t d_algorithm;
    int d_n_sources;
    int d_n_snapshots;

    // Precomputed steering vectors: n_angles x num_elements
    std::vector<std::vector<std::complex<float>>> d_steering_vectors;

    // Precomputed steering vectors as Eigen column vectors for MUSIC
    std::vector<Eigen::VectorXcf> d_steering_vectors_eigen;

    // Angle values for scan
    std::vector<float> d_angles_deg;

    // Output storage
    std::vector<aoa_result_t> d_aoa_results;
    std::vector<float> d_spectrum;

    // Snapshot ring buffer for MUSIC covariance estimation
    // Keyed by (range_bin << 16 | doppler_bin) for per-detection-bin accumulation
    struct SnapshotBuffer {
        std::deque<Eigen::VectorXcf> snapshots;
        int max_size;
    };
    std::unordered_map<int64_t, SnapshotBuffer> d_snapshot_buffers;
    static constexpr int MAX_SNAPSHOT_ENTRIES = 256;

    // Thread safety
    mutable gr::thread::mutex d_mutex;

    /**
     * compute_steering_vectors - Precompute steering vectors for all scan angles
     */
    void compute_steering_vectors();

    /**
     * steering_vector_ula - Compute the steering vector for a Uniform Linear Array at a given angle
     */
    void steering_vector_ula(float angle_rad, std::vector<std::complex<float>>& sv);

    /**
     * steering_vector_uca - Compute the steering vector for a Uniform Circular Array at a given angle
     */
    void steering_vector_uca(float angle_rad, std::vector<std::complex<float>>& sv);

    /**
     * bartlett_spectrum - Compute the Bartlett beamformer angular power spectrum P(theta) = |a^H * x|^2
     */
    void bartlett_spectrum(const std::complex<float>* array_response,
                           std::vector<float>& spectrum);

    /**
     * music_spectrum - Compute the MUSIC pseudo-spectrum from eigendecomposition of the covariance matrix
     */
    void music_spectrum(const Eigen::MatrixXcf& covariance,
                        std::vector<float>& spectrum);

    /**
     * build_covariance - Estimate the spatial covariance matrix from accumulated snapshots
     */
    Eigen::MatrixXcf build_covariance(const SnapshotBuffer& buf);

    /**
     * add_snapshot - Insert a new array snapshot into the per-bin ring buffer for covariance estimation
     */
    void add_snapshot(int64_t bin_key, const Eigen::VectorXcf& snapshot);

    /**
     * spatial_smooth_fb - Apply forward-backward spatial smoothing for single-snapshot MUSIC on ULA
     */
    Eigen::MatrixXcf spatial_smooth_fb(const Eigen::VectorXcf& snapshot);

    /**
     * prune_snapshot_buffers - Remove oldest snapshot buffer entries when total exceeds MAX_SNAPSHOT_ENTRIES
     */
    void prune_snapshot_buffers();

    /**
     * find_peak_angle - Locate the peak angle in the spectrum with parabolic interpolation
     */
    float find_peak_angle(const std::vector<float>& spectrum,
                          float& confidence, float& peak_width);

public:
    /**
     * aoa_estimator_impl - Construct the AoA estimator with array geometry and algorithm parameters
     */
    aoa_estimator_impl(int num_elements,
                       float d_lambda,
                       int n_angles,
                       float min_angle_deg,
                       float max_angle_deg,
                       int array_type,
                       int num_range_bins,
                       int num_doppler_bins,
                       int max_detections,
                       int algorithm,
                       int n_sources,
                       int n_snapshots);
    ~aoa_estimator_impl();

    /**
     * work - Process input CAF data and produce AoA estimates for each detection
     */
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    /** set_d_lambda - Update element spacing in wavelengths and recompute steering vectors */
    void set_d_lambda(float d_lambda) override;
    /** set_scan_range - Update angular scan limits and recompute steering vectors */
    void set_scan_range(float min_deg, float max_deg) override;
    /** set_array_type - Switch array geometry (ULA/UCA) and recompute steering vectors */
    void set_array_type(int type) override;
    /** set_algorithm - Switch between Bartlett and MUSIC algorithms */
    void set_algorithm(int algorithm) override;
    /** set_n_sources - Set assumed number of sources for MUSIC */
    void set_n_sources(int n_sources) override;
    /** set_n_snapshots - Set snapshot buffer depth for MUSIC covariance estimation */
    void set_n_snapshots(int n_snapshots) override;

    /** get_aoa_results - Return AoA results from the last processed frame */
    std::vector<aoa_result_t> get_aoa_results() const override;
    /** get_spectrum - Return the full angular spectrum from the last processed frame */
    std::vector<float> get_spectrum() const override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_AOA_ESTIMATOR_IMPL_H */
