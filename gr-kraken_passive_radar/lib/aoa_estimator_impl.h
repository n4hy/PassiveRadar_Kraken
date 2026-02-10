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

    // Compute steering vectors for all scan angles
    void compute_steering_vectors();

    // Steering vector for single angle
    void steering_vector_ula(float angle_rad, std::vector<std::complex<float>>& sv);
    void steering_vector_uca(float angle_rad, std::vector<std::complex<float>>& sv);

    // Bartlett beamformer spectrum
    void bartlett_spectrum(const std::complex<float>* array_response,
                           std::vector<float>& spectrum);

    // MUSIC spectrum computation
    void music_spectrum(const Eigen::MatrixXcf& covariance,
                        std::vector<float>& spectrum);

    // Build covariance matrix from snapshot buffer
    Eigen::MatrixXcf build_covariance(const SnapshotBuffer& buf);

    // Add a snapshot to the per-bin ring buffer
    void add_snapshot(int64_t bin_key, const Eigen::VectorXcf& snapshot);

    // Forward-backward spatial smoothing for single-snapshot MUSIC (ULA only)
    Eigen::MatrixXcf spatial_smooth_fb(const Eigen::VectorXcf& snapshot);

    // Prune snapshot buffers that exceed MAX_SNAPSHOT_ENTRIES total
    void prune_snapshot_buffers();

    // Find spectrum peak with interpolation
    float find_peak_angle(const std::vector<float>& spectrum,
                          float& confidence, float& peak_width);

public:
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

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    void set_d_lambda(float d_lambda) override;
    void set_scan_range(float min_deg, float max_deg) override;
    void set_array_type(int type) override;
    void set_algorithm(int algorithm) override;
    void set_n_sources(int n_sources) override;
    void set_n_snapshots(int n_snapshots) override;

    std::vector<aoa_result_t> get_aoa_results() const override;
    std::vector<float> get_spectrum() const override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_AOA_ESTIMATOR_IMPL_H */
