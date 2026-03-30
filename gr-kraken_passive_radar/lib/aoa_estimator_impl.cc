/*
 * AoA Estimator Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Bartlett beamformer and MUSIC for Angle-of-Arrival estimation.
 * Supports ULA and UCA array configurations.
 */

#include "aoa_estimator_impl.h"
#include <gnuradio/io_signature.h>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#endif

namespace gr {
namespace kraken_passive_radar {

// Pad float count to next multiple of 1024 (= 4096 bytes) for buffer alignment
static inline int pad4k(int n) { return ((n + 1023) & ~1023); }

static constexpr float PI = 3.14159265358979323846f;
static constexpr float TWO_PI = 2.0f * PI;

aoa_estimator::sptr aoa_estimator::make(int num_elements,
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
                                         int n_snapshots)
{
    return gnuradio::make_block_sptr<aoa_estimator_impl>(
        num_elements, d_lambda, n_angles, min_angle_deg, max_angle_deg,
        array_type, num_range_bins, num_doppler_bins, max_detections,
        algorithm, n_sources, n_snapshots);
}

aoa_estimator_impl::aoa_estimator_impl(int num_elements,
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
                                       int n_snapshots)
    : gr::sync_block("aoa_estimator",
                     // Inputs: 4 CAF maps + detection list
                     gr::io_signature::make(5, 5,
                         // First 4 inputs: CAF maps from each surveillance channel
                         // Last input: detection list
                         sizeof(float)),  // Will use vlen
                     // Output: AoA-augmented detection list (padded to 4096-byte boundary)
                     gr::io_signature::make(1, 1, pad4k(max_detections * 12) * sizeof(float))),
      d_num_elements(num_elements),
      d_d_lambda(d_lambda),
      d_n_angles(std::max(2, n_angles)),
      d_min_angle_deg(min_angle_deg),
      d_max_angle_deg(max_angle_deg),
      d_array_type(static_cast<array_type_t>(array_type)),
      d_num_range_bins(num_range_bins),
      d_num_doppler_bins(num_doppler_bins),
      d_max_detections(max_detections),
      d_algorithm(static_cast<aoa_algorithm_t>(algorithm)),
      d_n_sources(std::max(1, std::min(n_sources, num_elements - 1))),
      d_n_snapshots(std::max(2, n_snapshots))
{
    // Set input signature with correct vlens (detection port padded to 4096-byte boundary)
    const int caf_size = num_range_bins * num_doppler_bins;
    const int det_padded = pad4k(max_detections * 10);

    set_input_signature(gr::io_signature::makev(5, 5,
        {caf_size * (int)sizeof(gr_complex),  // CAF 0 (complex)
         caf_size * (int)sizeof(gr_complex),  // CAF 1
         caf_size * (int)sizeof(gr_complex),  // CAF 2
         caf_size * (int)sizeof(gr_complex),  // CAF 3
         det_padded * (int)sizeof(float)}));  // Detections (padded)

    d_steering_vectors.resize(n_angles);
    d_steering_vectors_eigen.resize(n_angles);
    d_angles_deg.resize(n_angles);
    d_spectrum.resize(n_angles);
    d_aoa_results.reserve(max_detections);

    compute_steering_vectors();
}

aoa_estimator_impl::~aoa_estimator_impl() {}

void aoa_estimator_impl::compute_steering_vectors()
{
    float angle_step = (d_max_angle_deg - d_min_angle_deg) / (d_n_angles - 1);

#ifdef HAVE_OPTMATHKERNELS
    // Batch compute all steering vectors: d_n_angles * d_num_elements phases
    int total = d_n_angles * d_num_elements;
    std::vector<float> phases(total);
    std::vector<float> sv_re(total);
    std::vector<float> sv_im(total);

    for (int i = 0; i < d_n_angles; i++) {
        d_angles_deg[i] = d_min_angle_deg + i * angle_step;
        float angle_rad = d_angles_deg[i] * PI / 180.0f;
        d_steering_vectors[i].resize(d_num_elements);

        if (d_array_type == array_type_t::ULA) {
            float sin_theta = std::sin(angle_rad);
            for (int n = 0; n < d_num_elements; n++) {
                phases[i * d_num_elements + n] = -TWO_PI * d_d_lambda * n * sin_theta;
            }
        } else {
            float radius = d_d_lambda * d_num_elements / TWO_PI;
            for (int n = 0; n < d_num_elements; n++) {
                float phi_n = TWO_PI * n / d_num_elements;
                phases[i * d_num_elements + n] = -TWO_PI * radius * std::cos(angle_rad - phi_n);
            }
        }
    }

    // Single batch complex_exp for ALL steering vectors
    optmath::neon::neon_complex_exp_f32(sv_re.data(), sv_im.data(), phases.data(), total);

    // Scatter back into per-angle vectors
    for (int i = 0; i < d_n_angles; i++) {
        d_steering_vectors_eigen[i].resize(d_num_elements);
        for (int n = 0; n < d_num_elements; n++) {
            int idx = i * d_num_elements + n;
            d_steering_vectors[i][n] = std::complex<float>(sv_re[idx], sv_im[idx]);
            d_steering_vectors_eigen[i](n) = std::complex<float>(sv_re[idx], sv_im[idx]);
        }
    }
#else
    for (int i = 0; i < d_n_angles; i++) {
        d_angles_deg[i] = d_min_angle_deg + i * angle_step;
        float angle_rad = d_angles_deg[i] * PI / 180.0f;

        d_steering_vectors[i].resize(d_num_elements);

        if (d_array_type == array_type_t::ULA) {
            steering_vector_ula(angle_rad, d_steering_vectors[i]);
        } else {
            steering_vector_uca(angle_rad, d_steering_vectors[i]);
        }

        // Copy to Eigen vectors
        d_steering_vectors_eigen[i].resize(d_num_elements);
        for (int n = 0; n < d_num_elements; n++) {
            d_steering_vectors_eigen[i](n) = d_steering_vectors[i][n];
        }
    }
#endif
}

void aoa_estimator_impl::steering_vector_ula(float angle_rad,
                                              std::vector<std::complex<float>>& sv)
{
    // ULA steering vector: a_n = exp(-j * 2 * pi * d * n * sin(theta) / lambda)
    // With d_lambda = d/lambda (typically 0.5)
    float sin_theta = std::sin(angle_rad);

    for (int n = 0; n < d_num_elements; n++) {
        float phase = -TWO_PI * d_d_lambda * n * sin_theta;
        sv[n] = std::complex<float>(std::cos(phase), std::sin(phase));
    }
}

void aoa_estimator_impl::steering_vector_uca(float angle_rad,
                                              std::vector<std::complex<float>>& sv)
{
    // UCA steering vector: a_n = exp(-j * 2 * pi * r * cos(theta - phi_n) / lambda)
    // where phi_n = 2 * pi * n / N is the element angular position
    // r = d_lambda * N / (2 * pi) for element spacing d_lambda

    float radius = d_d_lambda * d_num_elements / TWO_PI;

    for (int n = 0; n < d_num_elements; n++) {
        float phi_n = TWO_PI * n / d_num_elements;
        float phase = -TWO_PI * radius * std::cos(angle_rad - phi_n);
        sv[n] = std::complex<float>(std::cos(phase), std::sin(phase));
    }
}

void aoa_estimator_impl::bartlett_spectrum(const std::complex<float>* array_response,
                                            std::vector<float>& spectrum)
{
    // Bartlett beamformer: P(theta) = |a(theta)^H * x|^2 / (|a|^2 * |x|^2)
    // Normalized to give max = 1 for matched direction

    // Compute |x|^2
    float x_power = 0.0f;
    for (int n = 0; n < d_num_elements; n++) {
        x_power += std::norm(array_response[n]);
    }

    if (x_power < 1e-10f) {
        std::fill(spectrum.begin(), spectrum.end(), 0.0f);
        return;
    }

    for (int i = 0; i < d_n_angles; i++) {
        // Compute a^H * x
        std::complex<float> dot(0.0f, 0.0f);
        for (int n = 0; n < d_num_elements; n++) {
            // a^H means conjugate of steering vector
            dot += std::conj(d_steering_vectors[i][n]) * array_response[n];
        }

        // |a^H * x|^2 normalized
        // |a|^2 = num_elements for normalized steering vectors
        spectrum[i] = std::norm(dot) / (d_num_elements * x_power);
    }
}

void aoa_estimator_impl::music_spectrum(const Eigen::MatrixXcf& covariance,
                                         std::vector<float>& spectrum)
{
    // Eigendecomposition of Hermitian covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> solver(covariance);

    // Eigenvalues are sorted ascending by SelfAdjointEigenSolver
    const Eigen::MatrixXcf& eigenvectors = solver.eigenvectors();

    // Noise subspace: columns corresponding to smallest eigenvalues
    // For d_n_sources sources, noise subspace has (N - d_n_sources) columns
    int noise_dim = d_num_elements - d_n_sources;
    Eigen::MatrixXcf En = eigenvectors.leftCols(noise_dim);

    // Precompute En * En^H (noise projection matrix)
    Eigen::MatrixXcf noise_proj = En * En.adjoint();

    // MUSIC pseudo-spectrum: P(theta) = 1 / (a^H * En * En^H * a)
    for (int i = 0; i < d_n_angles; i++) {
        const Eigen::VectorXcf& a = d_steering_vectors_eigen[i];
        std::complex<float> denom = a.adjoint() * noise_proj * a;
        float denom_real = denom.real();
        spectrum[i] = 1.0f / (denom_real + 1e-10f);
    }
}

Eigen::MatrixXcf aoa_estimator_impl::build_covariance(const SnapshotBuffer& buf)
{
    int N = d_num_elements;
    Eigen::MatrixXcf R = Eigen::MatrixXcf::Zero(N, N);

    int K = static_cast<int>(buf.snapshots.size());
    if (K == 0) return R;

    for (const auto& x : buf.snapshots) {
        R += x * x.adjoint();
    }
    R /= static_cast<float>(K);

    // Diagonal loading for numerical stability: eps = trace(R) * 1e-6
    float trace_val = R.trace().real();
    float eps = trace_val * 1e-6f;
    R += eps * Eigen::MatrixXcf::Identity(N, N);

    return R;
}

void aoa_estimator_impl::add_snapshot(int64_t bin_key, const Eigen::VectorXcf& snapshot)
{
    auto it = d_snapshot_buffers.find(bin_key);
    if (it == d_snapshot_buffers.end()) {
        SnapshotBuffer buf;
        buf.max_size = d_n_snapshots;
        buf.snapshots.push_back(snapshot);
        d_snapshot_buffers[bin_key] = std::move(buf);
    } else {
        auto& buf = it->second;
        buf.max_size = d_n_snapshots;
        buf.snapshots.push_back(snapshot);
        while (static_cast<int>(buf.snapshots.size()) > buf.max_size) {
            buf.snapshots.pop_front();
        }
    }

    // Prune if total entries exceed limit
    if (static_cast<int>(d_snapshot_buffers.size()) > MAX_SNAPSHOT_ENTRIES) {
        prune_snapshot_buffers();
    }
}

Eigen::MatrixXcf aoa_estimator_impl::spatial_smooth_fb(const Eigen::VectorXcf& snapshot)
{
    // Forward-backward spatial smoothing for single-snapshot MUSIC on ULA
    // Sub-array size: L = N - d_n_sources (must have L > d_n_sources)
    int N = d_num_elements;
    int L = N - d_n_sources;
    if (L <= d_n_sources || L < 2) {
        // Can't smooth, return rank-1 covariance
        return snapshot * snapshot.adjoint();
    }

    int n_subarrays = N - L + 1;
    Eigen::MatrixXcf R = Eigen::MatrixXcf::Zero(L, L);

    // Forward averaging
    for (int i = 0; i < n_subarrays; i++) {
        Eigen::VectorXcf sub = snapshot.segment(i, L);
        R += sub * sub.adjoint();
    }

    // Exchange matrix J (anti-diagonal identity)
    Eigen::MatrixXcf J = Eigen::MatrixXcf::Zero(L, L);
    for (int i = 0; i < L; i++) {
        J(i, L - 1 - i) = 1.0f;
    }

    // Backward averaging: R_fb = (R + J * R^* * J) / (2 * n_subarrays)
    Eigen::MatrixXcf R_fb = (R + J * R.conjugate() * J) / (2.0f * n_subarrays);

    // Diagonal loading
    float trace_val = R_fb.trace().real();
    float eps = trace_val * 1e-6f;
    R_fb += eps * Eigen::MatrixXcf::Identity(L, L);

    return R_fb;
}

void aoa_estimator_impl::prune_snapshot_buffers()
{
    // Simple strategy: remove entries with fewest snapshots
    while (static_cast<int>(d_snapshot_buffers.size()) > MAX_SNAPSHOT_ENTRIES) {
        auto min_it = d_snapshot_buffers.begin();
        size_t min_size = min_it->second.snapshots.size();
        for (auto it = d_snapshot_buffers.begin(); it != d_snapshot_buffers.end(); ++it) {
            if (it->second.snapshots.size() < min_size) {
                min_size = it->second.snapshots.size();
                min_it = it;
            }
        }
        d_snapshot_buffers.erase(min_it);
    }
}

float aoa_estimator_impl::find_peak_angle(const std::vector<float>& spectrum,
                                           float& confidence, float& peak_width)
{
    // Find maximum
    auto max_it = std::max_element(spectrum.begin(), spectrum.end());
    int max_idx = std::distance(spectrum.begin(), max_it);
    float max_val = *max_it;

    if (max_val < 1e-10f) {
        confidence = 0.0f;
        peak_width = 180.0f;
        return 0.0f;
    }

    // Parabolic interpolation for sub-bin accuracy
    float peak_angle = d_angles_deg[max_idx];

    if (max_idx > 0 && max_idx < d_n_angles - 1) {
        float y0 = spectrum[max_idx - 1];
        float y1 = spectrum[max_idx];
        float y2 = spectrum[max_idx + 1];

        float denom = y0 - 2*y1 + y2;
        if (std::abs(denom) > 1e-10f) {
            float delta = 0.5f * (y0 - y2) / denom;
            float angle_step = (d_max_angle_deg - d_min_angle_deg) / (d_n_angles - 1);
            peak_angle = d_angles_deg[max_idx] + delta * angle_step;
        }
    }

    // Compute 3dB width
    float half_power = max_val / 2.0f;
    int left_idx = max_idx;
    int right_idx = max_idx;

    while (left_idx > 0 && spectrum[left_idx] > half_power) left_idx--;
    while (right_idx < d_n_angles - 1 && spectrum[right_idx] > half_power) right_idx++;

    float angle_step = (d_max_angle_deg - d_min_angle_deg) / (d_n_angles - 1);
    peak_width = (right_idx - left_idx) * angle_step;

    // Confidence based on peak sharpness and value
    // High peak with narrow width = high confidence
    float expected_width = 2.0f * 180.0f / (PI * d_num_elements * d_d_lambda);
    confidence = max_val * std::min(1.0f, expected_width / std::max(peak_width, 1.0f));
    confidence = std::min(1.0f, std::max(0.0f, confidence));

    return peak_angle;
}

int aoa_estimator_impl::work(int noutput_items,
                              gr_vector_const_void_star& input_items,
                              gr_vector_void_star& output_items)
{
    // Input CAFs from surveillance channels (dynamic, not hardcoded to 4)
    const int num_caf_inputs = static_cast<int>(input_items.size()) - 1;
    std::vector<const gr_complex*> caf(num_caf_inputs);
    for (int ch = 0; ch < num_caf_inputs; ch++) {
        caf[ch] = static_cast<const gr_complex*>(input_items[ch]);
    }
    const float* detections = static_cast<const float*>(input_items[num_caf_inputs]);
    float* out = static_cast<float*>(output_items[0]);

    const int caf_size = d_num_range_bins * d_num_doppler_bins;
    const int det_input_size = pad4k(d_max_detections * 10);
    const int det_output_size = pad4k(d_max_detections * 12);

    for (int frame = 0; frame < noutput_items; frame++) {
        const float* frame_dets = detections + frame * det_input_size;
        float* frame_out = out + frame * det_output_size;

        // Clear output
        std::fill(frame_out, frame_out + det_output_size, 0.0f);

        {
            gr::thread::scoped_lock lock(d_mutex);
            d_aoa_results.clear();
        }

        // Process each detection
        int out_idx = 0;
        for (int d = 0; d < d_max_detections && out_idx < d_max_detections; d++) {
            // Detection format: [id, range_bin, doppler_bin, range_m, doppler_hz, ...]
            float det_id = frame_dets[d * 10 + 0];
            float range_bin = frame_dets[d * 10 + 1];
            float doppler_bin = frame_dets[d * 10 + 2];
            float range_m = frame_dets[d * 10 + 3];
            float doppler_hz = frame_dets[d * 10 + 4];

            // Skip empty detections
            if (range_m <= 0.0f) continue;

            // Get integer bin indices
            int r_bin = static_cast<int>(range_bin + 0.5f);
            int d_bin = static_cast<int>(doppler_bin + 0.5f);

            // Bounds check
            r_bin = std::max(0, std::min(r_bin, d_num_range_bins - 1));
            d_bin = std::max(0, std::min(d_bin, d_num_doppler_bins - 1));

            int bin_idx = d_bin * d_num_range_bins + r_bin;

            // Extract array response at this bin from each CAF
            std::vector<std::complex<float>> array_response(num_caf_inputs);
            for (int ch = 0; ch < num_caf_inputs; ch++) {
                const gr_complex* ch_caf = caf[ch] + frame * caf_size;
                array_response[ch] = ch_caf[bin_idx];
            }

            // Algorithm dispatch
            if (d_algorithm == aoa_algorithm_t::MUSIC) {
                // Build Eigen snapshot vector
                Eigen::VectorXcf snapshot(d_num_elements);
                for (int n = 0; n < d_num_elements && n < num_caf_inputs; n++) {
                    snapshot(n) = array_response[n];
                }

                // Key for this detection bin
                int64_t bin_key = (static_cast<int64_t>(r_bin) << 16) | d_bin;

                // Add snapshot to ring buffer
                add_snapshot(bin_key, snapshot);

                auto it = d_snapshot_buffers.find(bin_key);
                int n_accumulated = (it != d_snapshot_buffers.end())
                    ? static_cast<int>(it->second.snapshots.size()) : 0;

                // Need at least d_n_sources + 1 snapshots for MUSIC
                if (n_accumulated >= d_n_sources + 1) {
                    // Build covariance from accumulated snapshots
                    Eigen::MatrixXcf R = build_covariance(it->second);
                    music_spectrum(R, d_spectrum);
                } else if (d_array_type == array_type_t::ULA && d_num_elements > 2 * d_n_sources) {
                    // Fall back to forward-backward spatial smoothing for single snapshot
                    Eigen::MatrixXcf R_fb = spatial_smooth_fb(snapshot);
                    // spatial_smooth_fb returns a smaller matrix; compute MUSIC on sub-array
                    int L = R_fb.rows();
                    int noise_dim = L - d_n_sources;
                    if (noise_dim > 0) {
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> solver(R_fb);
                        Eigen::MatrixXcf En = solver.eigenvectors().leftCols(noise_dim);
                        Eigen::MatrixXcf noise_proj = En * En.adjoint();

                        // Scan with sub-array steering vectors (length L)
                        for (int i = 0; i < d_n_angles; i++) {
                            Eigen::VectorXcf a_sub = d_steering_vectors_eigen[i].head(L);
                            std::complex<float> denom_c = a_sub.adjoint() * noise_proj * a_sub;
                            d_spectrum[i] = 1.0f / (denom_c.real() + 1e-10f);
                        }
                    } else {
                        // Can't do MUSIC, fall back to Bartlett
                        bartlett_spectrum(array_response.data(), d_spectrum);
                    }
                } else {
                    // Fall back to Bartlett until enough snapshots
                    bartlett_spectrum(array_response.data(), d_spectrum);
                }
            } else {
                // Bartlett (default)
                bartlett_spectrum(array_response.data(), d_spectrum);
            }

            // Find peak angle
            float confidence, peak_width;
            float aoa_deg = find_peak_angle(d_spectrum, confidence, peak_width);

            // Create AoA result
            aoa_result_t result;
            result.detection_id = static_cast<int>(det_id);
            result.aoa_deg = aoa_deg;
            result.aoa_confidence = confidence;
            result.spectrum_peak = *std::max_element(d_spectrum.begin(), d_spectrum.end());
            result.spectrum_width_deg = peak_width;

            {
                gr::thread::scoped_lock lock(d_mutex);
                d_aoa_results.push_back(result);
            }

            // Pack output: original detection + AoA info
            // [id, range_bin, doppler_bin, range_m, doppler_hz, snr_db,
            //  power_sum, cluster_size, peak_range, peak_doppler, aoa_deg, aoa_confidence]
            int base = out_idx * 12;
            for (int i = 0; i < 10; i++) {
                frame_out[base + i] = frame_dets[d * 10 + i];
            }
            frame_out[base + 10] = aoa_deg;
            frame_out[base + 11] = confidence;

            out_idx++;
        }
    }

    return noutput_items;
}

void aoa_estimator_impl::set_d_lambda(float d_lambda)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_d_lambda = d_lambda;
    compute_steering_vectors();
}

void aoa_estimator_impl::set_scan_range(float min_deg, float max_deg)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_min_angle_deg = min_deg;
    d_max_angle_deg = max_deg;
    compute_steering_vectors();
}

void aoa_estimator_impl::set_array_type(int type)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_array_type = static_cast<array_type_t>(type);
    compute_steering_vectors();
}

void aoa_estimator_impl::set_algorithm(int algorithm)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_algorithm = static_cast<aoa_algorithm_t>(algorithm);
    d_snapshot_buffers.clear();
}

void aoa_estimator_impl::set_n_sources(int n_sources)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_n_sources = std::max(1, std::min(n_sources, d_num_elements - 1));
    d_snapshot_buffers.clear();
}

void aoa_estimator_impl::set_n_snapshots(int n_snapshots)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_n_snapshots = std::max(2, n_snapshots);
    d_snapshot_buffers.clear();
}

std::vector<aoa_result_t> aoa_estimator_impl::get_aoa_results() const
{
    gr::thread::scoped_lock lock(d_mutex);
    return d_aoa_results;
}

std::vector<float> aoa_estimator_impl::get_spectrum() const
{
    gr::thread::scoped_lock lock(d_mutex);
    return d_spectrum;
}

} // namespace kraken_passive_radar
} // namespace gr
