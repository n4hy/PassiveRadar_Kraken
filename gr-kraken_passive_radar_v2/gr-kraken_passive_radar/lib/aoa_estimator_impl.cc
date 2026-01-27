/*
 * AoA Estimator Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Bartlett beamformer for Angle-of-Arrival estimation.
 * Supports ULA and UCA array configurations.
 */

#include "aoa_estimator_impl.h"
#include <gnuradio/io_signature.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace gr {
namespace kraken_passive_radar {

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
                                         int max_detections)
{
    return gnuradio::make_block_sptr<aoa_estimator_impl>(
        num_elements, d_lambda, n_angles, min_angle_deg, max_angle_deg,
        array_type, num_range_bins, num_doppler_bins, max_detections);
}

aoa_estimator_impl::aoa_estimator_impl(int num_elements,
                                       float d_lambda,
                                       int n_angles,
                                       float min_angle_deg,
                                       float max_angle_deg,
                                       int array_type,
                                       int num_range_bins,
                                       int num_doppler_bins,
                                       int max_detections)
    : gr::sync_block("aoa_estimator",
                     // Inputs: 4 CAF maps + detection list
                     gr::io_signature::make(5, 5,
                         // First 4 inputs: CAF maps from each surveillance channel
                         // Last input: detection list
                         sizeof(float)),  // Will use vlen
                     // Output: AoA-augmented detection list
                     gr::io_signature::make(1, 1, max_detections * 12 * sizeof(float))),
      d_num_elements(num_elements),
      d_d_lambda(d_lambda),
      d_n_angles(n_angles),
      d_min_angle_deg(min_angle_deg),
      d_max_angle_deg(max_angle_deg),
      d_array_type(static_cast<array_type_t>(array_type)),
      d_num_range_bins(num_range_bins),
      d_num_doppler_bins(num_doppler_bins),
      d_max_detections(max_detections)
{
    // Set input signature with correct vlens
    const int caf_size = num_range_bins * num_doppler_bins;
    const int det_size = max_detections * 10;

    // Note: GNU Radio doesn't support mixed vlens easily, so we use message passing
    // or restructure. For now, use fixed sizes.
    set_input_signature(gr::io_signature::makev(5, 5,
        {caf_size * (int)sizeof(gr_complex),  // CAF 0 (complex)
         caf_size * (int)sizeof(gr_complex),  // CAF 1
         caf_size * (int)sizeof(gr_complex),  // CAF 2
         caf_size * (int)sizeof(gr_complex),  // CAF 3
         det_size * (int)sizeof(float)}));    // Detections

    d_steering_vectors.resize(n_angles);
    d_angles_deg.resize(n_angles);
    d_spectrum.resize(n_angles);
    d_aoa_results.reserve(max_detections);

    compute_steering_vectors();
}

aoa_estimator_impl::~aoa_estimator_impl() {}

void aoa_estimator_impl::compute_steering_vectors()
{
    float angle_step = (d_max_angle_deg - d_min_angle_deg) / (d_n_angles - 1);

    for (int i = 0; i < d_n_angles; i++) {
        d_angles_deg[i] = d_min_angle_deg + i * angle_step;
        float angle_rad = d_angles_deg[i] * PI / 180.0f;

        d_steering_vectors[i].resize(d_num_elements);

        if (d_array_type == array_type_t::ULA) {
            steering_vector_ula(angle_rad, d_steering_vectors[i]);
        } else {
            steering_vector_uca(angle_rad, d_steering_vectors[i]);
        }
    }
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
    // Input CAFs from 4 surveillance channels
    const gr_complex* caf[4];
    for (int ch = 0; ch < 4; ch++) {
        caf[ch] = static_cast<const gr_complex*>(input_items[ch]);
    }
    const float* detections = static_cast<const float*>(input_items[4]);
    float* out = static_cast<float*>(output_items[0]);

    const int caf_size = d_num_range_bins * d_num_doppler_bins;
    const int det_input_size = d_max_detections * 10;
    const int det_output_size = d_max_detections * 12;

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
            std::complex<float> array_response[4];
            for (int ch = 0; ch < 4; ch++) {
                const gr_complex* ch_caf = caf[ch] + frame * caf_size;
                array_response[ch] = ch_caf[bin_idx];
            }

            // Compute Bartlett spectrum
            bartlett_spectrum(array_response, d_spectrum);

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
