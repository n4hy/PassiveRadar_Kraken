/*
 * Detection Clustering Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Connected components clustering for CFAR detection outputs.
 * Uses 8-connectivity BFS for robust clustering of adjacent cells.
 */

#include "detection_cluster_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace gr {
namespace kraken_passive_radar {

// Static member definition
constexpr int detection_cluster_impl::d_neighbors[8][2];

detection_cluster::sptr detection_cluster::make(int num_range_bins,
                                                 int num_doppler_bins,
                                                 int min_cluster_size,
                                                 int max_cluster_extent,
                                                 float range_resolution_m,
                                                 float doppler_resolution_hz,
                                                 int max_detections)
{
    return gnuradio::make_block_sptr<detection_cluster_impl>(
        num_range_bins, num_doppler_bins, min_cluster_size,
        max_cluster_extent, range_resolution_m, doppler_resolution_hz,
        max_detections);
}

detection_cluster_impl::detection_cluster_impl(int num_range_bins,
                                               int num_doppler_bins,
                                               int min_cluster_size,
                                               int max_cluster_extent,
                                               float range_resolution_m,
                                               float doppler_resolution_hz,
                                               int max_detections)
    : gr::sync_block("detection_cluster",
                     // Input: detection mask + power map
                     gr::io_signature::make(2, 2, num_range_bins * num_doppler_bins * sizeof(float)),
                     // Output: packed detection vector
                     gr::io_signature::make(1, 1, max_detections * 10 * sizeof(float))),
      d_num_range_bins(num_range_bins),
      d_num_doppler_bins(num_doppler_bins),
      d_min_cluster_size(min_cluster_size),
      d_max_cluster_extent(max_cluster_extent),
      d_range_res_m(range_resolution_m),
      d_doppler_res_hz(doppler_resolution_hz),
      d_max_detections(max_detections)
{
    const int total_cells = num_range_bins * num_doppler_bins;
    d_labels.resize(total_cells, 0);
    d_visited.resize(total_cells, false);
    d_detections.reserve(max_detections);
}

detection_cluster_impl::~detection_cluster_impl() {}

int detection_cluster_impl::find_connected_components(const float* det_mask)
{
    const int total_cells = d_num_range_bins * d_num_doppler_bins;

    // Reset state
    std::fill(d_labels.begin(), d_labels.end(), 0);
    std::fill(d_visited.begin(), d_visited.end(), false);

    int current_label = 0;
    std::queue<int> bfs_queue;

    for (int i = 0; i < total_cells; i++) {
        // Skip if not a detection or already visited
        if (det_mask[i] < 0.5f || d_visited[i]) {
            continue;
        }

        // Start new component
        current_label++;
        int component_size = 0;

        bfs_queue.push(i);
        d_visited[i] = true;

        while (!bfs_queue.empty()) {
            int cell = bfs_queue.front();
            bfs_queue.pop();

            d_labels[cell] = current_label;
            component_size++;

            // Check if cluster is too large
            if (component_size > d_max_cluster_extent) {
                // Mark remaining cells but don't add neighbors
                while (!bfs_queue.empty()) {
                    int c = bfs_queue.front();
                    bfs_queue.pop();
                    d_labels[c] = current_label;
                }
                break;
            }

            int r = range_from_idx(cell);
            int d = doppler_from_idx(cell);

            // Check 8-connected neighbors
            for (int n = 0; n < 8; n++) {
                int nr = r + d_neighbors[n][0];
                int nd = d + d_neighbors[n][1];

                // Bounds check
                if (nr < 0 || nr >= d_num_range_bins ||
                    nd < 0 || nd >= d_num_doppler_bins) {
                    continue;
                }

                int neighbor_idx = idx(nr, nd);

                // Skip if not a detection or already visited
                if (det_mask[neighbor_idx] < 0.5f || d_visited[neighbor_idx]) {
                    continue;
                }

                d_visited[neighbor_idx] = true;
                bfs_queue.push(neighbor_idx);
            }
        }
    }

    return current_label;
}

void detection_cluster_impl::compute_cluster_stats(int label,
                                                   const float* det_mask,
                                                   const float* power_db,
                                                   detection_t& det)
{
    // Initialize statistics
    double weighted_range_sum = 0.0;
    double weighted_doppler_sum = 0.0;
    double total_weight = 0.0;
    float peak_power = -std::numeric_limits<float>::infinity();
    int peak_range = 0;
    int peak_doppler = 0;
    int cluster_size = 0;

    const int total_cells = d_num_range_bins * d_num_doppler_bins;

    for (int i = 0; i < total_cells; i++) {
        if (d_labels[i] != label) {
            continue;
        }

        int r = range_from_idx(i);
        int d = doppler_from_idx(i);
        float power = power_db[i];

        // Convert dB to linear for weighting (with floor to avoid issues)
        float linear_power = std::pow(10.0f, std::max(power, -60.0f) / 10.0f);

        weighted_range_sum += r * linear_power;
        weighted_doppler_sum += d * linear_power;
        total_weight += linear_power;
        cluster_size++;

        // Track peak
        if (power > peak_power) {
            peak_power = power;
            peak_range = r;
            peak_doppler = d;
        }
    }

    // Compute centroid (power-weighted)
    if (total_weight > 0) {
        det.range_bin = static_cast<float>(weighted_range_sum / total_weight);
        det.doppler_bin = static_cast<float>(weighted_doppler_sum / total_weight);
    } else {
        det.range_bin = static_cast<float>(peak_range);
        det.doppler_bin = static_cast<float>(peak_doppler);
    }

    // Convert to physical units
    det.range_m = det.range_bin * d_range_res_m;

    // Doppler: centered at zero
    float doppler_center = d_num_doppler_bins / 2.0f;
    det.doppler_hz = (det.doppler_bin - doppler_center) * d_doppler_res_hz;

    // Statistics
    det.snr_db = peak_power;
    det.power_sum = static_cast<float>(10.0 * std::log10(total_weight));
    det.cluster_size = cluster_size;
    det.peak_range = peak_range;
    det.peak_doppler = peak_doppler;
}

int detection_cluster_impl::work(int noutput_items,
                                 gr_vector_const_void_star& input_items,
                                 gr_vector_void_star& output_items)
{
    const float* det_mask = static_cast<const float*>(input_items[0]);
    const float* power_db = static_cast<const float*>(input_items[1]);
    float* out = static_cast<float*>(output_items[0]);

    const int map_size = d_num_range_bins * d_num_doppler_bins;

    // Process each frame
    for (int frame = 0; frame < noutput_items; frame++) {
        const float* frame_mask = det_mask + frame * map_size;
        const float* frame_power = power_db + frame * map_size;
        float* frame_out = out + frame * d_max_detections * 10;

        // Find connected components
        int num_components = find_connected_components(frame_mask);

        // Clear output and detection list
        std::fill(frame_out, frame_out + d_max_detections * 10, 0.0f);

        {
            gr::thread::scoped_lock lock(d_mutex);
            d_detections.clear();
        }

        // Process each cluster
        int det_count = 0;
        for (int label = 1; label <= num_components && det_count < d_max_detections; label++) {
            // Count cluster size first
            int size = 0;
            for (int i = 0; i < map_size; i++) {
                if (d_labels[i] == label) size++;
            }

            // Skip clusters that are too small
            if (size < d_min_cluster_size) {
                continue;
            }

            detection_t det;
            det.id = det_count;
            compute_cluster_stats(label, frame_mask, frame_power, det);

            // Pack into output
            int base = det_count * 10;
            frame_out[base + 0] = static_cast<float>(det.id);
            frame_out[base + 1] = det.range_bin;
            frame_out[base + 2] = det.doppler_bin;
            frame_out[base + 3] = det.range_m;
            frame_out[base + 4] = det.doppler_hz;
            frame_out[base + 5] = det.snr_db;
            frame_out[base + 6] = det.power_sum;
            frame_out[base + 7] = static_cast<float>(det.cluster_size);
            frame_out[base + 8] = static_cast<float>(det.peak_range);
            frame_out[base + 9] = static_cast<float>(det.peak_doppler);

            {
                gr::thread::scoped_lock lock(d_mutex);
                d_detections.push_back(det);
            }

            det_count++;
        }
    }

    return noutput_items;
}

void detection_cluster_impl::set_min_cluster_size(int size)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_min_cluster_size = std::max(1, size);
}

void detection_cluster_impl::set_max_cluster_extent(int extent)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_max_cluster_extent = std::max(1, extent);
}

void detection_cluster_impl::set_range_resolution(float res_m)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_range_res_m = res_m;
}

void detection_cluster_impl::set_doppler_resolution(float res_hz)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_doppler_res_hz = res_hz;
}

std::vector<detection_t> detection_cluster_impl::get_detections() const
{
    gr::thread::scoped_lock lock(d_mutex);
    return d_detections;
}

int detection_cluster_impl::get_num_detections() const
{
    gr::thread::scoped_lock lock(d_mutex);
    return static_cast<int>(d_detections.size());
}

} // namespace kraken_passive_radar
} // namespace gr
