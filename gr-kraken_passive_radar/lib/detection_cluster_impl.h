/*
 * Detection Clustering Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_DETECTION_CLUSTER_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_DETECTION_CLUSTER_IMPL_H

#include <gnuradio/kraken_passive_radar/detection_cluster.h>
#include <gnuradio/thread/thread.h>
#include <vector>
#include <queue>

namespace gr {
namespace kraken_passive_radar {

class detection_cluster_impl : public detection_cluster
{
private:
    int d_num_range_bins;
    int d_num_doppler_bins;
    int d_min_cluster_size;
    int d_max_cluster_extent;
    float d_range_res_m;
    float d_doppler_res_hz;
    int d_max_detections;

    // Working buffers
    std::vector<int> d_labels;          // Connected component labels
    std::vector<bool> d_visited;        // BFS visited flags
    std::vector<detection_t> d_detections;  // Output detections

    // Thread safety
    mutable gr::thread::mutex d_mutex;

    // 8-connectivity neighbor offsets (dr, dd)
    static constexpr int d_neighbors[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    // Connected components using BFS
    int find_connected_components(const float* det_mask);

    // Compute cluster statistics
    void compute_cluster_stats(int label,
                               const float* det_mask,
                               const float* power_db,
                               detection_t& det);

    // Index conversion helpers
    inline int idx(int r, int d) const { return d * d_num_range_bins + r; }
    inline int range_from_idx(int i) const { return i % d_num_range_bins; }
    inline int doppler_from_idx(int i) const { return i / d_num_range_bins; }

public:
    detection_cluster_impl(int num_range_bins,
                           int num_doppler_bins,
                           int min_cluster_size,
                           int max_cluster_extent,
                           float range_resolution_m,
                           float doppler_resolution_hz,
                           int max_detections);
    ~detection_cluster_impl();

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    void set_min_cluster_size(int size) override;
    void set_max_cluster_extent(int extent) override;
    void set_range_resolution(float res_m) override;
    void set_doppler_resolution(float res_hz) override;

    std::vector<detection_t> get_detections() const override;
    int get_num_detections() const override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_DETECTION_CLUSTER_IMPL_H */
