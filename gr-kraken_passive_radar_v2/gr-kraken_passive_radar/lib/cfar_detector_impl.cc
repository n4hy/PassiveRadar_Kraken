/*
 * CFAR Detector Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "cfar_detector_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace gr {
namespace kraken_passive_radar {

cfar_detector::sptr
cfar_detector::make(int num_range_bins,
                    int num_doppler_bins,
                    int guard_cells_range,
                    int guard_cells_doppler,
                    int ref_cells_range,
                    int ref_cells_doppler,
                    float pfa,
                    int cfar_type,
                    int os_k)
{
    return gnuradio::make_block_sptr<cfar_detector_impl>(
        num_range_bins, num_doppler_bins,
        guard_cells_range, guard_cells_doppler,
        ref_cells_range, ref_cells_doppler,
        pfa, cfar_type, os_k);
}

cfar_detector_impl::cfar_detector_impl(int num_range_bins,
                                       int num_doppler_bins,
                                       int guard_cells_range,
                                       int guard_cells_doppler,
                                       int ref_cells_range,
                                       int ref_cells_doppler,
                                       float pfa,
                                       int cfar_type,
                                       int os_k)
    : gr::sync_block("cfar_detector",
                     gr::io_signature::make(1, 1, sizeof(float) * num_range_bins * num_doppler_bins),
                     gr::io_signature::make(1, 1, sizeof(float) * num_range_bins * num_doppler_bins)),
      d_num_range_bins(num_range_bins),
      d_num_doppler_bins(num_doppler_bins),
      d_guard_cells_range(guard_cells_range),
      d_guard_cells_doppler(guard_cells_doppler),
      d_ref_cells_range(ref_cells_range),
      d_ref_cells_doppler(ref_cells_doppler),
      d_pfa(pfa),
      d_cfar_type(cfar_type),
      d_os_k(os_k),
      d_num_detections(0)
{
    compute_threshold_factor();
    d_ref_samples.reserve(4 * ref_cells_range * ref_cells_doppler);
}

cfar_detector_impl::~cfar_detector_impl()
{
}

void cfar_detector_impl::compute_threshold_factor()
{
    // Number of reference cells (2D window minus guard region)
    int total_window_range = 2 * (d_guard_cells_range + d_ref_cells_range) + 1;
    int total_window_doppler = 2 * (d_guard_cells_doppler + d_ref_cells_doppler) + 1;
    int guard_window_range = 2 * d_guard_cells_range + 1;
    int guard_window_doppler = 2 * d_guard_cells_doppler + 1;
    
    d_num_ref_cells = total_window_range * total_window_doppler - 
                      guard_window_range * guard_window_doppler;
    
    // For CA-CFAR, threshold factor alpha = N * (Pfa^(-1/N) - 1)
    // where N = number of reference cells
    if (d_num_ref_cells > 0) {
        d_threshold_factor = d_num_ref_cells * (pow(d_pfa, -1.0 / d_num_ref_cells) - 1.0);
    } else {
        d_threshold_factor = 10.0; // Fallback
    }
    
    // For OS-CFAR, adjust k if not set
    if (d_cfar_type == 3 && d_os_k == 0) {
        d_os_k = static_cast<int>(0.75 * d_num_ref_cells); // 75th percentile
    }
}

void cfar_detector_impl::set_pfa(float pfa)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_pfa = pfa;
    compute_threshold_factor();
}

void cfar_detector_impl::set_cfar_type(int cfar_type)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_cfar_type = cfar_type;
    compute_threshold_factor();
}

void cfar_detector_impl::set_guard_cells(int range, int doppler)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_guard_cells_range = range;
    d_guard_cells_doppler = doppler;
    compute_threshold_factor();
}

void cfar_detector_impl::set_ref_cells(int range, int doppler)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_ref_cells_range = range;
    d_ref_cells_doppler = doppler;
    compute_threshold_factor();
    d_ref_samples.reserve(4 * range * doppler);
}

int cfar_detector_impl::get_num_detections() const
{
    return d_num_detections;
}

float cfar_detector_impl::estimate_noise_level(const float *data, int r, int d)
{
    d_ref_samples.clear();
    
    int window_range = d_guard_cells_range + d_ref_cells_range;
    int window_doppler = d_guard_cells_doppler + d_ref_cells_doppler;
    
    // Collect reference cell samples (outside guard region)
    for (int dr = -window_range; dr <= window_range; dr++) {
        for (int dd = -window_doppler; dd <= window_doppler; dd++) {
            // Skip guard region and CUT
            if (abs(dr) <= d_guard_cells_range && abs(dd) <= d_guard_cells_doppler) {
                continue;
            }
            
            // Handle boundary conditions with wraparound
            int rr = (r + dr + d_num_range_bins) % d_num_range_bins;
            int dd_idx = (d + dd + d_num_doppler_bins) % d_num_doppler_bins;
            
            d_ref_samples.push_back(data[dd_idx * d_num_range_bins + rr]);
        }
    }
    
    if (d_ref_samples.empty()) {
        return 0.0f;
    }
    
    float noise_estimate = 0.0f;
    
    switch (d_cfar_type) {
        case 0: // CA-CFAR: Cell Averaging
        {
            float sum = std::accumulate(d_ref_samples.begin(), d_ref_samples.end(), 0.0f);
            noise_estimate = sum / d_ref_samples.size();
            break;
        }
        
        case 1: // GO-CFAR: Greatest-Of
        {
            // Split into leading and lagging windows
            size_t half = d_ref_samples.size() / 2;
            float leading_sum = std::accumulate(d_ref_samples.begin(), 
                                                 d_ref_samples.begin() + half, 0.0f);
            float lagging_sum = std::accumulate(d_ref_samples.begin() + half, 
                                                 d_ref_samples.end(), 0.0f);
            float leading_avg = leading_sum / half;
            float lagging_avg = lagging_sum / (d_ref_samples.size() - half);
            noise_estimate = std::max(leading_avg, lagging_avg);
            break;
        }
        
        case 2: // SO-CFAR: Smallest-Of
        {
            size_t half = d_ref_samples.size() / 2;
            float leading_sum = std::accumulate(d_ref_samples.begin(), 
                                                 d_ref_samples.begin() + half, 0.0f);
            float lagging_sum = std::accumulate(d_ref_samples.begin() + half, 
                                                 d_ref_samples.end(), 0.0f);
            float leading_avg = leading_sum / half;
            float lagging_avg = lagging_sum / (d_ref_samples.size() - half);
            noise_estimate = std::min(leading_avg, lagging_avg);
            break;
        }
        
        case 3: // OS-CFAR: Order Statistics
        {
            std::sort(d_ref_samples.begin(), d_ref_samples.end());
            int k = std::min(std::max(d_os_k - 1, 0), (int)d_ref_samples.size() - 1);
            noise_estimate = d_ref_samples[k];
            break;
        }
        
        default:
            noise_estimate = std::accumulate(d_ref_samples.begin(), 
                                              d_ref_samples.end(), 0.0f) / d_ref_samples.size();
    }
    
    return noise_estimate;
}

int cfar_detector_impl::work(int noutput_items,
                              gr_vector_const_void_star &input_items,
                              gr_vector_void_star &output_items)
{
    gr::thread::scoped_lock lock(d_mutex);
    
    const float *in = (const float *)input_items[0];
    float *out = (float *)output_items[0];
    
    int frame_size = d_num_range_bins * d_num_doppler_bins;
    
    for (int i = 0; i < noutput_items; i++) {
        const float *frame_in = &in[i * frame_size];
        float *frame_out = &out[i * frame_size];
        
        int detections = 0;
        
        // Process each cell
        for (int d = 0; d < d_num_doppler_bins; d++) {
            for (int r = 0; r < d_num_range_bins; r++) {
                int idx = d * d_num_range_bins + r;
                float cut_value = frame_in[idx];
                
                // Estimate noise level from reference cells
                float noise_level = estimate_noise_level(frame_in, r, d);
                
                // Compute threshold
                float threshold = d_threshold_factor * noise_level;
                
                // Detection decision
                if (cut_value > threshold) {
                    frame_out[idx] = 1.0f;
                    detections++;
                } else {
                    frame_out[idx] = 0.0f;
                }
            }
        }
        
        d_num_detections = detections;
    }
    
    return noutput_items;
}

} /* namespace kraken_passive_radar */
} /* namespace gr */
