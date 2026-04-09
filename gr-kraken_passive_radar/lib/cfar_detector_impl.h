/*
 * CFAR Detector Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_CFAR_DETECTOR_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_CFAR_DETECTOR_IMPL_H

#include <gnuradio/kraken_passive_radar/cfar_detector.h>
#include <gnuradio/thread/thread.h>
#include <vector>

namespace gr {
namespace kraken_passive_radar {

/**
 * cfar_detector_impl - Implementation of the CFAR detector block
 *
 * Technique: 2D Cell-Averaging CFAR with support for CA, GO, SO, and OS
 * variants. Uses leading/lagging reference windows with guard cells and
 * computes an adaptive noise threshold per cell under test.
 */
class cfar_detector_impl : public cfar_detector
{
private:
    int d_num_range_bins;
    int d_num_doppler_bins;
    int d_guard_cells_range;
    int d_guard_cells_doppler;
    int d_ref_cells_range;
    int d_ref_cells_doppler;
    float d_pfa;
    int d_cfar_type;
    int d_os_k;
    
    float d_threshold_factor;
    int d_num_ref_cells;
    int d_num_detections;
    
    std::vector<float> d_ref_samples;
    std::vector<float> d_leading_samples;   // GO/SO-CFAR: range-leading cells
    std::vector<float> d_lagging_samples;   // GO/SO-CFAR: range-lagging cells
    
    mutable gr::thread::mutex d_mutex;
    
    /**
     * compute_threshold_factor - Derive the CFAR threshold multiplier from the Pfa and number of reference cells
     */
    void compute_threshold_factor();

    /**
     * estimate_noise_level - Estimate the local noise level around a cell under test using reference cells
     */
    float estimate_noise_level(const float *data, int range_idx, int doppler_idx);

public:
    /**
     * cfar_detector_impl - Construct CFAR detector with grid dimensions and algorithm parameters
     */
    cfar_detector_impl(int num_range_bins,
                       int num_doppler_bins,
                       int guard_cells_range,
                       int guard_cells_doppler,
                       int ref_cells_range,
                       int ref_cells_doppler,
                       float pfa,
                       int cfar_type,
                       int os_k);
    ~cfar_detector_impl();

    /** set_pfa - Update Pfa and recompute threshold factor */
    void set_pfa(float pfa) override;
    /** set_cfar_type - Switch CFAR algorithm variant */
    void set_cfar_type(int cfar_type) override;
    /** set_guard_cells - Update guard cell counts in both dimensions */
    void set_guard_cells(int range, int doppler) override;
    /** set_ref_cells - Update reference cell counts in both dimensions */
    void set_ref_cells(int range, int doppler) override;
    /** get_num_detections - Return detection count from the last frame */
    int get_num_detections() const override;

    /**
     * work - Apply CFAR detection across the range-Doppler map and output detection mask
     */
    int work(int noutput_items,
             gr_vector_const_void_star &input_items,
             gr_vector_void_star &output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_CFAR_DETECTOR_IMPL_H */
