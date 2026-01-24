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
    
    gr::thread::mutex d_mutex;
    
    void compute_threshold_factor();
    float estimate_noise_level(const float *data, int range_idx, int doppler_idx);

public:
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

    void set_pfa(float pfa) override;
    void set_cfar_type(int cfar_type) override;
    void set_guard_cells(int range, int doppler) override;
    void set_ref_cells(int range, int doppler) override;
    int get_num_detections() const override;

    int work(int noutput_items,
             gr_vector_const_void_star &input_items,
             gr_vector_void_star &output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_CFAR_DETECTOR_IMPL_H */
