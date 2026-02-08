/*
 * Doppler Processor Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_IMPL_H

#include <gnuradio/kraken_passive_radar/doppler_processor.h>
#include <gnuradio/thread/thread.h>
#include <fftw3.h>
#include <vector>

namespace gr {
namespace kraken_passive_radar {

class doppler_processor_impl : public doppler_processor
{
private:
    int d_num_range_bins;
    int d_num_doppler_bins;
    int d_window_type;
    bool d_output_power;
    int d_cpi_count;
    
    std::vector<gr_complex> d_accumulator;
    std::vector<float> d_window;
    std::vector<gr_complex> d_output_buffer;
    
    fftwf_complex *d_fft_in;
    fftwf_complex *d_fft_out;
    fftwf_plan d_fft_plan;
    
    gr::thread::mutex d_mutex;
    
    void generate_window();

public:
    doppler_processor_impl(int num_range_bins,
                           int num_doppler_bins,
                           int window_type,
                           bool output_power);
    ~doppler_processor_impl();

    void set_num_doppler_bins(int num_doppler_bins) override;
    void set_window_type(int window_type) override;

    int work(int noutput_items,
             gr_vector_const_void_star &input_items,
             gr_vector_void_star &output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_IMPL_H */
