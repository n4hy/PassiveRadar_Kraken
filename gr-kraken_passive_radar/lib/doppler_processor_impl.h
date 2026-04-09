/*
 * Doppler Processor Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_IMPL_H

#include <gnuradio/kraken_passive_radar/doppler_processor.h>
#include <gnuradio/sync_decimator.h>
#include <gnuradio/thread/thread.h>
#include <fftw3.h>
#include <vector>

namespace gr {
namespace kraken_passive_radar {

/**
 * doppler_processor_impl - Implementation of the Doppler processor block
 *
 * Technique: Accumulates range profiles across CPIs, transposes to column-major
 * layout, and applies batched FFTW slow-time FFTs with configurable window
 * functions to produce the range-Doppler map.
 */
class doppler_processor_impl : public doppler_processor
{
private:
    int d_num_range_bins;
    int d_num_doppler_bins;
    int d_window_type;
    bool d_output_power;
    std::vector<float> d_window;

    // Transposed layout for contiguous batched FFT
    // Input [n_doppler × n_range] → transpose to [n_range × n_doppler] → batch FFT
    fftwf_complex* d_transposed;  // [n_range × n_doppler] transposed buffer
    fftwf_plan d_batch_plan;      // n_range contiguous n_doppler-point FFTs

    gr::thread::mutex d_mutex;

    /**
     * generate_window - Compute the selected window function coefficients (Hamming/Hann/Blackman/rect)
     */
    void generate_window();

    /**
     * create_batch_plan - Create the batched FFTW plan for n_range contiguous n_doppler-point FFTs
     */
    void create_batch_plan();

public:
    /**
     * doppler_processor_impl - Construct Doppler processor with FFT dimensions and window type
     */
    doppler_processor_impl(int num_range_bins,
                           int num_doppler_bins,
                           int window_type,
                           bool output_power);
    ~doppler_processor_impl();

    /** set_num_doppler_bins - Update Doppler FFT size and recreate batch plan */
    void set_num_doppler_bins(int num_doppler_bins) override;
    /** set_window_type - Switch window function and regenerate coefficients */
    void set_window_type(int window_type) override;

    /**
     * work - Accumulate range profiles, apply windowed slow-time FFT, and output range-Doppler map
     */
    int work(int noutput_items,
             gr_vector_const_void_star &input_items,
             gr_vector_void_star &output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif
