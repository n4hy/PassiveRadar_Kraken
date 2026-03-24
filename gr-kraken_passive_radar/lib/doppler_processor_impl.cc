/*
 * Doppler Processor Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "doppler_processor_impl.h"
#include <gnuradio/io_signature.h>
#include <volk/volk.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace gr {
namespace kraken_passive_radar {

doppler_processor::sptr
doppler_processor::make(int num_range_bins,
                        int num_doppler_bins,
                        int window_type,
                        bool output_power)
{
    return gnuradio::make_block_sptr<doppler_processor_impl>(
        num_range_bins, num_doppler_bins, window_type, output_power);
}

doppler_processor_impl::doppler_processor_impl(int num_range_bins,
                                               int num_doppler_bins,
                                               int window_type,
                                               bool output_power)
    : gr::sync_decimator("doppler_processor",
                     gr::io_signature::make(1, 1, sizeof(gr_complex) * num_range_bins),
                     gr::io_signature::make(1, 1,
                         output_power ? sizeof(float) * num_range_bins * num_doppler_bins
                                      : sizeof(gr_complex) * num_range_bins * num_doppler_bins),
                     num_doppler_bins),  // decimation factor
      d_num_range_bins(num_range_bins),
      d_num_doppler_bins(num_doppler_bins),
      d_window_type(window_type),
      d_output_power(output_power)
{
    // Validate input parameters
    if (num_range_bins < 1) {
        throw std::invalid_argument("num_range_bins must be >= 1");
    }
    if (num_doppler_bins < 1) {
        throw std::invalid_argument("num_doppler_bins must be >= 1");
    }

    // Allocate accumulation buffer: [doppler_bins x range_bins]
    d_accumulator.resize(d_num_doppler_bins * d_num_range_bins);

    // Allocate FFT input/output buffers
    d_fft_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d_num_doppler_bins);
    d_fft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d_num_doppler_bins);

    // Check for allocation failure
    if (!d_fft_in || !d_fft_out) {
        if (d_fft_in) fftwf_free(d_fft_in);
        if (d_fft_out) fftwf_free(d_fft_out);
        throw std::runtime_error("Failed to allocate FFTW buffers");
    }

    // Create FFT plan for slow-time (Doppler) dimension
    d_fft_plan = fftwf_plan_dft_1d(d_num_doppler_bins, d_fft_in, d_fft_out,
                                   FFTW_FORWARD, FFTW_ESTIMATE);

    if (!d_fft_plan) {
        fftwf_free(d_fft_in);
        fftwf_free(d_fft_out);
        throw std::runtime_error("Failed to create FFTW plan");
    }

    // Generate window coefficients
    generate_window();

    // Allocate output buffer
    d_output_buffer.resize(d_num_range_bins * d_num_doppler_bins);
}

doppler_processor_impl::~doppler_processor_impl()
{
    fftwf_destroy_plan(d_fft_plan);
    fftwf_free(d_fft_in);
    fftwf_free(d_fft_out);
}

void doppler_processor_impl::generate_window()
{
    d_window.resize(d_num_doppler_bins);

    // Handle edge case: single bin (no windowing needed)
    if (d_num_doppler_bins == 1) {
        d_window[0] = 1.0f;
        return;
    }

    for (int i = 0; i < d_num_doppler_bins; i++) {
        double x = 2.0 * M_PI * i / (d_num_doppler_bins - 1);
        switch (d_window_type) {
            case 0: // Rectangular
                d_window[i] = 1.0f;
                break;
            case 1: // Hamming
                d_window[i] = 0.54f - 0.46f * cos(x);
                break;
            case 2: // Hann
                d_window[i] = 0.5f * (1.0f - cos(x));
                break;
            case 3: // Blackman
                d_window[i] = 0.42f - 0.5f * cos(x) + 0.08f * cos(2.0 * x);
                break;
            default:
                d_window[i] = 1.0f;
        }
    }
}

void doppler_processor_impl::set_num_doppler_bins(int num_doppler_bins)
{
    gr::thread::scoped_lock lock(d_mutex);

    if (num_doppler_bins != d_num_doppler_bins) {
        d_num_doppler_bins = num_doppler_bins;
        d_accumulator.resize(d_num_doppler_bins * d_num_range_bins);

        fftwf_destroy_plan(d_fft_plan);
        fftwf_free(d_fft_in);
        fftwf_free(d_fft_out);

        d_fft_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d_num_doppler_bins);
        d_fft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d_num_doppler_bins);

        if (!d_fft_in || !d_fft_out) {
            if (d_fft_in) { fftwf_free(d_fft_in); d_fft_in = nullptr; }
            if (d_fft_out) { fftwf_free(d_fft_out); d_fft_out = nullptr; }
            throw std::runtime_error("Failed to reallocate FFTW buffers");
        }

        d_fft_plan = fftwf_plan_dft_1d(d_num_doppler_bins, d_fft_in, d_fft_out,
                                       FFTW_FORWARD, FFTW_MEASURE);

        if (!d_fft_plan) {
            fftwf_free(d_fft_in); d_fft_in = nullptr;
            fftwf_free(d_fft_out); d_fft_out = nullptr;
            throw std::runtime_error("Failed to create FFTW plan on resize");
        }

        generate_window();
        d_output_buffer.resize(d_num_range_bins * d_num_doppler_bins);
        set_decimation(d_num_doppler_bins);
    }
}

void doppler_processor_impl::set_window_type(int window_type)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_window_type = window_type;
    generate_window();
}

int doppler_processor_impl::work(int noutput_items,
                                  gr_vector_const_void_star &input_items,
                                  gr_vector_void_star &output_items)
{
    gr::thread::scoped_lock lock(d_mutex);

    const gr_complex *in = (const gr_complex *)input_items[0];
    const int rd_size = d_num_range_bins * d_num_doppler_bins;

    // sync_decimator guarantees: noutput_items output items to produce,
    // noutput_items * decimation() input items available.
    // Each output item consumes d_num_doppler_bins input range profiles.

    for (int out_i = 0; out_i < noutput_items; out_i++) {
        // Pointer to this output frame's d_num_doppler_bins input range profiles
        const gr_complex *frame_in = in + out_i * d_num_doppler_bins * d_num_range_bins;

        // Process each range bin: extract slow-time column, window, FFT
        for (int r = 0; r < d_num_range_bins; r++) {
            // Extract slow-time samples for this range bin and apply window
            for (int d = 0; d < d_num_doppler_bins; d++) {
                gr_complex sample = frame_in[d * d_num_range_bins + r];
                d_fft_in[d][0] = sample.real() * d_window[d];
                d_fft_in[d][1] = sample.imag() * d_window[d];
            }

            // Compute Doppler FFT
            fftwf_execute(d_fft_plan);

            // FFT shift and store output (NumPy fftshift convention)
            for (int d = 0; d < d_num_doppler_bins; d++) {
                int shifted_d = (d + (d_num_doppler_bins + 1) / 2) % d_num_doppler_bins;
                int out_idx = shifted_d * d_num_range_bins + r;

                if (d_output_power) {
                    float *out = (float *)output_items[0];
                    float power = d_fft_out[d][0] * d_fft_out[d][0] +
                                  d_fft_out[d][1] * d_fft_out[d][1];
                    out[out_i * rd_size + out_idx] = power;
                } else {
                    gr_complex *out = (gr_complex *)output_items[0];
                    out[out_i * rd_size + out_idx] =
                        gr_complex(d_fft_out[d][0], d_fft_out[d][1]);
                }
            }
        }
    }

    return noutput_items;
}

} /* namespace kraken_passive_radar */
} /* namespace gr */
