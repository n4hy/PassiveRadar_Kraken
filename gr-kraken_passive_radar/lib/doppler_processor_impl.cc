/*
 * Doppler Processor — Transpose + Contiguous Batched FFT
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Input: [n_doppler × n_range] range profiles (row-major)
 * Transpose to [n_range × n_doppler] for contiguous memory access,
 * then batched FFT all range bins in one FFTW call with stride=1.
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
                     num_doppler_bins),
      d_num_range_bins(num_range_bins),
      d_num_doppler_bins(num_doppler_bins),
      d_window_type(window_type),
      d_output_power(output_power),
      d_transposed(nullptr),
      d_batch_plan(nullptr)
{
    if (num_range_bins < 1 || num_doppler_bins < 1)
        throw std::invalid_argument("bins must be >= 1");

    generate_window();
    create_batch_plan();
}

doppler_processor_impl::~doppler_processor_impl()
{
    if (d_batch_plan) fftwf_destroy_plan(d_batch_plan);
    if (d_transposed) fftwf_free(d_transposed);
}

void doppler_processor_impl::generate_window()
{
    d_window.resize(d_num_doppler_bins);
    if (d_num_doppler_bins == 1) { d_window[0] = 1.0f; return; }

    for (int i = 0; i < d_num_doppler_bins; i++) {
        double x = 2.0 * M_PI * i / (d_num_doppler_bins - 1);
        switch (d_window_type) {
            case 0: d_window[i] = 1.0f; break;
            case 1: d_window[i] = 0.54f - 0.46f * cos(x); break;
            case 2: d_window[i] = 0.5f * (1.0f - cos(x)); break;
            case 3: d_window[i] = 0.42f - 0.5f * cos(x) + 0.08f * cos(2.0 * x); break;
            default: d_window[i] = 1.0f;
        }
    }
}

void doppler_processor_impl::create_batch_plan()
{
    if (d_batch_plan) fftwf_destroy_plan(d_batch_plan);
    if (d_transposed) fftwf_free(d_transposed);

    const int total = d_num_range_bins * d_num_doppler_bins;
    d_transposed = fftwf_alloc_complex(total);
    if (!d_transposed) throw std::runtime_error("doppler: alloc failed");

    // Batched FFT on transposed [n_range × n_doppler] layout:
    // n_range contiguous n_doppler-point FFTs, stride=1, dist=n_doppler
    int n = d_num_doppler_bins;
    d_batch_plan = fftwf_plan_many_dft(
        1, &n, d_num_range_bins,
        d_transposed, nullptr,
        1, d_num_doppler_bins,     // stride=1, dist=n_doppler (contiguous!)
        d_transposed, nullptr,
        1, d_num_doppler_bins,
        FFTW_FORWARD, FFTW_ESTIMATE);

    if (!d_batch_plan) {
        fftwf_free(d_transposed);
        d_transposed = nullptr;
        throw std::runtime_error("doppler: FFTW batch plan failed");
    }
}

int doppler_processor_impl::work(int noutput_items,
                                  gr_vector_const_void_star &input_items,
                                  gr_vector_void_star &output_items)
{
    gr::thread::scoped_lock lock(d_mutex);

    const gr_complex *in = (const gr_complex *)input_items[0];
    const int NR = d_num_range_bins;
    const int ND = d_num_doppler_bins;
    const int rd_size = NR * ND;

    for (int out_i = 0; out_i < noutput_items; out_i++) {
        const gr_complex *frame_in = in + out_i * rd_size;

        // Transpose [ND × NR] → [NR × ND] with windowing
        // Input:  frame_in[d * NR + r]
        // Output: d_transposed[r * ND + d] = frame_in[d * NR + r] * window[d]
        for (int d = 0; d < ND; d++) {
            const float w = d_window[d];
            const gr_complex* row = frame_in + d * NR;
            for (int r = 0; r < NR; r++) {
                d_transposed[r * ND + d][0] = row[r].real() * w;
                d_transposed[r * ND + d][1] = row[r].imag() * w;
            }
        }

        // Batched FFT: all range bins, contiguous n_doppler elements each
        fftwf_execute(d_batch_plan);

        // FFT shift + output
        const int half = (ND + 1) / 2;

        if (d_output_power) {
            float *out = (float *)output_items[0] + out_i * rd_size;
            for (int r = 0; r < NR; r++) {
                const fftwf_complex* src = d_transposed + r * ND;
                // Second half → first half of output
                for (int d = 0; d < ND - half; d++) {
                    float re = src[half + d][0], im = src[half + d][1];
                    out[d * NR + r] = re * re + im * im;
                }
                // First half → second half of output
                for (int d = 0; d < half; d++) {
                    float re = src[d][0], im = src[d][1];
                    out[(ND - half + d) * NR + r] = re * re + im * im;
                }
            }
        } else {
            gr_complex *out = (gr_complex *)output_items[0] + out_i * rd_size;
            for (int r = 0; r < NR; r++) {
                const fftwf_complex* src = d_transposed + r * ND;
                for (int d = 0; d < ND - half; d++) {
                    out[d * NR + r] = gr_complex(src[half + d][0], src[half + d][1]);
                }
                for (int d = 0; d < half; d++) {
                    out[(ND - half + d) * NR + r] = gr_complex(src[d][0], src[d][1]);
                }
            }
        }
    }

    return noutput_items;
}

void doppler_processor_impl::set_num_doppler_bins(int num_doppler_bins)
{
    gr::thread::scoped_lock lock(d_mutex);
    if (num_doppler_bins != d_num_doppler_bins) {
        d_num_doppler_bins = num_doppler_bins;
        generate_window();
        create_batch_plan();
        set_decimation(d_num_doppler_bins);
    }
}

void doppler_processor_impl::set_window_type(int window_type)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_window_type = window_type;
    generate_window();
}

} /* namespace kraken_passive_radar */
} /* namespace gr */
