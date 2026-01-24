/*
 * Doppler Processor Block for KrakenSDR Passive Radar
 * 
 * Computes range-Doppler map via slow-time FFT across multiple CPIs.
 * Input: Range profiles (from CAF IFFT output)
 * Output: 2D range-Doppler map
 *
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_H

#include <gnuradio/sync_block.h>
#include <gnuradio/kraken_passive_radar/api.h>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief Doppler processor for passive radar
 * \ingroup kraken_passive_radar
 *
 * Accumulates range profiles over multiple CPIs and computes
 * FFT across slow-time to extract Doppler information.
 *
 * Processing:
 * 1. Accumulate num_doppler_bins range profiles
 * 2. Apply window function (Hamming/Hann/Blackman)
 * 3. Compute FFT across slow-time for each range bin
 * 4. Output |FFT|² as range-Doppler map
 */
class KRAKEN_PASSIVE_RADAR_API doppler_processor : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<doppler_processor> sptr;

    /*!
     * \brief Create doppler processor
     * \param num_range_bins Number of range bins per CPI (input vector length)
     * \param num_doppler_bins Number of CPIs to accumulate (Doppler FFT size)
     * \param window_type Window function: 0=rect, 1=hamming, 2=hann, 3=blackman
     * \param output_power If true, output |X|²; if false, output complex X
     */
    static sptr make(int num_range_bins,
                     int num_doppler_bins,
                     int window_type = 1,
                     bool output_power = true);

    virtual void set_num_doppler_bins(int num_doppler_bins) = 0;
    virtual void set_window_type(int window_type) = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_DOPPLER_PROCESSOR_H */
