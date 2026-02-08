/*
 * CFAR Detector Block for KrakenSDR Passive Radar
 * 
 * Constant False Alarm Rate detector for range-Doppler maps.
 * Supports CA-CFAR, GO-CFAR, SO-CFAR, and OS-CFAR algorithms.
 *
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_CFAR_DETECTOR_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_CFAR_DETECTOR_H

#include <gnuradio/sync_block.h>
#include <gnuradio/kraken_passive_radar/api.h>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief CFAR detector for passive radar
 * \ingroup kraken_passive_radar
 *
 * Implements Cell-Averaging CFAR and variants:
 * - CA-CFAR: Cell Averaging (mean of reference cells)
 * - GO-CFAR: Greatest-Of (max of leading/lagging windows)
 * - SO-CFAR: Smallest-Of (min of leading/lagging windows)
 * - OS-CFAR: Order Statistics (k-th ordered sample)
 *
 * Layout per dimension:
 * [ref cells] [guard cells] [CUT] [guard cells] [ref cells]
 *
 * Output is binary detection map (1.0 = detection, 0.0 = no detection)
 * plus optional target list with range/Doppler indices.
 */
class KRAKEN_PASSIVE_RADAR_API cfar_detector : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<cfar_detector> sptr;

    /*!
     * \brief Create CFAR detector
     * \param num_range_bins Number of range bins
     * \param num_doppler_bins Number of Doppler bins
     * \param guard_cells_range Guard cells in range dimension (each side)
     * \param guard_cells_doppler Guard cells in Doppler dimension (each side)
     * \param ref_cells_range Reference cells in range dimension (each side)
     * \param ref_cells_doppler Reference cells in Doppler dimension (each side)
     * \param pfa Probability of false alarm (determines threshold multiplier)
     * \param cfar_type 0=CA, 1=GO, 2=SO, 3=OS
     * \param os_k For OS-CFAR: k-th ordered statistic (1 to num_ref_cells)
     */
    static sptr make(int num_range_bins,
                     int num_doppler_bins,
                     int guard_cells_range = 2,
                     int guard_cells_doppler = 2,
                     int ref_cells_range = 8,
                     int ref_cells_doppler = 8,
                     float pfa = 1e-6,
                     int cfar_type = 0,
                     int os_k = 0);

    virtual void set_pfa(float pfa) = 0;
    virtual void set_cfar_type(int cfar_type) = 0;
    virtual void set_guard_cells(int range, int doppler) = 0;
    virtual void set_ref_cells(int range, int doppler) = 0;
    
    // Get detection count from last processed frame
    virtual int get_num_detections() const = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_CFAR_DETECTOR_H */
