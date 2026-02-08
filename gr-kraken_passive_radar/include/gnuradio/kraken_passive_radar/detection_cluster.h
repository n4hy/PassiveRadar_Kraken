/*
 * Detection Clustering Block for KrakenSDR Passive Radar
 *
 * Merges adjacent CFAR detections into target clusters using
 * connected components analysis with centroid computation.
 *
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_DETECTION_CLUSTER_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_DETECTION_CLUSTER_H

#include <gnuradio/sync_block.h>
#include <gnuradio/kraken_passive_radar/api.h>
#include <vector>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief Detection structure for clustered targets
 */
struct detection_t {
    int id;              // Detection ID (per frame)
    float range_bin;     // Centroid range bin (fractional)
    float doppler_bin;   // Centroid Doppler bin (fractional)
    float range_m;       // Range in meters
    float doppler_hz;    // Doppler shift in Hz
    float snr_db;        // Peak SNR in dB
    float power_sum;     // Total power in cluster
    int cluster_size;    // Number of cells in cluster
    int peak_range;      // Range bin of peak cell
    int peak_doppler;    // Doppler bin of peak cell
};

/*!
 * \brief Detection Clustering for passive radar
 * \ingroup kraken_passive_radar
 *
 * Takes a binary detection mask from CFAR and the original power
 * map, performs connected components labeling to group adjacent
 * detections, then computes centroid and statistics for each cluster.
 *
 * Connected components uses 8-connectivity (includes diagonals).
 *
 * Input 0: CFAR detection mask (float, 1.0 = detection)
 * Input 1: Power map in dB (float, for centroid weighting)
 * Output 0: Detection list as packed float vector
 *
 * Output format per detection (10 floats):
 *   [id, range_bin, doppler_bin, range_m, doppler_hz,
 *    snr_db, power_sum, cluster_size, peak_range, peak_doppler]
 */
class KRAKEN_PASSIVE_RADAR_API detection_cluster : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<detection_cluster> sptr;

    /*!
     * \brief Create detection clustering block
     * \param num_range_bins Number of range bins
     * \param num_doppler_bins Number of Doppler bins
     * \param min_cluster_size Minimum cells to form valid detection
     * \param max_cluster_extent Maximum cluster size before split
     * \param range_resolution_m Range resolution in meters
     * \param doppler_resolution_hz Doppler resolution in Hz
     * \param max_detections Maximum detections to output per frame
     */
    static sptr make(int num_range_bins,
                     int num_doppler_bins,
                     int min_cluster_size = 1,
                     int max_cluster_extent = 50,
                     float range_resolution_m = 600.0f,
                     float doppler_resolution_hz = 3.9f,
                     int max_detections = 100);

    virtual void set_min_cluster_size(int size) = 0;
    virtual void set_max_cluster_extent(int extent) = 0;
    virtual void set_range_resolution(float res_m) = 0;
    virtual void set_doppler_resolution(float res_hz) = 0;

    // Get detections from last frame
    virtual std::vector<detection_t> get_detections() const = 0;
    virtual int get_num_detections() const = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_DETECTION_CLUSTER_H */
