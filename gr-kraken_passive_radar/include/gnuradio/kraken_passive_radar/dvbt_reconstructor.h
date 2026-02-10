/*
 * DVB-T Reconstructor Public Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_DVBT_RECONSTRUCTOR_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_DVBT_RECONSTRUCTOR_H

#include <gnuradio/kraken_passive_radar/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief Multi-Signal Reference Signal Reconstructor (Block B3)
 * \ingroup kraken_passive_radar
 *
 * Reconstructs a clean reference signal using demodulation-remodulation
 * with signal-specific error correction. This provides a "perfect" reference
 * signal for passive radar processing, improving sensitivity by 10-20 dB in
 * weak signal environments.
 *
 * Supported Signal Types:
 *   - FM Radio: Analog FM broadcast (88-108 MHz, US/worldwide)
 *   - ATSC 3.0: NextGen TV OFDM (US digital TV)
 *   - DVB-T: European/Australian digital TV (OFDM)
 *   - Future: WiFi, LTE, 5G NR
 *
 * FM Signal Flow:
 *   Input: Noisy FM @ 2.4 MSPS
 *     ↓
 *   FM Demodulation (quadrature demod)
 *     ↓
 *   Audio Filtering (0-53 kHz)
 *     ↓
 *   Optional: Pilot Regeneration (19 kHz stereo pilot)
 *     ↓
 *   FM Remodulation (frequency modulator)
 *     ↓
 *   Output: Clean FM reference @ 2.4 MSPS
 *
 * ATSC 3.0 / DVB-T Signal Flow (OFDM):
 *   Input: Noisy OFDM signal @ 2.4 MSPS
 *     ↓
 *   OFDM Demodulation
 *     ↓
 *   FEC Decoding (LDPC for ATSC 3.0, Viterbi+RS for DVB-T)
 *     ↓
 *   Clean Bitstream
 *     ↓
 *   FEC Encoding
 *     ↓
 *   OFDM Remodulation
 *     ↓
 *   SVD Pilot Enhancement (optional)
 *     ↓
 *   Output: Clean OFDM reference @ 2.4 MSPS
 */
class KRAKEN_PASSIVE_RADAR_API dvbt_reconstructor : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<dvbt_reconstructor> sptr;

    /*!
     * \brief Create a multi-signal reconstructor block with runtime selector
     *
     * Unified factory method that creates a reconstructor for the specified
     * signal type. Parameters not applicable to the selected signal type are
     * ignored.
     *
     * Supported signal types:
     *   - "fm": FM Radio (88-108 MHz, US/worldwide)
     *   - "atsc3": ATSC 3.0 NextGen TV (US digital TV)
     *   - "dvbt": DVB-T (Europe/Australia digital TV)
     *   - "passthrough": No processing (Phase 1 behavior)
     *
     * \param signal_type Signal type string ("fm", "atsc3", "dvbt", "passthrough")
     *
     * FM parameters (used when signal_type == "fm"):
     * \param fm_deviation FM deviation in Hz (75e3 for US, 50e3 for Europe)
     * \param enable_stereo Process stereo signal (L+R and L-R)
     * \param enable_pilot_regen Regenerate 19 kHz stereo pilot
     * \param audio_bw Audio bandwidth in Hz (15e3 mono, 53e3 stereo)
     *
     * OFDM parameters (used when signal_type == "atsc3" or "dvbt"):
     * \param fft_size OFDM FFT size
     *                 ATSC 3.0: 8192, 16384, 32768
     *                 DVB-T: 2048 (2K), 4096 (4K), 8192 (8K)
     * \param guard_interval Guard interval
     *                       ATSC 3.0: samples (e.g., 192, 384, 768)
     *                       DVB-T: ratio (4=1/4, 8=1/8, 16=1/16, 32=1/32)
     * \param constellation Modulation (DVB-T: 0=QPSK, 1=16QAM, 2=64QAM)
     * \param code_rate FEC code rate (DVB-T: 0=1/2, 1=2/3, 2=3/4, 3=5/6, 4=7/8)
     * \param pilot_pattern Pilot pattern (ATSC 3.0 specific)
     * \param enable_svd Enable SVD pilot enhancement (OFDM only)
     *
     * \return Shared pointer to the block
     *
     * Example usage:
     * \code
     *   // FM Radio mode (US)
     *   auto fm_recon = make("fm", 75e3, true, true, 15e3);
     *
     *   // ATSC 3.0 mode (US NextGen TV)
     *   auto atsc_recon = make("atsc3", 0, false, false, 0, 8192, 192);
     *
     *   // DVB-T mode (Europe)
     *   auto dvbt_recon = make("dvbt", 0, false, false, 0, 2048, 4, 2, 2);
     * \endcode
     */
    static sptr make(const std::string& signal_type = "passthrough",
                     float fm_deviation = 75e3,
                     bool enable_stereo = true,
                     bool enable_pilot_regen = true,
                     float audio_bw = 15e3,
                     int fft_size = 2048,
                     int guard_interval = 4,
                     int constellation = 2,
                     int code_rate = 2,
                     int pilot_pattern = 0,
                     bool enable_svd = true);

    /*!
     * \brief Enable/disable SVD noise reduction (OFDM signals only)
     *
     * \param enable True to enable SVD pilot enhancement
     */
    virtual void set_enable_svd(bool enable) = 0;

    /*!
     * \brief Enable/disable stereo pilot regeneration (FM only)
     *
     * \param enable True to regenerate 19 kHz pilot
     */
    virtual void set_enable_pilot_regen(bool enable) = 0;

    /*!
     * \brief Get current SNR estimate
     *
     * \return Estimated SNR in dB (constellation error for OFDM, audio SNR for FM)
     */
    virtual float get_snr_estimate() = 0;

    /*!
     * \brief Get current signal type
     *
     * \return Signal type string ("fm", "atsc3", "dvbt", "passthrough")
     */
    virtual std::string get_signal_type() = 0;

    /*!
     * \brief Set signal type (runtime switching)
     *
     * Allows changing the signal type at runtime. The block will reinitialize
     * with the new signal type parameters.
     *
     * \param signal_type New signal type ("fm", "atsc3", "dvbt", "passthrough")
     */
    virtual void set_signal_type(const std::string& signal_type) = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_DVBT_RECONSTRUCTOR_H */
