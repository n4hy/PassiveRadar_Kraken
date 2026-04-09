/*
 * Multi-Signal Reconstructor Implementation Header
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_DVBT_RECONSTRUCTOR_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_DVBT_RECONSTRUCTOR_IMPL_H

#include <gnuradio/kraken_passive_radar/dvbt_reconstructor.h>
#include <gnuradio/thread/thread.h>
#include <gnuradio/filter/firdes.h>
#include <fftw3.h>
#include <volk/volk.h>
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <string>

namespace gr {
namespace kraken_passive_radar {

/**
 * dvbt_reconstructor_impl - Implementation of the multi-signal reference reconstructor block
 *
 * Technique: Demodulation-remodulation with signal-specific error correction.
 * FM uses quadrature demod/remod with optional stereo pilot regeneration.
 * OFDM (ATSC 3.0, DVB-T) uses FFT-based demod, FEC decode/encode, and
 * optional SVD pilot enhancement for improved reference signal quality.
 */
class dvbt_reconstructor_impl : public dvbt_reconstructor
{
public:
    /**
     * signal_type_t - Enumeration of supported signal types for reconstruction
     */
    enum signal_type_t {
        SIGNAL_PASSTHROUGH,  // Phase 1: no processing
        SIGNAL_FM,           // FM Radio
        SIGNAL_ATSC3,        // ATSC 3.0 OFDM
        SIGNAL_DVBT          // DVB-T OFDM
    };

private:
    signal_type_t d_signal_type;
    std::string d_signal_type_str;

    // Common parameters
    float d_sample_rate;
    float d_snr_estimate;
    bool d_enable_svd;

    // === FM-specific members ===
    float d_fm_deviation;
    bool d_enable_stereo;
    bool d_enable_pilot_regen;
    float d_audio_bw;
    float d_fm_gain;              // sample_rate / (2*pi*deviation)
    float d_fm_phase;             // Phase accumulator for demod
    std::vector<float> d_audio_lpf_taps;
    std::vector<float> d_pilot_bpf_taps;
    std::vector<float> d_audio_buffer;
    float d_preemph_state;        // Pre-emphasis filter state

    // === OFDM-specific members (ATSC 3.0 / DVB-T) ===
    int d_fft_size;
    int d_guard_interval;
    int d_constellation;
    int d_code_rate;
    int d_pilot_pattern;          // ATSC 3.0 specific

    // OFDM symbol parameters
    int d_symbol_length;          // FFT size + guard interval
    int d_useful_carriers;        // Number of active carriers
    int d_num_pilots;             // Number of pilot carriers

    // === FFTW resources ===
    fftwf_complex *d_fft_in;
    fftwf_complex *d_fft_out;
    fftwf_plan d_fft_plan;
    fftwf_plan d_ifft_plan;

    // === Internal buffers ===
    std::vector<gr_complex> d_symbol_buffer;
    std::vector<gr_complex> d_freq_domain;
    std::vector<uint8_t> d_bit_buffer;

    // Thread safety
    gr::thread::mutex d_mutex;

    /** init_fm_mode - Initialize FM demodulation/remodulation pipeline and filter taps */
    void init_fm_mode();
    /** init_atsc3_mode - Initialize ATSC 3.0 OFDM demod/remod with LDPC FEC */
    void init_atsc3_mode();
    /** init_dvbt_mode - Initialize DVB-T OFDM demod/remod with Viterbi+RS FEC */
    void init_dvbt_mode();
    /** init_fft_plans - Allocate FFTW buffers and create forward/inverse FFT plans */
    void init_fft_plans();
    /** cleanup_fft_plans - Destroy FFTW plans and free aligned buffers */
    void cleanup_fft_plans();

    /** fm_demodulate - Perform quadrature FM demodulation to extract audio samples */
    void fm_demodulate(const gr_complex* in, float* audio_out, int n);
    /** fm_apply_audio_filter - Apply low-pass filter to demodulated audio */
    void fm_apply_audio_filter(float* audio, int n);
    /** fm_regenerate_pilot - Regenerate the 19 kHz stereo pilot tone in the audio signal */
    void fm_regenerate_pilot(float* audio, int n);
    /** fm_modulate - Frequency-modulate filtered audio back to a clean FM reference signal */
    void fm_modulate(const float* audio, gr_complex* fm_out, int n);
    /** fm_estimate_snr - Estimate audio-domain SNR from the demodulated signal */
    void fm_estimate_snr(const float* audio, int n);

    /** ofdm_demodulate - FFT-based OFDM demodulation to extract frequency-domain subcarriers */
    void ofdm_demodulate(const gr_complex* in, gr_complex* freq_out, int n_symbols);
    /** ofdm_modulate - IFFT-based OFDM remodulation from clean frequency-domain data */
    void ofdm_modulate(const gr_complex* freq_in, gr_complex* out, int n_symbols);
    /** ofdm_svd_enhancement - Apply SVD-based pilot enhancement for noise reduction on OFDM subcarriers */
    void ofdm_svd_enhancement(gr_complex* freq_data, int n_symbols);
    /** ofdm_estimate_snr - Estimate SNR from constellation error on demodulated OFDM symbols */
    void ofdm_estimate_snr(const gr_complex* symbols, int count);

    /** atsc3_decode_fec - Decode ATSC 3.0 LDPC forward error correction from OFDM symbols to bits */
    void atsc3_decode_fec(const gr_complex* symbols, uint8_t* bits, int n);
    /** atsc3_encode_fec - Encode bits back to ATSC 3.0 LDPC-coded OFDM symbols */
    void atsc3_encode_fec(const uint8_t* bits, gr_complex* symbols, int n);

    /** dvbt_decode_fec - Decode DVB-T Viterbi+Reed-Solomon FEC from OFDM symbols to bits */
    void dvbt_decode_fec(const gr_complex* symbols, uint8_t* bits, int n);
    /** dvbt_encode_fec - Encode bits back to DVB-T Viterbi+RS coded OFDM symbols */
    void dvbt_encode_fec(const uint8_t* bits, gr_complex* symbols, int n);

public:
    /**
     * dvbt_reconstructor_impl - Construct reconstructor for the specified signal type with given parameters
     */
    dvbt_reconstructor_impl(signal_type_t signal_type,
                            float fm_deviation,
                            bool enable_stereo,
                            bool enable_pilot_regen,
                            float audio_bw,
                            int fft_size,
                            int guard_interval,
                            int constellation,
                            int code_rate,
                            int pilot_pattern,
                            bool enable_svd);

    ~dvbt_reconstructor_impl();

    /** set_enable_svd - Enable or disable SVD pilot enhancement for OFDM signals */
    void set_enable_svd(bool enable) override;
    /** set_enable_pilot_regen - Enable or disable 19 kHz stereo pilot regeneration for FM */
    void set_enable_pilot_regen(bool enable) override;
    /** get_snr_estimate - Return the current SNR estimate in dB */
    float get_snr_estimate() override;
    /** get_signal_type - Return the current signal type string */
    std::string get_signal_type() override;
    /** set_signal_type - Switch signal type at runtime and reinitialize processing pipeline */
    void set_signal_type(const std::string& signal_type) override;

    /**
     * work - Demodulate, error-correct, and remodulate input to produce clean reference signal
     */
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_DVBT_RECONSTRUCTOR_IMPL_H */
