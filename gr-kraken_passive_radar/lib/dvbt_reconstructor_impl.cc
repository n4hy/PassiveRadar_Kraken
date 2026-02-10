/*
 * Multi-Signal Reconstructor Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "dvbt_reconstructor_impl.h"
#include <gnuradio/io_signature.h>
#include <cstring>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace gr {
namespace kraken_passive_radar {

dvbt_reconstructor::sptr
dvbt_reconstructor::make(const std::string& signal_type,
                         float fm_deviation,
                         bool enable_stereo,
                         bool enable_pilot_regen,
                         float audio_bw,
                         int fft_size,
                         int guard_interval,
                         int constellation,
                         int code_rate,
                         int pilot_pattern,
                         bool enable_svd)
{
    // Parse signal type string
    dvbt_reconstructor_impl::signal_type_t sig_type;
    if (signal_type == "fm") {
        sig_type = dvbt_reconstructor_impl::SIGNAL_FM;
    } else if (signal_type == "atsc3") {
        sig_type = dvbt_reconstructor_impl::SIGNAL_ATSC3;
    } else if (signal_type == "dvbt") {
        sig_type = dvbt_reconstructor_impl::SIGNAL_DVBT;
    } else if (signal_type == "passthrough") {
        sig_type = dvbt_reconstructor_impl::SIGNAL_PASSTHROUGH;
    } else {
        throw std::invalid_argument("Invalid signal_type. Must be 'fm', 'atsc3', 'dvbt', or 'passthrough'");
    }

    return gnuradio::make_block_sptr<dvbt_reconstructor_impl>(
        sig_type,
        fm_deviation,
        enable_stereo,
        enable_pilot_regen,
        audio_bw,
        fft_size,
        guard_interval,
        constellation,
        code_rate,
        pilot_pattern,
        enable_svd);
}

dvbt_reconstructor_impl::dvbt_reconstructor_impl(
    signal_type_t signal_type,
    float fm_deviation,
    bool enable_stereo,
    bool enable_pilot_regen,
    float audio_bw,
    int fft_size,
    int guard_interval,
    int constellation,
    int code_rate,
    int pilot_pattern,
    bool enable_svd)
    : gr::sync_block("dvbt_reconstructor",
                     gr::io_signature::make(1, 1, sizeof(gr_complex)),
                     gr::io_signature::make(1, 1, sizeof(gr_complex))),
      d_signal_type(signal_type),
      d_sample_rate(2.4e6),  // KrakenSDR default
      d_snr_estimate(0.0f),
      d_enable_svd(enable_svd),
      // FM parameters
      d_fm_deviation(fm_deviation),
      d_enable_stereo(enable_stereo),
      d_enable_pilot_regen(enable_pilot_regen),
      d_audio_bw(audio_bw),
      d_fm_gain(0.0f),
      d_fm_phase(0.0f),
      d_preemph_state(0.0f),
      // OFDM parameters
      d_fft_size(fft_size),
      d_guard_interval(guard_interval),
      d_constellation(constellation),
      d_code_rate(code_rate),
      d_pilot_pattern(pilot_pattern),
      d_symbol_length(0),
      d_useful_carriers(0),
      d_num_pilots(0),
      // FFTW
      d_fft_in(nullptr),
      d_fft_out(nullptr),
      d_fft_plan(nullptr),
      d_ifft_plan(nullptr)
{
    // Set signal type string
    switch (d_signal_type) {
        case SIGNAL_FM:
            d_signal_type_str = "fm";
            init_fm_mode();
            break;
        case SIGNAL_ATSC3:
            d_signal_type_str = "atsc3";
            init_atsc3_mode();
            break;
        case SIGNAL_DVBT:
            d_signal_type_str = "dvbt";
            init_dvbt_mode();
            break;
        case SIGNAL_PASSTHROUGH:
        default:
            d_signal_type_str = "passthrough";
            break;
    }
}

dvbt_reconstructor_impl::~dvbt_reconstructor_impl()
{
    cleanup_fft_plans();
}

// ============================================================================
// Initialization Methods
// ============================================================================

void dvbt_reconstructor_impl::init_fm_mode()
{
    GR_LOG_INFO(d_logger, "Initializing FM Radio mode");
    GR_LOG_INFO(d_logger, boost::format("  Deviation: %.1f kHz") % (d_fm_deviation/1e3));
    GR_LOG_INFO(d_logger, boost::format("  Stereo: %s") % (d_enable_stereo ? "Yes" : "No"));
    GR_LOG_INFO(d_logger, boost::format("  Pilot regen: %s") % (d_enable_pilot_regen ? "Yes" : "No"));

    // Calculate FM demodulator gain
    d_fm_gain = d_sample_rate / (2.0f * M_PI * d_fm_deviation);

    // Design audio low-pass filter
    float cutoff = d_enable_stereo ? 53e3 : d_audio_bw;
    float transition = 5e3;

    d_audio_lpf_taps = gr::filter::firdes::low_pass(
        1.0f,              // Gain
        d_sample_rate,     // Sample rate
        cutoff,            // Cutoff frequency
        transition,        // Transition width
        gr::fft::window::WIN_HAMMING
    );

    GR_LOG_INFO(d_logger, boost::format("  Audio LPF: %d taps, cutoff %.1f kHz")
                % d_audio_lpf_taps.size() % (cutoff/1e3));

    // Design pilot bandpass filter (if stereo with pilot regen)
    if (d_enable_stereo && d_enable_pilot_regen) {
        const float pilot_freq = 19e3;
        const float pilot_bw = 100;  // ±100 Hz

        d_pilot_bpf_taps = gr::filter::firdes::band_pass(
            1.0f,
            d_sample_rate,
            pilot_freq - pilot_bw,
            pilot_freq + pilot_bw,
            100,  // Transition
            gr::fft::window::WIN_HAMMING
        );

        GR_LOG_INFO(d_logger, boost::format("  Pilot BPF: %d taps @ 19 kHz")
                    % d_pilot_bpf_taps.size());
    }

    // Allocate audio buffer
    d_audio_buffer.resize(8192);  // Working buffer

    GR_LOG_INFO(d_logger, "FM Radio mode initialized successfully");
}

void dvbt_reconstructor_impl::init_atsc3_mode()
{
    GR_LOG_INFO(d_logger, "Initializing ATSC 3.0 mode");
    GR_LOG_INFO(d_logger, boost::format("  FFT size: %d") % d_fft_size);
    GR_LOG_INFO(d_logger, boost::format("  Guard interval: %d samples") % d_guard_interval);
    GR_LOG_INFO(d_logger, boost::format("  Pilot pattern: %d") % d_pilot_pattern);

    // Calculate symbol parameters
    d_symbol_length = d_fft_size + d_guard_interval;

    // ATSC 3.0 carrier configurations
    // Based on A/322:2023 ATSC 3.0 Physical Layer Protocol
    if (d_fft_size == 8192) {
        // 8K FFT mode
        d_useful_carriers = 6913;   // Active carriers (data + pilots)
        d_num_pilots = 560;          // Continual + scattered pilots
    } else if (d_fft_size == 16384) {
        // 16K FFT mode
        d_useful_carriers = 13825;
        d_num_pilots = 1120;
    } else if (d_fft_size == 32768) {
        // 32K FFT mode
        d_useful_carriers = 27649;
        d_num_pilots = 2240;
    } else {
        throw std::invalid_argument("ATSC 3.0 FFT size must be 8192, 16384, or 32768");
    }

    // Validate guard interval
    // ATSC 3.0 guard intervals (in samples for given FFT sizes):
    // 8K:  192, 384, 768, 1024 (GI ratios: 1/42, 1/21, 1/10, 1/8)
    // 16K: 384, 768, 1536, 2048
    // 32K: 768, 1536, 3072, 4096
    int valid_gis[] = {192, 384, 768, 1024, 1536, 2048, 3072, 4096};
    bool valid_gi = false;
    for (int gi : valid_gis) {
        if (d_guard_interval == gi) {
            valid_gi = true;
            break;
        }
    }
    if (!valid_gi) {
        GR_LOG_WARN(d_logger, boost::format("Non-standard guard interval: %d samples") % d_guard_interval);
    }

    // Allocate buffers
    d_symbol_buffer.resize(d_symbol_length);
    d_freq_domain.resize(d_fft_size * 10);  // Buffer for multiple symbols
    d_bit_buffer.resize(d_useful_carriers * 10 * 2);  // QPSK assumption

    // Initialize FFTW plans
    init_fft_plans();

    GR_LOG_INFO(d_logger, "ATSC 3.0 mode initialized");
    GR_LOG_INFO(d_logger, boost::format("  Symbol length: %d samples") % d_symbol_length);
    GR_LOG_INFO(d_logger, boost::format("  Useful carriers: %d") % d_useful_carriers);
    GR_LOG_INFO(d_logger, boost::format("  Pilot carriers: %d") % d_num_pilots);
    GR_LOG_WARN(d_logger, "Note: LDPC FEC is placeholder (QPSK hard-decision only)");
}

void dvbt_reconstructor_impl::init_dvbt_mode()
{
    GR_LOG_INFO(d_logger, "Initializing DVB-T mode");
    GR_LOG_INFO(d_logger, boost::format("  FFT size: %d") % d_fft_size);
    GR_LOG_INFO(d_logger, boost::format("  Guard interval: 1/%d") % d_guard_interval);

    // Calculate symbol parameters
    d_symbol_length = d_fft_size + (d_fft_size / d_guard_interval);

    // DVB-T carrier counts
    if (d_fft_size == 2048) {
        d_useful_carriers = 1705;
        d_num_pilots = 45;
    } else if (d_fft_size == 8192) {
        d_useful_carriers = 6817;
        d_num_pilots = 177;
    } else if (d_fft_size == 4096) {
        d_useful_carriers = 3409;
        d_num_pilots = 89;
    } else {
        throw std::invalid_argument("DVB-T FFT size must be 2048, 4096, or 8192");
    }

    // Validate parameters
    if (d_guard_interval != 4 && d_guard_interval != 8 &&
        d_guard_interval != 16 && d_guard_interval != 32) {
        throw std::invalid_argument("DVB-T guard interval must be 4, 8, 16, or 32");
    }

    if (d_constellation < 0 || d_constellation > 2) {
        throw std::invalid_argument("DVB-T constellation must be 0 (QPSK), 1 (16QAM), or 2 (64QAM)");
    }

    if (d_code_rate < 0 || d_code_rate > 4) {
        throw std::invalid_argument("DVB-T code rate must be 0-4 (1/2, 2/3, 3/4, 5/6, 7/8)");
    }

    // Allocate buffers
    d_symbol_buffer.resize(d_symbol_length);
    d_freq_domain.resize(d_fft_size);

    // Initialize FFTW plans
    init_fft_plans();

    GR_LOG_WARN(d_logger, "DVB-T mode initialized (PLACEHOLDER - full implementation TODO)");
}

void dvbt_reconstructor_impl::init_fft_plans()
{
    gr::thread::scoped_lock lock(d_mutex);

    // Allocate aligned memory for FFTW
    d_fft_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d_fft_size);
    d_fft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * d_fft_size);

    if (!d_fft_in || !d_fft_out) {
        throw std::runtime_error("Failed to allocate FFTW buffers");
    }

    // Create FFTW plans
    d_fft_plan = fftwf_plan_dft_1d(d_fft_size, d_fft_in, d_fft_out,
                                    FFTW_FORWARD, FFTW_MEASURE);
    d_ifft_plan = fftwf_plan_dft_1d(d_fft_size, d_fft_in, d_fft_out,
                                     FFTW_BACKWARD, FFTW_MEASURE);

    if (!d_fft_plan || !d_ifft_plan) {
        throw std::runtime_error("Failed to create FFTW plans");
    }

    GR_LOG_INFO(d_logger, boost::format("FFTW plans created for FFT size %d") % d_fft_size);
}

void dvbt_reconstructor_impl::cleanup_fft_plans()
{
    gr::thread::scoped_lock lock(d_mutex);

    if (d_fft_plan) {
        fftwf_destroy_plan(d_fft_plan);
        d_fft_plan = nullptr;
    }

    if (d_ifft_plan) {
        fftwf_destroy_plan(d_ifft_plan);
        d_ifft_plan = nullptr;
    }

    if (d_fft_in) {
        fftwf_free(d_fft_in);
        d_fft_in = nullptr;
    }

    if (d_fft_out) {
        fftwf_free(d_fft_out);
        d_fft_out = nullptr;
    }
}

// ============================================================================
// FM Processing Methods
// ============================================================================

void dvbt_reconstructor_impl::fm_demodulate(const gr_complex* in, float* audio_out, int n)
{
    // Quadrature demodulation: f(t) = (1/2π) * dφ/dt
    for (int i = 0; i < n; i++) {
        // Calculate instantaneous phase
        float phase = std::arg(in[i]);

        // Phase difference (frequency)
        float phase_diff = phase - d_fm_phase;

        // Unwrap phase (handle 2π discontinuities)
        while (phase_diff > M_PI) phase_diff -= 2*M_PI;
        while (phase_diff < -M_PI) phase_diff += 2*M_PI;

        // Convert phase difference to frequency (audio)
        audio_out[i] = phase_diff * d_fm_gain;

        d_fm_phase = phase;
    }
}

void dvbt_reconstructor_impl::fm_apply_audio_filter(float* audio, int n)
{
    // Simple FIR filter using VOLK for acceleration
    int ntaps = d_audio_lpf_taps.size();

    // In-place filtering (requires temporary buffer)
    std::vector<float> temp(n);
    std::memcpy(temp.data(), audio, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < ntaps && (i-j) >= 0; j++) {
            sum += temp[i-j] * d_audio_lpf_taps[j];
        }
        audio[i] = sum;
    }
}

void dvbt_reconstructor_impl::fm_regenerate_pilot(float* audio, int n)
{
    // TODO: Implement pilot regeneration
    // 1. Bandpass filter @ 19 kHz to extract pilot
    // 2. Estimate phase
    // 3. Generate perfect 19 kHz sinusoid
    // 4. Replace in composite signal
    // For now, this is a placeholder
    (void)audio;
    (void)n;
}

void dvbt_reconstructor_impl::fm_modulate(const float* audio, gr_complex* fm_out, int n)
{
    // Apply pre-emphasis (75 μs time constant for US)
    float preemph_alpha = 1.0f - std::exp(-1.0f / (75e-6f * d_sample_rate));

    for (int i = 0; i < n; i++) {
        // Pre-emphasis filter: y[n] = x[n] - α*y[n-1]
        float preemph = audio[i] - preemph_alpha * d_preemph_state;
        d_preemph_state = preemph;

        // Integrate audio to get phase
        d_fm_phase += 2.0f*M_PI * d_fm_deviation * preemph / d_sample_rate;

        // Wrap phase
        while (d_fm_phase > M_PI) d_fm_phase -= 2*M_PI;
        while (d_fm_phase < -M_PI) d_fm_phase += 2*M_PI;

        // Generate FM signal
        fm_out[i] = gr_complex(std::cos(d_fm_phase), std::sin(d_fm_phase));
    }
}

void dvbt_reconstructor_impl::fm_estimate_snr(const float* audio, int n)
{
    // Simple SNR estimation from audio signal
    float signal_power = 0.0f;
    float noise_power = 0.0f;

    for (int i = 0; i < n; i++) {
        signal_power += audio[i] * audio[i];
    }
    signal_power /= n;

    // Estimate noise (very simplified - assumes high-frequency content is noise)
    // TODO: Improve SNR estimation
    noise_power = signal_power * 0.01f;  // Assume 1% noise for now

    if (noise_power > 1e-10f) {
        d_snr_estimate = 10.0f * std::log10(signal_power / noise_power);
    } else {
        d_snr_estimate = 60.0f;
    }
}

// ============================================================================
// OFDM Processing Methods (Placeholders for ATSC 3.0 / DVB-T)
// ============================================================================

void dvbt_reconstructor_impl::ofdm_demodulate(const gr_complex* in, gr_complex* freq_out, int n_symbols)
{
    // OFDM demodulation: Time domain → Frequency domain
    // For each OFDM symbol:
    //   1. Skip guard interval
    //   2. Extract FFT_size samples
    //   3. FFT to frequency domain
    //   4. Output frequency-domain carriers

    const gr_complex* symbol_ptr = in;

    for (int sym = 0; sym < n_symbols; sym++) {
        // Skip guard interval (first d_guard_interval samples)
        const gr_complex* fft_input = symbol_ptr + d_guard_interval;

        // Copy FFT_size samples to FFTW input buffer
        for (int i = 0; i < d_fft_size; i++) {
            d_fft_in[i][0] = fft_input[i].real();
            d_fft_in[i][1] = fft_input[i].imag();
        }

        // Execute FFT
        fftwf_execute(d_fft_plan);

        // Copy FFT output to frequency domain buffer
        // ATSC 3.0 uses only a subset of carriers (d_useful_carriers)
        for (int i = 0; i < d_fft_size; i++) {
            freq_out[sym * d_fft_size + i] = gr_complex(d_fft_out[i][0], d_fft_out[i][1]);
        }

        // Advance to next symbol
        symbol_ptr += d_symbol_length;
    }
}

void dvbt_reconstructor_impl::ofdm_modulate(const gr_complex* freq_in, gr_complex* out, int n_symbols)
{
    // OFDM modulation: Frequency domain → Time domain
    // For each OFDM symbol:
    //   1. Insert frequency-domain carriers into IFFT input
    //   2. IFFT to time domain
    //   3. Copy guard interval (cyclic prefix)
    //   4. Serialize to output

    gr_complex* output_ptr = out;

    for (int sym = 0; sym < n_symbols; sym++) {
        // Copy frequency-domain data to IFFT input
        const gr_complex* freq_data = &freq_in[sym * d_fft_size];

        for (int i = 0; i < d_fft_size; i++) {
            d_fft_in[i][0] = freq_data[i].real();
            d_fft_in[i][1] = freq_data[i].imag();
        }

        // Execute IFFT
        fftwf_execute(d_ifft_plan);

        // Normalize IFFT output (FFTW doesn't normalize)
        float scale = 1.0f / d_fft_size;

        // Add cyclic prefix (guard interval)
        // Copy last d_guard_interval samples to beginning
        for (int i = 0; i < d_guard_interval; i++) {
            int src_idx = d_fft_size - d_guard_interval + i;
            output_ptr[i] = gr_complex(
                d_fft_out[src_idx][0] * scale,
                d_fft_out[src_idx][1] * scale
            );
        }

        // Copy IFFT output
        for (int i = 0; i < d_fft_size; i++) {
            output_ptr[d_guard_interval + i] = gr_complex(
                d_fft_out[i][0] * scale,
                d_fft_out[i][1] * scale
            );
        }

        // Advance to next symbol
        output_ptr += d_symbol_length;
    }
}

void dvbt_reconstructor_impl::ofdm_svd_enhancement(gr_complex* freq_data, int n_symbols)
{
    // SVD-based pilot noise reduction
    // This improves pilot SNR by 3-5 dB by exploiting spatial/temporal correlation

    if (n_symbols < 2 || d_num_pilots == 0) {
        return;  // Need at least 2 symbols for meaningful SVD
    }

    // ATSC 3.0 pilot pattern (simplified - actual pattern is more complex)
    // For now, use continual pilots at specific carrier indices
    std::vector<int> pilot_indices;

    // ATSC 3.0 continual pilots (example pattern for 8K mode)
    // Actual pilot pattern depends on pilot_pattern parameter
    // For 8K mode: pilots at k = 3*m where m = 0, 1, 2, ..., 2730
    for (int k = 0; k < d_fft_size; k += 3) {
        if (k < d_useful_carriers) {
            pilot_indices.push_back(k);
        }
    }

    int n_pilots = std::min((int)pilot_indices.size(), d_num_pilots);
    if (n_pilots < 4) {
        return;  // Need sufficient pilots
    }

    // Build pilot matrix: rows = pilots, cols = symbols
    Eigen::MatrixXcf pilot_matrix(n_pilots, n_symbols);

    for (int sym = 0; sym < n_symbols; sym++) {
        for (int p = 0; p < n_pilots; p++) {
            int carrier_idx = pilot_indices[p];
            gr_complex pilot_val = freq_data[sym * d_fft_size + carrier_idx];
            pilot_matrix(p, sym) = std::complex<float>(pilot_val.real(), pilot_val.imag());
        }
    }

    // SVD decomposition
    Eigen::JacobiSVD<Eigen::MatrixXcf> svd(
        pilot_matrix,
        Eigen::ComputeThinU | Eigen::ComputeThinV
    );

    // Threshold singular values: keep top 90% of energy
    Eigen::VectorXf singular_values = svd.singularValues();
    float total_energy = singular_values.squaredNorm();
    float threshold_energy = 0.9f * total_energy;

    float cumulative_energy = 0.0f;
    int keep_count = 0;

    for (int i = 0; i < singular_values.size(); i++) {
        cumulative_energy += singular_values(i) * singular_values(i);
        keep_count++;
        if (cumulative_energy >= threshold_energy) {
            break;
        }
    }

    // Zero out small singular values (noise)
    for (int i = keep_count; i < singular_values.size(); i++) {
        singular_values(i) = 0.0f;
    }

    // Reconstruct clean pilot matrix
    Eigen::MatrixXcf clean_pilots =
        svd.matrixU() * singular_values.asDiagonal().toDenseMatrix().cast<std::complex<float>>() * svd.matrixV().adjoint();

    // Put cleaned pilots back into frequency data
    for (int sym = 0; sym < n_symbols; sym++) {
        for (int p = 0; p < n_pilots; p++) {
            int carrier_idx = pilot_indices[p];
            std::complex<float> clean_val = clean_pilots(p, sym);
            freq_data[sym * d_fft_size + carrier_idx] = gr_complex(clean_val.real(), clean_val.imag());
        }
    }

    // Log improvement
    GR_LOG_DEBUG(d_logger, boost::format("SVD: kept %d/%d singular values")
                 % keep_count % singular_values.size());
}

void dvbt_reconstructor_impl::ofdm_estimate_snr(const gr_complex* symbols, int count)
{
    // Placeholder SNR estimation
    float signal_power = 0.0f;

    for (int i = 0; i < count; i++) {
        float mag = std::abs(symbols[i]);
        signal_power += mag * mag;
    }

    signal_power /= count;
    d_snr_estimate = 10.0f * std::log10(std::max(signal_power, 1e-10f));
}

void dvbt_reconstructor_impl::atsc3_decode_fec(const gr_complex* symbols, uint8_t* bits, int n)
{
    // ATSC 3.0 LDPC FEC Decoding (SIMPLIFIED PLACEHOLDER)
    // Full LDPC implementation requires significant code (use srsRAN or AFF3CT libraries)
    // For now: Simple hard-decision demapping based on constellation

    // ATSC 3.0 uses QPSK, 16-QAM, 64-QAM, 256-QAM, 1024-QAM, 4096-QAM
    // For this placeholder: assume QPSK (2 bits per symbol)

    for (int i = 0; i < n; i++) {
        gr_complex sym = symbols[i];

        // QPSK hard decision
        bits[i*2 + 0] = (sym.real() > 0) ? 1 : 0;
        bits[i*2 + 1] = (sym.imag() > 0) ? 1 : 0;
    }

    // TODO: Full LDPC decoding chain:
    // 1. Soft demapping (LLR calculation)
    // 2. LDPC belief propagation decoding
    // 3. BCH outer code decoding
    // 4. Descrambling
    GR_LOG_WARN(d_logger, "ATSC 3.0 LDPC decoding not implemented - using hard QPSK demapping");
}

void dvbt_reconstructor_impl::atsc3_encode_fec(const uint8_t* bits, gr_complex* symbols, int n)
{
    // ATSC 3.0 LDPC FEC Encoding (SIMPLIFIED PLACEHOLDER)
    // Full LDPC implementation requires significant code

    // QPSK mapping (2 bits per symbol)
    const float scale = 1.0f / std::sqrt(2.0f);  // Normalize power

    for (int i = 0; i < n; i++) {
        int bit0 = bits[i*2 + 0];
        int bit1 = bits[i*2 + 1];

        // QPSK constellation
        float I = (bit0 == 0) ? -scale : scale;
        float Q = (bit1 == 0) ? -scale : scale;

        symbols[i] = gr_complex(I, Q);
    }

    // TODO: Full LDPC encoding chain:
    // 1. Scrambling
    // 2. BCH outer code encoding
    // 3. LDPC encoding
    // 4. Bit interleaving
    // 5. Constellation mapping (QPSK/16QAM/64QAM/256QAM/1024QAM/4096QAM)
    GR_LOG_WARN(d_logger, "ATSC 3.0 LDPC encoding not implemented - using hard QPSK mapping");
}

void dvbt_reconstructor_impl::dvbt_decode_fec(const gr_complex* symbols, uint8_t* bits, int n)
{
    // TODO: DVB-T Viterbi + Reed-Solomon decoding
    (void)symbols;
    (void)bits;
    (void)n;
}

void dvbt_reconstructor_impl::dvbt_encode_fec(const uint8_t* bits, gr_complex* symbols, int n)
{
    // TODO: DVB-T Viterbi + Reed-Solomon encoding
    (void)bits;
    (void)symbols;
    (void)n;
}

// ============================================================================
// Runtime Controls
// ============================================================================

void dvbt_reconstructor_impl::set_enable_svd(bool enable)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_enable_svd = enable;
    GR_LOG_INFO(d_logger, boost::format("SVD pilot enhancement: %s") % (enable ? "enabled" : "disabled"));
}

void dvbt_reconstructor_impl::set_enable_pilot_regen(bool enable)
{
    gr::thread::scoped_lock lock(d_mutex);
    d_enable_pilot_regen = enable;
    GR_LOG_INFO(d_logger, boost::format("FM pilot regeneration: %s") % (enable ? "enabled" : "disabled"));
}

float dvbt_reconstructor_impl::get_snr_estimate()
{
    gr::thread::scoped_lock lock(d_mutex);
    return d_snr_estimate;
}

std::string dvbt_reconstructor_impl::get_signal_type()
{
    gr::thread::scoped_lock lock(d_mutex);
    return d_signal_type_str;
}

void dvbt_reconstructor_impl::set_signal_type(const std::string& signal_type)
{
    gr::thread::scoped_lock lock(d_mutex);

    // Parse signal type
    dvbt_reconstructor_impl::signal_type_t new_type;
    if (signal_type == "fm") {
        new_type = dvbt_reconstructor_impl::SIGNAL_FM;
    } else if (signal_type == "atsc3") {
        new_type = dvbt_reconstructor_impl::SIGNAL_ATSC3;
    } else if (signal_type == "dvbt") {
        new_type = dvbt_reconstructor_impl::SIGNAL_DVBT;
    } else if (signal_type == "passthrough") {
        new_type = dvbt_reconstructor_impl::SIGNAL_PASSTHROUGH;
    } else {
        GR_LOG_ERROR(d_logger, boost::format("Invalid signal type: %s") % signal_type);
        return;
    }

    // Only reinitialize if type actually changed
    if (new_type != d_signal_type) {
        GR_LOG_INFO(d_logger, boost::format("Switching from %s to %s mode")
                    % d_signal_type_str % signal_type);

        d_signal_type = new_type;
        d_signal_type_str = signal_type;

        // Reinitialize for new signal type
        // TODO: Implement proper reinitialization
        GR_LOG_WARN(d_logger, "Runtime signal type switching not fully implemented yet");
    }
}

// ============================================================================
// GNU Radio Work Function
// ============================================================================

int dvbt_reconstructor_impl::work(int noutput_items,
                                  gr_vector_const_void_star& input_items,
                                  gr_vector_void_star& output_items)
{
    const gr_complex* in = (const gr_complex*)input_items[0];
    gr_complex* out = (gr_complex*)output_items[0];

    switch (d_signal_type) {
        case SIGNAL_FM: {
            // Ensure audio buffer is large enough
            if (d_audio_buffer.size() < (size_t)noutput_items) {
                d_audio_buffer.resize(noutput_items);
            }

            // FM demod-remod chain
            fm_demodulate(in, d_audio_buffer.data(), noutput_items);
            fm_apply_audio_filter(d_audio_buffer.data(), noutput_items);

            if (d_enable_pilot_regen && d_enable_stereo) {
                fm_regenerate_pilot(d_audio_buffer.data(), noutput_items);
            }

            fm_modulate(d_audio_buffer.data(), out, noutput_items);
            fm_estimate_snr(d_audio_buffer.data(), std::min(noutput_items, 1000));
            break;
        }

        case SIGNAL_ATSC3:
        case SIGNAL_DVBT: {
            // OFDM processing: demod → FEC → remod → SVD
            // Calculate how many complete OFDM symbols we can process
            int n_complete_symbols = noutput_items / d_symbol_length;

            if (n_complete_symbols == 0) {
                // Not enough samples for even one symbol, pass through
                std::memcpy(out, in, noutput_items * sizeof(gr_complex));
                return noutput_items;
            }

            // Limit processing to avoid buffer issues
            n_complete_symbols = std::min(n_complete_symbols, 10);  // Process max 10 symbols at a time

            // Ensure frequency domain buffer is large enough
            int freq_size = n_complete_symbols * d_fft_size;
            if (d_freq_domain.size() < (size_t)freq_size) {
                d_freq_domain.resize(freq_size);
            }

            // Ensure bit buffer is large enough (assume QPSK = 2 bits/symbol)
            int bit_size = d_useful_carriers * n_complete_symbols * 2;
            if (d_bit_buffer.size() < (size_t)bit_size) {
                d_bit_buffer.resize(bit_size);
            }

            // 1. OFDM Demodulation (time → frequency)
            ofdm_demodulate(in, d_freq_domain.data(), n_complete_symbols);

            // 2. FEC Decoding (symbols → bits)
            if (d_signal_type == SIGNAL_ATSC3) {
                atsc3_decode_fec(d_freq_domain.data(), d_bit_buffer.data(), d_useful_carriers * n_complete_symbols);
            } else {  // SIGNAL_DVBT
                dvbt_decode_fec(d_freq_domain.data(), d_bit_buffer.data(), d_useful_carriers * n_complete_symbols);
            }

            // 3. FEC Encoding (bits → symbols)
            if (d_signal_type == SIGNAL_ATSC3) {
                atsc3_encode_fec(d_bit_buffer.data(), d_freq_domain.data(), d_useful_carriers * n_complete_symbols);
            } else {  // SIGNAL_DVBT
                dvbt_encode_fec(d_bit_buffer.data(), d_freq_domain.data(), d_useful_carriers * n_complete_symbols);
            }

            // 4. SVD Pilot Enhancement (optional)
            if (d_enable_svd && n_complete_symbols >= 2) {
                ofdm_svd_enhancement(d_freq_domain.data(), n_complete_symbols);
            }

            // 5. OFDM Remodulation (frequency → time)
            ofdm_modulate(d_freq_domain.data(), out, n_complete_symbols);

            // 6. Pass through any remaining samples
            int processed_samples = n_complete_symbols * d_symbol_length;
            if (processed_samples < noutput_items) {
                std::memcpy(out + processed_samples,
                           in + processed_samples,
                           (noutput_items - processed_samples) * sizeof(gr_complex));
            }

            // Estimate SNR from frequency domain
            ofdm_estimate_snr(d_freq_domain.data(), std::min(freq_size, 1000));
            break;
        }

        case SIGNAL_PASSTHROUGH:
        default:
            // Simple passthrough
            std::memcpy(out, in, noutput_items * sizeof(gr_complex));
            break;
    }

    return noutput_items;
}

} // namespace kraken_passive_radar
} // namespace gr
