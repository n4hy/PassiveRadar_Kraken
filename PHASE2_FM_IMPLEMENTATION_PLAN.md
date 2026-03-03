# Phase 2: FM Radio Reconstruction - Implementation Plan

**Priority:** 🎯 IMMEDIATE (US Market Critical)
**Timeline:** 2 weeks
**Status:** Ready to begin

---

## Why FM Radio First?

✅ **US Market Applicability**
- Ubiquitous 88-108 MHz FM stations across entire USA
- High-power transmitters (50-100 kW) = excellent range
- 24/7 continuous transmission

✅ **Simplest Implementation**
- No OFDM processing needed
- No complex FEC chains
- Analog modulation (straightforward demod/remod)
- Can implement in ~2 weeks vs. 4-7 weeks for OFDM signals

✅ **Proven for Passive Radar**
- Extensively documented in research literature
- Known to work well for bistatic radar
- Many successful deployments worldwide

✅ **Low Computational Load**
- <10% CPU usage (vs. 60-90% for LTE/5G)
- No GPU needed
- Real-time guaranteed at 2.4 MSPS

---

## FM Signal Technical Background

### Frequency Modulation Basics

FM radio encodes audio by varying the carrier frequency:
```
f(t) = f_c + k_f * m(t)
```
Where:
- `f_c` = carrier frequency (e.g., 100.1 MHz)
- `k_f` = frequency sensitivity
- `m(t)` = audio message signal

### FM Stereo (Multiplexed)

```
Composite Baseband Signal (0-53 kHz):
  0-15 kHz:     L+R (mono-compatible audio)
  19 kHz:       Stereo pilot (phase reference)
  23-53 kHz:    L-R (stereo difference) modulated on 38 kHz subcarrier
  57 kHz:       RDS data subcarrier (optional)
```

### Parameters for Block B3
- **FM Deviation:** 75 kHz (US standard)
- **Audio Bandwidth:** 15 kHz (mono) or 53 kHz (stereo)
- **Pre-emphasis:** 75 μs time constant (US)
- **Channel Spacing:** 200 kHz

---

## Implementation Architecture

### Signal Flow Diagram

```
Input: Noisy FM @ 2.4 MSPS (IQ samples from KrakenSDR)
    ↓
┌─────────────────────────────────────────────────────┐
│  DEMODULATION                                       │
│  • Quadrature demodulator (GNU Radio)              │
│  • Output: baseband audio (0-53 kHz)               │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  AUDIO PROCESSING                                   │
│  • Low-pass filter (15 kHz mono or 53 kHz stereo)  │
│  • Optional: Bandpass filter 19 kHz pilot          │
│  • Noise reduction (spectral subtraction)          │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  PILOT REGENERATION (Optional)                      │
│  • Extract 19 kHz pilot                            │
│  • Generate perfect sinusoid @ 19 kHz              │
│  • Improves phase coherence                        │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  REMODULATION                                       │
│  • Apply pre-emphasis (75 μs)                      │
│  • Frequency modulator (GNU Radio)                 │
│  • Output: clean FM signal @ 2.4 MSPS              │
└─────────────────────────────────────────────────────┘
    ↓
Output: Reconstructed FM reference → ECA & CAF
```

---

## Week 1: Demodulation & Audio Processing

### Day 1-2: Demodulation

**Task:** Integrate GNU Radio FM demodulation

**Code Changes:**
```cpp
// In dvbt_reconstructor_impl.h
class dvbt_reconstructor_impl {
private:
    // Signal type
    enum signal_type_t {
        SIGNAL_FM,
        SIGNAL_DVBT,
        SIGNAL_WIFI,
        SIGNAL_LTE,
        SIGNAL_5G
    };
    signal_type_t d_signal_type;

    // FM-specific
    float d_fm_deviation;
    bool d_enable_stereo;
    bool d_enable_pilot_regen;
    float d_audio_bw;

    // FM demod
    float d_fm_gain;  // = sample_rate / (2*pi*deviation)
    std::vector<gr_complex> d_fm_history;  // For demod
    float d_fm_phase;  // Track phase

    // Audio processing
    std::vector<float> d_audio_buffer;
    std::vector<float> d_lpf_taps;

    // Methods
    void fm_demodulate(const gr_complex* in, float* audio_out, int n);
    void init_fm_filters();
};

// In dvbt_reconstructor_impl.cc
void dvbt_reconstructor_impl::fm_demodulate(
    const gr_complex* in, float* audio_out, int n)
{
    // Quadrature demodulation: f(t) = (1/2π) * dφ/dt
    // φ = angle of IQ sample

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
```

**GNU Radio Alternative:**
```cpp
// Option: Use gr::analog blocks instead of custom code
#include <gnuradio/analog/quadrature_demod_cf.h>

// In constructor
d_fm_demod = gr::analog::quadrature_demod_cf::make(
    sample_rate / (2*M_PI*d_fm_deviation)
);
```

**Validation:**
- Feed in real FM signal
- Listen to demodulated audio (should be clear)
- Measure SNR of audio

---

### Day 3-4: Audio Filtering

**Task:** Clean up demodulated audio

**Low-Pass Filter Design:**
```cpp
void dvbt_reconstructor_impl::init_fm_filters()
{
    // Design low-pass filter for audio
    // Cutoff: 15 kHz (mono) or 53 kHz (stereo)
    // Transition width: 5 kHz
    // Attenuation: 60 dB

    float cutoff = d_enable_stereo ? 53e3 : 15e3;
    float transition = 5e3;

    // Use GNU Radio firdes for filter design
    d_lpf_taps = gr::filter::firdes::low_pass(
        1.0,              // Gain
        d_sample_rate,    // Sample rate
        cutoff,           // Cutoff frequency
        transition,       // Transition width
        gr::fft::window::WIN_HAMMING
    );

    // Apply filter using VOLK-accelerated convolution
}

void dvbt_reconstructor_impl::apply_audio_filter(
    float* audio, int n)
{
    // Convolve audio with LPF taps
    // Use volk_32f_x2_dot_prod_32f for efficiency

    for (int i = 0; i < n; i++) {
        float filtered = 0.0f;
        for (int j = 0; j < d_lpf_taps.size(); j++) {
            int idx = i - j;
            if (idx >= 0) {
                filtered += audio[idx] * d_lpf_taps[j];
            }
        }
        audio[i] = filtered;
    }
}
```

**VOLK Optimization:**
```cpp
// Use VOLK for faster filtering
#include <volk/volk.h>

volk_32f_x2_dot_prod_32f(&filtered_sample,
                          &audio[i],
                          d_lpf_taps.data(),
                          d_lpf_taps.size());
```

**Validation:**
- Verify audio quality improvement
- Check frequency response (should roll off above 15/53 kHz)
- Measure THD (Total Harmonic Distortion)

---

### Day 5: Noise Reduction (Optional)

**Task:** Spectral subtraction or Wiener filtering

**Spectral Subtraction Algorithm:**
```cpp
void dvbt_reconstructor_impl::noise_reduction(
    float* audio, int n)
{
    // 1. Estimate noise spectrum from silence periods
    // 2. Subtract noise spectrum from signal spectrum
    // 3. Convert back to time domain

    // FFT audio to frequency domain
    for (int i = 0; i < n; i++) {
        d_fft_in[i][0] = audio[i];
        d_fft_in[i][1] = 0.0f;
    }
    fftwf_execute(d_fft_plan);

    // Spectral subtraction
    for (int i = 0; i < n/2; i++) {
        float magnitude = std::sqrt(
            d_fft_out[i][0]*d_fft_out[i][0] +
            d_fft_out[i][1]*d_fft_out[i][1]
        );

        // Subtract estimated noise magnitude
        float clean_mag = std::max(magnitude - d_noise_estimate[i], 0.0f);

        // Preserve phase
        float phase = std::atan2(d_fft_out[i][1], d_fft_out[i][0]);

        d_fft_out[i][0] = clean_mag * std::cos(phase);
        d_fft_out[i][1] = clean_mag * std::sin(phase);
    }

    // IFFT back to time domain
    fftwf_execute(d_ifft_plan);

    // Copy result
    for (int i = 0; i < n; i++) {
        audio[i] = d_fft_in[i][0] / n;  // Normalize
    }
}
```

**Note:** This is optional - may not be needed if signal is strong.

---

## Week 2: Remodulation & Integration

### Day 6-7: Pilot Regeneration (Optional)

**Task:** Extract and regenerate 19 kHz stereo pilot

**Implementation:**
```cpp
void dvbt_reconstructor_impl::regenerate_pilot(
    float* audio, int n)
{
    // Extract 19 kHz component using narrow bandpass filter
    const float pilot_freq = 19e3;
    const float pilot_bw = 100;  // ±100 Hz

    // Bandpass filter @ 19 kHz
    std::vector<float> pilot_taps = gr::filter::firdes::band_pass(
        1.0,
        d_sample_rate,
        pilot_freq - pilot_bw,
        pilot_freq + pilot_bw,
        100,  // Transition
        gr::fft::window::WIN_HAMMING
    );

    // Filter to extract pilot
    float extracted_pilot[n];
    apply_filter(audio, extracted_pilot, n, pilot_taps);

    // Estimate phase from extracted pilot
    float avg_phase = estimate_phase(extracted_pilot, n, pilot_freq);

    // Generate perfect pilot
    for (int i = 0; i < n; i++) {
        float t = (float)i / d_sample_rate;
        float perfect_pilot = std::sin(2*M_PI*pilot_freq*t + avg_phase);

        // Replace in composite signal (at 19 kHz)
        // This requires careful spectral manipulation
        // ... (implementation details)
    }
}
```

**Benefit:** Improves phase coherence for bistatic radar processing.

---

### Day 8-9: FM Remodulation

**Task:** Modulate audio back to FM

**Implementation:**
```cpp
void dvbt_reconstructor_impl::fm_modulate(
    const float* audio, gr_complex* fm_out, int n)
{
    // Apply pre-emphasis (75 μs time constant for US)
    float preemph_alpha = 1.0f - std::exp(-1.0f / (75e-6 * d_sample_rate));

    float preemph_state = 0.0f;
    for (int i = 0; i < n; i++) {
        // Pre-emphasis filter: y[n] = x[n] - α*y[n-1]
        float preemph = audio[i] - preemph_alpha * preemph_state;
        preemph_state = preemph;

        // Integrate audio to get phase
        d_fm_phase += 2*M_PI * d_fm_deviation * preemph / d_sample_rate;

        // Wrap phase
        while (d_fm_phase > M_PI) d_fm_phase -= 2*M_PI;
        while (d_fm_phase < -M_PI) d_fm_phase += 2*M_PI;

        // Generate FM signal
        fm_out[i] = gr_complex(std::cos(d_fm_phase), std::sin(d_fm_phase));
    }
}
```

**GNU Radio Alternative:**
```cpp
#include <gnuradio/analog/frequency_modulator_fc.h>

d_fm_mod = gr::analog::frequency_modulator_fc::make(
    2*M_PI*d_fm_deviation / d_sample_rate
);
```

**Validation:**
- Demodulate reconstructed FM and listen (should match input audio)
- Check deviation (should be ±75 kHz)
- Measure harmonic distortion

---

### Day 10: Integration & Testing

**Task:** Wire everything together and validate

**Complete work() Function:**
```cpp
int dvbt_reconstructor_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items)
{
    const gr_complex* in = (const gr_complex*)input_items[0];
    gr_complex* out = (gr_complex*)output_items[0];

    if (d_signal_type == SIGNAL_FM) {
        // Allocate temp audio buffer
        std::vector<float> audio(noutput_items);

        // 1. Demodulate FM to audio
        fm_demodulate(in, audio.data(), noutput_items);

        // 2. Filter audio
        apply_audio_filter(audio.data(), noutput_items);

        // 3. Optional: Noise reduction
        if (d_enable_noise_reduction) {
            noise_reduction(audio.data(), noutput_items);
        }

        // 4. Optional: Regenerate pilot
        if (d_enable_pilot_regen && d_enable_stereo) {
            regenerate_pilot(audio.data(), noutput_items);
        }

        // 5. Remodulate to FM
        fm_modulate(audio.data(), out, noutput_items);

        // 6. Estimate SNR
        estimate_snr(in, out, noutput_items);

    } else {
        // Other signal types (passthrough for now)
        std::memcpy(out, in, noutput_items * sizeof(gr_complex));
    }

    return noutput_items;
}
```

**Field Testing:**
1. Capture real FM station with KrakenSDR
2. Run through Block B3
3. Feed to ECA and CAF
4. Compare CAF peaks with/without B3
5. Measure improvement in dB

**Expected Results:**
- 10-15 dB CAF peak improvement in urban multipath
- Cleaner audio when demodulated
- Reduced sidelobes in CAF
- Better target detection

---

## Updated Block Parameters

### New make() signature for FM:

```cpp
// C++ API
static sptr make_fm(
    float fm_deviation = 75e3,          // FM deviation (Hz)
    bool enable_stereo = true,          // Process stereo
    bool enable_pilot_regen = true,     // Regenerate 19 kHz pilot
    float audio_bw = 15e3,              // Audio bandwidth (Hz)
    bool enable_noise_reduction = false // Spectral subtraction
);

// Legacy make() for backward compatibility
static sptr make(...);  // Keep DVB-T parameters
```

### Python/GRC:
```python
from gnuradio import kraken_passive_radar

# FM mode
fm_recon = kraken_passive_radar.dvbt_reconstructor.make_fm(
    fm_deviation=75e3,
    enable_stereo=True,
    enable_pilot_regen=True,
    audio_bw=15e3
)
```

---

## GRC Block Update

Update `kraken_passive_radar_dvbt_reconstructor.block.yml`:

```yaml
parameters:
  - id: signal_type
    label: Signal Type
    dtype: string
    default: 'fm'
    options: ['fm', 'dvbt', 'wifi', 'lte', '5g']
    option_labels: ['FM Radio', 'DVB-T', 'WiFi', 'LTE', '5G NR']
    hide: none

  - id: fm_deviation
    label: FM Deviation (Hz)
    dtype: float
    default: '75e3'
    hide: ${'none' if signal_type == 'fm' else 'all'}

  - id: enable_stereo
    label: Enable Stereo
    dtype: bool
    default: 'True'
    hide: ${'part' if signal_type == 'fm' else 'all'}

  - id: enable_pilot_regen
    label: Regenerate Pilot
    dtype: bool
    default: 'True'
    hide: ${'part' if signal_type == 'fm' else 'all'}

  # ... DVB-T parameters (hide if signal_type != 'dvbt')
```

---

## Testing Plan

### Unit Tests (qa_dvbt_reconstructor.py)

Add FM-specific tests:

```python
def test_fm_demodulation(self):
    """Test FM demodulation produces clean audio"""
    # Generate FM test signal
    sample_rate = 2.4e6
    fm_dev = 75e3
    audio_freq = 1e3  # 1 kHz tone

    # ... generate FM signal

    # Process through block
    fm_recon = kraken_passive_radar.dvbt_reconstructor.make_fm()
    # ... run flowgraph

    # Demodulate output and check for 1 kHz tone
    # Should be cleaner than input

def test_fm_pilot_regeneration(self):
    """Test 19 kHz pilot is regenerated cleanly"""
    # Generate FM stereo with noisy pilot
    # Process through block with enable_pilot_regen=True
    # Check that output has clean 19 kHz pilot

def test_fm_snr_improvement(self):
    """Test SNR improvement with noisy FM"""
    # Add AWGN to FM signal (SNR = 10 dB)
    # Process through Block B3
    # Demodulate both input and output
    # Measure SNR improvement in audio domain
    # Should see 8-12 dB gain
```

### Integration Tests

```python
def test_fm_caf_improvement(self):
    """Test CAF peak improvement with FM reconstruction"""
    # Load real FM capture
    # Process through full chain (with and without B3)
    # Measure CAF peak amplitude
    # Should see 10-15 dB improvement
```

---

## Performance Requirements

### Computational Budget
- Target: <10% single CPU core @ 2.4 MSPS
- FM demod: ~2%
- Audio filtering: ~3%
- FM remod: ~2%
- Overhead: ~3%

### Memory
- Audio buffer: ~10 KB (1024 samples × 10 KB)
- Filter taps: ~5 KB
- Total: <100 KB

### Latency
- Target: <5 ms end-to-end
- FM demod: <1 ms
- Filtering: <2 ms
- FM remod: <1 ms
- Margin: ~1 ms

**All requirements are easily met** - FM is very lightweight.

---

## Success Criteria

### Functional
- ✅ FM demodulation produces intelligible audio
- ✅ Remodulated FM can be demodulated correctly
- ✅ Block runs in real-time at 2.4 MSPS
- ✅ Integrates into passive radar chain
- ✅ All unit tests pass

### Performance
- ✅ SNR improvement: >10 dB (audio domain)
- ✅ CAF peak improvement: >10 dB (bistatic radar)
- ✅ CPU usage: <10% single core
- ✅ Latency: <5 ms
- ✅ Detection range increase: 1.5-2× (with strong FM transmitter)

---

## Deliverables Checklist

### Code
- [ ] FM demodulation implementation
- [ ] Audio filtering (LPF)
- [ ] Optional noise reduction
- [ ] Optional pilot regeneration
- [ ] FM remodulation
- [ ] Updated make_fm() factory method
- [ ] Signal type parameter (enum)

### Testing
- [ ] Unit tests for FM mode
- [ ] Integration test with real FM signal
- [ ] CAF improvement measurement
- [ ] Performance benchmarks

### Documentation
- [ ] FM mode usage guide
- [ ] Parameter reference
- [ ] Performance results
- [ ] Comparison with DVB-T (for future reference)

### GRC
- [ ] Updated block YAML with signal type selector
- [ ] Example flowgraph (FM passive radar)

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Audio quality degraded | Low | Medium | Validate filter design, test with real signals |
| Pilot regen introduces artifacts | Medium | Low | Make it optional, validate carefully |
| CPU budget exceeded | Very Low | Low | FM is very lightweight, plenty of margin |
| Stereo processing complexity | Low | Medium | Implement mono-only mode as fallback |

---

## Next Steps After Phase 2

Once FM is working:
1. Document performance results
2. Create user guide ("How to use FM for passive radar")
3. Collect field test data (range, accuracy)
4. Begin Phase 4 (WiFi) or Phase 5 (LTE) based on user needs
5. Add auto-detection (scan for strongest FM station)

---

## References

### FM Technical Standards
- ITU-R BS.450: FM Broadcasting Standard
- NRSC-1: US FM Pre-emphasis/De-emphasis
- NRSC-2: FM Stereo Specification

### Passive Radar with FM
- "Passive Bistatic Radar using FM Radio Illuminators" (various papers)
- "FM-Based Passive Radar: A Review" (IEEE)

### GNU Radio FM Blocks
- gr::analog::quadrature_demod_cf
- gr::analog::frequency_modulator_fc
- gr::filter::firdes (filter design)

---

**Ready to begin Phase 2 FM implementation!**

**Estimated completion:** 2 weeks from start
**Next milestone:** FM demodulation working (end of Week 1)
