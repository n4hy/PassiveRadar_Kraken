# Block B3: Reference Signal Reconstruction - Multi-Signal Roadmap

**Date:** 2026-02-09
**Status:** Phase 1 skeleton complete, pivoting to FM Radio (US market priority)
**Goal:** Support multiple signals of opportunity for passive radar worldwide

---

## Executive Summary

Block B3 (Reference Signal Reconstruction) improves passive radar sensitivity by 10-20 dB through demodulation-remodulation with error correction. The core infrastructure built in Phase 1 is **signal-agnostic** and will support multiple transmission standards.

**Immediate Priority:** FM Radio (US market, simplest implementation)
**Future Expansion:** DVB-T, ATSC 3.0, WiFi, LTE, 5G

---

## Shared Infrastructure (Complete ✅)

The following components are **already built** and work for ALL signal types:

- ✅ GNU Radio sync_block framework
- ✅ Python bindings (pybind11 + GNU Radio)
- ✅ Build system integration (CMake)
- ✅ GRC block definition template
- ✅ Parameter validation framework
- ✅ FFTW3 FFT/IFFT infrastructure
- ✅ Runtime control methods (enable/disable features)
- ✅ SNR estimation framework
- ✅ Unit test framework
- ✅ Thread-safe parameter access

**What this means:** We don't rebuild the block structure for each signal type - we just change the signal processing algorithm inside `work()`.

---

## Signal Type Comparison

| Signal | US Availability | Complexity | Power | Range | OFDM? | FEC? |
|--------|----------------|------------|-------|-------|-------|------|
| **FM Radio** | ✅ Ubiquitous | ⭐ Simple | 🔥 High (50-100 kW) | 60+ km | ❌ No | ❌ No |
| **DVB-T** | ❌ Not in US | ⭐⭐⭐ Complex | 🔥 High (10-100 kW) | 40+ km | ✅ Yes | ✅ Yes |
| **ATSC 3.0** | 🟡 Major cities | ⭐⭐⭐⭐ Very Complex | 🔥 High (10-100 kW) | 40+ km | ✅ Yes | ✅ Yes |
| **WiFi** | ✅ Everywhere | ⭐⭐ Moderate | 🔥 Low (100 mW) | 100 m | ✅ Yes | ✅ Yes |
| **LTE** | ✅ Ubiquitous | ⭐⭐⭐⭐ Very Complex | 🔥 Medium (5-40 W) | 5-30 km | ✅ Yes | ✅ Yes |
| **5G NR** | ✅ Growing | ⭐⭐⭐⭐⭐ Extremely Complex | 🔥 Medium (5-40 W) | 1-10 km | ✅ Yes | ✅ Yes |
| **ATSC 1.0** | ✅ Ubiquitous | ⭐⭐⭐ Complex | 🔥 High (10-100 kW) | 40+ km | ❌ No (8-VSB) | ✅ Yes |

---

## Implementation Roadmap

### Phase 1: Block Skeleton ✅ COMPLETE (Week 1)

**Status:** Done
**Applies to:** ALL signal types

**Deliverables:**
- GNU Radio block framework
- Build system integration
- Python bindings
- FFTW infrastructure
- Parameter framework
- Passthrough implementation

**Time:** 1 week (completed)

---

### Phase 2: FM Radio Implementation 🎯 IMMEDIATE PRIORITY (Weeks 2-3)

**Status:** Next
**US Market Priority:** Critical

#### FM Signal Characteristics
- **Frequency:** 88-108 MHz
- **Modulation:** FM (Frequency Modulation)
- **Bandwidth:** ±75 kHz deviation (200 kHz channel spacing)
- **Stereo:** 19 kHz pilot + 38 kHz subcarrier
- **RDS:** 57 kHz data subcarrier (optional)
- **Power:** 50-100 kW ERP (very strong signals)

#### Implementation Approach

**Signal Flow:**
```
Input: FM signal @ 2.4 MSPS
    ↓
[FM Demodulation] - GNU Radio blocks
    ↓
[Audio Processing] - L+R extraction, filtering
    ↓
[Stereo Pilot Enhancement] - Clean 19 kHz pilot (optional)
    ↓
[FM Remodulation] - Reconstruct carrier
    ↓
Output: Clean FM reference @ 2.4 MSPS
```

#### Technical Details

1. **Demodulation**
   - Use `gr::analog::quadrature_demod_cf` for FM demod
   - Extract audio baseband (0-53 kHz for mono, 0-53 kHz + 23-53 kHz stereo)
   - Optional: Extract 19 kHz stereo pilot for phase reference

2. **Signal Cleaning**
   - Low-pass filter audio (15 kHz for mono, 53 kHz for stereo)
   - Optional: Bandpass filter stereo pilot (19 kHz ± 100 Hz)
   - Remove noise using Wiener filtering or spectral subtraction

3. **Remodulation**
   - Use `gr::analog::frequency_modulator_fc`
   - Apply pre-emphasis (50 μs or 75 μs time constant)
   - Reconstruct stereo if needed (L-R modulation @ 38 kHz)

4. **Pilot Enhancement (Optional)**
   - Extract 19 kHz pilot with narrow bandpass
   - Regenerate perfect sinusoid at 19 kHz
   - Improves phase coherence for bistatic processing

#### Advantages for Passive Radar
- ✅ Simple implementation (no OFDM, no FEC)
- ✅ Very strong signals (easy to detect)
- ✅ Continuous transmission (24/7 availability)
- ✅ Long range (60+ km with high-power transmitters)
- ✅ Well-studied for passive radar applications
- ✅ Minimal computational load

#### Parameters for Block B3 (FM Mode)
```cpp
static sptr make_fm(
    float deviation = 75e3,      // FM deviation (Hz)
    bool enable_stereo = true,   // Process stereo
    bool enable_pilot_regen = true,  // Regenerate 19 kHz pilot
    float audio_bw = 15e3        // Audio bandwidth (Hz)
);
```

#### Deliverables (2 weeks)
- FM demodulation integration
- Audio filtering and processing
- FM remodulation
- Optional pilot regeneration
- Unit tests with real FM signals
- Documentation

**Expected Improvement:** 10-15 dB in urban multipath environments

---

### Phase 3: DVB-T Implementation (Weeks 4-7)

**Status:** Deferred (not in US)
**Markets:** Europe, Australia, Africa, Asia

#### Already Designed (from previous work)
- OFDM demodulation (gr-dtv blocks)
- Viterbi + Reed-Solomon FEC
- SVD pilot enhancement
- 2K/4K/8K mode support

#### Use Cases
- International deployments
- DVB-T signal generator testing
- Academic research
- Europe/Australia market

**Time:** 4 weeks (when needed)

---

### Phase 4: WiFi Implementation (Weeks 8-10)

**Status:** Planned
**US Market Priority:** High (short-range applications)

#### WiFi Signal Characteristics
- **Standards:** 802.11a/g/n/ac (OFDM-based)
- **Frequencies:** 2.4 GHz, 5 GHz
- **Bandwidth:** 20/40/80/160 MHz channels
- **Modulation:** BPSK to 256-QAM
- **Power:** 100 mW - 1 W
- **Range:** 50-200 m

#### Implementation Approach

**Signal Flow:**
```
Input: WiFi OFDM @ 20 MHz
    ↓
[OFDM Demodulation] - gr-ieee802-11 blocks
    ↓
[Pilot Extraction] - 802.11 pilot carriers
    ↓
[FEC Decoding] - Convolutional + optional LDPC
    ↓
[FEC Encoding]
    ↓
[OFDM Remodulation]
    ↓
[Pilot Enhancement] - SVD on pilot subcarriers
    ↓
Output: Clean WiFi reference
```

#### Advantages for Passive Radar
- ✅ Very common (homes, businesses, public spaces)
- ✅ Multiple transmitters (spatial diversity)
- ✅ OFDM-based (similar to DVB-T infrastructure)
- ✅ Good for short-range detection (indoor, urban)
- ✅ High bandwidth → good range resolution

#### Challenges
- ⚠️ Low power (limited range)
- ⚠️ Bursty transmission (not continuous)
- ⚠️ Many interfering networks
- ⚠️ Frequent channel changes

#### Dependencies
- gr-ieee802-11 OOT module (or custom OFDM implementation)
- 802.11a/g/n frame structure knowledge
- Pilot pattern awareness

**Time:** 3 weeks

**Expected Improvement:** 12-18 dB (compensates for low power)

---

### Phase 5: LTE Implementation (Weeks 11-15)

**Status:** Planned
**US Market Priority:** Very High (ubiquitous cellular)

#### LTE Signal Characteristics
- **Frequencies:** 700 MHz - 2.6 GHz (multiple bands)
- **Bandwidth:** 1.4, 3, 5, 10, 15, 20 MHz
- **Downlink:** OFDMA (Orthogonal Frequency Division Multiple Access)
- **Modulation:** QPSK to 64-QAM
- **Power:** 5-40 W per eNodeB
- **Range:** 5-30 km

#### Implementation Approach

**Signal Flow:**
```
Input: LTE downlink @ 20 MHz
    ↓
[Cell Search] - Detect PSS/SSS (cell ID)
    ↓
[OFDM Demodulation] - Extract resource grid
    ↓
[Reference Signal Extraction] - LTE RS (cell-specific)
    ↓
[Channel Estimation] - From RS
    ↓
[PDSCH Decoding] - Turbo code FEC
    ↓
[PDSCH Encoding]
    ↓
[OFDM Remodulation] - With RS
    ↓
[RS Enhancement] - SVD on reference signals
    ↓
Output: Clean LTE reference
```

#### Advantages for Passive Radar
- ✅ Extremely common (everywhere cellular coverage exists)
- ✅ High power (better range than WiFi)
- ✅ Continuous transmission
- ✅ Well-defined reference signals (RS)
- ✅ Good for medium-range detection (5-30 km)
- ✅ Multiple cells provide spatial diversity

#### Challenges
- ⚠️ Very complex (cell search, synchronization)
- ⚠️ Turbo codes (complex FEC)
- ⚠️ Resource allocation varies (dynamic scheduling)
- ⚠️ Need to decode control channels (PDCCH)

#### Dependencies
- gr-lte OOT module or custom implementation
- LTE frame structure (10 ms frames, 1 ms subframes)
- Reference signal patterns (varies by antenna ports)
- Turbo decoder

**Time:** 5 weeks

**Expected Improvement:** 15-20 dB (very strong reference signals)

---

### Phase 6: 5G NR Implementation (Weeks 16-22)

**Status:** Planned (Future)
**US Market Priority:** Growing (future-proof)

#### 5G NR Signal Characteristics
- **Frequencies:** Sub-6 GHz (600 MHz - 6 GHz), mmWave (24-40 GHz)
- **Bandwidth:** Up to 100 MHz (sub-6), up to 400 MHz (mmWave)
- **Waveform:** CP-OFDM (cyclic prefix OFDM)
- **Numerology:** Flexible subcarrier spacing (15, 30, 60, 120 kHz)
- **Modulation:** QPSK to 256-QAM
- **Power:** Varies widely

#### Implementation Approach

**Signal Flow:**
```
Input: 5G NR @ flexible BW
    ↓
[SSB Decoding] - Synchronization Signal Block
    ↓
[OFDM Demodulation] - CP-OFDM, flexible numerology
    ↓
[DMRS Extraction] - Demodulation Reference Signals
    ↓
[PDSCH Decoding] - LDPC FEC
    ↓
[PDSCH Encoding]
    ↓
[OFDM Remodulation]
    ↓
[DMRS Enhancement] - SVD
    ↓
Output: Clean 5G reference
```

#### Advantages for Passive Radar
- ✅ Future-proof (5G deployment growing)
- ✅ Very high bandwidth (excellent range resolution)
- ✅ Flexible numerology (adaptable)
- ✅ Massive MIMO (spatial diversity)
- ✅ Beamforming (directional coverage)

#### Challenges
- ⚠️ Extremely complex (most complex signal type)
- ⚠️ LDPC codes (computationally intensive)
- ⚠️ Variable frame structure
- ⚠️ Beamforming complicates reference extraction
- ⚠️ mmWave has very short range

#### Dependencies
- gr-5g-nr OOT module (if available) or custom implementation
- 5G NR specifications (3GPP TS 38.xxx series)
- LDPC decoder/encoder
- SSB and DMRS pattern knowledge

**Time:** 7 weeks

**Expected Improvement:** 20-25 dB (best performance, most complex)

---

### Phase 7: ATSC 3.0 Implementation (Weeks 23-27)

**Status:** Planned (US NextGen TV)
**US Market Priority:** Medium (limited deployment)

#### ATSC 3.0 Signal Characteristics
- **Frequencies:** 54-698 MHz (TV bands)
- **Bandwidth:** 6 MHz channels
- **Waveform:** OFDM
- **Modulation:** QPSK to 4096-QAM
- **FEC:** LDPC + BCH
- **Power:** 10-100 kW

#### Implementation Approach
- Similar to DVB-T but with ATSC 3.0 specific parameters
- LDPC decoding (more complex than DVB-T Viterbi)
- Bootstrap signal for initial acquisition
- Pilot boosting option

#### Advantages
- ✅ High power (excellent range)
- ✅ US digital TV standard (future)
- ✅ OFDM-based (similar to DVB-T)

#### Challenges
- ⚠️ Limited deployment (major cities only)
- ⚠️ Complex LDPC codes
- ⚠️ Variable frame structure
- ⚠️ No widespread GNU Radio support yet

**Time:** 5 weeks

---

### Phase 8 (Optional): ATSC 1.0 Implementation

**Status:** Considered (legacy)
**US Market Priority:** Medium (being phased out)

#### Characteristics
- 8-VSB modulation (NOT OFDM)
- Completely different architecture needed
- Trellis coding + Reed-Solomon FEC
- High power, good range
- Still widely broadcast but being replaced by ATSC 3.0

**Decision:** Likely **not worth implementing** due to:
- Legacy standard being phased out
- Complex non-OFDM processing
- Better alternatives (FM, ATSC 3.0) available

---

## Implementation Priority Order

### Immediate (2026 Q1-Q2)
1. ✅ **Phase 1:** Block skeleton (COMPLETE)
2. 🎯 **Phase 2:** FM Radio (NEXT - 2 weeks)

### Short Term (2026 Q3)
3. **Phase 4:** WiFi (short-range applications)
4. **Phase 5:** LTE (ubiquitous coverage)

### Medium Term (2026 Q4)
5. **Phase 3:** DVB-T (international markets)
6. **Phase 7:** ATSC 3.0 (US NextGen TV)

### Long Term (2027+)
7. **Phase 6:** 5G NR (future-proof)

---

## Technical Commonalities

### All OFDM-based signals share:
- FFT/IFFT operations (FFTW infrastructure)
- Pilot extraction and enhancement (SVD)
- FEC decoding/encoding (varies by type)
- Symbol synchronization
- Channel estimation from reference signals

### FM is unique:
- No OFDM (analog modulation)
- No FEC (analog signal)
- Simpler processing
- Audio-domain reconstruction

---

## Block B3 Parameter Evolution

### Current (FM Mode)
```python
dvbt_reconstructor.make(
    signal_type="fm",
    fm_deviation=75e3,
    enable_stereo=True,
    enable_pilot_regen=True
)
```

### Future (Multi-Signal)
```python
# Auto-detect or user-specified
dvbt_reconstructor.make(
    signal_type="auto",  # or "fm", "dvbt", "wifi", "lte", "5g", "atsc3"

    # Signal-specific parameters
    fm_deviation=75e3,        # FM only
    fft_size=2048,            # OFDM signals
    bandwidth=20e6,           # WiFi, LTE, 5G
    center_freq=100.1e6,      # All

    # Common parameters
    enable_fec=True,          # OFDM signals
    enable_svd=True,          # All
    enable_pilot_regen=True   # All
)
```

---

## Performance Expectations

| Signal | Expected SNR Gain | Range Improvement | Complexity | Implementation Time |
|--------|-------------------|-------------------|------------|---------------------|
| FM | 10-15 dB | 1.5-2× | Low | 2 weeks |
| DVB-T | 15-20 dB | 2-2.5× | High | 4 weeks |
| WiFi | 12-18 dB | 2-3× | Medium | 3 weeks |
| LTE | 15-20 dB | 2-2.5× | Very High | 5 weeks |
| 5G NR | 20-25 dB | 2.5-3× | Extreme | 7 weeks |
| ATSC 3.0 | 15-20 dB | 2-2.5× | Very High | 5 weeks |

---

## Resource Requirements

### Computational Load (per signal type)

| Signal | CPU (single core) | FFT Size | FEC Complexity | Real-time at 2.4 MSPS? |
|--------|-------------------|----------|----------------|------------------------|
| FM | 5-10% | N/A (no FFT) | None | ✅ Yes |
| DVB-T | 40-60% | 2048-8192 | Medium | ✅ Yes |
| WiFi | 30-50% | 64-256 | Medium | ✅ Yes |
| LTE | 60-80% | 512-2048 | High | 🟡 Tight |
| 5G NR | 80-95% | 512-4096 | Very High | ⚠️ May need GPU |
| ATSC 3.0 | 70-90% | 4096-32768 | Very High | 🟡 Tight |

### GPU Acceleration
- FM: Not needed
- DVB-T/WiFi: Optional (helpful but not critical)
- LTE/5G/ATSC 3.0: Recommended for real-time performance

---

## Dependencies by Signal Type

### FM Radio
- ✅ GNU Radio analog blocks (`gr-analog`)
- ✅ Basic filtering blocks
- No external dependencies

### DVB-T
- ✅ gr-dtv (already added)
- ✅ FFTW3
- ✅ Eigen3 (for SVD)

### WiFi
- ⚠️ gr-ieee802-11 (OOT module, may need custom install)
- ✅ FFTW3
- ✅ Eigen3

### LTE
- ⚠️ gr-lte or srsRAN libraries
- ✅ FFTW3
- ✅ Eigen3
- ⚠️ Turbo decoder library

### 5G NR
- ⚠️ Custom implementation or future gr-5g-nr
- ✅ FFTW3
- ✅ Eigen3
- ⚠️ LDPC decoder library (srsRAN or custom)

### ATSC 3.0
- ⚠️ Custom implementation (no GR support yet)
- ✅ FFTW3
- ✅ Eigen3
- ⚠️ LDPC decoder library

---

## Integration with Existing System

Block B3 (regardless of signal type) fits in the same location:

```
KrakenSDR Source (Ch 0 = Reference)
    ↓
Phase Correction
    ↓
AGC Conditioning
    ↓
[Block B3: Signal Reconstructor] ← Auto-detect or user-select signal type
    ↓
Split to:
  → ECA Clutter Canceller (port 0)
  → CAF Cross-Ambiguity Function (port 0)
```

**No changes needed** to upstream or downstream blocks!

---

## Testing Strategy

### Per-Signal Validation
1. **Synthetic Signals:** Generate test signal in GNU Radio
2. **Add Noise:** Degrade SNR to 5-10 dB
3. **Process through B3:** Measure output SNR
4. **Compare:** Input vs. output correlation
5. **Real Signals:** Capture actual transmissions, validate improvement

### Integration Testing
- Connect to full passive radar chain
- Measure CAF peak improvement
- Measure detection range increase
- Measure false alarm rate reduction

---

## Market Applicability

| Region | Primary Signal | Backup Signal | Notes |
|--------|---------------|---------------|-------|
| **United States** | FM, LTE | WiFi, 5G | ATSC 3.0 in major cities |
| **Europe** | FM, DVB-T | LTE, WiFi | DVB-T2 also available |
| **China** | FM, DTMB | LTE, 5G | DTMB similar to DVB-T |
| **Australia** | FM, DVB-T | LTE, WiFi | Strong FM coverage |
| **Worldwide** | FM, LTE | WiFi | FM almost universal |

---

## Success Metrics

### Phase 2 (FM) - Target Metrics
- ✅ 10 dB minimum CAF improvement
- ✅ No sample drops at 2.4 MSPS
- ✅ Real-time processing on x86 (single core <80%)
- ✅ Detection of targets not visible without B3
- ✅ 50+ km range with 50 kW FM transmitter

### Future Phases - Target Metrics
- ✅ Signal-specific SNR gains (see table above)
- ✅ Interchangeable signal types (runtime selection)
- ✅ Graceful fallback if signal not available
- ✅ Auto-detection of available signals

---

## Documentation Deliverables

### Per Signal Type
- Implementation guide
- Parameter reference
- Performance benchmarks
- Real-world test results
- Troubleshooting guide

### Overall
- Multi-signal selection guide
- "Which signal should I use?" decision tree
- Geographic availability map
- Computational requirements comparison

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| Complex signals (5G) exceed CPU budget | Implement GPU acceleration path |
| Signal not available in user's area | Support multiple fallback options |
| gr-* dependencies unavailable | Implement custom algorithms where needed |
| FEC libraries missing | Use srsRAN or implement simplified versions |

### Market Risks
| Risk | Mitigation |
|------|------------|
| Signal phased out (ATSC 1.0) | Focus on future-proof signals (FM, LTE, 5G) |
| New standards emerge | Modular design allows adding new signals |
| Regional variations | Support multiple standards per region |

---

## Next Actions

### Immediate (This Week)
1. ✅ Document Phase 1 completion
2. ✅ Create multi-signal roadmap (this document)
3. 🎯 Begin Phase 2: FM Radio implementation
   - Design FM demod/remod architecture
   - Identify GNU Radio blocks needed
   - Create Phase 2 implementation plan

### Short Term (Next 2 Weeks)
4. Implement FM demodulation
5. Implement audio filtering
6. Implement FM remodulation
7. Test with real FM stations
8. Validate CAF improvement

### Medium Term (Next 3 Months)
9. Complete WiFi implementation
10. Complete LTE implementation
11. Document performance comparisons
12. Create user selection guide

---

## Conclusion

The Block B3 infrastructure is **signal-agnostic** and ready to support multiple transmission standards. By starting with FM Radio, we'll have a working US-compatible system quickly, then expand to more sophisticated signals (LTE, 5G) for enhanced performance and future-proofing.

**Estimated total time to support all signals:** ~6 months part-time development

**Priority order ensures:**
- ✅ Immediate US market applicability (FM)
- ✅ Future-proof with modern cellular (LTE, 5G)
- ✅ International market support (DVB-T, ATSC 3.0)
- ✅ Flexibility to use best available signal in any location

---

**Status:** Phase 1 complete, Phase 2 (FM) ready to begin
**Next Milestone:** FM Radio reconstruction functional in 2 weeks
