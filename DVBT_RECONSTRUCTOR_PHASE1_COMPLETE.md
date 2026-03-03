# DVB-T Reconstructor (Block B3) - Phase 1 Complete

**Date:** 2026-02-09
**Status:** ✓ Phase 1 Block Skeleton Complete
**Next:** Phase 2 - OFDM Demodulation

---

## Overview

Implemented the block skeleton for the DVB-T Reference Signal Reconstructor (Block B3), which will use demodulation-remodulation with Forward Error Correction to create a "perfect" reference signal for passive radar processing.

**Expected Impact:** 10-20 dB sensitivity improvement in weak signal environments

---

## Phase 1 Deliverables - All Complete ✓

### Files Created

1. **Public API**
   - `gr-kraken_passive_radar/include/gnuradio/kraken_passive_radar/dvbt_reconstructor.h`
   - Defines public interface with factory method `make()`
   - Parameters: fft_size, guard_interval, constellation, code_rate, enable_svd

2. **Implementation**
   - `gr-kraken_passive_radar/lib/dvbt_reconstructor_impl.h`
   - `gr-kraken_passive_radar/lib/dvbt_reconstructor_impl.cc`
   - Implements GNU Radio sync_block
   - FFTW3 initialization/cleanup (forward and inverse plans)
   - Parameter validation (FFT size, guard interval, constellation, code rate)
   - Current behavior: Simple passthrough (memcpy)

3. **Python Bindings**
   - `gr-kraken_passive_radar/python/kraken_passive_radar/bindings/dvbt_reconstructor_python.cc`
   - Pybind11 bindings with gr::sync_block inheritance
   - Factory method: `dvbt_reconstructor.make()`
   - Methods: `set_enable_svd()`, `get_snr_estimate()`

4. **GRC Block Definition**
   - `gr-kraken_passive_radar/grc/kraken_passive_radar_dvbt_reconstructor.block.yml`
   - GNU Radio Companion block definition
   - Dropdown menus for FFT size, guard interval, constellation, code rate
   - Boolean toggle for SVD enable/disable

5. **Unit Tests**
   - `gr-kraken_passive_radar/python/kraken_passive_radar/qa_dvbt_reconstructor.py`
   - Tests for block creation, parameter validation, passthrough, SNR estimation
   - Placeholder tests for Phase 2-4 functionality (commented out)

### Files Modified

1. **Build System**
   - `gr-kraken_passive_radar/CMakeLists.txt` - Added `dtv` to GNU Radio components
   - `gr-kraken_passive_radar/lib/CMakeLists.txt` - Added dvbt_reconstructor_impl.cc, linked gnuradio::gnuradio-dtv
   - `gr-kraken_passive_radar/python/kraken_passive_radar/bindings/CMakeLists.txt` - Added dvbt_reconstructor_python.cc

2. **Python Bindings**
   - `gr-kraken_passive_radar/python/kraken_passive_radar/bindings/python_bindings.cc` - Added `bind_dvbt_reconstructor()`

3. **Module Init**
   - `gr-kraken_passive_radar/python/kraken_passive_radar/__init__.py` - Updated docstring

---

## Validation Results ✓

All tests passed:

```
Test 1: Passthrough with constant signal
  ✓ Perfect passthrough: True

Test 2: Passthrough with complex test signal
  Samples: 10000
  ✓ Perfect passthrough: True
  ✓ SNR estimate: 60.00 dB

Test 3: SVD enable/disable
  ✓ SVD control functional

Test 4: Parameter validation
  ✓ Valid FFT sizes accepted: 3/3 (2048, 4096, 8192)
  ✓ Invalid FFT size rejected: True
```

### Tested Features

- ✓ Block compilation and linking with gr-dtv
- ✓ Python bindings (pybind11 with GNU Radio base class)
- ✓ Block instantiation via `make()` factory method
- ✓ Parameter validation (throws exceptions for invalid values)
- ✓ GNU Radio flowgraph integration
- ✓ Passthrough functionality (memcpy implementation)
- ✓ SNR estimate getter (placeholder implementation)
- ✓ SVD enable/disable runtime control
- ✓ FFTW plan initialization and cleanup
- ✓ Processes 10k complex samples without errors

---

## Implementation Details

### Block Parameters

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `fft_size` | int | 2048, 4096, 8192 | OFDM FFT size (DVB-T 2K/4K/8K mode) |
| `guard_interval` | int | 4, 8, 16, 32 | Guard interval ratio (1/4, 1/8, 1/16, 1/32) |
| `constellation` | int | 0, 1, 2 | Modulation (0=QPSK, 1=16QAM, 2=64QAM) |
| `code_rate` | int | 0-4 | FEC rate (0=1/2, 1=2/3, 2=3/4, 3=5/6, 4=7/8) |
| `enable_svd` | bool | true/false | Enable SVD pilot noise reduction |

### DVB-T Symbol Parameters (Calculated)

For 2K mode (2048 FFT):
- Symbol length: 2560 samples (2048 + 2048/4)
- Useful carriers: 1705
- Continual pilots: 45

For 8K mode (8192 FFT):
- Symbol length: 10240 samples (8192 + 8192/4)
- Useful carriers: 6817
- Continual pilots: 177

### Memory Management

- FFTW plans created with `FFTW_MEASURE` for optimal performance
- Aligned memory allocation via `fftwf_malloc()`
- Proper cleanup in destructor (plans and buffers)
- Thread-safe parameter access via `gr::thread::mutex`

---

## Build Instructions

```bash
cd /home/n4hy/PassiveRadar_Kraken/gr-kraken_passive_radar
mkdir -p build && cd build
cmake ..
cmake --build .
sudo cmake --install .  # Optional: install system-wide
```

### Build from source directory:
```bash
# From build directory
PYTHONPATH=python/kraken_passive_radar/bindings:python:$PYTHONPATH \
LD_LIBRARY_PATH=lib:$LD_LIBRARY_PATH \
python3 -c "from gnuradio import gr; import kraken_passive_radar_python as kpr; \
            blk = kpr.dvbt_reconstructor.make(); print('Block created:', blk)"
```

---

## Dependencies (All Met)

- ✓ GNU Radio 3.10.9.2 (runtime, blocks, fft, filter, **dtv**)
- ✓ gr-dtv (DVB-T demod/remod/FEC blocks) - **newly added**
- ✓ FFTW3f (FFT operations)
- ✓ Eigen3 (for SVD in Phase 4)
- ✓ VOLK (vector optimizations)
- ✓ pybind11 (Python bindings)

---

## Next Steps: Phase 2 - OFDM Demodulation

### Objectives

1. **Integrate gr-dtv OFDM demodulation blocks**
   - Add `gr::dtv::dvbt_ofdm_sym_acquisition` for symbol sync
   - Add `gr::dtv::dvbt_demap` for constellation demapping
   - Extract soft bits from DVB-T signal

2. **Implement OFDM symbol buffering**
   - Add `set_history(d_symbol_length)` for symbol alignment
   - Align to OFDM symbol boundaries in `work()`
   - Buffer full symbols for demodulation

3. **Refine SNR estimation**
   - Estimate SNR from constellation error vector magnitude (EVM)
   - Use pilot carriers for reference
   - Implement `get_snr_estimate()` with real calculation

4. **Testing**
   - Validate with DVB-T test signals
   - Measure soft bit quality
   - Verify symbol synchronization

### Expected Deliverable

Block extracts soft bits from DVB-T signal, ready for FEC decoding (Phase 3).

---

## Architecture Context

Block B3 integrates into the passive radar signal chain:

```
KrakenSDR Source (Ch 0 = Reference)
    ↓
Phase Correction
    ↓
AGC Conditioning
    ↓
[Block B3: DVB-T Reconstructor] ← Phase 1 Complete (passthrough)
    ↓                              Phase 2: OFDM demod
Split to:                          Phase 3: FEC decode/encode
  → ECA Clutter Canceller          Phase 4: OFDM remod + SVD
  → CAF Cross-Ambiguity Function
```

---

## Timeline

- **Week 1:** ✓ Block skeleton and build integration **(COMPLETE)**
- **Week 2:** Phase 2 - OFDM demodulation **(NEXT)**
- **Week 3:** Phase 3 - FEC decoding/encoding
- **Week 4:** Phase 4 - Remodulation, SVD, integration, testing

**Total:** 4 weeks for core functionality

---

## Notes

- Phase 1 provides a fully functional GNU Radio block that compiles, installs, and runs
- Passthrough implementation allows integration testing without affecting signal chain
- All parameter validation is in place for DVB-T modes
- FFTW infrastructure ready for Phase 2 OFDM processing
- Block follows established patterns from eca_canceller and doppler_processor
- Ready for gr-dtv integration in Phase 2

---

**Phase 1 Status:** ✅ COMPLETE
**Next Action:** Begin Phase 2 - OFDM Demodulation with gr-dtv blocks
