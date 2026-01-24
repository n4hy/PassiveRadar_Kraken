# Passive Radar Module v2 - New Blocks and Tests

**License:** MIT  
**Copyright:** (c) 2026 Dr Robert W McGwier, PhD

## New Blocks Added

### 1. Doppler Processor (`doppler_processor`)
Generates range-Doppler maps via slow-time FFT.

**Algorithm:**
1. Accumulate `num_doppler_bins` range profiles
2. Apply window function (Hamming/Hann/Blackman)
3. Compute FFT along slow-time for each range bin
4. FFT-shift to center zero Doppler
5. Output power or complex RDM

**Parameters:**
- `num_range_bins`: Input vector length
- `num_doppler_bins`: Number of CPIs (Doppler FFT size)
- `window_type`: 0=rect, 1=hamming, 2=hann, 3=blackman
- `output_power`: True for |X|², False for complex

**Performance:**
- Doppler resolution: PRF / num_doppler_bins ≈ 3.8 Hz (64 bins, 250 kHz)
- Max unambiguous Doppler: ±PRF/2 = ±125 Hz

---

### 2. CFAR Detector (`cfar_detector`)
Constant False Alarm Rate detection for range-Doppler maps.

**Algorithm Variants:**
| Type | Method |
|------|--------|
| CA-CFAR | Mean of reference cells |
| GO-CFAR | Max of leading/lagging averages |
| SO-CFAR | Min of leading/lagging averages |
| OS-CFAR | k-th ordered statistic |

**Threshold Calculation:**
```
α = N × (Pfa^(-1/N) - 1)
threshold = α × noise_estimate
```

**Parameters:**
- Guard cells: Exclude target energy leakage
- Reference cells: Estimate noise level
- Pfa: 1e-6 typical (1 false alarm per million cells)

---

### 3. Coherence Monitor (`coherence_monitor`)
Automatic calibration verification and triggering.

**Metrics Monitored:**
1. Cross-correlation coefficient (ref vs each surv channel)
2. Phase variance across measurement windows

**Calibration Trigger Logic:**
- Correlation < threshold (default 0.95)
- Phase variance > threshold (default 5°)
- 3 consecutive failures required (hysteresis)

**Measurement Schedule:**
- Not every sample (minimizes CPU load)
- Default: 10ms measurement every 1000ms
- ~0.5% duty cycle

**Message Interface:**
- `cal_request` (output): PMT dict when recalibration needed
- `cal_complete` (input): Acknowledge calibration done

---

## Unit Tests

### Standalone Tests (no GNU Radio required)
`tests/test_algorithms_standalone.py`

| Test | Verified |
|------|----------|
| ECA matrix formulation | 55 dB suppression, target preservation |
| Doppler FFT detection | Correct bin location (45 ± 1) |
| CA-CFAR threshold | Pfa within expected range |
| 2D CA-CFAR | All targets detected, low false alarms |
| Correlation coefficient | Self=1.0, phase-shifted=1.0, independent≈0 |
| Phase variance | Good cal: 0.96°, Bad cal: 14.3° |
| Calibration hysteresis | Triggers on 3rd consecutive failure |

### GNU Radio Tests
`tests/test_passive_radar.py`

Requires GNU Radio installed. Tests actual block implementations.

---

## Calibration System Architecture

```
                    ┌─────────────────────┐
                    │  KrakenSDR Source   │
                    │    (5 channels)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Coherence Monitor  │◄───── cal_complete
                    │                     │
                    │  - Periodic check   ├─────► cal_request
                    │  - Correlation      │
                    │  - Phase variance   │
                    └──────────┬──────────┘
                               │
            ┌─────────┬────────┴────────┬─────────┐
            ▼         ▼                 ▼         ▼
         ch0       ch1              ch2-ch4
        (ref)    (surv1)           (surv2-4)
```

**Decision Flow:**
```
Every measure_interval_ms:
  1. Compute correlation(ref, surv_i) for each i
  2. Compute phase_variance(surv_i) for each i
  3. If any corr < threshold OR phase_var > threshold:
       consecutive_failures++
     Else:
       consecutive_failures = 0
  4. If consecutive_failures >= 3:
       Send cal_request message
       Set calibration_in_progress = true
  5. On cal_complete message:
       Force immediate verification
       Clear flags if passed
```

---

## File Manifest

```
gr-kraken_passive_radar/
├── include/gnuradio/kraken_passive_radar/
│   ├── doppler_processor.h      # NEW
│   ├── cfar_detector.h          # NEW
│   └── coherence_monitor.h      # NEW
├── lib/
│   ├── doppler_processor_impl.cc/h   # NEW
│   ├── cfar_detector_impl.cc/h       # NEW
│   └── coherence_monitor_impl.cc/h   # NEW
├── python/kraken_passive_radar/bindings/
│   ├── doppler_processor_python.cc   # NEW
│   ├── cfar_detector_python.cc       # NEW
│   └── coherence_monitor_python.cc   # NEW
├── grc/
│   ├── kraken_passive_radar_doppler_processor.block.yml   # NEW
│   ├── kraken_passive_radar_cfar_detector.block.yml       # NEW
│   └── kraken_passive_radar_coherence_monitor.block.yml   # NEW
└── tests/
    ├── test_passive_radar.py           # GNU Radio tests
    └── test_algorithms_standalone.py   # Standalone tests (NEW)
```

---

## Build Instructions

```bash
cd gr-kraken_passive_radar
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

**New Dependency:** FFTW3 (for Doppler processor)
```bash
sudo apt install libfftw3-dev
```

---

## Running Tests

```bash
# Standalone tests (no GNU Radio)
python3 tests/test_algorithms_standalone.py

# Full tests (requires GNU Radio)
python3 -m pytest tests/test_passive_radar.py -v
```
