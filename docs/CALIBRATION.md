# KrakenSDR Phase Calibration Guide

## Overview

The KrakenSDR contains 5 independent RTL-SDR receivers, each with its own R820T2 tuner
and crystal oscillator. The oscillator frequency offsets cause inter-channel phase drift
of 1-3 deg/s. Without calibration, the phase errors accumulate to tens of degrees within
seconds, degrading ECA clutter cancellation and AoA estimation.

The calibration system uses the KrakenSDR's internal noise source to inject a common
correlated signal into all 5 receivers simultaneously, then measures the inter-channel
phase offsets and drift rates via FFT cross-correlation.

## Hardware

### Noise Source

The KrakenSDR has a wideband noise source controlled by GPIO 0 on the RTL2832U chip
with serial number 1000 (the first dongle). The noise source connects to a silicon
RF switch that simultaneously:

1. **Disconnects** all 5 antenna inputs
2. **Routes** the noise source to all 5 receiver inputs through a power splitter

The switch provides >30 dB of isolation between antenna and noise paths.

### GPIO Control

The noise source is toggled via USB vendor control transfers to the RTL2832U:

| Register | Address | Block | Description |
|----------|---------|-------|-------------|
| GPD      | 0x3004  | SYSB=2 | GPIO direction (0=output) |
| GPOE     | 0x3003  | SYSB=2 | GPIO output enable (1=enabled) |
| GPO      | 0x3001  | SYSB=2 | GPIO output value (1=noise ON) |

Write transfers use `wIndex = (block << 8) | 0x10`.
Read transfers use `wIndex = block << 8`.

The GPIO control works via raw libusb control transfers even while osmosdr/GNU Radio
holds the device for streaming. This was verified with the concurrent hardware test.

## Calibration Architecture

### Primary: Inline Flowgraph Calibrator

The main calibration system is built into `kraken_pbr_flowgraph.py`. It calibrates
all 4 surveillance channels (ch1-ch4) against the reference channel (ch0) using:

1. **Vector probes** (`probe_signal_vc` with `stream_to_vector`, 65536 samples)
   for capturing contiguous time-aligned sample blocks
2. **FFT cross-correlation** to find inter-channel delays (handles delays up to
   +/-32768 samples = +/-123ms at 266 kHz decimated rate)
3. **Parabolic drift fitting** with exponential time weighting
4. **NCO-based phase correction** applying continuous per-sample compensation

### Legacy: CalibrationController

`calibration_controller.py` is used by the headless pipeline (`run_passive_radar.py`).
It uses scalar probes and simple cross-correlation. Superseded by the inline calibrator.

## Calibration Flow

Each calibration cycle (default: every 60 seconds):

```
1. Enable noise source (GPIO 0 HIGH on SN 1000)
2. Wait 1.5s for pipeline flush
   - USB buffers: ~1.7s at 2.4 MHz with 128x65536 byte buffers
   - GNU Radio block buffers: ~0.1s
   - Measured arrival time: ~0.6s (1.5s provides margin)
3. Read reference vector probe (65536 complex64 samples)
4. For each surveillance channel (ch1-ch4):
   a. Read surv vector probe
   b. FFT cross-correlation: IFFT(Xr * conj(Xs))
   c. Find peak delay, align vectors, compute phase + correlation
   d. If correlation < 0.3, retry (probe vectors may not overlap)
   e. Update drift model (parabolic fit with exponential weighting)
   f. Apply correction to phase_corrector block
5. Disable noise source (GPIO 0 LOW)
```

## Cross-Correlation Convention

The inline calibrator uses `IFFT(R * conj(S))`, which gives:

```
xcorr[k] = sum(r[n+k] * conj(s[n]))
```

Peak at index `k = delay` means reference leads surveillance by `delay` samples.
Alignment: `ref[delay:]` matches `surv[:N-delay]` for positive delay.

The standalone calibrator (`krakensdr_calibrate.py`) uses the conjugate convention
`IFFT(conj(R) * S)` with `np.abs()`, which gives the same delay magnitude.

## Phase Corrector

The `phase_corrector` GNU Radio block applies continuous per-sample phase correction
to each surveillance channel:

```
output[n] = input[n] * exp(j * (phi0 + omega * n/fs))
```

Where:
- `phi0 = -phase_offset` (negated measured offset)
- `omega = -drift_rate` (negated measured drift in rad/s)
- `fs` = decimated sample rate (266 kHz)

### Implementation Details

- **Thread safety**: Lock protects `phi0`, `omega`, `sample_count` between
  the calibration thread and GNU Radio scheduler thread
- **NCO optimization**: Uses `np.cumprod()` (2 exp() calls + N complex multiplies)
  instead of per-sample `np.exp()`. ~3.6x faster on RPi5.
- **Static fast path**: When drift=0, uses single-phasor multiply (no NCO)
- **Precision**: Complex128 accumulator, cast to complex64 for output

## Measured Performance

Tested on KrakenSDR with 5 RTL-SDR receivers at 103.7 MHz:

| Metric | Value |
|--------|-------|
| Correlation (noise source ON) | 0.59-0.95 across channels |
| Phase stability (60s, drift-compensated) | < 1 degree |
| Drift rate (typical) | 0-1.7 deg/s per channel |
| Pipeline flush time | ~0.6s |
| Calibration cycle duration | ~3s (1.5s flush + reads + noise off) |
| Max recal interval (<5 deg) | 50s (all channels) |
| Recommended recal interval | 60s (with drift compensation) |

### Recalibration Interval Sweep Results

| Interval | Worst compensated deviation | Status |
|----------|----------------------------|--------|
| 10s | 1.2 deg | OK |
| 20s | 1.9 deg | OK |
| 30s | 2.7 deg | OK |
| 40s | 3.9 deg | OK |
| 50s | 4.8 deg | OK |
| 60s | 5.5 deg | Marginal |

## Test Procedures

### Hardware Verification

```bash
# Verify noise source GPIO works (no flowgraph required)
python3 test_noise_source_hw.py

# Verify GPIO works while osmosdr holds device
python3 test_libusb_concurrent.py
```

### Pipeline Lag Measurement

```bash
# Measure noise-source arrival time through GNU Radio pipeline
python3 test_noise_source_lag.py --timeout 10
```

### Phase Stability

```bash
# Single-point stability over time
python3 test_phase_stability.py

# Progressive interval sweep (finds max recal interval)
python3 test_recal_sweep.py --threshold 5
```

### Full Flowgraph

```bash
python3 kraken_pbr_flowgraph.py
```

Watch for: all 4 channels reporting correlation > 0.5 and consistent drift rates.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Correlation always ~0.01 | Cross-correlation alignment bug | Verify `ref[delay:]` matches `surv[:N-delay]` (not swapped) |
| Correlation drops intermittently | Vector probe timing mismatch | Retry loop reads vectors until overlap achieved |
| "PLL not locked" warnings | R820T tuner initialization | Harmless for noise source calibration; noise is wideband |
| No power increase on noise ON | GPIO not toggling | Run `test_noise_source_hw.py` to verify hardware |
| Power increase but no correlation | Probes reading stale data | Increase settle time (default 1.5s) |
