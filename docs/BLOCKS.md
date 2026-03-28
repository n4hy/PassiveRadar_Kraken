# Signal Processing Blocks Reference

Comprehensive technical documentation for all signal processing blocks in PassiveRadar_Kraken.

## Table of Contents

1. [Source Blocks](#source-blocks)
   - [krakensdr_source](#krakensdr-source)
   - [rspduo_source](#rspduo-source)
2. [Preprocessing Blocks](#preprocessing-blocks)
   - [conditioning](#conditioning)
   - [dvbt_reconstructor (Block B3)](#dvbt-reconstructor-block-b3)
3. [Core Processing Blocks](#core-processing-blocks)
   - [eca_canceller](#eca-canceller)
   - [caf_processing](#caf-processing)
   - [doppler_processor](#doppler-processor)
4. [Detection Blocks](#detection-blocks)
   - [cfar_detector](#cfar-detector)
   - [detection_cluster](#detection-cluster)
5. [Estimation & Tracking](#estimation--tracking)
   - [aoa_estimator](#aoa-estimator)
   - [tracker](#tracker)
6. [Monitoring Blocks](#monitoring-blocks)
   - [coherence_monitor](#coherence-monitor)

---

## Source Blocks

### krakensdr_source

**Purpose**: 5-channel coherent SDR source for KrakenSDR hardware.

**Technical Description**:

The KrakenSDR consists of 5 RTL-SDR tuners sharing a common clock for phase coherence. Channel 0 serves as the reference (illuminator-facing), while channels 1-4 form a surveillance array for angle-of-arrival estimation.

**Key Features**:
- **Direct GPIO control**: Uses libusb vendor control transfers to the RTL2832U chip (endpoint 0) to toggle the internal noise source, bypassing rtlsdr_open() which would reset the device state.
- **Phase calibration**: When enabled, injects noise into all channels simultaneously, measures cross-correlation phase offsets, and applies corrections.
- **Sample synchronization**: All 5 channels share a common 28.8 MHz reference clock, ensuring sample-level coherence.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_freq` | float | 98.1e6 | Center frequency (Hz) |
| `sample_rate` | float | 2.4e6 | Sample rate (Hz) |
| `gain` | float | 40 | RF gain (dB) |
| `num_channels` | int | 5 | Number of channels |
| `enable_cal` | bool | True | Enable phase calibration |
| `cal_interval` | float | 60 | Calibration interval (seconds) |

**Output Format**: 5 streams of `complex64` (interleaved I/Q)

**Implementation**: `kraken_passive_radar/krakensdr_source.py`

**Noise Source GPIO Control**:
```python
# Direct libusb vendor control transfer to RTL2832U system registers
# Block SYSB=2, registers: GPO=0x3001, GPOE=0x3003, GPD=0x3004
# Matches rtlsdr_set_gpio_output() + rtlsdr_set_gpio_bit() from librtlsdr
libusb_control_transfer(
    handle, 0x40,    # bmRequestType: vendor, host-to-device
    0,               # bRequest: register access
    addr,            # wValue: register address (e.g. 0x3001 for GPO)
    block << 8,      # wIndex: block selector (SYSB=2 → 0x0200)
    data, 1, 1000    # 1 byte data, 1000ms timeout
)
```

---

### rspduo_source

**Purpose**: Dual-tuner SDR source for SDRplay RSPduo.

**Technical Description**:

The RSPduo operates in dual-tuner (diversity) mode with Tuner A as reference and Tuner B as surveillance. Both tuners share a common ADC clock, ensuring phase coherence within the device.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_freq` | float | 98.1e6 | Center frequency (Hz) |
| `sample_rate` | float | 2e6 | Sample rate (Hz, max 2 MHz dual-tuner) |
| `if_gain` | int | 40 | IF gain (20-59 dB) |
| `lna_state` | int | 3 | LNA attenuation state (0-9) |
| `agc_enabled` | bool | False | Enable IF AGC |

**Output Format**: 2 streams of `complex64`

**Implementation**: `kraken_passive_radar/rspduo_source.py`

**Note**: Requires gr-sdrplay3 GNU Radio OOT module.

---

## Preprocessing Blocks

### phase_corrector

**Purpose**: Per-sample phase and drift compensation for surveillance channels.

**Technical Description**:

Applied to each of the 4 surveillance channels independently. Multiplies every sample
by a complex exponential that cancels the measured phase offset and linear drift rate
relative to the reference channel. Updated every calibration cycle (default 60s).

**Algorithm** (NCO-based):
```
# On calibration update:
phi0 = -measured_phase_offset        (radians)
omega = -measured_drift_rate         (radians/sec)
sample_count = 0

# Per work() call (N samples):
start_phase = phi0 + omega * sample_count / fs
phase_inc = omega / fs
phasors[0] = exp(j * start_phase)
phasors[1:N] = cumprod(exp(j * phase_inc))    # only 2 exp() calls total
output[n] = input[n] * phasors[n]
sample_count += N
```

**Key Features**:
- **Thread-safe**: Lock protects shared state between calibration and GR scheduler threads
- **NCO optimization**: `np.cumprod()` replaces per-sample `np.exp()` (3.6x faster on RPi5)
- **Static fast path**: Single-phasor multiply when drift rate is zero
- **4-channel**: One independent instance per surveillance channel

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | float | 250000 | Decimated sample rate (Hz) |

**Runtime Methods**:

| Method | Description |
|--------|-------------|
| `set_correction(phase_rad, drift_rad_per_sec)` | Update from calibration measurement |

### conditioning

**Purpose**: Automatic Gain Control (AGC) with configurable time constants.

**Technical Description**:

The conditioning block normalizes signal amplitude to prevent saturation in subsequent processing stages. It uses an exponential moving average of signal magnitude with separate attack and decay time constants.

**Algorithm**:
```
magnitude = |x[n]|
if magnitude > current_level:
    current_level = attack * magnitude + (1 - attack) * current_level
else:
    current_level = decay * magnitude + (1 - decay) * current_level

gain = target_level / max(current_level, epsilon)
gain = clamp(gain, min_gain, max_gain)  # max_gain=1000 to prevent spikes
y[n] = x[n] * gain
```

**Key Features**:
- **Decay-only mode**: When signal drops below threshold, only decay is applied (prevents gain explosion on silence).
- **Clamped gain**: Maximum gain limited to 1000x to prevent numerical overflow.
- **NEON optimization**: Uses `neon_complex_magnitude_f32()` for batch magnitude computation on ARM.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_level` | float | 1.0 | Target output RMS level |
| `attack_time` | float | 0.001 | Attack time constant (seconds) |
| `decay_time` | float | 0.1 | Decay time constant (seconds) |
| `max_gain` | float | 1000 | Maximum gain limit |
| `sample_rate` | float | 2.4e6 | Sample rate for time constant conversion |

**Implementation**: `src/conditioning.cpp`, `libkraken_conditioning.so`

---

### dvbt_reconstructor (Block B3)

**Purpose**: Reference signal reconstruction for improved correlation gain.

**Technical Description**:

Block B3 implements reference signal reconstruction to improve the quality of the reference channel before cross-correlation. This provides 10-20 dB improvement in detection sensitivity.

#### FM Radio Mode

**Algorithm**:
1. **Quadrature FM demodulation**: Extract audio from FM signal
   ```
   phase[n] = atan2(Q[n], I[n])
   audio[n] = (phase[n] - phase[n-1]) * sample_rate / (2 * pi * max_deviation)
   ```
2. **Audio lowpass filter**: 1157-tap FIR, 15 kHz cutoff
3. **Pilot regeneration**: 57819-tap bandpass filter centered at 19 kHz (stereo pilot)
4. **Pre-emphasis**: 75 μs US standard (τ = 75e-6)
5. **FM remodulation**: Generate clean FM signal
   ```
   phase_out[n] = phase_out[n-1] + 2 * pi * max_deviation * audio[n] / sample_rate
   y[n] = cos(phase_out[n]) + j * sin(phase_out[n])
   ```

**Performance**: 8% CPU usage, 10-15 dB SNR improvement

#### ATSC 3.0 Mode

**Algorithm**:
1. **OFDM demodulation**: FFT to extract subcarriers
   ```
   X[k] = FFT(x[n] * window[n])
   ```
2. **Pilot extraction**: Identify pilot subcarrier locations
3. **SVD enhancement**: Singular value decomposition for noise reduction
   ```
   U, S, Vh = SVD(pilot_matrix)
   S_thresholded = S * (S > 0.9 * S[0])  # Keep 90% energy
   pilots_enhanced = U @ diag(S_thresholded) @ Vh
   ```
4. **Channel estimation**: Interpolate pilots to all subcarriers
5. **OFDM remodulation**: IFFT to regenerate time-domain signal

**FFT Sizes**: 8K, 16K, 32K (ATSC 3.0 standard)

**Performance**: 49% CPU usage, 15-20 dB SNR improvement (with full LDPC)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal_type` | enum | 'fm' | Signal type: fm, atsc3, dvbt, passthrough |
| `fft_size` | int | 8192 | FFT size for OFDM modes |
| `sample_rate` | float | 2.4e6 | Sample rate (Hz) |

**Implementation**: `gr-kraken_passive_radar/lib/dvbt_reconstructor_impl.cc`

---

## Core Processing Blocks

### eca_canceller

**Purpose**: Extended Cancellation Algorithm for clutter suppression.

**Technical Description**:

The ECA-B (Batch) clutter canceller uses NLMS (Normalized Least Mean Squares) adaptive filtering to remove direct-path interference and multipath clutter from the surveillance channel.

**Algorithm**:

The filter minimizes the error between surveillance and filtered reference:
```
e[n] = surv[n] - sum(w[k] * ref[n - k]) for k = 0 to num_taps-1
```

**NLMS Update** (batch mode):
1. Build autocorrelation matrix R:
   ```
   R[j,k] = sum(ref[n-j] * conj(ref[n-k])) for n = 0 to N-1
   ```
2. Build cross-correlation vector p:
   ```
   p[j] = sum(surv[n] * conj(ref[n-j])) for n = 0 to N-1
   ```
3. Add diagonal loading for stability:
   ```
   R_reg = R + lambda * I  (lambda = reg_factor, default 0.001)
   ```
4. Solve for optimal weights:
   ```
   w = R_reg^(-1) * p  (via Cholesky decomposition)
   ```
5. Apply filter and compute error:
   ```
   y[n] = surv[n] - sum(w[k] * ref[n-k])
   ```

**Key Features**:
- **Batch processing**: Entire CPI processed at once for optimal weights
- **Diagonal loading**: Prevents ill-conditioning of R matrix
- **NEON optimization**: Uses OptMathKernels for complex dot products

**Parameters** (OOT block: `eca_canceller`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_taps` | int | 128 | Filter length (clutter delay spread) |
| `reg_factor` | float | 0.001 | Tikhonov/diagonal loading regularization |
| `num_surv` | int | 4 | Number of surveillance channels |

**GPU Acceleration**: 10x speedup with CUDA (cuBLAS for matrix operations)

**Implementation**:
- CPU: `src/eca_b_clutter_canceller.cpp`, `libkraken_eca_b_clutter_canceller.so`
- GPU: `src/gpu/eca_gpu.cu`, `libkraken_eca_gpu.so`

---

### caf_processing

**Purpose**: Cross-Ambiguity Function computation for range profiling.

**Technical Description**:

The CAF computes the cross-correlation between surveillance and reference signals across multiple Doppler shifts to generate a range-Doppler map.

**Algorithm**:

For each Doppler bin fd:
1. Apply Doppler shift to reference:
   ```
   ref_shifted[n] = ref[n] * exp(-j * 2 * pi * fd * n / sample_rate)
   ```
2. Compute cross-correlation via FFT:
   ```
   Ref_FFT = FFT(ref_shifted, fft_len)
   Surv_FFT = FFT(surv, fft_len)
   XCorr = IFFT(Surv_FFT * conj(Ref_FFT))
   ```
3. Extract magnitude:
   ```
   CAF[fd, range] = |XCorr[range]|
   ```

**Linear Correlation**:

To avoid circular correlation artifacts:
```
fft_len = next_power_of_2(2 * n_samples)
```

Zero-padding ensures linear (not circular) convolution.

**Key Features**:
- **Precomputed phasors**: Doppler shift phasors computed once at initialization
- **FFTW/cuFFT**: Uses optimized FFT libraries
- **Linear correlation**: Zero-padded to next power-of-2 to avoid circular artifacts

**Parameters** (GRC block: `caf`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | int | 4096 | CPI length (samples per processing interval) |

The CAF block computes the range profile (cross-correlation) only. Doppler processing
is handled by the separate `doppler_processor` block, which applies windowing and
slow-time FFT across multiple CPI outputs.

The standalone C++ kernel (`caf_create_full()`) additionally accepts `n_doppler`,
`n_range`, `doppler_start`, `doppler_step`, and `sample_rate` for full range-Doppler
map computation, but these are not exposed in the GRC block.

**GPU Acceleration**: 23x speedup with CUDA (cuFFT batched)

**Implementation**:
- CPU: `src/caf_processing.cpp`, `libkraken_caf_processing.so`
- GPU: `src/gpu/caf_gpu.cu`, `libkraken_caf_gpu.so`

---

### doppler_processor

**Purpose**: Slow-time FFT for Doppler velocity extraction.

**Technical Description**:

The Doppler processor accumulates multiple range profiles (from CAF output) and computes FFT across the slow-time dimension to extract Doppler frequency shifts.

**Algorithm**:

Input: Matrix of range profiles [num_doppler_bins × num_range_bins]

For each range bin r:
1. Extract slow-time column:
   ```
   x[d] = input[d, r] for d = 0 to num_doppler_bins-1
   ```
2. Apply window:
   ```
   x_windowed[d] = x[d] * window[d]
   ```
3. Compute FFT:
   ```
   X[k] = FFT(x_windowed)
   ```
4. Apply FFT shift (NumPy convention):
   ```
   X_shifted[k] = X[(k + N/2) % N]
   ```

**Windowing Options**:

| Type | Formula | Sidelobe Level |
|------|---------|----------------|
| Rectangular | 1 | -13 dB |
| Hamming | 0.54 - 0.46*cos(2πn/N) | -43 dB |
| Hann | 0.5*(1 - cos(2πn/N)) | -32 dB |
| Blackman | 0.42 - 0.5*cos(2πn/N) + 0.08*cos(4πn/N) | -58 dB |

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_range_bins` | int | 4096 | Range dimension |
| `num_doppler_bins` | int | 256 | Doppler FFT size |
| `window_type` | int | 1 | 0=rect, 1=hamming, 2=hann, 3=blackman |
| `output_power` | bool | True | Output |X|² (true) or complex X (false) |

**GPU Acceleration**: 1.2x speedup with CUDA (cuFFT batched)

**Implementation**:
- CPU: `src/doppler_processing.cpp`, `libkraken_doppler_processing.so`
- GPU: `src/gpu/doppler_gpu.cu`, `libkraken_doppler_gpu.so`

---

## Detection Blocks

### cfar_detector

**Purpose**: Constant False Alarm Rate detection in range-Doppler maps.

**Technical Description**:

CFAR detection maintains a constant probability of false alarm (Pfa) by adapting the detection threshold to local noise levels.

**Algorithms**:

**CA-CFAR (Cell Averaging)**:
```
noise_estimate = mean(training_cells)
threshold = noise_estimate + threshold_factor
detection = (cell_under_test > threshold)
```

**GO-CFAR (Greatest Of)**:
```
noise_left = mean(left_training_cells)
noise_right = mean(right_training_cells)
noise_estimate = max(noise_left, noise_right)
```
Better for multiple targets (avoids target masking).

**SO-CFAR (Smallest Of)**:
```
noise_estimate = min(noise_left, noise_right)
```
Better at clutter edges but higher Pfa in clutter.

**OS-CFAR (Ordered Statistic)**:
```
sorted_cells = sort(training_cells)
noise_estimate = sorted_cells[k]  # k-th smallest
```
Robust to interfering targets.

**2D Optimization (O(n) via Prefix Sum)**:

Traditional 2D CFAR is O(n² × window_size²). The optimized version uses integral images:

```
// Build prefix sum
prefix[i,j] = input[i,j] + prefix[i-1,j] + prefix[i,j-1] - prefix[i-1,j-1]

// Compute region sum in O(1)
region_sum = prefix[r2,c2] - prefix[r1-1,c2] - prefix[r2,c1-1] + prefix[r1-1,c1-1]
```

This reduces 2D CFAR to O(n) per cell regardless of window size.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfar_type` | int | 0 | 0=CA, 1=GO, 2=SO, 3=OS |
| `guard_cells` | int | 2 | Guard band around CUT |
| `training_cells` | int | 8 | Training window size |
| `threshold_db` | float | 10.0 | Detection threshold (dB above noise) |
| `os_rank` | int | N/A | OS-CFAR rank (k) |

**GPU Acceleration**: 305x speedup with CUDA (parallel per-cell)

**Implementation**:
- CPU: `src/backend.cpp`, `libkraken_backend.so`
- GPU: `src/gpu/cfar_gpu.cu`, `libkraken_cfar_gpu.so`
- OOT: `gr-kraken_passive_radar/lib/cfar_detector_impl.cc`

---

### detection_cluster

**Purpose**: Group adjacent CFAR detections into single targets.

**Technical Description**:

Uses 8-connected component labeling to cluster spatially adjacent detections:

**Algorithm**:
```
for each detection (r, d):
    if not visited[r, d]:
        cluster = BFS/DFS from (r, d) visiting 8 neighbors
        centroid = weighted_mean(cluster, weights=snr)
        output.append(centroid)
```

**Centroid Computation**:
```
range_centroid = sum(range[i] * snr[i]) / sum(snr[i])
doppler_centroid = sum(doppler[i] * snr[i]) / sum(snr[i])
snr_combined = 10 * log10(sum(10^(snr[i]/10)))
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_cluster_size` | int | 1 | Minimum detections per cluster |
| `max_cluster_size` | int | 100 | Maximum cluster size (reject if exceeded) |

**Implementation**: `gr-kraken_passive_radar/lib/detection_cluster_impl.cc`

---

## Estimation & Tracking

### aoa_estimator

**Purpose**: Angle-of-Arrival estimation using antenna array.

**Technical Description**:

#### Bartlett Beamforming

Conventional beamformer that maximizes output power:

```
P(θ) = |a(θ)^H * x|² / (a(θ)^H * a(θ))

where:
  a(θ) = steering vector for angle θ
  x = complex samples from antenna array
```

**Steering Vector (ULA)**:
```
a(θ)[k] = exp(-j * 2 * pi * d * k * sin(θ) / lambda)

where:
  d = element spacing
  k = element index (0 to N-1)
  lambda = wavelength
```

#### MUSIC (Multiple Signal Classification)

Subspace method with super-resolution capability:

1. Compute spatial covariance matrix:
   ```
   R = (1/K) * sum(x[k] * x[k]^H)
   ```

2. Eigendecomposition:
   ```
   R = U * Lambda * U^H
   ```

3. Partition into signal and noise subspaces:
   ```
   U_signal = U[:, 0:num_targets]
   U_noise = U[:, num_targets:]
   ```

4. MUSIC pseudo-spectrum:
   ```
   P(θ) = 1 / (a(θ)^H * U_noise * U_noise^H * a(θ))
   ```

5. Find peaks in P(θ) for AoA estimates.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | int | 0 | 0=Bartlett, 1=MUSIC |
| `num_elements` | int | 4 | Array elements (KrakenSDR: 4) |
| `element_spacing` | float | 0.5 | Spacing in wavelengths |
| `angle_resolution` | float | 1.0 | Angular scan step (degrees) |
| `num_sources` | int | 1 | Expected targets (MUSIC only) |

**Implementation**:
- `src/aoa_processing.cpp` (Eigen3 for SVD)
- `gr-kraken_passive_radar/lib/aoa_estimator_impl.cc`

---

### tracker

**Purpose**: Multi-target tracking with Kalman filter and GNN association.

**Technical Description**:

#### State Vector

```
x = [x, y, vx, vy]^T  (position and velocity in bistatic coordinates)
```

#### Kalman Filter

**Prediction**:
```
x_pred = F * x
P_pred = F * P * F^T + Q

where F = [1 0 dt  0]
          [0 1  0 dt]
          [0 0  1  0]
          [0 0  0  1]
```

**Update**:
```
y = z - H * x_pred  (innovation)
S = H * P_pred * H^T + R  (innovation covariance)
K = P_pred * H^T * S^(-1)  (Kalman gain)
x = x_pred + K * y
P = (I - K * H) * P_pred
```

#### Global Nearest Neighbor (GNN) Association

1. Compute cost matrix C[i,j] = Mahalanobis distance between track i and detection j
2. Apply Hungarian algorithm to find optimal assignment
3. Gating: reject assignments where C[i,j] > gate_threshold

#### Track Management

- **Tentative**: New tracks start tentative
- **Confirmed**: Promoted after M hits in N scans (default: 3/5)
- **Coasting**: Track without detection (prediction only)
- **Deleted**: Remove after K consecutive misses (default: 5)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `process_noise` | float | 1.0 | Process noise (Q scaling) |
| `measurement_noise` | float | 10.0 | Measurement noise (R scaling) |
| `gate_threshold` | float | 9.21 | Chi-square gate (95% with 2 DoF) |
| `confirm_hits` | int | 3 | Hits to confirm track |
| `confirm_window` | int | 5 | Window for confirmation |
| `delete_misses` | int | 5 | Misses to delete track |

**Implementation**: `gr-kraken_passive_radar/lib/tracker_impl.cc`

---

## Monitoring Blocks

### coherence_monitor

**Purpose**: Monitor phase coherence across channels and trigger calibration.

**Technical Description**:

The coherence monitor tracks phase relationships between channels to detect clock drift, temperature effects, or cable issues that degrade phase coherence.

**Algorithm**:

1. Cross-correlate each channel with reference:
   ```
   xcorr[ch] = FFT_corr(ch_data, ref_data)
   peak_phase[ch] = angle(xcorr[ch][peak_idx])
   ```

2. Compute phase drift rate:
   ```
   drift_rate[ch] = (current_phase[ch] - last_phase[ch]) / dt
   ```

3. Compute coherence metric:
   ```
   coherence = |mean(exp(j * phase[ch]))| for ch in 1..4
   ```
   Range: 0 (random phases) to 1 (perfect coherence)

4. Trigger calibration if:
   - `coherence < coherence_threshold` (default: 0.9)
   - `max(|drift_rate|) > drift_threshold` (default: 0.1 rad/s)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coherence_threshold` | float | 0.9 | Trigger calibration below this |
| `drift_threshold` | float | 0.1 | Max allowed drift (rad/s) |
| `averaging_time` | float | 1.0 | Phase averaging window (s) |

**Implementation**: `gr-kraken_passive_radar/lib/coherence_monitor_impl.cc`

---

## GPU Backend Selection

All GPU-accelerated blocks support runtime backend selection:

```python
from kraken_passive_radar.gpu_backend import set_processing_backend

# Auto-select (GPU if available, else CPU)
set_processing_backend('auto')

# Force GPU (raises error if unavailable)
set_processing_backend('gpu')

# Force CPU (always works)
set_processing_backend('cpu')
```

Or via environment variable:
```bash
export KRAKEN_GPU_BACKEND=gpu  # or cpu, auto
```

---

## Performance Summary

| Block | CPU (ms) | GPU (ms) | Speedup |
|-------|----------|----------|---------|
| conditioning | 0.5 | - | - |
| dvbt_reconstructor | 2-5 | - | - |
| eca_canceller | 5 | <1 | 10x |
| caf_processing | 47 | 2 | 23x |
| doppler_processor | 1.5 | 1.3 | 1.2x |
| cfar_detector | 592 | 1.9 | 305x |
| detection_cluster | 0.1 | - | - |
| aoa_estimator | 0.2 | - | - |
| tracker | 0.1 | - | - |

Typical full-pipeline latency: 15 ms (GPU) / 650 ms (CPU)
