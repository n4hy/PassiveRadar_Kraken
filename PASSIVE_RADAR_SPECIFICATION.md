# Passive Radar Processing Chain Specification

This document specifies a complete **passive bistatic radar (PBR)** processing chain based on **KrakenSDR** with **Explicit Clutter Cancellation (ECA)**.  
Blocks are named as if they were GNU Radio blocks, but this is a **system-level specification**, not tied to a specific framework.

---

## 1. Top-Level Logical Block Diagram

```
KrakenSDR Source (5ch coherent IQ)
        |
DC Block + IQ Balance (per channel)
        |
Sample Clock / Time Alignment (fractional delay + phase)
        |
Frequency Translate + Decimate (channel of interest)
        |
        +--> Reference Conditioning (AGC / Whitening) ---------+
        |                                                      |
        +--> Surveillance Conditioning #1 -------------------- |
        +--> Surveillance Conditioning #2 -------------------- |--> ECA Clutter Cancel
        +--> Surveillance Conditioning #3 -------------------- |
        +--> Surveillance Conditioning #4 -------------------- |
                                                               |
                                   +---------------------------+---------------------------+---------------------------+---------------------------+
                                   |                           |                           |                           |
                            Range-Doppler (CAF) #1      Range-Doppler (CAF) #2      Range-Doppler (CAF) #3      Range-Doppler (CAF) #4
                                   |                           |                           |                           |
                                   +---------------------------+----------- Multi-channel Fusion -----------+---------------------------+
                                                                       |
                                                                   CFAR Detector
                                                                       |
                                                                Detection Clustering
                                                                       |
                                                                    Tracker
                                                                       |
                                                             Angle / AoA Estimation
                                                                       |
                                                            Geolocation (Bistatic)
                                                                       |
                                                               Display / Logging
```

---

## 2. Signal Representation and Global Assumptions

* All signals are **complex analytic (I/Q)** after the KrakenSDR source.
* Five coherent channels:
  * `r(t)` – reference channel
  * `s1(t) … s4(t)` – surveillance channels
* All channels share a common sample rate `Fs` and must remain phase-coherent.
* Processing transitions from streaming to **vector (CPI-based)** processing prior to ECA and CAF stages.

---

## 3. Block Specifications

### 3.1 KrakenSDR Source (5ch coherent IQ)

**Purpose**  
Acquire five synchronous RF channels for passive radar processing.

**Function**
* Outputs five complex baseband streams at common `Fs`
* Preserves relative phase coherence across channels

**Efficiency Requirements**
* Single producer, multi-output
* Zero-copy buffers where possible
* Monotonic timestamping for CPI alignment

---

### 3.2 DC Block + IQ Balance (per channel)

**Purpose**  
Remove DC offsets, LO leakage, and correct IQ imbalance.

**Function**
* Single-pole IIR DC blocker or equivalent
* Optional fixed or calibrated IQ correction matrix

**Efficiency**
* Lightweight per-sample processing
* Implement using vectorized complex kernels

---

### 3.3 Sample Clock / Time Alignment

**Purpose**  
Achieve sub-sample alignment between reference and surveillance channels.

**Function**
* Integer sample delay correction
* Fractional delay correction (Farrow or polyphase)
* Constant phase offset correction
* Optional slow phase-drift tracking

**Notes**
* Alignment updated on a slow control loop (every K CPIs)
* Fractional delay filters should be short (8–16 taps)

---

### 3.4 Frequency Translation + Decimation

**Purpose**  
Isolate the illuminator-of-opportunity band and reduce computational load.

**Function**
* Complex frequency shift (NCO)
* Low-pass filtering
* Decimation to `Fs'` where `Fs' ≈ 1.2–1.5 × B`

**Efficiency**
* Polyphase FIR or PFB-based decimator
* Identical group delay across all channels

---

### 3.5 Reference Conditioning

**Purpose**  
Prepare a stable reference signal for clutter cancellation and matched filtering.

**Function**
* Slow AGC or amplitude limiting
* Optional spectral whitening

**Notes**
* Whitening may be done via FIR pre-emphasis or FFT-domain normalization

---

### 3.6 Surveillance Conditioning (×4)

**Purpose**  
Normalize surveillance channels and improve numerical conditioning.

**Function**
* Slow AGC or limiter
* Optional whitening consistent with reference processing

**Constraints**
* AGC time constant must be much longer than CPI duration

---

## 4. ECA Clutter Cancellation (Core Block)

### 4.1 Block Name
`eca_clutter_cancel_5in_4out`

### 4.2 Purpose
Remove direct-path, multipath, and static clutter from surveillance channels using the reference channel.

### 4.3 Signal Model

For surveillance channel `k`:

```
ĉ_k[n] = Σ_{l=0}^{L-1} w_{k,l} · r[n-l]
y_k[n] = s_k[n] − ĉ_k[n]
```

Where:
* `r[n]` – reference signal
* `s_k[n]` – surveillance signal
* `L` – clutter delay span (taps)
* `w_k` – complex filter weights

### 4.4 Implementation Requirements

* Four independent adaptive filters (one per surveillance channel)
* Shared reference signal
* Support both:
  * Streaming adaptive (LMS / NLMS / block-LMS)
  * Block LS / Wiener solutions with regularization

### 4.5 Efficiency Constraints

* No explicit Toeplitz matrix construction in hot path
* Use FFT-based convolution or correlation
* Diagonal loading for numerical stability
* Output clutter-suppressed channels plus optional diagnostics

---

## 5. Range–Doppler / Cross-Ambiguity Processing

### 5.1 Block Name
`caf_range_doppler_fft_vcc`

### 5.2 Purpose
Compute bistatic delay–Doppler maps.

### 5.3 Function
* Segment CPI into `M` blocks of length `N`
* FFT-based cross-correlation between reference and surveillance
* Produce `RD_k(τ, ν)` for each channel

### 5.4 Efficiency

* Overlap-save FFT correlation
* Batched FFT plans
* Prefer `complex<float>` unless precision demands otherwise

---

## 6. Multi-Channel Fusion

**Purpose**
Combine detections from multiple surveillance channels.

**Methods**
* Noncoherent integration: `Σ |RD_k|²`
* Coherent fusion if phase-calibrated

---

## 7. Detection and Tracking

### 7.1 CFAR Detector

**Purpose**
Control false alarm rate in range–Doppler domain.

**Function**
* 2D CFAR (CA, GOCA, SOCA, or OS)
* Guard and training regions

---

### 7.2 Detection Clustering

**Purpose**
Merge adjacent detections into plots.

**Function**
* Local maxima detection
* Connected-component labeling
* Centroid extraction

---

### 7.3 Tracker

**Purpose**
Maintain target state estimates over time.

**Methods**
* MHT, JPDA, or Kalman/UKF-based tracking
* Track confirmation and deletion logic

---

## 8. Angle-of-Arrival Estimation

**Purpose**
Estimate bearing using the four surveillance channels as an array.

**Methods**
* Phase interferometry
* MUSIC / ESPRIT (optional)

**Requirements**
* Stable inter-channel calibration
* Known antenna geometry

---

## 9. Geolocation (Bistatic)

**Purpose**
Estimate target position using bistatic range, Doppler, and AoA.

**Function**
* Solve bistatic ellipse equations
* Incorporate known transmitter and receiver geometry

---

## 10. Display and Logging

**Purpose**
Operational visibility and offline analysis.

**Outputs**
* Range–Doppler plots
* Detection overlays
* Track histories
* QA and performance metrics

---

## 11. System-Level Efficiency Requirements

* Vectorized processing beyond front-end
* Reuse FFT plans
* Aligned, contiguous memory
* Regularization in all adaptive solvers
* Identical multi-rate paths across channels

---

## 12. Conclusion

This document defines a complete, efficient, and operationally realistic passive radar processing chain suitable for KrakenSDR-based systems with ECA clutter cancellation.
