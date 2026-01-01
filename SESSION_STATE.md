# Session State & Architectural Roadmap

## 1. Project Goal
Implement a high-performance, coherent Passive Radar system using KrakenSDR (5-channel RTL-SDR). The architecture follows a strict "Passive Radar Done Right" topology, prioritizing mathematical correctness, phase coherence, and computational efficiency.

## 2. Current Session Accomplishments
*   **GRC Cleanup:** Repaired `kraken_passive_radar_system.grc`.
    *   **Visibility:** All hidden blocks (Filters, DC Blockers, ECA) are now exposed and visibly selectable on the canvas.
    *   **Fixes:** Renamed invalid `fft_vxx` blocks to `fft_vcc`.
    *   **Connectivity:** Resolved conflicting input connections. Retained the implicit `Source -> Filter -> DC -> ECA` chain but made it visible.
*   **Audit:** Verified that `krakensdr_source.py` exposes hardware control for the noise source (`set_noise_source`) but **lacks** any signal processing logic for calibration.

## 3. Master Architecture & Requirements

### A. Frontend & Calibration (The Foundation)
*   **Source:** KrakenSDR (5ch coherent IQ).
*   **DC Block:** Per-channel removal of DC offset.
*   **Time/Phase Alignment:**
    *   *Requirement:* Use hardware noise source to correlate and calculate fractional delay + phase offsets.
    *   *Status:* **Missing**. Needs implementation.
*   **Freq Translation / Decimation:**
    *   *Requirement:* Polyphase FIR or PFB implementations to preserve phase coherence.
    *   *Status:* Implemented in GRC via `freq_xlating_fir_filter`, needs verification/wrapping.

### B. Conditioning (Pre-Processing)
*   **Reference & Surveillance Conditioning:**
    *   *Requirement:* **AGC** (Slow relative to CPI) and optional **Spectral Whitening**.
    *   *Purpose:* Improve numerical conditioning for ECA/CAF.
    *   *Status:* Partially implicitly done, needs explicit blocks.

### C. Clutter Cancellation (The Filter)
*   **ECA (Extensive Cancellation Algorithm):**
    *   *Model:* $y_k[n] = s_k[n] - \sum w_{k,l} r[n-l]$
    *   *Requirement:* Efficient Block/Batch Least Squares. **Diagonal Loading** (Regularization) is mandatory for stability.
    *   *Status:* C++ implementation exists (`eca_b_clutter_canceller.cpp`). Needs audit for Diagonal Loading compliance.

### D. Range-Doppler Processing (The Map)
*   **CAF (Cross-Ambiguity Function):**
    *   *Requirement:* **Overlap-Save** FFT correlation or Batched FFTs.
    *   *Status:* GRC uses basic circular convolution (FFT->Mult->IFFT). Needs upgrade to Overlap-Save logic for continuous streams.

### E. Backend (Fusion & Detection)
*   **Fusion:** Combine 4 Surv channels.
    *   *Modes:* Non-coherent (Power sum) or Coherent (Voltage sum, requires precise calibration).
*   **CFAR:** Adaptive thresholding.
*   **Clustering:** Centroiding multiple hits into Plots.
*   **Tracking:** State estimation over time.
*   **AoA:** Estimation using array geometry (Music/Bartlett) on detected Plots.
*   **Geolocation:** Bistatic equation solving.

### F. Efficiency & Implementation Standards
*   **Hot Path:** Vectorized C++ (AVX/Neon) or GPU kernels.
*   **Memory:** Contiguous and aligned buffers. Zero-copy where possible.
*   **FFT:** Plans must be reused (e.g., `FFTW_MEASURE` cached).
*   **Stability:** Regularization (Diagonal Loading) and Condition Number checks are required.

## 4. Next Immediate Steps (Plan)
1.  **Implement "Calibration Block":** Create a Python/C++ block to handle the Time/Phase alignment using the noise source.
    *   *Test:* Verify 0-phase difference on correlated noise inputs.
2.  **Unit Test Audit:** Systematically verify `ECA` implementation for Diagonal Loading and `CAF` for Overlap-Save correctness.
3.  **Conditioning Blocks:** Implement/Verify Slow-AGC and Whitening blocks.

---
*End of Session Report - [Date]*
