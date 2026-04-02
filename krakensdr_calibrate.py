"""
KrakenSDR Phase Calibration via Internal Noise Source
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Uses librtlsdr directly (ctypes) to calibrate inter-channel phase offsets.

The KrakenSDR's 5 RTL-SDR dongles have independent crystal oscillators,
causing both frequency offsets (~50-100 Hz) and phase drift over time.

Calibration process:
  1. Enable internal noise source (common signal to all channels)
  2. Capture samples simultaneously (barrier-synchronized threads)
  3. Find inter-channel sample delays via short-segment cross-correlation
  4. Align samples and estimate phase offsets with frequency compensation
"""

import ctypes
import ctypes.util
import numpy as np
import json
import time
import sys
import threading
from typing import Optional

# --- librtlsdr ctypes bindings (lazy loaded) ---

_lib = None
_dev_p = ctypes.c_void_p


def _get_lib():
    """Lazy-load librtlsdr to allow module import without the library installed."""
    global _lib
    if _lib is None:
        # Try find_library first (handles platform differences)
        lib_path = ctypes.util.find_library('rtlsdr')
        if lib_path:
            try:
                _lib = ctypes.CDLL(lib_path)
                return _lib
            except OSError:
                pass
        # Fallback to common library names (including ARM variants)
        for candidate in ['librtlsdr.so.0', 'librtlsdr.so.2', 'librtlsdr.so']:
            try:
                _lib = ctypes.CDLL(candidate)
                return _lib
            except OSError:
                continue
        raise RuntimeError(
            "librtlsdr not found. Install with: sudo apt install librtlsdr-dev\n"
            "Calibration requires RTL-SDR library for direct device access."
        )
    return _lib


def _check(ret, msg=""):
    if ret != 0:
        raise RuntimeError(f"librtlsdr error {ret}: {msg}")


def get_device_count():
    return _get_lib().rtlsdr_get_device_count()


def get_index_by_serial(serial: str) -> int:
    idx = _get_lib().rtlsdr_get_index_by_serial(serial.encode())
    if idx < 0:
        raise RuntimeError(f"Device with serial {serial} not found")
    return idx


def open_device(index: int):
    dev = ctypes.c_void_p()
    _check(_get_lib().rtlsdr_open(ctypes.byref(dev), ctypes.c_uint32(index)),
           f"open device {index}")
    return dev


def close_device(dev):
    _get_lib().rtlsdr_close(dev)


def configure_device(dev, freq_hz: int, sample_rate: int, gain_tenth_db: int):
    lib = _get_lib()
    _check(lib.rtlsdr_set_center_freq(dev, ctypes.c_uint32(freq_hz)),
           "set_center_freq")
    _check(lib.rtlsdr_set_sample_rate(dev, ctypes.c_uint32(sample_rate)),
           "set_sample_rate")
    _check(lib.rtlsdr_set_tuner_gain_mode(dev, ctypes.c_int(1)),
           "set_gain_mode")
    _check(lib.rtlsdr_set_tuner_gain(dev, ctypes.c_int(gain_tenth_db)),
           "set_tuner_gain")


def set_noise_source(dev, gpio: int, enable: bool):
    _check(_get_lib().rtlsdr_set_bias_tee_gpio(dev, ctypes.c_int(gpio), ctypes.c_int(int(enable))),
           f"set_bias_tee_gpio({gpio}, {enable})")


def read_samples(dev, num_samples: int) -> np.ndarray:
    lib = _get_lib()
    num_bytes = num_samples * 2
    buf = (ctypes.c_uint8 * num_bytes)()
    n_read = ctypes.c_int(0)
    _check(lib.rtlsdr_reset_buffer(dev), "reset_buffer")
    _check(lib.rtlsdr_read_sync(dev, buf, ctypes.c_int(num_bytes),
                                  ctypes.byref(n_read)), "read_sync")
    actual = n_read.value // 2
    raw = np.frombuffer(buf, dtype=np.uint8)[:actual * 2]
    iq = raw.astype(np.float32).reshape(-1, 2)
    samples = (iq[:, 0] - 127.5 + 1j * (iq[:, 1] - 127.5)).astype(np.complex64)
    return samples / 127.5


# --- Sample delay estimation ---

def find_sample_delay(ref, surv, max_delay=20000):
    """Find inter-channel sample delay using short-segment FFT cross-correlation.

    Short segments (2048 samples ~1ms at 2MHz) are immune to inter-channel
    frequency offset (50-100 Hz causes <0.05 cycle rotation per segment).
    Multiple segments are used and the median delay is returned for robustness.

    Returns (delay, snr) where delay d means surv[n+d] ~= alpha * ref[n].
    """
    seg_len = 2048
    n = min(len(ref), len(surv))
    n_segs = 4
    delays = []
    snrs = []

    for i in range(n_segs):
        # Evenly spaced reference segments
        r_start = n // (n_segs + 1) * (i + 1)
        if r_start + seg_len > n:
            break

        r = ref[r_start:r_start + seg_len]

        # Wide surveillance window to capture the delay
        s_start = max(0, r_start - max_delay)
        s_end = min(n, r_start + seg_len + max_delay)
        s = surv[s_start:s_end]

        # FFT cross-correlation
        # Convention: IFFT(conj(R) * S) gives xcorr[k] = sum(conj(r[n]) * s[n+k]).
        # Peak at k means surv[s_start+k] aligns with ref[r_start].
        # np.abs() makes conjugation order irrelevant for delay magnitude.
        n_fft = int(2 ** np.ceil(np.log2(len(s) + seg_len)))
        R = np.fft.fft(r.astype(np.complex128), n_fft)
        S = np.fft.fft(s.astype(np.complex128), n_fft)
        xcorr = np.abs(np.fft.ifft(np.conj(R) * S))

        # Peak in valid range
        valid_len = len(s) - seg_len + 1
        peak_k = int(np.argmax(xcorr[:valid_len]))

        # Convert FFT lag to actual sample delay
        # xcorr[k] peaks when ref[r_start+n] matches surv[s_start+k+n]
        # => delay d = s_start + k - r_start
        delay = s_start + peak_k - r_start

        noise_floor = float(np.median(xcorr[:valid_len]))
        snr = float(xcorr[peak_k]) / (noise_floor + 1e-20)

        delays.append(delay)
        snrs.append(snr)

    if not delays:
        print("  WARNING: No valid segments for delay estimation")
        return 0, 0.0

    # Filter out low-SNR segments before taking median
    min_snr = 5.0
    valid = [(d, s) for d, s in zip(delays, snrs) if s >= min_snr]
    if not valid:
        print(f"  WARNING: All {len(delays)} segments below SNR threshold "
              f"({min_snr}), max SNR={max(snrs):.1f}")
        # Fall back to all segments but warn
        valid = list(zip(delays, snrs))

    valid_delays, valid_snrs = zip(*valid)
    delay = int(np.median(valid_delays))
    snr = float(np.median(valid_snrs))
    return delay, snr


def align_samples(ref, surv, delay):
    """Align samples by compensating for inter-channel delay.

    Returns (ref_aligned, surv_aligned) of equal length.
    """
    if delay > 0:
        n = min(len(ref), len(surv) - delay)
        if n < 1:
            raise ValueError(f"Delay {delay} too large for {len(surv)} samples")
        return ref[:n].copy(), surv[delay:delay + n].copy()
    elif delay < 0:
        d = -delay
        n = min(len(ref) - d, len(surv))
        if n < 1:
            raise ValueError(f"Delay {delay} too large for {len(ref)} samples")
        return ref[d:d + n].copy(), surv[:n].copy()
    else:
        n = min(len(ref), len(surv))
        return ref[:n].copy(), surv[:n].copy()


def check_clipping(samples, threshold=0.95):
    """Check for ADC clipping (I or Q component near full scale).

    Returns (clip_fraction, max_component_value).
    """
    i_comp = np.abs(samples.real)
    q_comp = np.abs(samples.imag)
    clip_frac = float(max(np.mean(i_comp > threshold), np.mean(q_comp > threshold)))
    max_val = float(max(np.max(i_comp), np.max(q_comp)))
    return clip_frac, max_val


# --- Phase estimation with frequency offset compensation ---

def estimate_phase_offset(reference, surveillance, sample_rate, block_size=1024):
    """Estimate inter-channel phase offset with frequency offset compensation.

    Samples MUST be delay-aligned before calling this function.

    Process:
    1. Divide into short blocks (block_size samples)
    2. Compute per-block normalized cross-correlation
    3. Estimate frequency offset from phase progression across blocks
    4. Derotate and average for final phase estimate

    Returns (phase_offset_rad, correlation_magnitude, freq_offset_hz)
    """
    n_blocks = min(len(reference), len(surveillance)) // block_size
    if n_blocks < 4:
        raise ValueError("Not enough samples for reliable phase estimation")

    # Per-block normalized cross-correlation
    block_xcorrs = np.zeros(n_blocks, dtype=np.complex128)
    for b in range(n_blocks):
        start = b * block_size
        r = reference[start:start + block_size]
        s = surveillance[start:start + block_size]
        rp = np.vdot(r, r).real
        sp = np.vdot(s, s).real
        if rp > 0 and sp > 0:
            block_xcorrs[b] = np.vdot(r, s) / np.sqrt(rp * sp)

    # Validate correlation quality before phase estimation
    corr_magnitudes = np.abs(block_xcorrs)
    mean_corr_mag = float(np.mean(corr_magnitudes))
    if mean_corr_mag < 0.1:
        print(f"  WARNING: Low correlation magnitude ({mean_corr_mag:.4f}), "
              f"frequency offset estimation unreliable")
        # Fall back to zero offset with reported correlation
        avg_corr = np.mean(block_xcorrs)
        return -np.angle(avg_corr), float(np.abs(avg_corr)), 0.0

    # Estimate frequency offset from phase progression
    phases = np.angle(block_xcorrs)
    phases_unwrapped = np.unwrap(phases)

    block_indices = np.arange(n_blocks, dtype=np.float64)
    coeffs = np.polyfit(block_indices, phases_unwrapped, 1)
    slope = coeffs[0]       # radians per block

    block_duration = block_size / sample_rate
    freq_offset_hz = slope / (2 * np.pi * block_duration)

    # Derotate and average
    derotation = np.exp(-1j * slope * block_indices)
    derotated = block_xcorrs * derotation
    avg_corr = np.mean(derotated)

    phase_offset = -np.angle(avg_corr)
    correlation = float(np.abs(avg_corr))

    return phase_offset, correlation, freq_offset_hz


# --- KrakenSDR device management ---

SERIALS = ["1000", "1001", "1002", "1003", "1004"]
NOISE_CTRL_SERIAL = "1000"
NOISE_GPIO = 0


class KrakenDevices:
    """Context manager for opening/closing all 5 KrakenSDR devices."""

    def __init__(self, freq_hz, sample_rate, gain_db):
        self.freq_hz = freq_hz
        self.sample_rate = sample_rate
        self.gain_tenth = int(gain_db * 10)
        self.devices = []
        self.ctrl_dev = None

    def __enter__(self):
        for serial in SERIALS:
            idx = get_index_by_serial(serial)
            dev = open_device(idx)
            configure_device(dev, self.freq_hz, self.sample_rate, self.gain_tenth)
            self.devices.append((serial, dev))
            if serial == NOISE_CTRL_SERIAL:
                self.ctrl_dev = dev
        return self

    def __exit__(self, *args):
        for serial, dev in self.devices:
            try:
                if serial == NOISE_CTRL_SERIAL:
                    set_noise_source(dev, NOISE_GPIO, False)
                close_device(dev)
            except Exception:
                pass
        self.devices = []

    def noise_on(self, settle_sec=0.3):
        set_noise_source(self.ctrl_dev, NOISE_GPIO, True)
        time.sleep(settle_sec)
        # Flush stale samples
        for _, dev in self.devices:
            read_samples(dev, 16384)

    def noise_off(self):
        set_noise_source(self.ctrl_dev, NOISE_GPIO, False)

    def capture_all(self, n_samples):
        """Capture from all channels simultaneously using barrier-synchronized threads.

        Broadband noise coherence time is ~1/BW, so all channels must start
        USB reads at approximately the same time. Remaining timing jitter is
        handled by the delay estimation step.
        """
        data = {}
        errors = {}
        barrier = threading.Barrier(len(self.devices))
        lib = _get_lib()  # Get lib once before spawning threads

        def capture_one(serial, dev):
            try:
                _check(lib.rtlsdr_reset_buffer(dev), "reset_buffer")
                barrier.wait(timeout=5.0)
                num_bytes = n_samples * 2
                buf = (ctypes.c_uint8 * num_bytes)()
                n_read = ctypes.c_int(0)
                _check(lib.rtlsdr_read_sync(dev, buf, ctypes.c_int(num_bytes),
                                              ctypes.byref(n_read)), "read_sync")
                actual = n_read.value // 2
                raw = np.frombuffer(buf, dtype=np.uint8)[:actual * 2]
                iq = raw.astype(np.float32).reshape(-1, 2)
                samples = (iq[:, 0] - 127.5 + 1j * (iq[:, 1] - 127.5)).astype(np.complex64)
                data[serial] = samples / 127.5
            except Exception as e:
                errors[serial] = e

        threads = []
        for serial, dev in self.devices:
            t = threading.Thread(target=capture_one, args=(serial, dev))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        if errors:
            raise RuntimeError(f"Capture errors: {errors}")
        return data


# --- Calibration ---

def calibrate(freq_hz: int = 103700000,
              sample_rate: int = 2000000,
              gain_db: float = 0.0,
              cal_samples: int = 524288,
              settle_time_sec: float = 0.3,
              drift_interval_sec: float = 2.0,
              cal_file: Optional[str] = "calibration.json") -> dict:
    """
    Run phase calibration using KrakenSDR internal noise source.

    Takes two snapshots separated by drift_interval_sec to estimate both
    initial phase offsets AND drift rates. With drift rate compensation,
    the calibration stays valid for minutes instead of seconds.

    Returns dict with phase_offsets, drift_rates, correlations per channel.
    """
    print(f"KrakenSDR Phase Calibration")
    print(f"  Frequency: {freq_hz/1e6:.3f} MHz")
    print(f"  Sample rate: {sample_rate/1e6:.3f} MHz")
    print(f"  Gain: {gain_db:.1f} dB")
    print(f"  Cal samples: {cal_samples}")
    print(f"  Drift interval: {drift_interval_sec:.1f} s")

    n_dev = get_device_count()
    if n_dev < 5:
        raise RuntimeError(f"Expected 5 RTL-SDR devices, found {n_dev}")

    with KrakenDevices(freq_hz, sample_rate, gain_db) as kd:
        print("\n[1/6] Devices opened and configured")

        print("[2/6] Enabling noise source...")
        kd.noise_on(settle_time_sec)

        # First snapshot
        print("[3/6] Capturing snapshot 1...")
        t1 = time.time()
        cal_data1 = kd.capture_all(cal_samples)

        # Power and clipping check (first snapshot only)
        for serial in SERIALS:
            p = 10 * np.log10(np.mean(np.abs(cal_data1[serial])**2) + 1e-20)
            clip_frac, max_val = check_clipping(cal_data1[serial])
            clip_warn = f"  CLIPPING {clip_frac*100:.1f}%!" if clip_frac > 0.01 else ""
            print(f"  SN {serial}: power={p:+.1f} dB  peak={max_val:.3f}{clip_warn}")

        # Wait and take second snapshot for drift rate estimation
        print(f"[4/6] Waiting {drift_interval_sec:.1f}s for drift measurement...")
        time.sleep(drift_interval_sec)

        t2 = time.time()
        cal_data2 = kd.capture_all(cal_samples)
        dt = t2 - t1

        kd.noise_off()

        # Process both snapshots
        def process_snapshot(cal_data):
            reference = cal_data[SERIALS[0]]
            phases = np.zeros(5, dtype=np.float64)
            corrs = np.zeros(5, dtype=np.float64)
            delays_out = [0]
            corrs[0] = 1.0

            for i in range(1, 5):
                delay, _ = find_sample_delay(reference, cal_data[SERIALS[i]])
                delays_out.append(delay)
                ref_a, surv_a = align_samples(reference, cal_data[SERIALS[i]], delay)
                phase, corr, _ = estimate_phase_offset(
                    ref_a, surv_a, sample_rate, block_size=1024)
                phases[i] = phase
                corrs[i] = corr

            return phases, corrs, delays_out

        print("[5/6] Computing phase offsets (snapshot 1)...")
        phases1, correlations, delays = process_snapshot(cal_data1)

        print("[6/6] Computing drift rates (snapshot 2)...")
        phases2, corrs2, _ = process_snapshot(cal_data2)

        # Drift rate from the two snapshots
        phase_diff = phases2 - phases1
        # Unwrap to handle wrapping around ±π
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        drift_rates_rad = phase_diff / dt  # radians per second
        drift_rates_deg = np.degrees(drift_rates_rad)

        phase_offsets = phases1  # Use first snapshot as reference time

        print(f"\n  Results (dt={dt:.2f}s):")
        for i, serial in enumerate(SERIALS):
            print(f"    Ch{i} (SN {serial}): phase={np.degrees(phase_offsets[i]):+7.2f}°"
                  f"  drift={drift_rates_deg[i]:+.3f}°/s"
                  f"  corr={correlations[i]:.4f}"
                  f"  delay={delays[i]:+d}")

    # Quality check
    min_corr = min(correlations[1:])
    if min_corr < 0.5:
        print(f"\n  WARNING: Low correlation ({min_corr:.3f}). "
              "Calibration may be unreliable.")
    elif min_corr < 0.8:
        print(f"\n  NOTE: Moderate correlation ({min_corr:.3f}). "
              "Results usable but not ideal.")

    # Estimate effective calibration lifetime with rate compensation
    # Residual drift (rate estimation error) scales as ~rate_error * time
    # With 2s measurement, rate accuracy is ~noise/dt ≈ 2°/2s = 1°/s uncertainty
    # More conservatively: calibration valid for drift_interval_sec * 50-100x
    max_drift_rate = max(abs(drift_rates_deg[i]) for i in range(1, 5))
    if max_drift_rate > 0.01:
        print(f"\n  Max drift rate: {max_drift_rate:.3f}°/s ({max_drift_rate*60:.1f}°/min)")
        print(f"  Without rate compensation: recal every ~{5.0/max_drift_rate:.1f}s")
        print(f"  With rate compensation: recal every ~300s (conservative)")

    # Build result
    result = {
        "timestamp": t1,
        "freq_hz": freq_hz,
        "sample_rate": sample_rate,
        "gain_db": gain_db,
        "drift_measurement_dt_sec": dt,
    }
    for i in range(5):
        result[f"ch{i}_phase_rad"] = float(phase_offsets[i])
        result[f"ch{i}_drift_rad_per_sec"] = float(drift_rates_rad[i])
        result[f"ch{i}_corr"] = float(correlations[i])
        result[f"ch{i}_delay"] = int(delays[i])

    if cal_file:
        with open(cal_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved to {cal_file}")

    print("\nCalibration complete.")
    return result


# --- Phase drift characterization ---

def measure_drift(freq_hz: int = 103700000,
                  sample_rate: int = 2000000,
                  gain_db: float = 0.0,
                  duration_sec: float = 300.0,
                  interval_sec: float = 5.0,
                  cal_samples: int = 131072,
                  output_file: str = "drift_log.json"):
    """
    Measure phase drift over time to determine recalibration interval.

    Keeps devices open with noise source ON the entire time.
    Takes snapshots every interval_sec and records phase offsets.
    """
    n_measurements = int(duration_sec / interval_sec) + 1
    print(f"KrakenSDR Phase Drift Characterization")
    print(f"  Duration: {duration_sec:.0f} s ({duration_sec/60:.1f} min)")
    print(f"  Interval: {interval_sec:.1f} s")
    print(f"  Measurements: {n_measurements}")
    print(f"  Frequency: {freq_hz/1e6:.3f} MHz")
    print(f"  Gain: {gain_db:.1f} dB")
    print()

    drift_log = {
        "freq_hz": freq_hz,
        "sample_rate": sample_rate,
        "gain_db": gain_db,
        "interval_sec": interval_sec,
        "measurements": []
    }

    with KrakenDevices(freq_hz, sample_rate, gain_db) as kd:
        print("Enabling noise source (stays ON for entire measurement)...")
        kd.noise_on(settle_sec=0.5)

        t0 = time.time()
        ref_phases = np.zeros(5)

        print(f"\n{'Time(s)':>8}  {'Ch1(°)':>8}  {'Ch2(°)':>8}  {'Ch3(°)':>8}  {'Ch4(°)':>8}"
              f"  {'Max|Δ|(°)':>10}  {'MinCorr':>8}")
        print("-" * 80)

        for m in range(n_measurements):
            if m > 0:
                wait_until = t0 + m * interval_sec
                remaining = wait_until - time.time()
                if remaining > 0:
                    time.sleep(remaining)

            elapsed = time.time() - t0

            # Capture and process
            data = kd.capture_all(cal_samples)
            reference = data[SERIALS[0]]

            phases = np.zeros(5)
            corrs = np.zeros(5)
            foffs = np.zeros(5)
            sample_delays = [0]
            corrs[0] = 1.0

            for i in range(1, 5):
                surv = data[SERIALS[i]]

                # Find delay (re-estimate each time as it drifts with freq offset)
                delay, _ = find_sample_delay(reference, surv, max_delay=20000)
                sample_delays.append(delay)

                # Align and estimate phase
                ref_aligned, surv_aligned = align_samples(reference, surv, delay)
                p, c, f = estimate_phase_offset(
                    ref_aligned, surv_aligned, sample_rate, block_size=1024)
                phases[i] = p
                corrs[i] = c
                foffs[i] = f

            if m == 0:
                ref_phases = phases.copy()

            # Phase drift from initial calibration
            drift = phases - ref_phases
            drift = (drift + np.pi) % (2 * np.pi) - np.pi
            drift_deg = np.degrees(drift)
            max_drift = float(np.max(np.abs(drift_deg[1:])))
            min_corr = float(np.min(corrs[1:]))

            marker = " ***" if max_drift > 5.0 else ""
            print(f"{elapsed:8.1f}  {drift_deg[1]:+8.2f}  {drift_deg[2]:+8.2f}  "
                  f"{drift_deg[3]:+8.2f}  {drift_deg[4]:+8.2f}  {max_drift:10.2f}"
                  f"  {min_corr:8.4f}{marker}")

            drift_log["measurements"].append({
                "time_sec": elapsed,
                "phases_deg": np.degrees(phases).tolist(),
                "drift_deg": drift_deg.tolist(),
                "correlations": corrs.tolist(),
                "freq_offsets_hz": foffs.tolist(),
                "delays": sample_delays,
                "max_drift_deg": max_drift,
            })

        kd.noise_off()

    # Save log
    with open(output_file, 'w') as f:
        json.dump(drift_log, f, indent=2)

    # Analyze results with proper phase unwrapping
    times = np.array([m["time_sec"] for m in drift_log["measurements"]])
    min_corrs = [min(m["correlations"][1:]) for m in drift_log["measurements"]]

    # Unwrap drift for each channel to get true cumulative drift
    raw_drift = np.array([m["drift_deg"] for m in drift_log["measurements"]])
    unwrapped_drift = np.zeros_like(raw_drift)
    for ch in range(1, 5):
        unwrapped_drift[:, ch] = np.unwrap(np.radians(raw_drift[:, ch])) * 180 / np.pi

    print(f"\n{'='*80}")
    print(f"Drift Analysis (phase-unwrapped):")
    print(f"  Mean min correlation: {np.mean(min_corrs):.4f}")

    # Without rate compensation: time to exceed 5°
    max_unwrapped = np.max(np.abs(unwrapped_drift[:, 1:]), axis=1)
    for i, md in enumerate(max_unwrapped):
        if md > 5.0:
            print(f"\n  Without rate compensation:")
            print(f"    Time to exceed 5° budget: {times[i]:.1f} s")
            print(f"    Recommended recal interval: {max(1, int(times[i] * 0.8))} s")
            break
    else:
        print(f"\n  Without rate compensation:")
        print(f"    Phase stayed within 5° for entire {duration_sec:.0f} s!")

    # Drift rates and residual with linear compensation
    print(f"\n  Per-channel drift rates:")
    max_residual = 0
    for ch in range(1, 5):
        rate, intercept = np.polyfit(times, unwrapped_drift[:, ch], 1)
        residual = unwrapped_drift[:, ch] - (rate * times + intercept)
        max_res = np.max(np.abs(residual))
        max_residual = max(max_residual, max_res)
        print(f"    Ch{ch}: rate={rate:+.4f} °/s ({rate*60:+.2f} °/min)"
              f"  max_residual={max_res:.2f}°")

    # With rate compensation: time for residual to exceed 5°
    print(f"\n  With linear rate compensation:")
    print(f"    Max residual across all channels: {max_residual:.2f}°")
    if max_residual < 5.0:
        print(f"    Residual stays within 5° for entire {duration_sec:.0f}s measurement!")
        print(f"    Recommended recal interval: >= {int(duration_sec)}s")
    else:
        # Find when residual first exceeds 5°
        for i in range(len(times)):
            worst = 0
            for ch in range(1, 5):
                rate, intercept = np.polyfit(times, unwrapped_drift[:, ch], 1)
                res = abs(unwrapped_drift[i, ch] - (rate * times[i] + intercept))
                worst = max(worst, res)
            if worst > 5.0:
                print(f"    Residual exceeds 5° at t={times[i]:.1f}s")
                print(f"    Recommended recal interval: {max(1, int(times[i] * 0.8))}s")
                break

    print(f"\n  Drift log saved to {output_file}")
    return drift_log


# --- Periodic recalibration with drift rate tracking ---

def periodic_recal(freq_hz: int = 103700000,
                   sample_rate: int = 2000000,
                   gain_db: float = 0.0,
                   recal_interval_sec: float = 120.0,
                   cal_samples: int = 524288,
                   duration_sec: float = 0.0,
                   output_file: str = "periodic_cal_log.json"):
    """
    Periodic recalibration with single-snapshot phase measurement.

    At each recalibration: noise ON ~1s, single capture, noise OFF.
    Drift rate is computed from phase CHANGE between consecutive recalibrations
    (120s time base gives much better accuracy than 2s snapshot pairs).

    This accurately tracks drift rate changes caused by temperature variations.
    The drift rate (ω = dφ/dt) and its rate of change (dω/dt) are both logged,
    which is what we need to characterize outdoor thermal sensitivity.

    Args:
        recal_interval_sec: Seconds between recalibrations (default: 120)
        duration_sec: Total run time (0 = run until Ctrl+C)
        output_file: JSON log file
    """
    if duration_sec > 0:
        n_recals = int(duration_sec / recal_interval_sec) + 1
    else:
        n_recals = 999999

    print(f"KrakenSDR Periodic Recalibration")
    print(f"  Recal interval: {recal_interval_sec:.0f} s ({recal_interval_sec/60:.1f} min)")
    print(f"  Frequency: {freq_hz/1e6:.3f} MHz, Gain: {gain_db:.1f} dB")
    if duration_sec > 0:
        print(f"  Duration: {duration_sec:.0f} s ({duration_sec/60:.1f} min), {n_recals} recals")
    else:
        print(f"  Duration: unlimited (Ctrl+C to stop)")
    print(f"  Noise source ON time per recal: ~1s")
    print()

    log = {
        "freq_hz": freq_hz,
        "sample_rate": sample_rate,
        "gain_db": gain_db,
        "recal_interval_sec": recal_interval_sec,
        "recalibrations": []
    }

    header = (f"{'Time':>8}  {'Ch1 φ°':>8}  {'Ch2 φ°':>8}  {'Ch3 φ°':>8}  {'Ch4 φ°':>8}"
              f"  │ {'Ch1 ω':>8}  {'Ch2 ω':>8}  {'Ch3 ω':>8}  {'Ch4 ω':>8}"
              f"  │ {'MinCorr':>7}  {'PredErr':>7}")
    print(header)
    print(f"{'(s)':>8}  {'(deg)':>8}  {'(deg)':>8}  {'(deg)':>8}  {'(deg)':>8}"
          f"  │ {'(°/s)':>8}  {'(°/s)':>8}  {'(°/s)':>8}  {'(°/s)':>8}"
          f"  │ {'':>7}  {'(°)':>7}")
    print("─" * len(header))

    with KrakenDevices(freq_hz, sample_rate, gain_db) as kd:
        t0 = time.time()
        # History for unwrapped phase tracking
        phase_history = []  # list of (time, phases_rad[5])
        prev_drift_rates = None

        for m in range(n_recals):
            if m > 0:
                wait_until = t0 + m * recal_interval_sec
                remaining = wait_until - time.time()
                if remaining > 0:
                    time.sleep(remaining)

            elapsed = time.time() - t0

            # --- Brief calibration burst (~1s) ---
            kd.noise_on(settle_sec=0.3)
            data = kd.capture_all(cal_samples)
            kd.noise_off()
            # --- End burst ---

            snap_time = time.time()

            # Compute phase offsets
            ref = data[SERIALS[0]]
            phases_rad = np.zeros(5)
            corrs = np.zeros(5)
            corrs[0] = 1.0
            delays = [0]

            for i in range(1, 5):
                delay, _ = find_sample_delay(ref, data[SERIALS[i]])
                delays.append(delay)
                r_a, s_a = align_samples(ref, data[SERIALS[i]], delay)
                p, c, _ = estimate_phase_offset(r_a, s_a, sample_rate, block_size=1024)
                phases_rad[i] = p
                corrs[i] = c

            phase_history.append((elapsed, phases_rad.copy()))

            # Compute drift rate from phase change (unwrapped)
            drift_rates = np.zeros(5)  # °/s
            prediction_error = 0.0

            if len(phase_history) >= 2:
                t_prev, ph_prev = phase_history[-2]
                t_now, ph_now = phase_history[-1]
                dt = t_now - t_prev

                if dt > 0:
                    dphase = ph_now - ph_prev
                    # Unwrap
                    dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
                    drift_rates = np.degrees(dphase) / dt

                # Prediction error: how far off was last recal's linear extrapolation?
                if len(phase_history) >= 3 and prev_drift_rates is not None:
                    t_pp, ph_pp = phase_history[-3]
                    dt_pred = t_now - t_prev
                    # Predicted phase = ph_prev + prev_drift_rate * dt
                    predicted = ph_prev + np.radians(prev_drift_rates) * dt_pred
                    actual = ph_now
                    pred_err = actual - predicted
                    pred_err = (pred_err + np.pi) % (2 * np.pi) - np.pi
                    prediction_error = float(np.max(np.abs(np.degrees(pred_err[1:]))))

            # Rate of change of drift rate (°/s²)
            drift_accel = np.zeros(5)
            if prev_drift_rates is not None:
                drift_accel = (drift_rates - prev_drift_rates) / recal_interval_sec

            min_corr = float(np.min(corrs[1:]))
            phases_deg = np.degrees(phases_rad)

            rate_str = "     ---" if m == 0 else f"{drift_rates[1]:+8.4f}"
            line = f"{elapsed:8.1f}  {phases_deg[1]:+8.2f}  {phases_deg[2]:+8.2f}" \
                   f"  {phases_deg[3]:+8.2f}  {phases_deg[4]:+8.2f}  │"
            if m == 0:
                line += f" {'---':>8}  {'---':>8}  {'---':>8}  {'---':>8}"
            else:
                line += (f" {drift_rates[1]:+8.4f}  {drift_rates[2]:+8.4f}"
                         f"  {drift_rates[3]:+8.4f}  {drift_rates[4]:+8.4f}")
            line += f"  │ {min_corr:7.4f}"
            if prediction_error > 0:
                marker = " ***" if prediction_error > 5.0 else ""
                line += f"  {prediction_error:7.2f}{marker}"

            print(line)

            # Print drift acceleration if significant
            if prev_drift_rates is not None and np.max(np.abs(drift_accel[1:])) > 0.001:
                print(f"         dω/dt (°/s²):"
                      f"  {drift_accel[1]:+.5f}  {drift_accel[2]:+.5f}"
                      f"  {drift_accel[3]:+.5f}  {drift_accel[4]:+.5f}")

            log["recalibrations"].append({
                "time_sec": elapsed,
                "phases_deg": phases_deg.tolist(),
                "drift_rates_deg_per_sec": drift_rates.tolist(),
                "drift_accel_deg_per_sec2": drift_accel.tolist(),
                "prediction_error_deg": prediction_error,
                "correlations": corrs.tolist(),
                "delays": delays,
                "min_corr": min_corr,
            })

            prev_drift_rates = drift_rates.copy()

            # Crash-safe save
            with open(output_file, 'w') as f:
                json.dump(log, f, indent=2)

    # Final analysis
    recals = log["recalibrations"]
    times = [r["time_sec"] for r in recals]
    pred_errors = [r["prediction_error_deg"] for r in recals if r["prediction_error_deg"] > 0]

    print(f"\n{'='*100}")
    print(f"Periodic Recalibration Analysis ({len(recals)} recals over {times[-1]:.0f}s)")

    if pred_errors:
        print(f"\n  Prediction error (linear extrapolation over {recal_interval_sec:.0f}s):")
        print(f"    Mean: {np.mean(pred_errors):.2f}°")
        print(f"    Max:  {np.max(pred_errors):.2f}°")
        print(f"    Std:  {np.std(pred_errors):.2f}°")
        n_exceeded = sum(1 for e in pred_errors if e > 5.0)
        print(f"    Exceeded 5° budget: {n_exceeded}/{len(pred_errors)} intervals")

    for ch in range(1, 5):
        rates = [r["drift_rates_deg_per_sec"][ch] for r in recals[1:]]  # skip first (no rate)
        if rates:
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            print(f"  Ch{ch} drift rate: mean={mean_rate:+.4f} °/s ({mean_rate*60:+.2f} °/min)"
                  f"  std={std_rate:.4f} °/s")

    # Key output: is 2-minute recal sufficient?
    if pred_errors:
        max_pred = max(pred_errors)
        if max_pred < 5.0:
            print(f"\n  RESULT: {recal_interval_sec:.0f}s recal interval SUFFICIENT")
            print(f"  Max prediction error: {max_pred:.2f}° (budget: 5°)")
        else:
            # Estimate what interval would work
            # prediction_error scales roughly with interval² (for rate changes)
            # or linearly with interval (for rate measurement noise)
            safe = recal_interval_sec * 5.0 / max_pred
            print(f"\n  RESULT: {recal_interval_sec:.0f}s recal interval INSUFFICIENT")
            print(f"  Max prediction error: {max_pred:.2f}° (budget: 5°)")
            print(f"  Suggested interval: {safe:.0f}s")

    print(f"\n  Log saved to {output_file}")
    return log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KrakenSDR Phase Calibration")
    sub = parser.add_subparsers(dest="command", help="Command")

    # Calibrate command
    cal_parser = sub.add_parser("calibrate", help="Run phase calibration")
    cal_parser.add_argument("--freq", type=float, default=103.7e6)
    cal_parser.add_argument("--gain", type=float, default=0.0)
    cal_parser.add_argument("--samples", type=int, default=524288)
    cal_parser.add_argument("--output", type=str, default="calibration.json")

    # Drift measurement command (noise source stays ON)
    drift_parser = sub.add_parser("drift", help="Continuous drift measurement (noise ON)")
    drift_parser.add_argument("--freq", type=float, default=103.7e6)
    drift_parser.add_argument("--gain", type=float, default=0.0)
    drift_parser.add_argument("--duration", type=float, default=300.0,
                              help="Duration in seconds (default: 300)")
    drift_parser.add_argument("--interval", type=float, default=5.0,
                              help="Measurement interval in seconds (default: 5)")
    drift_parser.add_argument("--output", type=str, default="drift_log.json")

    # Periodic recalibration (noise source cycles ON/OFF)
    per_parser = sub.add_parser("periodic",
                                help="Periodic recal: measure phase + drift rate every N seconds")
    per_parser.add_argument("--freq", type=float, default=103.7e6)
    per_parser.add_argument("--gain", type=float, default=0.0)
    per_parser.add_argument("--interval", type=float, default=120.0,
                            help="Recalibration interval in seconds (default: 120)")
    per_parser.add_argument("--duration", type=float, default=0.0,
                            help="Total duration (0 = run until Ctrl+C)")
    per_parser.add_argument("--output", type=str, default="periodic_cal_log.json")

    args = parser.parse_args()

    if args.command is None:
        args.command = "calibrate"
        args.freq = 103.7e6
        args.gain = 0.0
        args.samples = 524288
        args.output = "calibration.json"

    try:
        if args.command == "calibrate":
            calibrate(
                freq_hz=int(args.freq),
                sample_rate=2000000,
                gain_db=args.gain,
                cal_samples=args.samples,
                cal_file=args.output,
            )
        elif args.command == "drift":
            measure_drift(
                freq_hz=int(args.freq),
                sample_rate=2000000,
                gain_db=args.gain,
                duration_sec=args.duration,
                interval_sec=args.interval,
                output_file=args.output,
            )
        elif args.command == "periodic":
            periodic_recal(
                freq_hz=int(args.freq),
                sample_rate=2000000,
                gain_db=args.gain,
                recal_interval_sec=args.interval,
                duration_sec=args.duration,
                output_file=args.output,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
