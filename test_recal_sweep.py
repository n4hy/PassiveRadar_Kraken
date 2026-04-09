#!/usr/bin/env python3
"""
Progressive recalibration interval test.

Calibrates all 4 surveillance channels, then tests increasing intervals
(10, 20, 30, ... seconds) with linear drift compensation. Terminates
when ANY channel exceeds 5° compensated deviation.
"""

import sys
import time
import io
import contextlib
import numpy as np
from gnuradio import gr, blocks, filter
from gnuradio.filter import firdes, window
from gnuradio.kraken_passive_radar import krakensdr_source


class sweep_flowgraph(gr.top_block):
    """Minimal 5-channel flowgraph with vector probes for recalibration sweep testing.

    Technique: DC block, LPF/decimate, stream-to-vector, probe per channel.
    """
    def __init__(self, freq=103.7e6, gain=49.6):
        """Initialize 5-channel flowgraph for recalibration interval testing.

        Technique: per-channel DC block, LPF/decimation, and vector probe chain.
        """
        gr.top_block.__init__(self, "Recal Sweep Test")
        samp_rate = 2_400_000
        signal_bw = 250_000
        decimation = int(samp_rate / signal_bw)
        lpf_taps = firdes.low_pass(
            1.0, samp_rate, signal_bw / 2, signal_bw / 10, window.WIN_HAMMING)

        self.src = krakensdr_source(
            frequency=freq, sample_rate=samp_rate, gain=gain)

        self.vec_size = 2**16
        self._probes = []
        for ch in range(5):
            dc = filter.dc_blocker_cc(32, True)
            lpf = filter.freq_xlating_fir_filter_ccc(
                decimation, lpf_taps, 0, samp_rate)
            s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.vec_size)
            probe = blocks.probe_signal_vc(self.vec_size)
            self._probes.append(probe)
            self.connect((self.src, ch), dc, lpf, s2v, probe)

    def read_vector(self, ch):
        """Return the latest contiguous sample vector from channel ch.

        Technique: reads probe_signal_vc output as complex64 array.
        """
        return np.array(self._probes[ch].level(), dtype=np.complex64)


def xcorr_phase(ref, surv):
    """Compute phase offset between ref and surv via FFT cross-correlation.

    Technique: FFT-based cross-correlation with delay alignment and normalized dot product.
    """
    N = len(ref)
    Xr = np.fft.fft(ref)
    Xs = np.fft.fft(surv)
    xc = np.fft.ifft(Xr * np.conj(Xs))
    peak = int(np.argmax(np.abs(xc)))
    delay = peak if peak <= N // 2 else peak - N
    overlap = N - abs(delay)
    if overlap < 256:
        return 0.0, 0.0, delay
    if delay >= 0:
        ra, sa = ref[delay:delay + overlap], surv[:overlap]
    else:
        ra, sa = ref[:overlap], surv[-delay:-delay + overlap]
    xc = np.vdot(ra, sa)
    rp = np.sqrt(np.vdot(ra, ra).real)
    sp = np.sqrt(np.vdot(sa, sa).real)
    if rp > 0 and sp > 0:
        return float(np.angle(xc)), float(np.abs(xc) / (rp * sp)), delay
    return 0.0, 0.0, delay


def measure_all(tb, n_surv=4):
    """Enable noise, measure all channels, disable noise. Returns list of (phase, corr, delay)."""
    with contextlib.redirect_stdout(io.StringIO()):
        tb.src.set_noise_source(True)
    time.sleep(1.5)

    results = [None] * n_surv
    for attempt in range(3):
        ref = tb.read_vector(0)
        for ch in range(n_surv):
            if results[ch] is not None:
                continue
            surv = tb.read_vector(ch + 1)
            phase, corr, delay = xcorr_phase(ref, surv)
            if corr >= 0.3:
                results[ch] = (phase, corr, delay)
        if all(r is not None for r in results):
            break
        time.sleep(0.3)

    with contextlib.redirect_stdout(io.StringIO()):
        tb.src.set_noise_source(False)
    return results


def main():
    """Run progressive recalibration interval sweep with linear drift compensation.

    Technique: measures phase at increasing intervals, applying drift rate correction,
    and stops when compensated deviation exceeds threshold.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=float, default=103.7e6)
    parser.add_argument('--gain', type=float, default=49.6)
    parser.add_argument('--max-interval', type=int, default=120,
                        help='Max recal interval to test (default 120s)')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Max allowed deviation in degrees (default 5)')
    args = parser.parse_args()

    N_SURV = 4
    THRESHOLD = args.threshold

    print("=" * 75)
    print("  Progressive Recalibration Interval Sweep (all 4 surv channels)")
    print("=" * 75)
    print(f"  Threshold: {THRESHOLD}°  |  Max interval: {args.max_interval}s")
    print()

    tb = sweep_flowgraph(freq=args.freq, gain=args.gain)
    tb.start()
    print("Settling 3s ...")
    time.sleep(3.0)

    # --- Initial calibration: two measurements 5s apart to establish drift ---
    print("\n--- Establishing drift rates (2 measurements, 5s apart) ---")
    m1 = measure_all(tb, N_SURV)
    t1 = time.time()
    for ch in range(N_SURV):
        if m1[ch]:
            print(f"  ch{ch+1}: phase={np.degrees(m1[ch][0]):+.1f}°  corr={m1[ch][1]:.3f}")
        else:
            print(f"  ch{ch+1}: FAILED")

    time.sleep(5.0)
    m2 = measure_all(tb, N_SURV)
    t2 = time.time()
    dt = t2 - t1

    # Compute per-channel drift rates
    drift_rates = [0.0] * N_SURV  # rad/s
    ref_phases = [0.0] * N_SURV   # rad
    for ch in range(N_SURV):
        if m1[ch] and m2[ch]:
            p1, p2 = m1[ch][0], m2[ch][0]
            # Unwrap
            dp = p2 - p1
            if dp > np.pi:
                dp -= 2 * np.pi
            elif dp < -np.pi:
                dp += 2 * np.pi
            drift_rates[ch] = dp / dt
            ref_phases[ch] = p2
            print(f"  ch{ch+1}: drift = {np.degrees(drift_rates[ch]):+.3f} °/s")
        else:
            print(f"  ch{ch+1}: could not establish drift")

    # --- Sweep intervals ---
    print(f"\n--- Interval sweep (10s steps, stop at >{THRESHOLD}° deviation) ---")
    header = (f"{'Intv':>5s}  " +
              "  ".join(f"{'ch'+str(ch+1)+' raw':>9s} {'comp':>7s}" for ch in range(N_SURV)) +
              f"  {'worst':>7s}")
    print(header)
    print("-" * len(header))

    t_ref = time.time()
    sweep_results = []
    max_ok_interval = 0

    for interval in range(10, args.max_interval + 1, 10):
        target = t_ref + interval
        now = time.time()
        if now < target:
            time.sleep(target - now)

        m = measure_all(tb, N_SURV)
        elapsed = time.time() - t_ref

        parts = [f"{interval:4d}s"]
        worst_comp = 0.0

        for ch in range(N_SURV):
            if m[ch] is None:
                parts.append(f"  {'FAIL':>9s} {'FAIL':>7s}")
                worst_comp = 999.0
                continue

            phase = m[ch][0]
            raw_dev = phase - ref_phases[ch]
            # Unwrap
            raw_dev_deg = np.degrees(raw_dev)
            if raw_dev_deg > 180:
                raw_dev_deg -= 360
            elif raw_dev_deg < -180:
                raw_dev_deg += 360

            # Compensated: subtract predicted drift
            predicted = drift_rates[ch] * elapsed
            comp_dev = phase - ref_phases[ch] - predicted
            comp_deg = np.degrees(comp_dev)
            if comp_deg > 180:
                comp_deg -= 360
            elif comp_deg < -180:
                comp_deg += 360

            parts.append(f"  {raw_dev_deg:+8.1f}° {comp_deg:+6.1f}°")
            worst_comp = max(worst_comp, abs(comp_deg))

        parts.append(f"  {worst_comp:6.1f}°")
        print("  ".join(parts))
        sweep_results.append((interval, worst_comp))

        if worst_comp <= THRESHOLD:
            max_ok_interval = interval

        if worst_comp > THRESHOLD:
            print(f"\n  STOPPED: worst compensated deviation {worst_comp:.1f}° > {THRESHOLD}° at {interval}s")
            break

    # Summary
    print(f"\n{'=' * 75}")
    print(f"  RESULT: Max recal interval with <{THRESHOLD}° deviation: {max_ok_interval}s")
    print(f"{'=' * 75}")

    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
