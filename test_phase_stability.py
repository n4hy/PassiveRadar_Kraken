#!/usr/bin/env python3
"""
Phase stability test: measure how fast the inter-channel phase drifts
after a single calibration snapshot.

Procedure:
  1. Start flowgraph, let pipeline settle
  2. Enable noise source, wait for flush, take reference measurement
  3. Disable noise source
  4. At 10, 20, 30, 40, 50, 60 seconds: re-enable noise, measure phase,
     disable noise, report deviation from reference
  5. Print summary table

This tells us the maximum usable recalibration interval.
"""

import sys
import time
import numpy as np
from gnuradio import gr, blocks, filter
from gnuradio.filter import firdes, window
from gnuradio.kraken_passive_radar import krakensdr_source


class stability_flowgraph(gr.top_block):
    def __init__(self, freq=103.7e6, gain=49.6):
        gr.top_block.__init__(self, "Phase Stability Test")
        samp_rate = 2_400_000
        signal_bw = 250_000
        decimation = int(samp_rate / signal_bw)
        lpf_taps = firdes.low_pass(
            1.0, samp_rate, signal_bw / 2, signal_bw / 10, window.WIN_HAMMING)

        self.src = krakensdr_source(
            frequency=freq, sample_rate=samp_rate, gain=gain)

        self.vec_size = 2**16
        self.decimated_rate = int(samp_rate / decimation)

        # Only need ref (ch0) and surv (ch1) for phase measurement
        # Plus null sinks for unused channels
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
        return np.array(self._probes[ch].level(), dtype=np.complex64)


def measure_phase(tb, ref_ch=0, surv_ch=1):
    """
    Enable noise source, wait for pipeline flush, measure phase offset
    between ref and surv channels, disable noise source.

    Returns (phase_deg, correlation, delay_samples).
    """
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        tb.src.set_noise_source(True)
    time.sleep(1.5)  # Wait for noise data to flush through pipeline

    ref = tb.read_vector(ref_ch)
    surv = tb.read_vector(surv_ch)

    N = len(ref)
    Xr = np.fft.fft(ref)
    Xs = np.fft.fft(surv)
    xc_full = np.fft.ifft(Xr * np.conj(Xs))
    peak = int(np.argmax(np.abs(xc_full)))
    delay = peak if peak <= N // 2 else peak - N

    overlap = N - abs(delay)
    if delay >= 0:
        ra = ref[delay:delay + overlap]
        sa = surv[:overlap]
    else:
        ra = ref[:overlap]
        sa = surv[-delay:-delay + overlap]

    xc = np.vdot(ra, sa)
    rp = np.sqrt(np.vdot(ra, ra).real)
    sp = np.sqrt(np.vdot(sa, sa).real)
    corr = float(np.abs(xc) / (rp * sp)) if rp > 0 and sp > 0 else 0.0
    phase = float(np.degrees(np.angle(xc)))

    with contextlib.redirect_stdout(io.StringIO()):
        tb.src.set_noise_source(False)
    return phase, corr, delay


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase stability test")
    parser.add_argument('--freq', type=float, default=103.7e6)
    parser.add_argument('--gain', type=float, default=49.6)
    parser.add_argument('--intervals', type=str, default='10,20,30,40,50,60',
                        help='Comma-separated wait intervals in seconds')
    args = parser.parse_args()

    intervals = [int(x) for x in args.intervals.split(',')]

    print("=" * 65)
    print("  Phase Stability Test — Drift vs. Recalibration Interval")
    print("=" * 65)
    print(f"  Freq: {args.freq / 1e6:.3f} MHz  |  Gain: {args.gain} dB")
    print(f"  Test intervals: {intervals} seconds")
    print()

    tb = stability_flowgraph(freq=args.freq, gain=args.gain)
    tb.start()
    print("Flowgraph started. Settling 3s ...")
    time.sleep(3.0)

    # Reference measurement
    print("\n--- Reference calibration ---")
    ref_phase, ref_corr, ref_delay = measure_phase(tb)
    print(f"  Phase: {ref_phase:+.2f} deg  |  Corr: {ref_corr:.4f}  |  Delay: {ref_delay:+d} samp")
    t_ref = time.time()

    # Measurements at each interval
    results = []
    print(f"\n--- Drift measurements (noise OFF between measurements) ---")
    print(f"{'Interval':>10s}  {'Phase':>10s}  {'Deviation':>10s}  {'Corr':>8s}  {'Delay':>8s}  {'Rate':>12s}")
    print("-" * 65)

    for interval in intervals:
        # Wait until the target time
        target = t_ref + interval
        now = time.time()
        if now < target:
            remaining = target - now
            print(f"  Waiting {remaining:.0f}s ...", end='\r')
            time.sleep(remaining)

        # Measure
        phase, corr, delay = measure_phase(tb)
        elapsed = time.time() - t_ref
        # Unwrap deviation relative to reference
        dev = phase - ref_phase
        if dev > 180:
            dev -= 360
        elif dev < -180:
            dev += 360
        rate = dev / elapsed if elapsed > 0 else 0

        results.append((interval, phase, dev, corr, delay, rate, elapsed))
        print(f"{interval:>8d}s  {phase:>+9.2f}°  {dev:>+9.2f}°  {corr:>8.4f}  {delay:>+7d}  {rate:>+.4f} °/s")

    # Fit linear and parabolic models to all measurements
    all_times = np.array([0.0] + [r[6] for r in results])
    all_phases = np.array([ref_phase] + [r[1] for r in results])
    all_phases_unwrapped = np.degrees(np.unwrap(np.radians(all_phases)))

    lin_coeffs = np.polyfit(all_times, all_phases_unwrapped, 1)   # [b, c]
    par_coeffs = np.polyfit(all_times, all_phases_unwrapped, 2)   # [a, b, c]

    print(f"\n{'=' * 75}")
    print(f"  SUMMARY")
    print(f"{'=' * 75}")
    print(f"  Reference phase:  {ref_phase:+.2f}° (corr={ref_corr:.4f})")
    print(f"  Linear fit:       phase(t) = {lin_coeffs[0]:+.4f}·t {lin_coeffs[1]:+.2f}")
    print(f"  Parabolic fit:    phase(t) = {par_coeffs[0]:+.6f}·t² {par_coeffs[1]:+.4f}·t {par_coeffs[2]:+.2f}")
    print()
    print(f"  {'Interval':>8s}  {'Raw dev':>9s}  {'Linear comp':>11s}  {'Parabol comp':>12s}  {'Corr':>6s}")
    print(f"  {'-'*55}")

    max_ok_raw = 0
    max_ok_lin = 0
    max_ok_par = 0
    for r in results:
        interval, phase, raw_dev, corr, delay, rate, elapsed = r
        # Unwrap this measurement relative to the reference
        phase_uw = np.degrees(np.unwrap(np.radians([ref_phase, phase])))[1]

        pred_lin = np.polyval(lin_coeffs, elapsed)
        pred_par = np.polyval(par_coeffs, elapsed)
        resid_lin = (phase_uw - pred_lin + 180) % 360 - 180
        resid_par = (phase_uw - pred_par + 180) % 360 - 180

        if abs(raw_dev) < 5.0:
            max_ok_raw = interval
        if abs(resid_lin) < 5.0:
            max_ok_lin = interval
        if abs(resid_par) < 5.0:
            max_ok_par = interval

        print(f"  {interval:6d}s  {raw_dev:+8.2f}°  {resid_lin:+10.2f}°  {resid_par:+11.2f}°  {corr:6.4f}")

    print()
    print(f"  Max interval <5° (no compensation):  "
          f"{'none' if max_ok_raw == 0 else f'{max_ok_raw}s'}")
    print(f"  Max interval <5° (linear):           "
          f"{'none' if max_ok_lin == 0 else f'{max_ok_lin}s'}")
    print(f"  Max interval <5° (parabolic):        "
          f"{'none' if max_ok_par == 0 else f'{max_ok_par}s'}")
    print()
    print(f"  Drift coefficients for PhaseCorrectorBlock:")
    print(f"    Linear rate:      {lin_coeffs[0]:+.4f} °/s")
    print(f"    Parabolic a:      {par_coeffs[0]:+.6f} °/s²")
    print(f"    Parabolic b:      {par_coeffs[1]:+.4f} °/s")

    tb.stop()
    tb.wait()
    print("\nDone.")


if __name__ == '__main__':
    main()
