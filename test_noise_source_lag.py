#!/usr/bin/env python3
"""
Test program: measure noise source lag through GNU Radio pipeline.

Turns on the KrakenSDR noise source and measures how long it takes for
the noise-source signal to propagate through the GNU Radio pipeline to
each of the 5 channels.  This measures the combined latency of:
  1. Hardware switch settling
  2. USB transfer buffers
  3. GNU Radio internal block buffers

The test uses vector probes (contiguous sample blocks) with FFT
cross-correlation to compensate for inter-channel scheduling delay.

Usage:
    python3 test_noise_source_lag.py [--freq 103.7e6] [--gain 49.6] [--timeout 15]
"""

import sys
import time
import argparse
import numpy as np
from gnuradio import gr, blocks, filter
from gnuradio.filter import firdes, window
from gnuradio.kraken_passive_radar import krakensdr_source


class noise_lag_flowgraph(gr.top_block):
    """Minimal 5-channel flowgraph with vector probes for lag measurement."""

    def __init__(self, freq=103.7e6, gain=49.6):
        """Initialize 5-channel flowgraph with vector probes for lag measurement.

        Technique: DC block, LPF/decimate, stream-to-vector, probe per channel.
        """
        gr.top_block.__init__(self, "Noise Source Lag Test")

        samp_rate = 2_400_000
        signal_bw = 250_000
        decimation = int(samp_rate / signal_bw)
        lpf_taps = firdes.low_pass(
            1.0, samp_rate, signal_bw / 2, signal_bw / 10, window.WIN_HAMMING)

        self.src = krakensdr_source(
            frequency=freq, sample_rate=samp_rate, gain=gain)

        # Per-channel: DC block -> LPF/decimate -> stream-to-vector -> vector probe
        # Large vectors needed: GR scheduler can offset channels by 10-100ms
        # on RPi5 (20+ threads). At 266 kHz, 100ms = 26667 samples.
        self.vec_size = 2**16  # 65536 samples = 245ms at 266 kHz
        self.decimated_rate = int(samp_rate / decimation)
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
        """Return the latest contiguous sample vector from *ch*."""
        return np.array(self._probes[ch].level(), dtype=np.complex64)


def correlate(ref, surv):
    """
    Cross-correlate *ref* and *surv* vectors, compensating for any
    inter-channel sample delay via FFT cross-correlation.

    Uses the FULL vector length so we can find delays up to N/2 samples.
    With N=65536 at 266 kHz, that's delays up to ±123ms — enough to
    cover GNU Radio scheduler jitter on RPi5.

    Returns (correlation, phase_rad, delay_samples).
    """
    N = len(ref)

    # Full-length FFT cross-correlation to find delay
    Xr = np.fft.fft(ref)
    Xs = np.fft.fft(surv)
    xc_full = np.fft.ifft(Xr * np.conj(Xs))
    peak = int(np.argmax(np.abs(xc_full)))
    delay = peak if peak <= N // 2 else peak - N

    # Align vectors using the detected delay.
    # Peak at k=delay means ref[n] ~ surv[n - delay]:
    #   delay > 0 → ref leads surv → ref[delay:] aligns with surv[:N-delay]
    #   delay < 0 → surv leads ref → surv[|delay|:] aligns with ref[:N-|delay|]
    overlap = N - abs(delay)
    if overlap < 256:
        return 0.0, 0.0, delay

    if delay >= 0:
        ra = ref[delay:delay + overlap]
        sa = surv[:overlap]
    else:
        ra = ref[:overlap]
        sa = surv[-delay:-delay + overlap]

    xc = np.vdot(ra, sa)
    rp = np.sqrt(np.vdot(ra, ra).real)
    sp = np.sqrt(np.vdot(sa, sa).real)

    if rp > 0 and sp > 0:
        return float(np.abs(xc) / (rp * sp)), float(np.angle(xc)), delay
    return 0.0, 0.0, delay


def main():
    """Measure noise source signal propagation latency through the GNU Radio pipeline.

    Technique: monitors cross-correlation between channels after noise source toggle
    to determine per-channel arrival time.
    """
    parser = argparse.ArgumentParser(
        description="Measure noise-source lag through GNU Radio pipeline")
    parser.add_argument('--freq', type=float, default=103.7e6,
                        help='Center frequency in Hz (default: 103.7 MHz)')
    parser.add_argument('--gain', type=float, default=49.6,
                        help='RF gain in dB (default: 49.6)')
    parser.add_argument('--timeout', type=float, default=15.0,
                        help='Max seconds to wait for noise data (default: 15)')
    args = parser.parse_args()

    print("=" * 60)
    print("  KrakenSDR Noise Source Pipeline Lag Test")
    print("=" * 60)
    print(f"  Freq:    {args.freq / 1e6:.3f} MHz")
    print(f"  Gain:    {args.gain} dB")
    print(f"  Timeout: {args.timeout} s")
    print()

    # ---- Start flowgraph ----
    tb = noise_lag_flowgraph(freq=args.freq, gain=args.gain)
    tb.start()
    print("Flowgraph started.  Waiting 3 s for pipeline to fill ...")
    time.sleep(3.0)

    # ---- Baseline (noise OFF) ----
    print("\n--- Baseline (noise source OFF) ---")
    vecs = [tb.read_vector(ch) for ch in range(5)]
    for ch in range(5):
        pwr = float(10 * np.log10(np.mean(np.abs(vecs[ch])**2) + 1e-20))
        print(f"  ch{ch} (SN via osmosdr dev #{ch}): power = {pwr:+.1f} dB")
    ref = vecs[0]
    for ch in range(1, 5):
        c, p, d = correlate(ref, vecs[ch])
        print(f"  ch0-ch{ch}: corr={c:.4f}  phase={np.degrees(p):+7.1f} deg  delay={d:+d} samp")

    # ---- Enable noise source ----
    print("\n--- Enabling noise source ---")
    tb.src.set_noise_source(True)
    t0 = time.time()

    THRESHOLD = 0.5
    first_arrival = {}          # ch -> seconds
    print(f"\nMonitoring power + correlations (threshold = {THRESHOLD}) ...")
    print(f"{'t(s)':>6}  {'ch0 pwr':>8}  {'ch1 pwr':>8}  "
          f"{'ch0-1 corr':>10}  {'ch0-2 corr':>10}  {'ch0-3 corr':>10}  {'ch0-4 corr':>10}")
    print("-" * 80)

    while time.time() - t0 < args.timeout:
        vecs = [tb.read_vector(ch) for ch in range(5)]
        elapsed = time.time() - t0
        p0 = float(10 * np.log10(np.mean(np.abs(vecs[0])**2) + 1e-20))
        p1 = float(10 * np.log10(np.mean(np.abs(vecs[1])**2) + 1e-20))
        parts = [f"{elapsed:6.2f}", f"{p0:+7.1f}dB", f"{p1:+7.1f}dB"]

        for ch in range(1, 5):
            c, p, d = correlate(vecs[0], vecs[ch])
            tag = ""
            if c >= THRESHOLD and ch not in first_arrival:
                first_arrival[ch] = elapsed
                tag = " <--"
            parts.append(f"{c:8.4f}{tag:4s}")

        print("  ".join(parts))

        if len(first_arrival) == 4:
            break
        time.sleep(0.2)

    # ---- Report ----
    print()
    if first_arrival:
        print("=== Noise-source arrival times ===")
        for ch in sorted(first_arrival):
            print(f"  ch0-ch{ch}: {first_arrival[ch]:.2f} s")
        max_lag = max(first_arrival.values())
        print(f"\n  Recommended minimum settle time: {max_lag + 0.5:.1f} s")
    else:
        print("WARNING: Noise source data NEVER arrived at any channel!")
        print("Possible causes:")
        print("  - Noise source GPIO not toggling (check USB control transfer errors)")
        print("  - Wrong serial number (expected SN 1000)")
        print("  - Hardware switch not connected")

    # ---- Disable noise source ----
    print("\n--- Disabling noise source ---")
    tb.src.set_noise_source(False)
    time.sleep(1.0)

    # ---- Post-test baseline ----
    print("\n--- Post-test baseline (noise source OFF) ---")
    ref = tb.read_vector(0)
    for ch in range(1, 5):
        c, p, d = correlate(ref, tb.read_vector(ch))
        print(f"  ch0-ch{ch}: corr={c:.4f}")

    tb.stop()
    tb.wait()
    print("\nDone.")


if __name__ == '__main__':
    main()
