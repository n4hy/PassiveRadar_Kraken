#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: KrakenSDR Passive Radar — Range-Doppler
# Author: N4HY
# Description: Passive Bistatic Radar with periodic phase calibration and drift tracking
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
import numpy as np
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import kraken_passive_radar
from gnuradio.kraken_passive_radar import krakensdr_source
import sip
import threading
import time

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class phase_corrector(gr.sync_block):
    """
    Applies phase correction to surveillance channel.
    Correction phasor updated by calibration routine.
    """
    def __init__(self):
        gr.sync_block.__init__(
            self, name="Phase Corrector",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.correction = np.complex64(1.0 + 0j)  # No correction initially

    def set_correction(self, phasor):
        self.correction = np.complex64(phasor)

    def work(self, input_items, output_items):
        output_items[0][:] = input_items[0] * self.correction
        return len(output_items[0])


class kraken_pbr_flowgraph(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "KrakenSDR Passive Radar — Range-Doppler", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("KrakenSDR Passive Radar — Range-Doppler")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "kraken_pbr_flowgraph")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.signal_bw = signal_bw = 250000
        self.samp_rate = samp_rate = 2400000
        self.decimation = decimation = int(samp_rate / signal_bw)
        self.rf_gain = rf_gain = 49.6
        self.lpf_taps = lpf_taps = firdes.low_pass(1.0, samp_rate, signal_bw/2, signal_bw/10, window.WIN_HAMMING)
        self.center_freq = center_freq = 103.7e6
        self.decimated_rate = decimated_rate = int(samp_rate / decimation)

        # Cross-correlation / range parameters
        self.cpi_samples = cpi_samples = 2048
        self.fft_size = fft_size = 2048

        # ECA parameters
        self.eca_taps = eca_taps = 256
        self.eca_reg = eca_reg = 0.0001

        # Doppler parameters
        self.num_doppler_bins = num_doppler_bins = 64
        self.rd_size = rd_size = num_doppler_bins * fft_size  # 131072

        # Phase calibration state (thread-safe via cal_lock)
        self.phase_history = []       # list of (elapsed_sec, phase_deg)
        self.drift_rate = 0.0         # deg/sec from linear fit
        self.start_time = time.time()
        self.cal_lock = threading.Lock()
        self._cal_stop = threading.Event()
        self._cal_wake = threading.Event()  # wakes cal thread when interval changes
        self._cal_status = "Waiting for startup..."
        self._cal_status_color = "orange"
        self._phase_plot_dirty = False  # flag for timer to redraw phase plot
        self.recal_interval = 10.0  # seconds between recalibrations

        ##################################################
        # Blocks — 2-channel with Doppler (ch0=ref, ch1=surv)
        ##################################################

        # Phase corrector on surveillance channel (applied before ECA)
        self.phase_corr = phase_corrector()

        # Probes for calibration (grab raw samples after LPF/decimate, before phase_corr)
        self.cal_probe_ref = blocks.probe_signal_c()
        self.cal_probe_surv = blocks.probe_signal_c()

        # Stream-to-vector for cross-correlation input
        self.s2v_surv0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, cpi_samples)
        self.s2v_ref = blocks.stream_to_vector(gr.sizeof_gr_complex*1, cpi_samples)

        # FFTs for cross-correlation (forward)
        self.fft_surv0 = fft.fft_vcc(fft_size, True, [], False, 1)
        self.fft_ref = fft.fft_vcc(fft_size, True, [], False, 1)

        # Conjugate reference in frequency domain
        self.v2s_ref_for_conj = blocks.vector_to_stream(gr.sizeof_gr_complex*1, fft_size)
        self.conjugate_ref = blocks.conjugate_cc()
        self.s2v_ref_conj = blocks.stream_to_vector(gr.sizeof_gr_complex*1, fft_size)

        # Cross-multiply and IFFT
        self.multiply_surv0 = blocks.multiply_vcc(fft_size)
        self.ifft_surv0 = fft.fft_vcc(fft_size, False, [], False, 1)

        # Doppler processor: accumulates CPIs, FFT across slow-time
        self.doppler_proc = kraken_passive_radar.doppler_processor.make(
            fft_size, num_doppler_bins,
            1,     # window_type: 1=Hamming
            True   # output_power: float |X|^2
        )

        # dB conversion for Range-Doppler map
        self.nlog10_rd = blocks.nlog10_ff(10, rd_size, 0)

        # Probe to grab latest RD map vector for matplotlib display
        self.rd_probe = blocks.probe_signal_vf(rd_size)

        # ---- Row 0: Matplotlib Range-Doppler 2D image ----
        self.rd_fig = Figure(figsize=(8, 4), dpi=100)
        self.rd_ax = self.rd_fig.add_subplot(111)
        cpi_dur = cpi_samples / decimated_rate
        doppler_max = 1.0 / (2.0 * cpi_dur)
        self.rd_display_bins = 200
        range_km_per_bin = 3e8 / decimated_rate / 1000.0
        range_max_km = self.rd_display_bins * range_km_per_bin
        self.rd_img = self.rd_ax.imshow(
            np.zeros((num_doppler_bins, self.rd_display_bins)),
            aspect='auto', origin='lower',
            cmap='inferno',
            extent=[0, range_max_km, -doppler_max, doppler_max],
            vmin=-10, vmax=50,
            interpolation='nearest'
        )
        self.rd_ax.set_xlabel('Bistatic Range (km)')
        self.rd_ax.set_ylabel('Doppler (Hz)')
        self.rd_ax.set_title('Range-Doppler Map (dB) — Calibrating...')
        self.rd_fig.colorbar(self.rd_img, ax=self.rd_ax, label='dB', shrink=0.8)
        self.rd_fig.tight_layout()
        self.rd_canvas = FigureCanvas(self.rd_fig)
        self.top_grid_layout.addWidget(self.rd_canvas, 0, 0, 1, 2)

        # ---- Row 1: Frequency sinks ----
        # Freq sink on ch0 (reference)
        self.qtgui_freq_sink_ref = qtgui.freq_sink_c(
            2048, window.WIN_BLACKMAN_hARRIS, 0, samp_rate,
            "Reference (ch0)", 1, None
        )
        self.qtgui_freq_sink_ref.set_update_time(0.10)
        self.qtgui_freq_sink_ref.set_y_axis((-140), 10)
        self.qtgui_freq_sink_ref.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_ref.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_ref.enable_autoscale(True)
        self.qtgui_freq_sink_ref.enable_grid(True)
        self.qtgui_freq_sink_ref.set_fft_average(0.2)
        self.qtgui_freq_sink_ref.enable_axis_labels(True)
        self.qtgui_freq_sink_ref.enable_control_panel(False)
        self.qtgui_freq_sink_ref.set_fft_window_normalized(False)
        self.qtgui_freq_sink_ref.set_line_label(0, 'Ref')
        self.qtgui_freq_sink_ref.set_line_color(0, "blue")
        self._qtgui_freq_sink_ref_win = sip.wrapinstance(self.qtgui_freq_sink_ref.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_ref_win, 1, 0, 1, 1)

        # Freq sink on ch1 (surveillance)
        self.qtgui_freq_sink_surv = qtgui.freq_sink_c(
            2048, window.WIN_BLACKMAN_hARRIS, 0, samp_rate,
            "Surveillance (ch1)", 1, None
        )
        self.qtgui_freq_sink_surv.set_update_time(0.10)
        self.qtgui_freq_sink_surv.set_y_axis((-140), 10)
        self.qtgui_freq_sink_surv.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_surv.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_surv.enable_autoscale(True)
        self.qtgui_freq_sink_surv.enable_grid(True)
        self.qtgui_freq_sink_surv.set_fft_average(0.2)
        self.qtgui_freq_sink_surv.enable_axis_labels(True)
        self.qtgui_freq_sink_surv.enable_control_panel(False)
        self.qtgui_freq_sink_surv.set_fft_window_normalized(False)
        self.qtgui_freq_sink_surv.set_line_label(0, 'Surv')
        self.qtgui_freq_sink_surv.set_line_color(0, "red")
        self._qtgui_freq_sink_surv_win = sip.wrapinstance(self.qtgui_freq_sink_surv.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_surv_win, 1, 1, 1, 1)

        # ---- Row 2: Phase offset history plot ----
        self.phase_fig = Figure(figsize=(8, 2.5), dpi=100)
        self.phase_ax = self.phase_fig.add_subplot(111)
        self.phase_line, = self.phase_ax.plot([], [], 'o-', color='cyan',
                                               markersize=5, linewidth=1.5,
                                               label='Measured')
        self.phase_trend, = self.phase_ax.plot([], [], '--', color='red',
                                                linewidth=1.5, alpha=0.7,
                                                label='Linear fit')
        self.phase_ax.set_xlabel('Time (s)')
        self.phase_ax.set_ylabel('Phase Offset (deg)')
        self.phase_ax.set_title('Inter-channel Phase Drift — Waiting...')
        self.phase_ax.grid(True, alpha=0.3)
        self.phase_ax.set_ylim(-180, 180)
        self.phase_ax.set_xlim(0, 30)
        self.phase_ax.legend(loc='upper left', fontsize=8)
        self.phase_fig.tight_layout()
        self.phase_canvas = FigureCanvas(self.phase_fig)
        self.top_grid_layout.addWidget(self.phase_canvas, 2, 0, 1, 2)

        # ---- Row 3: Calibration status label ----
        self.cal_label = Qt.QLabel("Phase Cal: Waiting for startup...")
        self.cal_label.setStyleSheet("font-weight: bold; font-size: 14px; color: orange;")
        self.top_grid_layout.addWidget(self.cal_label, 3, 0, 1, 2)

        # ---- Row 4: Recalibration interval control ----
        recal_hbox = Qt.QHBoxLayout()
        recal_lbl = Qt.QLabel("Recalibration interval (s):")
        recal_lbl.setStyleSheet("font-size: 13px;")
        recal_hbox.addWidget(recal_lbl)
        self.recal_spinbox = Qt.QSpinBox()
        self.recal_spinbox.setRange(1, 600)
        self.recal_spinbox.setValue(int(self.recal_interval))
        self.recal_spinbox.setSuffix(" s")
        recal_hbox.addWidget(self.recal_spinbox)
        self.recal_btn = Qt.QPushButton("Execute")
        self.recal_btn.clicked.connect(self._apply_recal_interval)
        recal_hbox.addWidget(self.recal_btn)
        recal_hbox.addStretch()
        self.top_grid_layout.addLayout(recal_hbox, 4, 0, 1, 2)

        # Row stretches
        self.top_grid_layout.setRowStretch(0, 3)  # RD map largest
        self.top_grid_layout.setRowStretch(1, 2)  # Freq sinks
        self.top_grid_layout.setRowStretch(2, 2)  # Phase plot
        self.top_grid_layout.setRowStretch(3, 0)  # Label
        self.top_grid_layout.setRowStretch(4, 0)  # Recal controls
        for c in range(2):
            self.top_grid_layout.setColumnStretch(c, 1)

        # Timer to update displays from probes (~3 Hz)
        self._rd_timer = Qt.QTimer()
        self._rd_timer.timeout.connect(self._update_displays)
        self._rd_timer.start(400)

        # Signal chain blocks
        self.kraken_src = krakensdr_source(frequency=center_freq, sample_rate=samp_rate, gain=rf_gain)
        self.dc_blocker_ref = filter.dc_blocker_cc(32, True)
        self.dc_blocker_surv0 = filter.dc_blocker_cc(32, True)
        self.freq_xlating_fir_ref = filter.freq_xlating_fir_filter_ccc(decimation, lpf_taps, 0, samp_rate)
        self.freq_xlating_fir_surv0 = filter.freq_xlating_fir_filter_ccc(decimation, lpf_taps, 0, samp_rate)
        self.agc_ref = analog.agc2_cc(0.01, 0.001, 1.0, 1.0, 65536)
        self.agc_surv0 = analog.agc2_cc(0.01, 0.001, 1.0, 1.0, 65536)
        self.eca_canceller = kraken_passive_radar.eca_canceller(eca_taps, eca_reg, 1)

        # Null sinks for unused channels 2-4
        self.null_sink_ch2 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.null_sink_ch3 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.null_sink_ch4 = blocks.null_sink(gr.sizeof_gr_complex*1)

        ##################################################
        # Connections
        ##################################################
        # KrakenSDR source -> DC blockers (ch0 ref, ch1 surv; ch2-4 to null)
        self.connect((self.kraken_src, 0), (self.dc_blocker_ref, 0))
        self.connect((self.kraken_src, 0), (self.qtgui_freq_sink_ref, 0))
        self.connect((self.kraken_src, 1), (self.dc_blocker_surv0, 0))
        self.connect((self.kraken_src, 1), (self.qtgui_freq_sink_surv, 0))
        self.connect((self.kraken_src, 2), (self.null_sink_ch2, 0))
        self.connect((self.kraken_src, 3), (self.null_sink_ch3, 0))
        self.connect((self.kraken_src, 4), (self.null_sink_ch4, 0))

        # Conditioning: DC block -> LPF/decimate -> Phase correction (surv only) -> AGC
        self.connect((self.dc_blocker_ref, 0), (self.freq_xlating_fir_ref, 0))
        self.connect((self.dc_blocker_surv0, 0), (self.freq_xlating_fir_surv0, 0))
        self.connect((self.freq_xlating_fir_ref, 0), (self.agc_ref, 0))
        self.connect((self.freq_xlating_fir_ref, 0), (self.cal_probe_ref, 0))
        self.connect((self.freq_xlating_fir_surv0, 0), (self.phase_corr, 0))
        self.connect((self.freq_xlating_fir_surv0, 0), (self.cal_probe_surv, 0))
        self.connect((self.phase_corr, 0), (self.agc_surv0, 0))

        # ECA clutter cancellation (1 ref + 1 surv -> 1 cleaned surv)
        self.connect((self.agc_ref, 0), (self.eca_canceller, 0))
        self.connect((self.agc_ref, 0), (self.s2v_ref, 0))
        self.connect((self.agc_surv0, 0), (self.eca_canceller, 1))
        self.connect((self.eca_canceller, 0), (self.s2v_surv0, 0))

        # Stream -> Vector -> FFT (ref and surv)
        self.connect((self.s2v_ref, 0), (self.fft_ref, 0))
        self.connect((self.s2v_surv0, 0), (self.fft_surv0, 0))

        # Cross-correlation: FFT(surv) * conj(FFT(ref)) -> IFFT
        self.connect((self.fft_ref, 0), (self.v2s_ref_for_conj, 0))
        self.connect((self.v2s_ref_for_conj, 0), (self.conjugate_ref, 0))
        self.connect((self.conjugate_ref, 0), (self.s2v_ref_conj, 0))
        self.connect((self.fft_surv0, 0), (self.multiply_surv0, 0))
        self.connect((self.s2v_ref_conj, 0), (self.multiply_surv0, 1))
        self.connect((self.multiply_surv0, 0), (self.ifft_surv0, 0))

        # IFFT -> Doppler processor -> dB -> probe for matplotlib display
        self.connect((self.ifft_surv0, 0), (self.doppler_proc, 0))
        self.connect((self.doppler_proc, 0), (self.nlog10_rd, 0))
        self.connect((self.nlog10_rd, 0), (self.rd_probe, 0))

    def run_calibration(self):
        """
        Run one phase calibration using KrakenSDR noise source.
        Thread-safe: only updates shared data under cal_lock.
        UI updates happen in the timer callback.
        """
        print("=== Phase Calibration Starting ===")
        with self.cal_lock:
            self._cal_status = "Noise source ON -- calibrating..."
            self._cal_status_color = "red"

        # Step 1: Enable noise source (hardware disconnects antennas)
        self.kraken_src.set_noise_source(True)
        print("Noise source ENABLED")

        # Step 2: Wait for switch settling + data pipeline to flush
        time.sleep(0.2)

        # Step 3: Grab samples from probes (after LPF/decimation)
        N = 2048
        ref_samples = np.zeros(N, dtype=np.complex64)
        surv_samples = np.zeros(N, dtype=np.complex64)

        for i in range(N):
            ref_samples[i] = self.cal_probe_ref.level()
            surv_samples[i] = self.cal_probe_surv.level()
            time.sleep(1.0 / self.decimated_rate)

        # Step 4: Compute phase offset via cross-correlation
        xcorr = np.vdot(ref_samples, surv_samples)  # conjugate(ref) . surv
        ref_power = np.sqrt(np.vdot(ref_samples, ref_samples).real)
        surv_power = np.sqrt(np.vdot(surv_samples, surv_samples).real)

        if ref_power > 0 and surv_power > 0:
            correlation = np.abs(xcorr) / (ref_power * surv_power)
            phase_offset = np.angle(xcorr)
            phase_deg = np.degrees(phase_offset)
            correction_phasor = np.exp(-1j * phase_offset)
            elapsed = time.time() - self.start_time

            print(f"Phase offset: {phase_deg:.1f} deg  (correlation: {correlation:.4f})")

            # Step 5: Apply correction
            self.phase_corr.set_correction(correction_phasor)

            # Step 6: Record measurement and compute drift rate
            with self.cal_lock:
                self.phase_history.append((elapsed, phase_deg))

                if len(self.phase_history) >= 2:
                    times = np.array([p[0] for p in self.phase_history])
                    phases_deg = np.array([p[1] for p in self.phase_history])
                    # Unwrap to handle +-180 wrapping
                    phases_unwrapped = np.degrees(np.unwrap(np.radians(phases_deg)))
                    # Linear fit: phase(t) = drift_rate * t + offset
                    coeffs = np.polyfit(times, phases_unwrapped, 1)
                    self.drift_rate = coeffs[0]  # deg/sec
                    drift_str = f"{self.drift_rate:+.3f} deg/s"
                else:
                    drift_str = "measuring..."

                self._cal_status = (f"offset={phase_deg:+.1f} deg, "
                                    f"corr={correlation:.3f}, "
                                    f"drift={drift_str}")
                self._cal_status_color = "green"
                self._phase_plot_dirty = True
        else:
            print("WARNING: Zero power during calibration!")
            with self.cal_lock:
                self._cal_status = "FAILED (zero power)"
                self._cal_status_color = "red"

        # Step 7: Disable noise source (hardware reconnects antennas)
        self.kraken_src.set_noise_source(False)
        print(f"Noise source DISABLED  |  {self._cal_status}")
        print("=== Phase Calibration Complete ===")

    def _apply_recal_interval(self):
        """Slot for Execute button: update recalibration interval from spinbox."""
        new_val = self.recal_spinbox.value()
        with self.cal_lock:
            self.recal_interval = float(new_val)
        # Wake the calibration thread so it picks up the new interval immediately
        self._cal_wake.set()
        print(f"Recalibration interval set to {new_val}s")

    def _periodic_calibration(self):
        """
        Background thread: runs calibration at fixed interval (recal_interval seconds).
        """
        time.sleep(1.0)  # Let pipeline settle after flowgraph start

        # --- Initial calibration ---
        self.run_calibration()

        # --- Continuous loop at fixed interval ---
        while not self._cal_stop.is_set():
            with self.cal_lock:
                interval = self.recal_interval

            print(f"Next calibration in {interval:.0f}s")

            # Wait for interval, but wake early if interval changed or stop requested
            self._cal_wake.clear()
            self._cal_wake.wait(timeout=interval)
            if self._cal_stop.is_set():
                return
            self.run_calibration()

    def _update_displays(self):
        """Timer callback: update RD map, phase plot, and cal label from main thread."""
        # --- Update Range-Doppler map ---
        try:
            rd_data = self.rd_probe.level()
            if len(rd_data) == self.rd_size:
                rd_map = np.array(rd_data).reshape(self.num_doppler_bins, self.fft_size)
                rd_map = np.clip(rd_map, -50, 100)
                rd_zoomed = rd_map[:, :self.rd_display_bins]
                self.rd_img.set_data(rd_zoomed)
                noise_floor = np.median(rd_zoomed)
                peak = np.max(rd_zoomed)
                vmin = noise_floor - 3
                vmax = min(peak, noise_floor + 40)
                self.rd_img.set_clim(vmin, vmax)
                self.rd_canvas.draw_idle()
        except Exception:
            pass

        # --- Update calibration status label (thread-safe read) ---
        with self.cal_lock:
            status = self._cal_status
            color = self._cal_status_color
            dirty = self._phase_plot_dirty
            self._phase_plot_dirty = False

        self.cal_label.setText(f"Phase Cal: {status}")
        self.cal_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")

        # --- Update phase offset plot (only when new data) ---
        if dirty:
            with self.cal_lock:
                if not self.phase_history:
                    return
                times = [p[0] for p in self.phase_history]
                phases_deg = [p[1] for p in self.phase_history]
                # Unwrap for display
                phases_unwrapped = np.degrees(np.unwrap(np.radians(phases_deg)))
                n_pts = len(self.phase_history)
                dr = self.drift_rate

            # Plot measured points (unwrapped)
            self.phase_line.set_data(times, phases_unwrapped)

            # Plot linear trend line
            if n_pts >= 2:
                t_arr = np.array(times)
                coeffs = np.polyfit(t_arr, phases_unwrapped, 1)
                # Extend trend 20% into the future
                t_extend = t_arr[-1] + (t_arr[-1] - t_arr[0]) * 0.2
                t_fit = np.array([t_arr[0], t_extend])
                p_fit = np.polyval(coeffs, t_fit)
                self.phase_trend.set_data(t_fit, p_fit)
                title = f"Inter-channel Phase Drift: {dr:+.3f} deg/s"
            else:
                self.phase_trend.set_data([], [])
                title = "Inter-channel Phase Drift: measuring..."

            self.phase_ax.set_title(title)
            self.phase_ax.set_xlim(0, max(times[-1] * 1.1, 10))
            ymin = min(phases_unwrapped) - 10
            ymax = max(phases_unwrapped) + 10
            self.phase_ax.set_ylim(ymin, ymax)
            self.phase_canvas.draw_idle()

    def closeEvent(self, event):
        self._rd_timer.stop()
        self._cal_stop.set()
        self._cal_wake.set()  # unblock cal thread so it exits
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "kraken_pbr_flowgraph")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    def get_signal_bw(self):
        return self.signal_bw

    def set_signal_bw(self, signal_bw):
        self.signal_bw = signal_bw
        self.set_decimation(int(self.samp_rate / self.signal_bw))
        self.set_lpf_taps(firdes.low_pass(1.0, self.samp_rate, self.signal_bw/2, self.signal_bw/10, window.WIN_HAMMING))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_decimated_rate(int(self.samp_rate / self.decimation))
        self.set_decimation(int(self.samp_rate / self.signal_bw))
        self.set_lpf_taps(firdes.low_pass(1.0, self.samp_rate, self.signal_bw/2, self.signal_bw/10, window.WIN_HAMMING))
        self.kraken_src.set_sample_rate(self.samp_rate)
        self.qtgui_freq_sink_ref.set_frequency_range(0, self.samp_rate)
        self.qtgui_freq_sink_surv.set_frequency_range(0, self.samp_rate)

    def get_decimation(self):
        return self.decimation

    def set_decimation(self, decimation):
        self.decimation = decimation
        self.set_decimated_rate(int(self.samp_rate / self.decimation))

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self.kraken_src.set_gain(self.rf_gain)

    def get_lpf_taps(self):
        return self.lpf_taps

    def set_lpf_taps(self, lpf_taps):
        self.lpf_taps = lpf_taps
        self.freq_xlating_fir_ref.set_taps(self.lpf_taps)
        self.freq_xlating_fir_surv0.set_taps(self.lpf_taps)

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size

    def get_eca_taps(self):
        return self.eca_taps

    def set_eca_taps(self, eca_taps):
        self.eca_taps = eca_taps
        self.eca_canceller.set_num_taps(self.eca_taps)

    def get_eca_reg(self):
        return self.eca_reg

    def set_eca_reg(self, eca_reg):
        self.eca_reg = eca_reg
        self.eca_canceller.set_reg_factor(self.eca_reg)

    def get_decimated_rate(self):
        return self.decimated_rate

    def set_decimated_rate(self, decimated_rate):
        self.decimated_rate = decimated_rate

    def get_cpi_samples(self):
        return self.cpi_samples

    def set_cpi_samples(self, cpi_samples):
        self.cpi_samples = cpi_samples

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.kraken_src.set_frequency(self.center_freq)


def main(top_block_cls=kraken_pbr_flowgraph, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    # Start periodic phase calibration in background thread
    cal_thread = threading.Thread(target=tb._periodic_calibration, daemon=True)
    cal_thread.start()

    def sig_handler(sig=None, frame=None):
        tb._cal_stop.set()
        tb._cal_wake.set()
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
