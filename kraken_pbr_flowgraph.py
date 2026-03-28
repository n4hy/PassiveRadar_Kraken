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

import os
import sys
import glob as _glob

# --- Auto-detect display environment before importing Qt/matplotlib ---
# When running over SSH, DISPLAY and WAYLAND_DISPLAY are typically unset even
# though a desktop session is active on the host.  Probe for available display
# sockets and configure the environment so Qt5 and matplotlib can start.
def _autodetect_display():
    """Set DISPLAY / WAYLAND_DISPLAY / QT_QPA_PLATFORM if not already present."""
    have_x = bool(os.environ.get('DISPLAY'))
    have_wl = bool(os.environ.get('WAYLAND_DISPLAY'))
    if have_x or have_wl:
        return  # already configured (includes ssh -X/-Y which sets DISPLAY)

    # If we are in an SSH session, look for X11 forwarding ports (6010+)
    # before falling back to the host's local display.
    if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT'):
        import socket
        for offset in range(10, 70):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.settimeout(0.05)
                s.connect(('127.0.0.1', 6000 + offset))
                s.close()
                os.environ['DISPLAY'] = f'localhost:{offset}.0'
                return
            except OSError:
                s.close()
        print("NOTE: SSH session detected but no X11 forwarding port found.\n"
              "  Run with 'ssh -Y' to forward the display, or set DISPLAY=:0\n"
              "  to render on the Pi's local screen.", file=sys.stderr)
        return

    uid = os.getuid()
    xdg = os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{uid}')
    if not os.environ.get('XDG_RUNTIME_DIR'):
        os.environ['XDG_RUNTIME_DIR'] = xdg

    # Prefer Wayland if a compositor socket exists
    wl_socks = sorted(_glob.glob(os.path.join(xdg, 'wayland-[0-9]*')))
    wl_socks = [s for s in wl_socks if not s.endswith('.lock')]
    if wl_socks:
        os.environ.setdefault('WAYLAND_DISPLAY', os.path.basename(wl_socks[0]))
        os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')
        # XWayland usually provides :0 alongside Wayland — set DISPLAY too
        # so that matplotlib Qt5Agg (which may use X11 internally) works.
        x_socks = sorted(_glob.glob('/tmp/.X11-unix/X*'))
        if x_socks:
            display_num = os.path.basename(x_socks[0]).lstrip('X')
            os.environ.setdefault('DISPLAY', f':{display_num}')
        return

    # Fall back to X11
    x_socks = sorted(_glob.glob('/tmp/.X11-unix/X*'))
    if x_socks:
        display_num = os.path.basename(x_socks[0]).lstrip('X')
        os.environ.setdefault('DISPLAY', f':{display_num}')
        # Look for Xauthority — common locations
        for candidate in [
            os.path.expanduser('~/.Xauthority'),
            *sorted(_glob.glob(f'{xdg}/.mutter-Xwaylandauth.*')),
        ]:
            if os.path.isfile(candidate):
                os.environ.setdefault('XAUTHORITY', candidate)
                break
        return

    print("WARNING: No X11 or Wayland display found. GUI will likely fail.",
          file=sys.stderr)

_autodetect_display()

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
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import kraken_passive_radar
from gnuradio.kraken_passive_radar import krakensdr_source
from gnuradio.kraken_passive_radar import rspduo_source
import sip
import threading
import time

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class phase_corrector(gr.sync_block):
    """
    Applies time-varying phase correction to surveillance channel.

    Corrects both the static phase offset and the linear drift rate
    measured during calibration.  Each sample is multiplied by
    exp(-j*(phi0 + omega*n/fs)) where omega is the drift in rad/s
    and n counts samples since the last calibration.
    """
    def __init__(self, sample_rate=250000):
        gr.sync_block.__init__(
            self, name="Phase Corrector",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self._fs = float(sample_rate)
        self._phi0 = 0.0          # initial phase offset (radians)
        self._omega = 0.0         # drift rate (radians/sec)
        self._sample_count = 0    # samples since last calibration
        self._lock = threading.Lock()

    def set_correction(self, phase_rad, drift_rad_per_sec=0.0):
        """Update correction from calibration measurement.

        Args:
            phase_rad: measured phase offset in radians (will be negated)
            drift_rad_per_sec: measured drift rate in radians/sec (negated)
        """
        with self._lock:
            self._phi0 = -phase_rad
            self._omega = -drift_rad_per_sec
            self._sample_count = 0

    def work(self, input_items, output_items):
        n = len(input_items[0])
        with self._lock:
            phi0 = self._phi0
            omega = self._omega
            count = self._sample_count
            self._sample_count = count + n

        if omega == 0.0:
            # Static correction — single complex multiply
            output_items[0][:] = input_items[0] * np.complex64(np.exp(1j * phi0))
            return n

        # NCO: 2 exp() calls + N complex multiplies via cumprod
        start_phase = phi0 + omega * count / self._fs
        phase_inc = omega / self._fs
        phasors = np.empty(n, dtype=np.complex128)
        phasors[0] = np.exp(1j * start_phase)
        phasors[1:] = np.exp(1j * phase_inc)
        np.cumprod(phasors, out=phasors)
        output_items[0][:] = input_items[0] * phasors.astype(np.complex64)
        return n


class kraken_pbr_flowgraph(gr.top_block, Qt.QWidget):

    def __init__(self, source_type='kraken',
                 rspduo_if_gain=40.0, rspduo_rf_gain=0.0):
        source_label = "RSPduo" if source_type == 'rspduo' else "KrakenSDR"
        title = f"{source_label} Passive Radar — Range-Doppler"
        gr.top_block.__init__(self, title, catch_exceptions=True)
        self.source_type = source_type
        Qt.QWidget.__init__(self)
        self.setWindowTitle(title)
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
        # Per-channel: 4 surveillance channels (indices 0-3 = sdr outputs 1-4)
        self.n_surv = 4
        self.phase_history = [[] for _ in range(self.n_surv)]
        self.drift_coeffs = [np.zeros(3) for _ in range(self.n_surv)]
        self.coeffs_history = [[] for _ in range(self.n_surv)]
        self.drift_tau = 10.0
        self.start_time = time.time()
        self.cal_lock = threading.Lock()
        self._cal_stop = threading.Event()
        self._cal_wake = threading.Event()  # wakes cal thread when interval changes
        self._cal_status = "Waiting for startup..."
        self._cal_status_color = "orange"
        self._phase_plot_dirty = False  # flag for timer to redraw phase plot
        self._frozen_trend_t = []  # accumulated frozen red curve t-values (NaN-separated)
        self._frozen_trend_p = []  # accumulated frozen red curve phase-values
        self._n_frozen_segs = 0    # number of coeffs_history entries already frozen
        self.recal_interval = 60.0  # seconds between recalibrations

        ##################################################
        # Blocks — 5-channel: ch0=ref, ch1-4=surv with phase correction
        ##################################################

        # Per-channel phase correctors (one per surv channel)
        self.phase_corrs = [phase_corrector(sample_rate=decimated_rate)
                            for _ in range(self.n_surv)]
        self.phase_corr = self.phase_corrs[0]  # alias for RD processing chain

        # Vector probes for calibration: contiguous sample blocks allow FFT
        # cross-correlation to find inter-channel delay from GR scheduling.
        self.cal_vec_size = 2**16  # 65536 samples ≈ 245ms at decimated rate
        self.s2v_cal_ref = blocks.stream_to_vector(gr.sizeof_gr_complex, self.cal_vec_size)
        self.cal_probe_ref = blocks.probe_signal_vc(self.cal_vec_size)
        self.s2v_cals = [blocks.stream_to_vector(gr.sizeof_gr_complex, self.cal_vec_size)
                         for _ in range(self.n_surv)]
        self.cal_probes = [blocks.probe_signal_vc(self.cal_vec_size)
                           for _ in range(self.n_surv)]

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
        self.doppler_proc = kraken_passive_radar.doppler_processor(
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
                                                label='Piecewise prediction')
        self.phase_extrap, = self.phase_ax.plot([], [], '--', color='orange',
                                                 linewidth=1.2, alpha=0.8,
                                                 label='Extrapolation')
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
        if self.source_type == 'rspduo':
            self.phase_ax.set_title('Phase Drift — N/A (RSPduo coherent clock)')
            self.cal_label = Qt.QLabel("Phase Cal: N/A (RSPduo coherent clock)")
            self.cal_label.setStyleSheet("font-weight: bold; font-size: 14px; color: gray;")
        else:
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
        if self.source_type == 'rspduo':
            self.sdr_src = rspduo_source(
                frequency=center_freq, sample_rate=samp_rate,
                if_gain=rspduo_if_gain, rf_gain=rspduo_rf_gain
            )
        else:
            self.sdr_src = krakensdr_source(
                frequency=center_freq, sample_rate=samp_rate, gain=rf_gain
            )
            # Keep legacy alias for calibration code
            self.kraken_src = self.sdr_src

        # Reference channel conditioning
        self.dc_blocker_ref = filter.dc_blocker_cc(32, True)
        self.freq_xlating_fir_ref = filter.freq_xlating_fir_filter_ccc(decimation, lpf_taps, 0, samp_rate)
        self.agc_ref = analog.agc2_cc(0.01, 0.001, 1.0, 1.0, 65536)

        # Surveillance channels conditioning (4 channels)
        self.dc_blockers = [filter.dc_blocker_cc(32, True) for _ in range(self.n_surv)]
        self.fir_filters = [filter.freq_xlating_fir_filter_ccc(decimation, lpf_taps, 0, samp_rate)
                            for _ in range(self.n_surv)]
        self.agcs = [analog.agc2_cc(0.01, 0.001, 1.0, 1.0, 65536) for _ in range(self.n_surv)]

        # ECA on surv0 (ch1) for range-Doppler processing
        self.eca_canceller = kraken_passive_radar.eca_canceller(eca_taps, eca_reg, 1)

        ##################################################
        # Connections — 5-channel: ch0=ref, ch1-4=surv
        ##################################################
        # Reference channel: source -> DC block -> freq sink
        self.connect((self.sdr_src, 0), (self.dc_blocker_ref, 0))
        self.connect((self.sdr_src, 0), (self.qtgui_freq_sink_ref, 0))

        # Surveillance channels: source -> DC block
        for i in range(self.n_surv):
            self.connect((self.sdr_src, i + 1), (self.dc_blockers[i], 0))
        # Freq sink on ch1 (surv0) only
        self.connect((self.sdr_src, 1), (self.qtgui_freq_sink_surv, 0))

        # Reference conditioning: DC block -> LPF/decimate -> AGC + cal probe
        self.connect((self.dc_blocker_ref, 0), (self.freq_xlating_fir_ref, 0))
        self.connect((self.freq_xlating_fir_ref, 0), (self.agc_ref, 0))
        self.connect((self.freq_xlating_fir_ref, 0), (self.s2v_cal_ref, 0))
        self.connect((self.s2v_cal_ref, 0), (self.cal_probe_ref, 0))

        # Surveillance conditioning: DC block -> LPF -> phase_corr -> AGC + cal probe
        for i in range(self.n_surv):
            self.connect((self.dc_blockers[i], 0), (self.fir_filters[i], 0))
            # Cal probe taps BEFORE phase corrector (measures raw offset)
            self.connect((self.fir_filters[i], 0), (self.s2v_cals[i], 0))
            self.connect((self.s2v_cals[i], 0), (self.cal_probes[i], 0))
            # Phase corrector -> AGC
            self.connect((self.fir_filters[i], 0), (self.phase_corrs[i], 0))
            self.connect((self.phase_corrs[i], 0), (self.agcs[i], 0))

        # ECA clutter cancellation on surv0 (ch1) for range-Doppler
        self.connect((self.agc_ref, 0), (self.eca_canceller, 0))
        self.connect((self.agc_ref, 0), (self.s2v_ref, 0))
        self.connect((self.agcs[0], 0), (self.eca_canceller, 1))
        self.connect((self.eca_canceller, 0), (self.s2v_surv0, 0))

        # Surv channels 1-3 (ch2-ch4): AGC outputs to null sinks for now
        # (future: feed into AoA processing)
        self.surv_null_sinks = [blocks.null_sink(gr.sizeof_gr_complex) for _ in range(self.n_surv - 1)]
        for i in range(self.n_surv - 1):
            self.connect((self.agcs[i + 1], 0), (self.surv_null_sinks[i], 0))

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

    def _xcorr_phase(self, ref_vec, surv_vec):
        """FFT cross-correlation with correct alignment.
        Returns (phase_rad, correlation, delay_samples).

        Convention: IFFT(R * conj(S)) gives xcorr[k] = sum(r[n+k] * conj(s[n])).
        Peak at k=delay means ref leads surv by 'delay' samples.
        Alignment: ref[delay:] matches surv[:N-delay] for delay>0.
        """
        N = len(ref_vec)
        Xr = np.fft.fft(ref_vec)
        Xs = np.fft.fft(surv_vec)
        xc_full = np.fft.ifft(Xr * np.conj(Xs))
        peak = int(np.argmax(np.abs(xc_full)))
        delay = peak if peak <= N // 2 else peak - N

        overlap = N - abs(delay)
        if overlap < 256:
            return 0.0, 0.0, delay

        if delay >= 0:
            ra = ref_vec[delay:delay + overlap]
            sa = surv_vec[:overlap]
        else:
            ra = ref_vec[:overlap]
            sa = surv_vec[-delay:-delay + overlap]

        xc = np.vdot(ra, sa)
        rp = np.sqrt(np.vdot(ra, ra).real)
        sp = np.sqrt(np.vdot(sa, sa).real)
        if rp > 0 and sp > 0:
            return float(np.angle(xc)), float(np.abs(xc) / (rp * sp)), delay
        return 0.0, 0.0, delay

    def run_calibration(self):
        """
        Run one phase calibration cycle for all 4 surveillance channels.
        Uses noise source + FFT cross-correlation with drift compensation.
        """
        if self.source_type == 'rspduo':
            return
        print("=== Phase Calibration Starting ===")
        with self.cal_lock:
            self._cal_status = "Noise source ON -- calibrating..."
            self._cal_status_color = "red"

        # Step 1: Enable noise source (hardware disconnects antennas)
        self.kraken_src.set_noise_source(True)
        print("Noise source ENABLED")

        # Step 2: Wait for noise data to flush through GNU Radio pipeline
        time.sleep(1.5)

        # Step 3: Read all vectors. Retry up to 3 times if any channel's
        # vector doesn't overlap with the ref (different s2v phase).
        elapsed = time.time() - self.start_time
        status_parts = []
        cal_results = [None] * self.n_surv  # (phase, corr, delay) per ch

        for attempt in range(3):
            ref_vec = np.array(self.cal_probe_ref.level(), dtype=np.complex64)
            for ch in range(self.n_surv):
                if cal_results[ch] is not None:
                    continue  # already got a good read
                surv_vec = np.array(self.cal_probes[ch].level(), dtype=np.complex64)
                phase_offset, correlation, delay = self._xcorr_phase(ref_vec, surv_vec)
                if correlation >= 0.3:
                    cal_results[ch] = (phase_offset, correlation, delay)
            if all(r is not None for r in cal_results):
                break
            time.sleep(0.3)  # Wait for probes to advance to overlapping window

        for ch in range(self.n_surv):
            if cal_results[ch] is None:
                print(f"  ch{ch+1}: no overlap after retries, skipping")
                continue
            phase_offset, correlation, delay = cal_results[ch]
            phase_deg = np.degrees(phase_offset)

            if correlation < 0.1:
                print(f"  ch{ch+1}: corr={correlation:.4f} (too low, skipping)")
                continue

            # Step 5: Compute drift rate from history
            drift_rad_per_sec = 0.0
            drift_str = "measuring..."
            with self.cal_lock:
                self.phase_history[ch].append((elapsed, phase_deg))
                hist = self.phase_history[ch]

                if len(hist) >= 2:
                    times = np.array([p[0] for p in hist])
                    ph = np.degrees(np.unwrap(np.radians([p[1] for p in hist])))

                    if len(hist) >= 3:
                        w = np.exp(-(times[-1] - times) / self.drift_tau)
                        c = np.polyfit(times, ph, 2, w=w)
                        self.drift_coeffs[ch] = c
                        self.coeffs_history[ch].append((elapsed, c[0], c[1], c[2]))
                        inst_drift = 2.0 * c[0] * elapsed + c[1]
                        drift_str = f"{inst_drift:+.3f} deg/s"
                    else:
                        lin = np.polyfit(times, ph, 1)
                        self.drift_coeffs[ch] = np.array([0.0, lin[0], lin[1]])
                        self.coeffs_history[ch].append((elapsed, 0.0, lin[0], lin[1]))
                        inst_drift = lin[0]
                        drift_str = f"{lin[0]:+.3f} deg/s (linear)"

                    drift_rad_per_sec = np.radians(inst_drift)

            # Step 6: Apply correction to this channel's phase corrector
            self.phase_corrs[ch].set_correction(phase_offset, drift_rad_per_sec)

            print(f"  ch{ch+1}: phase={phase_deg:+.1f}°  corr={correlation:.3f}"
                  f"  delay={delay:+d}  drift={drift_str}")
            status_parts.append(f"ch{ch+1}={phase_deg:+.0f}°/{correlation:.2f}")

        # Step 7: Disable noise source
        self.kraken_src.set_noise_source(False)

        with self.cal_lock:
            self._cal_status = "  ".join(status_parts) if status_parts else "FAILED"
            self._cal_status_color = "green" if status_parts else "red"
            self._phase_plot_dirty = True

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
                rd_map = np.asarray(rd_data).reshape(self.num_doppler_bins, self.fft_size)
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
                # Display surv0 (ch1) phase history on the plot
                if not self.phase_history[0]:
                    return
                times = [p[0] for p in self.phase_history[0]]
                phases_deg = [p[1] for p in self.phase_history[0]]
                phases_unwrapped = np.degrees(np.unwrap(np.radians(phases_deg)))
                n_pts = len(self.phase_history[0])
                coeffs = self.drift_coeffs[0].copy()
                ch = list(self.coeffs_history[0])

            # Plot measured points (unwrapped)
            self.phase_line.set_data(times, phases_unwrapped)

            # Freeze new red segments: each cal produces a segment from
            # previous cal time to this cal time, using that cal's coefficients.
            # Once frozen, these segments never change.
            while self._n_frozen_segs < len(ch):
                i = self._n_frozen_segs
                _, a_i, b_i, c_i = ch[i]
                t_start = ch[i - 1][0] if i > 0 else 0.0
                t_end = ch[i][0]
                if t_end > t_start:
                    seg_t = np.linspace(t_start, t_end, 40)
                    seg_p = np.polyval([a_i, b_i, c_i], seg_t)
                    self._frozen_trend_t.extend(seg_t.tolist())
                    self._frozen_trend_p.extend(seg_p.tolist())
                    # NaN separator so matplotlib doesn't connect segments
                    self._frozen_trend_t.append(np.nan)
                    self._frozen_trend_p.append(np.nan)
                self._n_frozen_segs += 1

            # Red dashed: all frozen past segments (immutable)
            self.phase_trend.set_data(self._frozen_trend_t, self._frozen_trend_p)

            # Orange dashed: forward extrapolation from latest coefficients
            if n_pts >= 2:
                t_now = times[-1]
                span = max(t_now - times[0], 10.0)
                t_ext = np.linspace(t_now, t_now + span * 0.25, 30)
                p_ext = np.polyval(coeffs, t_ext)
                self.phase_extrap.set_data(t_ext, p_ext)

                inst_drift = 2.0 * coeffs[0] * t_now + coeffs[1]
                title = f"Phase Drift: {inst_drift:+.3f} deg/s  (curvature a={coeffs[0]:+.5f})"
            else:
                self.phase_extrap.set_data([], [])
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
        self.sdr_src.set_sample_rate(self.samp_rate)
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
        if self.source_type == 'rspduo':
            self.sdr_src.set_rf_gain(self.rf_gain)
        else:
            self.sdr_src.set_gain(self.rf_gain)

    def get_lpf_taps(self):
        return self.lpf_taps

    def set_lpf_taps(self, lpf_taps):
        self.lpf_taps = lpf_taps
        self.freq_xlating_fir_ref.set_taps(self.lpf_taps)
        for fir in self.fir_filters:
            fir.set_taps(self.lpf_taps)

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
        self.sdr_src.set_frequency(self.center_freq)


def main(top_block_cls=kraken_pbr_flowgraph, options=None):
    parser = ArgumentParser(description="KrakenSDR/RSPduo Passive Radar GUI")
    parser.add_argument("--source", choices=['kraken', 'rspduo'], default='kraken',
                        help="SDR source: kraken (5-ch KrakenSDR) or rspduo (2-ch SDRplay RSPduo)")
    parser.add_argument("--if-gain", type=float, default=40.0,
                        help="RSPduo IF gain in dB (default: 40, range: 20-59)")
    parser.add_argument("--rf-gain", type=float, default=0.0,
                        help="RSPduo RF gain reduction in dB (default: 0, range: 0-27)")
    args, remaining = parser.parse_known_args()

    qapp = Qt.QApplication([''] + remaining)

    tb = top_block_cls(
        source_type=args.source,
        rspduo_if_gain=args.if_gain,
        rspduo_rf_gain=args.rf_gain,
    )

    tb.start()
    tb.show()

    # Start periodic phase calibration in background thread (KrakenSDR only)
    if tb.source_type != 'rspduo':
        cal_thread = threading.Thread(target=tb._periodic_calibration, daemon=True)
        cal_thread.start()
    else:
        print("RSPduo mode: Phase calibration skipped (coherent clock)")

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
