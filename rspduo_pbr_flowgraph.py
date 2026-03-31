#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: RSPduo Passive Radar - Range-Doppler
# Author: N4HY
# Copyright: 2026
# Description: Passive Bistatic Radar using RSPduo dual-tuner (coherent clock, no phase calibration needed)
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
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import kraken_passive_radar
from gnuradio.kraken_passive_radar import rspduo_source
import sip
import threading



class rspduo_pbr_flowgraph(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "RSPduo Passive Radar - Range-Doppler", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("RSPduo Passive Radar - Range-Doppler")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "rspduo_pbr_flowgraph")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.signal_bw = signal_bw = 0
        self.samp_rate = samp_rate = 1000000
        self.num_doppler_bins = num_doppler_bins = 64
        self.fft_size = fft_size = 2048
        self.decimation = decimation = max(1, int(samp_rate / signal_bw)) if signal_bw > 0 else 1
        self.rf_gain_reduction = rf_gain_reduction = 0.0
        self.rd_size = rd_size = num_doppler_bins * fft_size
        self.lpf_taps = lpf_taps = firdes.low_pass(1.0, samp_rate, signal_bw/2, signal_bw/10, window.WIN_HAMMING) if signal_bw > 0 else [1.0]
        self.if_gain = if_gain = 40.0
        self.eca_taps = eca_taps = 128
        self.eca_reg = eca_reg = 0.0001
        self.decimated_rate = decimated_rate = int(samp_rate / decimation)
        self.cpi_samples = cpi_samples = 2048
        self.center_freq = center_freq = 103.7e6

        ##################################################
        # Blocks
        ##################################################

        self.v2s_ref_for_conj = blocks.vector_to_stream(gr.sizeof_gr_complex*1, fft_size)
        self.s2v_surv = blocks.stream_to_vector(gr.sizeof_gr_complex*1, cpi_samples)
        self.s2v_ref_conj = blocks.stream_to_vector(gr.sizeof_gr_complex*1, fft_size)
        self.s2v_ref = blocks.stream_to_vector(gr.sizeof_gr_complex*1, cpi_samples)
        self.rspduo_src = rspduo_source(frequency=center_freq, sample_rate=samp_rate, if_gain=if_gain, rf_gain=rf_gain_reduction, bandwidth=0, bias_t=False, rf_notch=True, dab_notch=True, am_notch=False)
        self.qtgui_vector_sink_rd = qtgui.vector_sink_f(
            rd_size,
            0,
            1.0,
            "Range-Doppler Bin",
            "Power (dB)",
            "Range-Doppler Map",
            1, # Number of inputs
            None # parent
        )
        self.qtgui_vector_sink_rd.set_update_time(0.40)
        self.qtgui_vector_sink_rd.set_y_axis((-20), 60)
        self.qtgui_vector_sink_rd.enable_autoscale(True)
        self.qtgui_vector_sink_rd.enable_grid(True)
        self.qtgui_vector_sink_rd.set_x_axis_units("")
        self.qtgui_vector_sink_rd.set_y_axis_units("")
        self.qtgui_vector_sink_rd.set_ref_level(0)


        labels = ['Range-Doppler (dB)', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_rd.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_rd.set_line_label(i, labels[i])
            self.qtgui_vector_sink_rd.set_line_width(i, widths[i])
            self.qtgui_vector_sink_rd.set_line_color(i, colors[i])
            self.qtgui_vector_sink_rd.set_line_alpha(i, alphas[i])

        self._qtgui_vector_sink_rd_win = sip.wrapinstance(self.qtgui_vector_sink_rd.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_vector_sink_rd_win, 0, 0, 1, 2)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_surv = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "Surveillance (Tuner 2)", #name
            1,
            None # parent
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



        labels = ['Surv (Tuner 2)', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["red", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_surv.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_surv.set_line_label(i, labels[i])
            self.qtgui_freq_sink_surv.set_line_width(i, widths[i])
            self.qtgui_freq_sink_surv.set_line_color(i, colors[i])
            self.qtgui_freq_sink_surv.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_surv_win = sip.wrapinstance(self.qtgui_freq_sink_surv.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_surv_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_ref = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "Reference (Tuner 1)", #name
            1,
            None # parent
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



        labels = ['Ref (Tuner 1)', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_ref.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_ref.set_line_label(i, labels[i])
            self.qtgui_freq_sink_ref.set_line_width(i, widths[i])
            self.qtgui_freq_sink_ref.set_line_color(i, colors[i])
            self.qtgui_freq_sink_ref.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_ref_win = sip.wrapinstance(self.qtgui_freq_sink_ref.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_ref_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.nlog10_rd = blocks.nlog10_ff(10, rd_size, 0)
        self.multiply_surv = blocks.multiply_vcc(fft_size)
        self.ifft_surv = fft.fft_vcc(fft_size, False, [], False, 1)
        self.freq_xlating_fir_surv = filter.freq_xlating_fir_filter_ccc(decimation, lpf_taps, 0, samp_rate)
        self.freq_xlating_fir_ref = filter.freq_xlating_fir_filter_ccc(decimation, lpf_taps, 0, samp_rate)
        self.fft_surv = fft.fft_vcc(fft_size, True, [], False, 1)
        self.fft_ref = fft.fft_vcc(fft_size, True, [], False, 1)
        self.eca_canceller = kraken_passive_radar.eca_canceller(eca_taps, eca_reg, 1)
        self.doppler_proc = kraken_passive_radar.doppler_processor(fft_size, num_doppler_bins, 1, True)
        self.dc_blocker_surv = filter.dc_blocker_cc(32, True)
        self.dc_blocker_ref = filter.dc_blocker_cc(32, True)
        self.conjugate_ref = blocks.conjugate_cc()
        self.agc_surv = analog.agc2_cc(0.01, 0.001, 1.0, 1.0, 65536)
        self.agc_ref = analog.agc2_cc(0.01, 0.001, 1.0, 1.0, 65536)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.agc_ref, 0), (self.eca_canceller, 0))
        self.connect((self.agc_ref, 0), (self.s2v_ref, 0))
        self.connect((self.agc_surv, 0), (self.eca_canceller, 1))
        self.connect((self.conjugate_ref, 0), (self.s2v_ref_conj, 0))
        self.connect((self.dc_blocker_ref, 0), (self.freq_xlating_fir_ref, 0))
        self.connect((self.dc_blocker_surv, 0), (self.freq_xlating_fir_surv, 0))
        self.connect((self.doppler_proc, 0), (self.nlog10_rd, 0))
        self.connect((self.eca_canceller, 0), (self.s2v_surv, 0))
        self.connect((self.fft_ref, 0), (self.v2s_ref_for_conj, 0))
        self.connect((self.fft_surv, 0), (self.multiply_surv, 0))
        self.connect((self.freq_xlating_fir_ref, 0), (self.agc_ref, 0))
        self.connect((self.freq_xlating_fir_surv, 0), (self.agc_surv, 0))
        self.connect((self.ifft_surv, 0), (self.doppler_proc, 0))
        self.connect((self.multiply_surv, 0), (self.ifft_surv, 0))
        self.connect((self.nlog10_rd, 0), (self.qtgui_vector_sink_rd, 0))
        self.connect((self.rspduo_src, 0), (self.dc_blocker_ref, 0))
        self.connect((self.rspduo_src, 1), (self.dc_blocker_surv, 0))
        self.connect((self.rspduo_src, 0), (self.qtgui_freq_sink_ref, 0))
        self.connect((self.rspduo_src, 1), (self.qtgui_freq_sink_surv, 0))
        self.connect((self.s2v_ref, 0), (self.fft_ref, 0))
        self.connect((self.s2v_ref_conj, 0), (self.multiply_surv, 1))
        self.connect((self.s2v_surv, 0), (self.fft_surv, 0))
        self.connect((self.v2s_ref_for_conj, 0), (self.conjugate_ref, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "rspduo_pbr_flowgraph")
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
        self.qtgui_freq_sink_ref.set_frequency_range(0, self.samp_rate)
        self.qtgui_freq_sink_surv.set_frequency_range(0, self.samp_rate)
        self.rspduo_src.set_sample_rate(self.samp_rate)

    def get_num_doppler_bins(self):
        return self.num_doppler_bins

    def set_num_doppler_bins(self, num_doppler_bins):
        self.num_doppler_bins = num_doppler_bins
        self.set_rd_size(self.num_doppler_bins * self.fft_size)
        self.doppler_proc.set_num_doppler_bins(self.num_doppler_bins)

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size
        self.set_rd_size(self.num_doppler_bins * self.fft_size)

    def get_decimation(self):
        return self.decimation

    def set_decimation(self, decimation):
        self.decimation = decimation
        self.set_decimated_rate(int(self.samp_rate / self.decimation))

    def get_rf_gain_reduction(self):
        return self.rf_gain_reduction

    def set_rf_gain_reduction(self, rf_gain_reduction):
        self.rf_gain_reduction = rf_gain_reduction
        self.rspduo_src.set_rf_gain(self.rf_gain_reduction)

    def get_rd_size(self):
        return self.rd_size

    def set_rd_size(self, rd_size):
        self.rd_size = rd_size

    def get_lpf_taps(self):
        return self.lpf_taps

    def set_lpf_taps(self, lpf_taps):
        self.lpf_taps = lpf_taps
        self.freq_xlating_fir_ref.set_taps(self.lpf_taps)
        self.freq_xlating_fir_surv.set_taps(self.lpf_taps)

    def get_if_gain(self):
        return self.if_gain

    def set_if_gain(self, if_gain):
        self.if_gain = if_gain
        self.rspduo_src.set_if_gain(self.if_gain)

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
        self.rspduo_src.set_frequency(self.center_freq)




def main(top_block_cls=rspduo_pbr_flowgraph, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
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
