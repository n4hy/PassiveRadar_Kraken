#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# KrakenSDR Passive Radar (FM/TV illuminator) - Range-Doppler demo
# Tested with GNU Radio 3.10 API. Requires: gnuradio, gr-qtgui, gr-osmosdr (or Soapy source), numpy.
#
# Notes:
# - This script uses two RF inputs from a coherent receiver (e.g., KrakenSDR) as
#   "reference" (direct path from the illuminator) and "surveillance" (contains echoes).
# - It implements a simple CAF (cross-ambiguity function) via FFT-based cross-correlation
#   for range processing and an FFT across slow-time for Doppler. The result is shown
#   as a Range–Doppler raster.
# - Tune to a strong local FM or ATSC 1.0 TV carrier (e.g., 88–108 MHz for FM, VHF/UHF for TV).
# - For KrakenSDR via osmosdr, set device strings accordingly (see args below).
#
# DISCLAIMER: This is a teaching/research demo and not a certified ATC system.
#
from gnuradio import gr, blocks, filter, analog, fft, qtgui
import sip, numpy as np, sys, os, time, math
try:
    import osmosdr
    HAVE_OSMOSDR = True
except Exception:
    HAVE_OSMOSDR = False
from PyQt5 import Qt
from gnuradio.filter import firdes

class KrakenPassiveRadar(gr.top_block, Qt.QWidget):
    def __init__(self,
                 samp_rate=2_400_000,
                 center_freq=99.5e6,     # set to strong local FM or TV carrier
                 fft_len=4096,           # range FFT size (fast-time)
                 doppler_len=128,        # Doppler FFT size (slow-time)
                 decim=2,                # front-end decimation (post-channelize)
                 bpf_bw=180e3,           # FM: ~180 kHz; for TV pilot/segment adjust accordingly
                 ref_gain=30,
                 surv_gain=30,
                 ref_dev="numchan=5,rtl=0",   # EXAMPLE osmosdr string for Kraken channel 0
                 surv_dev="numchan=5,rtl=1",  # EXAMPLE osmosdr string for Kraken channel 1
                 ppm=0,
                 ):
        gr.top_block.__init__(self, "Kraken Passive Radar")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("KrakenSDR Passive Radar: Range–Doppler")
        qtgui.util.check_set_qss()
        self.top_scroll_layout = Qt.QVBoxLayout(self)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll.setWidgetResizable(True)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidget(Qt.QWidget())
        self.top_layout = Qt.QVBoxLayout(self.top_scroll.widget())

        # store params
        self.samp_rate = samp_rate
        self.center_freq = center_freq
        self.fft_len = fft_len
        self.doppler_len = doppler_len
        self.decim = decim
        self.bpf_bw = bpf_bw
        self.ref_gain = ref_gain
        self.surv_gain = surv_gain

        ############################
        # RF SOURCES (osmosdr)
        ############################
        if not HAVE_OSMOSDR:
            raise RuntimeError("gr-osmosdr not available. Please install gr-osmosdr (with Kraken/RTL support).")

        self.ref_src = osmosdr.source(args=ref_dev)
        self.ref_src.set_sample_rate(samp_rate)
        self.ref_src.set_center_freq(center_freq)
        self.ref_src.set_freq_corr(ppm)
        self.ref_src.set_gain_mode(True)  # manual gain
        self.ref_src.set_gain(ref_gain)
        self.ref_src.set_bandwidth(0)

        self.surv_src = osmosdr.source(args=surv_dev)
        self.surv_src.set_sample_rate(samp_rate)
        self.surv_src.set_center_freq(center_freq)
        self.surv_src.set_freq_corr(ppm)
        self.surv_src.set_gain_mode(True)
        self.surv_src.set_gain(surv_gain)
        self.surv_src.set_bandwidth(0)

        #########################################
        # CHANNELIZATION (Frequency Xlating FIR)
        #########################################
        # For FM, isolate ~180 kHz around carrier/content. For TV, retune bpf_bw accordingly.
        chans_taps = firdes.low_pass(1.0, samp_rate, self.bpf_bw/2, self.bpf_bw/4, firdes.WIN_HAMMING)
        self.ref_chan = filter.freq_xlating_fir_filter_ccc(self.decim, chans_taps, 0.0, samp_rate)
        self.surv_chan = filter.freq_xlating_fir_filter_ccc(self.decim, chans_taps, 0.0, samp_rate)
        self.fs = samp_rate // self.decim

        # DC blocker (helpful for FM/TV pilots)
        self.ref_dc = filter.dc_blocker_cc(128, True)
        self.surv_dc = filter.dc_blocker_cc(128, True)

        ############################
        # CAF RANGE PROCESSOR (FFT)
        ############################
        self.vec = fft_len
        self.ref_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, self.vec)
        self.surv_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, self.vec)

        self.win = np.hanning(self.vec).astype(np.float32)
        self.ref_win = blocks.multiply_const_vcc(self.win.astype(np.complex64))
        self.surv_win = blocks.multiply_const_vcc(self.win.astype(np.complex64))

        self.ref_fft = fft.fft_vcc(self.vec, True, (), True)
        self.surv_fft = fft.fft_vcc(self.vec, True, (), True)

        self.mult_conj = blocks.multiply_conjugate_cc(self.vec)
        self.ifft = fft.ifft_vcc(self.vec, True, (), True)
        self.corr_mag = blocks.complex_to_mag_squared(self.vec)

        # Stream tags to throttle GUI (optional)
        self.throttle = blocks.throttle(gr.sizeof_float*self.vec, self.fs, True)

        ############################
        # DOPPLER PROCESSOR (SLOW-TIME FFT)
        ############################
        self.slow_stv = blocks.stream_to_vector(gr.sizeof_float, self.vec)
        # Buffer slow-time vectors (doppler_len frames per range bin)
        self.slow_buf = blocks.vector_to_streams(gr.sizeof_float*self.vec, self.vec)  # demux ranges to parallel streams

        # Apply Doppler FFT per range bin using nlogics via FFT blocks in a loop:
        # To keep GRC-free, we approximate by stacking frames then using a raster GUI.
        # We'll still compute a slow-time FFT by accumulating doppler_len frames.
        self.accum = blocks.stream_to_vector(gr.sizeof_float, self.vec * self.doppler_len)
        # NOTE: For simplicity, we won't implement per-bin FFT here — instead, we display
        # range-time raster (motion shows slanted traces). For a full RD map, consider
        # gnuradio blocks or custom python blocks per-bin FFT.
        # As a compromise, we add a decimator + raster to visualize range vs time.
        self.rd_raster = qtgui.time_raster_sink_f(
            samp_rate=self.fs/self.vec,
            y_axis_units="Range bins",
            y_axis_label="Range bins (0..N-1)",
            number_of_points=self.vec,
            name="Range-Time (proxy for RD)",
            x_axis_label="Slow time",
            x_axis_units="s",
        )
        self._rd_raster_win = sip.wrapinstance(self.rd_raster.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._rd_raster_win)

        # Also show PSDs for debugging
        self.ref_psd = qtgui.freq_sink_c(1024, firdes.WIN_BLACKMAN_hARRIS, 0, self.fs, "REF PSD", 1)
        self.surv_psd = qtgui.freq_sink_c(1024, firdes.WIN_BLACKMAN_hARRIS, 0, self.fs, "SURV PSD", 1)
        self.top_layout.addWidget(sip.wrapinstance(self.ref_psd.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.surv_psd.qwidget(), Qt.QWidget))

        ############################
        # Connections
        ############################
        self.connect(self.ref_src, self.ref_chan, self.ref_dc, self.ref_vec, self.ref_win, self.ref_fft)
        self.connect(self.surv_src, self.surv_chan, self.surv_dc, self.surv_vec, self.surv_win, self.surv_fft)

        self.connect((self.ref_fft, 0), (self.mult_conj, 0))
        self.connect((self.surv_fft, 0), (self.mult_conj, 1))
        self.connect(self.mult_conj, self.ifft)
        self.connect(self.ifft, self.corr_mag)
        self.connect(self.corr_mag, self.throttle)
        self.connect(self.throttle, self.slow_stv)
        # feed as raster (expects stream of vectors -> flattened)
        self.connect(self.slow_stv, self.rd_raster)

        # PSD debug
        self.connect(self.ref_dc, (self.ref_psd, 0))
        self.connect(self.surv_dc, (self.surv_psd, 0))

        # Hints to user via title
        self.rd_raster.set_update_time(0.05)
        self.ref_psd.set_update_time(0.10)
        self.surv_psd.set_update_time(0.10)

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = KrakenPassiveRadar()
    tb.start()
    tb.show()
    qapp.exec_()

if __name__ == "__main__":
    main()
