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
from PyQt5 import Qt, QtCore
from gnuradio.filter import firdes

# In a real installation, we would import from the OOT module:
# from kraken_passive_radar import DopplerProcessingBlock, RangeDopplerWidget, ClutterCanceller
# But here we inline the new ClutterCanceller or define it for the standalone script.
# Since we already inlined DopplerProcessingBlock, we should inline ClutterCanceller too
# OR actually import it if we set PYTHONPATH.
# Given the user wants a standalone script (presumably), I will add the class here.

class ClutterCanceller(gr.basic_block):
    """
    Adaptive Clutter Canceller using NLMS.

    Inputs:
    0: Reference Signal (Predictor)
    1: Surveillance Signal (Desired)

    Outputs:
    0: Error Signal (Surveillance - Estimated Clutter)
    """
    def __init__(self, num_taps=32, mu=0.1):
        gr.basic_block.__init__(self,
            name="ClutterCanceller",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64])

        self.num_taps = num_taps
        self.mu = mu
        self.w = np.zeros(num_taps, dtype=np.complex64)

        # Buffer for reference history
        self.history = np.zeros(num_taps, dtype=np.complex64)

    def general_work(self, input_items, output_items):
        ref_in = input_items[0]
        surv_in = input_items[1]
        out_err = output_items[0]

        n_input = min(len(ref_in), len(surv_in), len(out_err))

        if n_input == 0:
            return 0

        # Prepare full reference stream including history
        full_ref = np.concatenate((self.history, ref_in[:n_input]))

        # Iterate
        # Note: In a production environment, this should be optimized (C++ or numba)
        for i in range(n_input):
            # x[n] = [ref[i], ref[i-1], ... ref[i-N+1]]
            # This corresponds to full_ref[i : i + num_taps] reversed
            x = full_ref[i : i + self.num_taps][::-1]

            # Filter output (Estimate of clutter)
            y = np.dot(self.w, x)

            # Error (Desired - Estimate)
            d = surv_in[i]
            e = d - y

            # Update weights (NLMS)
            norm_x_sq = np.real(np.dot(x, np.conj(x))) + 1e-12
            self.w += self.mu * e * np.conj(x) / norm_x_sq

            out_err[i] = e

        # Update history
        self.history = full_ref[n_input:]

        self.consume(0, n_input)
        self.consume(1, n_input)
        return n_input

class DopplerProcessingBlock(gr.basic_block):
    """
    Custom Python block to perform slow-time FFT for Range-Doppler processing.
    Consumes doppler_len vectors of size fft_len (Range bins).
    Outputs a single Range-Doppler map (flattened) or triggers a GUI update.
    """
    def __init__(self, fft_len, doppler_len, callback=None):
        gr.basic_block.__init__(self,
            name="DopplerProcessingBlock",
            in_sig=[(np.complex64, fft_len)],
            out_sig=None) # We don't output to GR stream, we send to GUI via callback

        self.fft_len = fft_len
        self.doppler_len = doppler_len
        self.callback = callback

        # Buffer to store 'doppler_len' range profiles (rows=slow_time, cols=fast_time)
        self.buffer = np.zeros((doppler_len, fft_len), dtype=np.complex64)
        self.buf_idx = 0

        # Window function for Doppler dimension
        self.win = np.hamming(doppler_len).astype(np.float32)

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        n_input = len(in0)

        # We need to process all input items
        # Since this is a basic_block, we must implement consume manually

        processed = 0
        while processed < n_input:
            # How much space is left in the buffer?
            space = self.doppler_len - self.buf_idx
            # How much data do we have?
            available = n_input - processed

            # Copy chunk
            to_copy = min(space, available)
            self.buffer[self.buf_idx : self.buf_idx + to_copy] = in0[processed : processed + to_copy]

            self.buf_idx += to_copy
            processed += to_copy

            if self.buf_idx >= self.doppler_len:
                # Buffer full, process map
                self.process_map()
                self.buf_idx = 0

        self.consume(0, n_input)
        return 0

    def process_map(self):
        # 1. Apply window along slow-time (axis 0)
        # buffer shape: (doppler_len, fft_len)
        # multiply column-wise by window
        cpi = self.buffer * self.win[:, np.newaxis]

        # 2. FFT along slow-time (axis 0) -> Doppler
        rd_complex = np.fft.fftshift(np.fft.fft(cpi, axis=0), axes=0)

        # 3. Magnitude squared or log mag
        rd_mag = 10 * np.log10(np.abs(rd_complex)**2 + 1e-12)

        # 4. Send to GUI
        if self.callback:
            self.callback(rd_mag)

class RangeDopplerWidget(Qt.QWidget):
    """
    Custom Qt Widget to display the Range-Doppler Map.
    """
    update_signal = QtCore.pyqtSignal(object)

    def __init__(self, fft_len, doppler_len, parent=None):
        super().__init__(parent)
        self.fft_len = fft_len
        self.doppler_len = doppler_len
        self.data = np.zeros((doppler_len, fft_len), dtype=np.float32)

        self.update_signal.connect(self.on_update)

        layout = Qt.QVBoxLayout(self)
        self.label = Qt.QLabel()
        self.label.setScaledContents(True) # Allow resizing
        layout.addWidget(self.label)

    def on_update(self, data):
        self.data = data
        # Normalize for display (simple min/max)
        d_min = np.min(data)
        d_max = np.max(data)
        if d_max > d_min:
            norm = (data - d_min) / (d_max - d_min) * 255
        else:
            norm = data * 0

        norm = norm.astype(np.uint8)

        # Map to colormap (grayscale for now)
        # In PyQt, creating an image from numpy array usually involves some shuffling
        # data is (doppler_len, fft_len) -> (Y, X)
        # We want X axis = Doppler, Y axis = Range ?
        # Standard RD Map: X=Doppler, Y=Range.
        # But our matrix is (doppler_len, fft_len).
        # So axis 0 is Doppler, axis 1 is Range.
        # So we should transpose to get (Range, Doppler) -> (Y, X)

        # display_data shape: (fft_len, doppler_len)
        display_data = norm.T

        h, w = display_data.shape
        # Create QImage from data
        # We need to make it contiguous and formatted correctly
        # This is a bit tricky with raw QImage and numpy without extra libs like qimage2ndarray
        # Simple hack: create RGB array

        rgb = np.dstack((display_data, display_data, display_data)).copy()

        qimg = Qt.QImage(rgb.data, w, h, 3*w, Qt.QImage.Format_RGB888)
        self.label.setPixmap(Qt.QPixmap.fromImage(qimg))

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

        #########################################
        # CLUTTER CANCELLATION (NLMS)
        #########################################
        # Clean surveillance channel by removing direct path (ref)
        self.clutter_cancel = ClutterCanceller(num_taps=64, mu=0.05)

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

        # New: Custom Doppler Processing Block
        # Create the widget first
        self.rd_widget = RangeDopplerWidget(self.fft_len, self.doppler_len)
        self.top_layout.addWidget(self.rd_widget)

        # Callback wrapper to be called from the block (in GR thread) to update GUI (in Qt thread)
        def update_gui(data):
            self.rd_widget.update_signal.emit(data)

        self.doppler_proc = DopplerProcessingBlock(self.fft_len, self.doppler_len, callback=update_gui)

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
        # self.top_layout.addWidget(self._rd_raster_win) # Disable old raster or keep it?
        # Let's keep it below for comparison
        self.top_layout.addWidget(self._rd_raster_win)

        # Also show PSDs for debugging
        self.ref_psd = qtgui.freq_sink_c(1024, firdes.WIN_BLACKMAN_hARRIS, 0, self.fs, "REF PSD", 1)
        self.surv_psd = qtgui.freq_sink_c(1024, firdes.WIN_BLACKMAN_hARRIS, 0, self.fs, "SURV PSD", 1)
        self.top_layout.addWidget(sip.wrapinstance(self.ref_psd.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.surv_psd.qwidget(), Qt.QWidget))

        ############################
        # Connections
        ############################
        # Reference Path: Src -> Chan -> DC -> [Clutter Cancel In 0] -> Vec -> Win -> FFT
        # Note: Ref DC also goes to CAF Reference input
        self.connect(self.ref_src, self.ref_chan, self.ref_dc)
        self.connect(self.surv_src, self.surv_chan, self.surv_dc)

        # Clutter Cancellation (NLMS)
        # Input 0: Ref (Predictor), Input 1: Surv (Desired)
        # Output 0: Error (Clean Surv)
        self.connect(self.ref_dc, (self.clutter_cancel, 0))
        self.connect(self.surv_dc, (self.clutter_cancel, 1))

        # CAF Processing
        # Reference branch for CAF
        self.connect(self.ref_dc, self.ref_vec, self.ref_win, self.ref_fft)

        # Surveillance branch for CAF (Now comes from Clutter Canceller)
        self.connect(self.clutter_cancel, self.surv_vec, self.surv_win, self.surv_fft)

        self.connect((self.ref_fft, 0), (self.mult_conj, 0))
        self.connect((self.surv_fft, 0), (self.mult_conj, 1))
        self.connect(self.mult_conj, self.ifft)

        # Branch 1: Existing path for Range-Time Raster
        self.connect(self.ifft, self.corr_mag)
        self.connect(self.corr_mag, self.throttle)
        self.connect(self.throttle, self.slow_stv)
        # feed as raster (expects stream of vectors -> flattened)
        self.connect(self.slow_stv, self.rd_raster)

        # Branch 2: New path for Range-Doppler Map
        # DopplerProcessingBlock expects complex input of size fft_len
        # The output of ifft is stream of vectors of size fft_len (complex)
        self.connect(self.ifft, self.doppler_proc)

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
