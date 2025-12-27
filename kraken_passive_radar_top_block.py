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
import ctypes
from ctypes import c_float, c_void_p, c_int, POINTER
import subprocess

try:
    import osmosdr
    HAVE_OSMOSDR = True
except Exception:
    HAVE_OSMOSDR = False
from PyQt5 import Qt, QtCore
from gnuradio.filter import firdes

# --- Helper: Compile and Load C++ Library ---

def compile_and_load_lib(lib_name, src_name, functions, extra_flags=None, extra_srcs=None):
    """
    Attempts to load a shared library. Compiles it if missing.
    functions: list of tuples (func_name, argtypes, restype)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    lib_path = os.path.join(src_dir, lib_name)
    cpp_file = os.path.join(src_dir, src_name)

    # Compile if missing
    if not os.path.exists(lib_path):
        if os.path.exists(cpp_file):
            print(f"Library {lib_name} not found. Compiling {cpp_file}...")
            try:
                cmd = [
                    'g++', '-O3', '-shared', '-fPIC', '-DLIBRARY_BUILD',
                    cpp_file
                ]
                if extra_srcs:
                    for s in extra_srcs:
                        s_path = os.path.join(src_dir, s)
                        if os.path.exists(s_path):
                            cmd.append(s_path)

                cmd.extend(['-o', lib_path])

                if extra_flags:
                    cmd.extend(extra_flags)
                subprocess.check_call(cmd)
                print(f"Compilation of {lib_name} successful.")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Compilation failed: {e}. Will use Python fallback.")
                return None
        else:
            print(f"Source file {cpp_file} not found. Cannot compile.")
            return None

    try:
        lib = ctypes.CDLL(lib_path)
        for name, args, res in functions:
            if hasattr(lib, name):
                f = getattr(lib, name)
                f.argtypes = args
                f.restype = res
            else:
                print(f"Warning: Function {name} not found in {lib_name}")
        print(f"Loaded C++ library: {lib_path}")
        return lib
    except Exception as e:
        print(f"Failed to load {lib_name}: {e}. Will use Python fallback.")
        return None

# --- Load Libraries ---

_eca_funcs = [
    ('eca_b_create', [c_int], c_void_p),
    ('eca_b_destroy', [c_void_p], None),
    ('eca_b_process', [c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int], None)
]
_eca_lib = compile_and_load_lib('libeca_b_clutter_canceller.so', 'eca_b_clutter_canceller.cpp', _eca_funcs, extra_srcs=['optmath/neon_kernels.cpp'])

_doppler_funcs = [
    ('doppler_create', [c_int, c_int], c_void_p),
    ('doppler_destroy', [c_void_p], None),
    ('doppler_process', [c_void_p, POINTER(c_float), POINTER(c_float)], None),
    ('doppler_process_complex', [c_void_p, POINTER(c_float), POINTER(c_float)], None)
]
# Add link flag for FFTW3f
_doppler_lib = compile_and_load_lib('libdoppler_processing.so', 'doppler_processing.cpp', _doppler_funcs, extra_flags=['-lfftw3f'])

_aoa_funcs = [
    ('aoa_create', [c_int, c_float], c_void_p),
    ('aoa_destroy', [c_void_p], None),
    ('aoa_process', [c_void_p, POINTER(c_float), c_float, POINTER(c_float), c_int], None)
]
_aoa_lib = compile_and_load_lib('libaoa_processing.so', 'aoa_processing.cpp', _aoa_funcs)

_resampler_funcs = [
    ('resampler_create', [c_int, c_int, POINTER(c_float), c_int], c_void_p),
    ('resampler_destroy', [c_void_p], None),
    ('resampler_process', [c_void_p, POINTER(c_float), c_int, POINTER(c_float), c_int], c_int)
]
_resampler_lib = compile_and_load_lib('libresampler.so', 'resampler.cpp', _resampler_funcs, extra_srcs=['optmath/neon_kernels.cpp'])


class ECABCanceller(gr.basic_block):
    """
    Extensive Cancellation Algorithm - Batch (ECA-B).
    Replaces NLMS.
    """
    def __init__(self, num_taps=64):
        gr.basic_block.__init__(self,
            name="ECABCanceller",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64])

        self.num_taps = num_taps
        self.use_cpp = (_eca_lib is not None)
        self.cpp_obj = None

        if self.use_cpp:
            self.cpp_obj = _eca_lib.eca_b_create(num_taps)
        else:
            print("Warning: ECA-B C++ library not available. No python fallback for ECA-B implemented (use C++).")
            # Usually we'd implement a slow numpy version here, but request was "Replace... in the library".
            # For robustness, let's implement a very simple pass-through or basic projection if possible,
            # but solving 64x64 matrix in python block every step might be slow.
            # We'll just pass through if missing to avoid crash.
            pass

    def __del__(self):
        if getattr(self, 'use_cpp', False) and getattr(self, 'cpp_obj', None):
            try:
                _eca_lib.eca_b_destroy(self.cpp_obj)
            except Exception:
                pass
            self.cpp_obj = None

    def general_work(self, input_items, output_items):
        ref_in = input_items[0]
        surv_in = input_items[1]
        out_err = output_items[0]

        n_input = min(len(ref_in), len(surv_in), len(out_err))
        if n_input == 0: return 0

        if self.use_cpp and self.cpp_obj:
            p_ref = ref_in.ctypes.data_as(POINTER(c_float))
            p_surv = surv_in.ctypes.data_as(POINTER(c_float))
            p_out = out_err.ctypes.data_as(POINTER(c_float))
            _eca_lib.eca_b_process(self.cpp_obj, p_ref, p_surv, p_out, n_input)
        else:
            # Pass through (Failure mode)
            out_err[:n_input] = surv_in[:n_input]

        self.consume(0, n_input)
        self.consume(1, n_input)
        return n_input

class PolyphaseResamplerBlock(gr.basic_block):
    """
    Custom C++ Polyphase Resampler Block
    """
    def __init__(self, interp, decim, taps):
        gr.basic_block.__init__(self,
            name="PolyphaseResamplerBlock",
            in_sig=[np.complex64],
            out_sig=[np.complex64])

        self.interp = interp
        self.decim = decim
        self.taps = np.array(taps, dtype=np.float32)

        self.use_cpp = (_resampler_lib is not None)
        self.cpp_obj = None

        if self.use_cpp:
            p_taps = self.taps.ctypes.data_as(POINTER(c_float))
            self.cpp_obj = _resampler_lib.resampler_create(interp, decim, p_taps, len(self.taps))
            # forecast hint handled in forecast() method
        else:
            raise RuntimeError("C++ Resampler Library not loaded. Cannot run Polyphase Resampler.")

    def __del__(self):
        if getattr(self, 'use_cpp', False) and getattr(self, 'cpp_obj', None):
            try:
                _resampler_lib.resampler_destroy(self.cpp_obj)
            except Exception:
                pass
            self.cpp_obj = None

    def forecast(self, noutput_items, ninput_items_required):
        # We need approximately noutput * decim / interp input items
        # Add some margin for taps and history
        req = int(noutput_items * self.decim / self.interp) + len(self.taps)
        ninput_items_required[0] = req

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]

        if not self.use_cpp:
            return 0

        n_input = len(in0)
        n_output_cap = len(out0)

        if n_input == 0:
            return 0

        # Call C++ process
        p_in = in0.ctypes.data_as(POINTER(c_float))
        p_out = out0.ctypes.data_as(POINTER(c_float))

        produced = _resampler_lib.resampler_process(self.cpp_obj, p_in, n_input, p_out, n_output_cap)

        self.consume(0, n_input)
        return produced

class DopplerProcessingBlock(gr.basic_block):
    """
    Custom Python block to perform slow-time FFT for Range-Doppler processing.
    """
    def __init__(self, fft_len, doppler_len, callback=None):
        gr.basic_block.__init__(self,
            name="DopplerProcessingBlock",
            in_sig=[(np.complex64, fft_len)],
            out_sig=None)

        self.fft_len = fft_len
        self.doppler_len = doppler_len
        self.callback = callback

        self.buffer = np.zeros((doppler_len, fft_len), dtype=np.complex64)
        self.buf_idx = 0

        # C++ Setup
        self.use_cpp = (_doppler_lib is not None)
        self.cpp_obj = None

        if self.use_cpp:
            self.cpp_obj = _doppler_lib.doppler_create(fft_len, doppler_len)
            self.out_buf_c = np.zeros((doppler_len, fft_len), dtype=np.float32)
        else:
            self.win = np.hamming(doppler_len).astype(np.float32)

    def __del__(self):
        if getattr(self, 'use_cpp', False) and getattr(self, 'cpp_obj', None):
            try:
                _doppler_lib.doppler_destroy(self.cpp_obj)
            except Exception:
                pass
            self.cpp_obj = None

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        n_input = len(in0)
        processed = 0
        while processed < n_input:
            space = self.doppler_len - self.buf_idx
            available = n_input - processed
            to_copy = min(space, available)
            self.buffer[self.buf_idx : self.buf_idx + to_copy] = in0[processed : processed + to_copy]
            self.buf_idx += to_copy
            processed += to_copy
            if self.buf_idx >= self.doppler_len:
                self.process_map()
                self.buf_idx = 0
        self.consume(0, n_input)
        return 0

    def process_map(self):
        if self.use_cpp:
            p_in = self.buffer.ctypes.data_as(POINTER(c_float))
            p_out = self.out_buf_c.ctypes.data_as(POINTER(c_float))
            _doppler_lib.doppler_process(self.cpp_obj, p_in, p_out)
            if self.callback:
                self.callback(self.out_buf_c.copy())
        else:
            cpi = self.buffer * self.win[:, np.newaxis]
            rd_complex = np.fft.fftshift(np.fft.fft(cpi, axis=0), axes=0)
            rd_mag = 10 * np.log10(np.abs(rd_complex)**2 + 1e-12)
            if self.callback:
                self.callback(rd_mag)

class AoAProcessor:
    def __init__(self, num_antennas, spacing=0.0):
        self.num_antennas = num_antennas
        self.spacing = spacing
        self.use_cpp = (_aoa_lib is not None)
        self.cpp_obj = None
        if self.use_cpp:
            self.cpp_obj = _aoa_lib.aoa_create(num_antennas, c_float(spacing))
    def __del__(self):
        if self.use_cpp and self.cpp_obj:
            _aoa_lib.aoa_destroy(self.cpp_obj)
            self.cpp_obj = None
    def compute_spectrum(self, antenna_data, freq_hz, n_angles=181):
        lambda_val = 299792458.0 / freq_hz
        if self.use_cpp and self.cpp_obj:
            inputs = np.array(antenna_data, dtype=np.complex64)
            output = np.zeros(n_angles, dtype=np.float32)
            p_in = inputs.ctypes.data_as(POINTER(c_float))
            p_out = output.ctypes.data_as(POINTER(c_float))
            _aoa_lib.aoa_process(self.cpp_obj, p_in, c_float(lambda_val), p_out, n_angles)
            return output
        return np.zeros(n_angles, dtype=np.float32)

class RangeDopplerWidget(Qt.QWidget):
    update_signal = QtCore.pyqtSignal(object)
    def __init__(self, fft_len, doppler_len, parent=None):
        super().__init__(parent)
        self.fft_len = fft_len
        self.doppler_len = doppler_len
        self.data = np.zeros((doppler_len, fft_len), dtype=np.float32)
        self.update_signal.connect(self.on_update)
        layout = Qt.QVBoxLayout(self)
        self.label = Qt.QLabel()
        self.label.setScaledContents(True)
        layout.addWidget(self.label)
    def on_update(self, data):
        self.data = data
        d_min = np.min(data)
        d_max = np.max(data)
        if d_max > d_min:
            norm = (data - d_min) / (d_max - d_min) * 255
        else:
            norm = data * 0
        norm = norm.astype(np.uint8)
        display_data = norm.T
        h, w = display_data.shape
        rgb = np.dstack((display_data, display_data, display_data)).copy()
        qimg = Qt.QImage(rgb.data, w, h, 3*w, Qt.QImage.Format_RGB888)
        self.label.setPixmap(Qt.QPixmap.fromImage(qimg))

class KrakenPassiveRadar(gr.top_block, Qt.QWidget):
    def __init__(self,
                 samp_rate=2048000,      # Modified to 2.048 MHz
                 center_freq=99.5e6,
                 fft_len=4096,
                 doppler_len=128,
                 decim=1,                # Decim handled by Resampler
                 bpf_bw=180e3,
                 ref_gain=30,
                 surv_gain=30,
                 ref_dev="numchan=5,rtl=0",
                 surv_dev="numchan=5,rtl=1",
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
        self.ref_src.set_gain_mode(True)
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
        # RESAMPLING (2.048 MHz -> 175 kHz)
        #########################################
        # Polyphase Resampler
        target_rate = 175000
        interp = 175
        decim_val = 2048

        fs_poly = samp_rate * interp

        # Taps calculation
        taps_len = 1000
        M = taps_len - 1
        h = np.zeros(taps_len, dtype=np.float32)
        fc = 80e3 / fs_poly # Normalized cutoff

        win = np.hamming(taps_len)

        for i in range(taps_len):
            if i == M/2.0:
                h[i] = 2 * fc
            else:
                n = i - M/2.0
                h[i] = np.sin(2 * np.pi * fc * n) / (np.pi * n)
            h[i] *= win[i]

        # Gain compensation
        h *= interp

        self.ref_resamp = PolyphaseResamplerBlock(interp, decim_val, h)
        self.surv_resamp = PolyphaseResamplerBlock(interp, decim_val, h)

        self.fs = target_rate

        self.ref_dc = filter.dc_blocker_cc(128, True)
        self.surv_dc = filter.dc_blocker_cc(128, True)

        #########################################
        # CLUTTER CANCELLATION (ECA-B)
        #########################################
        # Use new ECABCanceller
        self.clutter_cancel = ECABCanceller(num_taps=64)

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

        self.throttle = blocks.throttle(gr.sizeof_float*self.vec, self.fs, True)

        ############################
        # DOPPLER PROCESSOR (SLOW-TIME FFT)
        ############################
        self.slow_stv = blocks.stream_to_vector(gr.sizeof_float, self.vec)

        self.rd_widget = RangeDopplerWidget(self.fft_len, self.doppler_len)
        self.top_layout.addWidget(self.rd_widget)

        def update_gui(data):
            self.rd_widget.update_signal.emit(data)

        self.doppler_proc = DopplerProcessingBlock(self.fft_len, self.doppler_len, callback=update_gui)

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

        self.ref_psd = qtgui.freq_sink_c(1024, firdes.WIN_BLACKMAN_hARRIS, 0, self.fs, "REF PSD", 1)
        self.surv_psd = qtgui.freq_sink_c(1024, firdes.WIN_BLACKMAN_hARRIS, 0, self.fs, "SURV PSD", 1)
        self.top_layout.addWidget(sip.wrapinstance(self.ref_psd.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.surv_psd.qwidget(), Qt.QWidget))

        ############################
        # Connections
        ############################
        # Src -> Resampler -> DC
        self.connect(self.ref_src, self.ref_resamp, self.ref_dc)
        self.connect(self.surv_src, self.surv_resamp, self.surv_dc)

        self.connect(self.ref_dc, (self.clutter_cancel, 0))
        self.connect(self.surv_dc, (self.clutter_cancel, 1))

        self.connect(self.ref_dc, self.ref_vec, self.ref_win, self.ref_fft)
        self.connect(self.clutter_cancel, self.surv_vec, self.surv_win, self.surv_fft)

        self.connect((self.ref_fft, 0), (self.mult_conj, 0))
        self.connect((self.surv_fft, 0), (self.mult_conj, 1))
        self.connect(self.mult_conj, self.ifft)

        self.connect(self.ifft, self.corr_mag)
        self.connect(self.corr_mag, self.throttle)
        self.connect(self.throttle, self.slow_stv)
        self.connect(self.slow_stv, self.rd_raster)

        self.connect(self.ifft, self.doppler_proc)

        self.connect(self.ref_dc, (self.ref_psd, 0))
        self.connect(self.surv_dc, (self.surv_psd, 0))

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
