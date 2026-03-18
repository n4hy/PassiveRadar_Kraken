import numpy as np
from gnuradio import gr
import osmosdr
import sys
import ctypes

class krakensdr_source(gr.hier_block2):
    """
    KrakenSDR Source Block
    Wraps osmosdr.source configured for 5 coherent channels.
    """
    def __init__(self, frequency=100e6, sample_rate=2.4e6, gain=30.0):
        gr.hier_block2.__init__(
            self,
            "KrakenSDR Source",
            gr.io_signature(0, 0, 0),  # No inputs
            gr.io_signature(5, 5, gr.sizeof_gr_complex)  # 5 Complex outputs
        )

        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain

        # Create the underlying OSMOSDR source
        # "numchan=5" is the key argument for KrakenSDR.
        # We use explicit SERIAL NUMBERS (1000..1004) to ensure Channel 0 maps to Port 1, etc.
        # Attach buffer settings to EACH device string to avoid parsing errors.
        # buffers=128 (increased from 32) and buflen=65536 (64kB) to absorb USB latency.

        channel_args = []
        for i in range(5):
            channel_args.append(f"rtl={1000+i},buffers=128,buflen=65536")

        source_args = "numchan=5 " + " ".join(channel_args)

        self.osmosdr = osmosdr.source(args=source_args)

        self.osmosdr.set_sample_rate(self.sample_rate)

        # Configure all 5 channels
        for i in range(5):
            self.osmosdr.set_center_freq(self.frequency, i)
            # Explicitly enable Manual Gain mode so the set_gain() call is respected.
            # For RTL-SDR in gr-osmosdr: set_gain_mode(False) = Manual Gain.
            self.osmosdr.set_gain_mode(False, i)
            self.osmosdr.set_gain(self.gain, i)
            self.osmosdr.set_bandwidth(0, i) # 0 = Auto
            self.osmosdr.set_freq_corr(0, i)
            self.osmosdr.set_dc_offset_mode(0, i) # 0 = Off (Manual)
            self.osmosdr.set_iq_balance_mode(0, i) # 0 = Off (Manual)

            # Ensure proper clock source (internal) and antenna selection
            try:
                self.osmosdr.set_clock_source("internal", i)
            except (AttributeError, RuntimeError, NotImplementedError):
                # Some drivers (e.g., RTL-SDR) don't implement clock source control
                pass

            # Connect the internal osmosdr ports to the hier block outputs
            self.connect((self.osmosdr, i), (self, i))

    def set_frequency(self, freq):
        self.frequency = freq
        for i in range(5):
            self.osmosdr.set_center_freq(freq, i)

    def set_sample_rate(self, rate):
        self.sample_rate = rate
        self.osmosdr.set_sample_rate(rate)

    def set_gain(self, gain):
        self.gain = gain
        for i in range(5):
            self.osmosdr.set_gain(gain, i)

    def set_noise_source(self, enable):
        """
        Enables or disables the KrakenSDR internal noise source.

        The noise source is controlled via GPIO 0 on the device with serial 1000.
        gr-osmosdr does NOT support GPIO control, so we use librtlsdr directly
        via ctypes. librtlsdr can open a device by serial even while osmosdr
        holds the main data interface, because rtlsdr_set_bias_tee_gpio only
        needs a brief USB control transfer.

        IMPORTANT: When enabled, the hardware silicon switch DISCONNECTS all
        antennas and routes ONLY the internal noise source to all receivers.

        Args:
            enable (bool): True to enable, False to disable.
        """
        try:
            lib = ctypes.CDLL('librtlsdr.so.0')
            # Find device index by serial "1000"
            idx = lib.rtlsdr_get_index_by_serial(b"1000")
            if idx < 0:
                print("Warning: KrakenSDR SN 1000 not found for noise source control",
                      file=sys.stderr)
                return
            dev = ctypes.c_void_p()
            ret = lib.rtlsdr_open(ctypes.byref(dev), ctypes.c_uint32(idx))
            if ret != 0:
                print(f"Warning: Could not open rtlsdr device for noise source: error {ret}",
                      file=sys.stderr)
                return
            try:
                gpio = 0
                val = 1 if enable else 0
                ret = lib.rtlsdr_set_bias_tee_gpio(dev, ctypes.c_int(gpio), ctypes.c_int(val))
                if ret != 0:
                    print(f"Warning: rtlsdr_set_bias_tee_gpio returned {ret}",
                          file=sys.stderr)
            finally:
                lib.rtlsdr_close(dev)
        except OSError as e:
            print(f"Warning: librtlsdr.so.0 not available for noise source control: {e}",
                  file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to set noise source: {e}", file=sys.stderr)
