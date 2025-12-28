import numpy as np
from gnuradio import gr
import osmosdr
import sys

class krakensdr_source(gr.hier_block2):
    """
    KrakenSDR Source Block
    Wraps osmosdr.source configured for 5 coherent channels.
    """
    def __init__(self, frequency=100e6, sample_rate=2.048e6, gain=30.0):
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
        self.osmosdr = osmosdr.source(args="numchan=5 rtl=1000 rtl=1001 rtl=1002 rtl=1003 rtl=1004")

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
            except Exception:
                pass # Some drivers might not implement this for RTL

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
        The noise source is controlled via GPIO 0 on Device Index 1 (Serial 1001).
        Device Index 1 corresponds to Channel 1 (since Ch0=1000, Ch1=1001).

        Args:
            enable (bool): True to enable, False to disable.
        """
        val = 1 if enable else 0
        try:
            # set_gpio_bit(bank, bit, value, channel)
            # We assume Bank 0, Bit 0 is the noise source on Channel 1.
            self.osmosdr.set_gpio_bit(0, 0, val, 1)
        except AttributeError:
            print("Warning: osmosdr.source.set_gpio_bit not available. Noise source control failed.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to set noise source: {e}", file=sys.stderr)
