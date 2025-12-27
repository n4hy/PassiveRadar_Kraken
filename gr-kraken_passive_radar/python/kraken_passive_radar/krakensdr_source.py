import numpy as np
from gnuradio import gr
import osmosdr

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
        # "numchan=5" is the key argument for KrakenSDR
        self.osmosdr = osmosdr.source(args="numchan=5")

        self.osmosdr.set_sample_rate(self.sample_rate)
        self.osmosdr.set_center_freq(self.frequency)

        # Configure all 5 channels
        for i in range(5):
            self.osmosdr.set_gain(self.gain, i)
            self.osmosdr.set_freq_corr(0, i)
            self.osmosdr.set_dc_offset_mode(0, i)
            self.osmosdr.set_iq_balance_mode(0, i)

            # Connect the internal osmosdr ports to the hier block outputs
            self.connect((self.osmosdr, i), (self, i))

    def set_frequency(self, freq):
        self.frequency = freq
        self.osmosdr.set_center_freq(freq)

    def set_sample_rate(self, rate):
        self.sample_rate = rate
        self.osmosdr.set_sample_rate(rate)

    def set_gain(self, gain):
        self.gain = gain
        for i in range(5):
            self.osmosdr.set_gain(gain, i)
