import numpy as np
from gnuradio import gr


class vector_zero_pad(gr.sync_block):
    """
    Zero-pad input vector to larger output vector size.
    Used to enable power-of-2 FFT sizes with arbitrary CPI lengths.
    """
    def __init__(self, in_size=1024, out_size=2048):
        """Initialize the zero-padding block with input and output vector sizes.

        Technique: Configures GNU Radio vector I/O signatures so that each
        input vector of in_size samples is copied into the first in_size
        elements of an out_size output vector, with the remainder zeroed.
        """
        gr.sync_block.__init__(
            self,
            name='Vector Zero Pad',
            in_sig=[(np.complex64, in_size)],
            out_sig=[(np.complex64, out_size)]
        )
        self.in_size = in_size
        self.out_size = out_size

    def work(self, input_items, output_items):
        """Copy each input vector into the output and zero-fill the remainder.

        Technique: Direct array slice assignment -- copies in_size samples from
        input to output, then sets the trailing (out_size - in_size) samples to zero.
        """
        for i in range(len(input_items[0])):
            output_items[0][i][:self.in_size] = input_items[0][i]
            output_items[0][i][self.in_size:] = 0
        return len(input_items[0])
