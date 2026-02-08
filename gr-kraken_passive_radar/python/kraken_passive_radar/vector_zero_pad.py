import numpy as np
from gnuradio import gr


class vector_zero_pad(gr.sync_block):
    """
    Zero-pad input vector to larger output vector size.
    Used to enable power-of-2 FFT sizes with arbitrary CPI lengths.
    """
    def __init__(self, in_size=1024, out_size=2048):
        gr.sync_block.__init__(
            self,
            name='Vector Zero Pad',
            in_sig=[(np.complex64, in_size)],
            out_sig=[(np.complex64, out_size)]
        )
        self.in_size = in_size
        self.out_size = out_size

    def work(self, input_items, output_items):
        for i in range(len(input_items[0])):
            output_items[0][i][:self.in_size] = input_items[0][i]
            output_items[0][i][self.in_size:] = 0
        return len(input_items[0])
