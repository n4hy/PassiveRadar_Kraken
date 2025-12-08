
from gnuradio import gr, blocks
import numpy as np
import sys

class DopplerProcessingBlock(gr.basic_block):
    """
    Custom Python block to perform slow-time FFT for Range-Doppler processing.
    Consumes doppler_len vectors of size fft_len (Range bins).
    Outputs a single Range-Doppler map (flattened) or triggers a GUI update via callback.
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
