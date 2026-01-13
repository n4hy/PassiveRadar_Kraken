#!/usr/bin/env python3
import sys
import time
import signal
import numpy as np
from gnuradio import gr, blocks
import os

# Ensure we can load local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "gr-kraken_passive_radar/python"))

from kraken_passive_radar.krakensdr_source import krakensdr_source
from kraken_passive_radar.eca_b_clutter_canceller import EcaBClutterCanceller
from kraken_passive_radar.custom_blocks import ConditioningBlock, CafBlock, BackendBlock, TimeAlignmentBlock
from kraken_passive_radar.doppler_processing import DopplerProcessingBlock

class PassiveRadarTopBlock(gr.top_block):
    def __init__(self, freq=100e6, gain=30):
        gr.top_block.__init__(self, "Passive Radar Run")

        # Parameters
        self.freq = freq
        self.gain = gain
        self.cpi_len = 4096 # Fast time
        self.doppler_len = 64 # Slow time
        self.num_taps = 16

        # 1. Source
        # Note: krakensdr_source wrapper might expect GRC params.
        # It's a Hier Block usually? No, it's defined in python as gr.hier_block2.
        # But for direct instantiation, we might need to handle args.
        # Assuming standard signature.
        self.source = krakensdr_source(
            ip_addr="127.0.0.1", port=5000,
            center_freq=freq, gain=gain,
            num_channels=5
        )

        # 2. Conditioning (5 channels)
        self.cond_blocks = []
        for i in range(5):
            blk = ConditioningBlock(rate=1e-5)
            self.cond_blocks.append(blk)
            # Connect Source -> Cond
            self.connect((self.source, i), (blk, 0))

        # 2b. Time Alignment Probe (Calibration)
        # Connect Ref (Cond 0) and Surv (Cond 1-4) to Alignment probes.
        # We need 4 aligners (Ch0 vs Ch1, Ch0 vs Ch2...)
        self.align_blocks = []
        for i in range(4):
            blk = TimeAlignmentBlock(n_samples=self.cpi_len, interval_sec=2.0)
            self.align_blocks.append(blk)
            self.connect((self.cond_blocks[0], 0), (blk, 0))
            self.connect((self.cond_blocks[i+1], 0), (blk, 1))

        # 3. ECA (4 Surveillance channels)
        # Ref is Ch0. Surv is Ch1-4.
        # ECA Inputs: Ref, Surv1, Surv2, Surv3, Surv4
        self.eca = EcaBClutterCanceller(
            num_taps=self.num_taps,
            num_surv_channels=4,
            lib_path=os.path.abspath("src/libkraken_eca_b_clutter_canceller.so")
        )

        # Connect Ref (Cond 0) to ECA Ref (0)
        self.connect((self.cond_blocks[0], 0), (self.eca, 0))

        # Connect Surv (Cond 1-4) to ECA Surv (1-4)
        for i in range(4):
            self.connect((self.cond_blocks[i+1], 0), (self.eca, i+1))

        # 4. CAF (4 channels)
        # We need 4 CAF blocks.
        # CAF Inputs: Ref (Cond 0), Surv (ECA Out i)
        self.caf_blocks = []
        for i in range(4):
            # Note: CafBlock expects stream inputs and outputs vectors
            blk = CafBlock(n_samples=self.cpi_len)
            self.caf_blocks.append(blk)

            # Connect Ref
            self.connect((self.cond_blocks[0], 0), (blk, 0))
            # Connect Surv (ECA Out i)
            self.connect((self.eca, i), (blk, 1))

        # 5. Doppler (4 channels)
        # Input: Range Profile Vectors. Output: Map Vectors.
        self.doppler_blocks = []
        for i in range(4):
            blk = DopplerProcessingBlock(
                fft_len=self.cpi_len,
                doppler_len=self.doppler_len,
                cpi_len=self.cpi_len, # wrapper checks this?
                log_mag=True,
                lib_path=os.path.abspath("src/libkraken_doppler_processing.so")
            )
            self.doppler_blocks.append(blk)
            self.connect((self.caf_blocks[i], 0), (blk, 0))

        # 6. Backend (Fusion + CFAR)
        # Fusion of 4 maps.
        self.backend = BackendBlock(
            rows=self.doppler_len,
            cols=self.cpi_len,
            num_inputs=4
        )
        for i in range(4):
            self.connect((self.doppler_blocks[i], 0), (self.backend, i))

        # 7. Sink (Visualization)
        # Using a Vector Sink to poll data from Python main loop?
        # Or a file sink?
        # Let's use a ZeroMQ PUSH sink or just a Vector Sink for testing.
        # User wants "Display".
        # I'll output to a file that another script reads, or just print stats.
        # "Casual observer".
        # Maybe I print ASCII map here?
        # Vector Sink grows indefinitely. Bad.
        # Custom Sink that prints?
        self.sink = PrintSink(self.doppler_len, self.cpi_len)
        self.connect((self.backend, 0), (self.sink, 0))

class PrintSink(gr.sync_block):
    def __init__(self, rows, cols):
        gr.sync_block.__init__(
            self,
            name="Print Sink",
            in_sig=[(np.float32, rows*cols)],
            out_sig=None
        )
        self.rows = rows
        self.cols = cols
        self.ctr = 0

    def work(self, input_items, output_items):
        for item in input_items[0]:
            self.ctr += 1
            if self.ctr % 10 == 0:
                # Simple visualization: Count detections
                dets = np.sum(item)
                print(f"[Map {self.ctr}] Detections: {int(dets)}")
                if dets > 0:
                    # Print location of max
                    idx = np.argmax(item)
                    r = idx // self.cols
                    c = idx % self.cols
                    print(f"  Target at Doppler Bin {r}, Range Bin {c}")

                # Write PPM image for visualization (heatmap of last map)
                # Reshape to Rows x Cols
                # Normalize to 0-255
                # map_vals = item.reshape((self.rows, self.cols)) # Or is it flat? item is flat.
                # Just take first 128 range bins for visualization to keep image small?
                # or full map.

                # Normalize
                mx = np.max(item)
                mn = np.min(item)
                if mx > mn:
                    norm = (item - mn) / (mx - mn) * 255.0
                    norm = norm.astype(np.uint8)

                    # Write simple PPM
                    with open("passive_radar_map.ppm", "wb") as f:
                        f.write(f"P5\n{self.cols} {self.rows}\n255\n".encode('ascii'))
                        f.write(norm.tobytes())

        return len(input_items[0])

def main():
    tb = PassiveRadarTopBlock()

    def sig_handler(sig, frame):
        print("Stopping...")
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)

    print("Starting Passive Radar Flowgraph...")
    tb.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    tb.stop()
    tb.wait()

if __name__ == "__main__":
    main()
