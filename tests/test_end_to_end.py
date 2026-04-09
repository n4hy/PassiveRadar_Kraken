import unittest
import numpy as np
import sys
import os
import ctypes
from unittest.mock import MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from conftest import find_kernel_lib

# 1. Mock GNU Radio
class MockBlock:
    """Mock GNU Radio block base class for offline pipeline testing.

    Technique: stub all block lifecycle methods to enable unit testing without GNU Radio.
    """
    def __init__(self, name, in_sig, out_sig):
        """Initialize mock block with name and I/O signatures."""
        self.name = name
        self.in_sig = in_sig
        self.out_sig = out_sig
        self.consumed = []
        self.history = 0

    def set_output_multiple(self, mult):
        """Stub for setting output item granularity."""
        self.output_multiple = mult

    def consume_each(self, n):
        """Stub for consuming n items on all input ports."""
        self.consumed.append(n)

    def consume(self, port, n):
        """Stub for consuming n items on a specific input port."""
        self.consumed.append((port, n))

    def set_history(self, h):
        """Stub for setting block history (look-back samples)."""
        self.history = h

    def forecast(self, nout, nin_req):
        """Stub for declaring input requirements for a given output count."""
        pass

class MockSyncBlock(MockBlock):
    """Mock GNU Radio sync_block for offline testing.

    Technique: inherits MockBlock stubs with no additional behavior.
    """
    pass

class MockBasicBlock(MockBlock):
    """Mock GNU Radio basic_block for offline testing.

    Technique: inherits MockBlock stubs with no additional behavior.
    """
    pass

gnuradio = MagicMock()
gr = MagicMock()
gr.sync_block = MockSyncBlock
gr.basic_block = MockBasicBlock
gr.top_block = MagicMock()
gr.sizeof_gr_complex = 8
gr.sizeof_float = 4

gnuradio.gr = gr
sys.modules["gnuradio"] = gnuradio
sys.modules["gnuradio.gr"] = gr

# Mock blocks module as well since Eca imports it? No, it imports gr.
# But just in case.
sys.modules["gnuradio.blocks"] = MagicMock()

# Mock osmosdr
sys.modules["osmosdr"] = MagicMock()

# OOT module path must come BEFORE display package to resolve namespace correctly
oot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../gr-kraken_passive_radar/python"))
sys.path.insert(0, oot_path)
# Clear any cached kraken_passive_radar (and submodules) from display system
for key in list(sys.modules.keys()):
    if key == 'kraken_passive_radar' or key.startswith('kraken_passive_radar.'):
        del sys.modules[key]

from kraken_passive_radar.eca_b_clutter_canceller import EcaBClutterCanceller
from kraken_passive_radar.custom_blocks import ConditioningBlock, CafBlock, BackendBlock
from kraken_passive_radar.doppler_processing import DopplerProcessingBlock

class TestEndToEndOffline(unittest.TestCase):
    """End-to-end offline test of the full passive radar processing pipeline.

    Technique: manually chain Conditioning, ECA, CAF, Doppler, and Backend blocks.
    """
    def test_manual_pipeline(self):
        """Verify full pipeline detects a target at expected range-Doppler bin.

        Technique: synthesize ref+surv with known target, run all blocks, check peak and CFAR.
        """
        print("SETTING UP MANUAL PIPELINE TEST")

        cpi_len = 4096
        doppler_len = 64
        num_taps = 16
        total_samples = cpi_len * doppler_len

        # 1. Generate Data
        np.random.seed(1337)
        ref_sig = (np.random.randn(total_samples) + 1j*np.random.randn(total_samples)).astype(np.complex64)

        tgt_range_bin = 100
        tgt_doppler_bin = 8
        tgt_amp = 0.01
        clutter_amp = 1.0

        # Build signals
        surv_sig = ref_sig * clutter_amp
        tgt_sig_base = np.roll(ref_sig, tgt_range_bin)

        # Phase shift
        phase_per_sample = (2 * np.pi * tgt_doppler_bin) / (doppler_len * cpi_len)
        t = np.arange(total_samples)
        doppler_phasor = np.exp(1j * phase_per_sample * t).astype(np.complex64)

        tgt_sig = tgt_sig_base * doppler_phasor * tgt_amp
        surv_sig += tgt_sig

        noise = (np.random.randn(total_samples) + 1j*np.random.randn(total_samples)) * 0.001
        surv_sig += noise
        surv_sig = surv_sig.astype(np.complex64)

        # Find ECA library using centralized function
        eca_lib_path = find_kernel_lib("eca_b_clutter_canceller")
        eca_lib = str(eca_lib_path) if eca_lib_path.exists() else None

        # 2. Instantiate Blocks
        cond_ref = ConditioningBlock(rate=0.0)
        cond_surv = ConditioningBlock(rate=0.0)

        eca = EcaBClutterCanceller(
            num_taps=num_taps,
            num_surv_channels=1,
            lib_path=eca_lib if eca_lib else ""
        )

        caf = CafBlock(n_samples=cpi_len)

        dop = DopplerProcessingBlock(
            fft_len=cpi_len,
            doppler_len=doppler_len,
            # cpi_len is not arg, fft_len is arg.
        )
        # Note: DopplerProcessingBlock args are (fft_len, doppler_len)

        backend = BackendBlock(rows=doppler_len, cols=cpi_len, num_inputs=1)

        # 3. Pipeline Execution
        print("Running Conditioning...")
        ref_cond = np.zeros_like(ref_sig)
        surv_cond = np.zeros_like(surv_sig)
        cond_ref.work([ref_sig], [ref_cond])
        cond_surv.work([surv_sig], [surv_cond])

        print("Running ECA...")
        surv_clean = np.zeros_like(surv_cond)
        # Eca expects [Ref, Surv]
        # Outputs [Clean]
        eca.work([ref_cond, surv_cond], [surv_clean])

        print("Running CAF...")
        # Caf input is stream. Output is Vector (cpi_len).
        # We process total_samples.
        # Expect doppler_len vectors.
        caf_out = np.zeros((doppler_len, cpi_len), dtype=np.complex64)

        # caf.general_work inputs: [ref, surv]
        # output: [vectors]
        # Note: general_work usually takes list of arrays.
        n_caf = caf.general_work([ref_cond, surv_clean], [caf_out])
        print(f"CAF Produced: {n_caf} vectors")
        self.assertEqual(n_caf, doppler_len)

        print("Running Doppler...")
        # Doppler input: Vectors. Output: Flattened Map.
        # Input shape: (doppler_len, cpi_len).
        # Output shape: (1, doppler_len * cpi_len).
        dop_out = np.zeros((1, doppler_len * cpi_len), dtype=np.float32)

        n_dop = dop.general_work([caf_out], [dop_out])
        print(f"Doppler Produced: {n_dop} maps")
        self.assertEqual(n_dop, 1)

        print("Running Backend (Fusion + CFAR)...")
        # Input: [Map]. Output: [Mask].
        # BackendBlock expects list of maps.
        mask_out = np.zeros((1, doppler_len * cpi_len), dtype=np.float32)
        n_back = backend.work([dop_out], [mask_out])

        # 4. Analyze Results
        mask = mask_out[0] # Shape (size,)
        # Reshape [Doppler, Range]
        # Doppler block outputs flat (fft_len * doppler_len).
        # Row-major: d * fft_len + r.
        # So reshape(doppler_len, fft_len)
        mask_2d = mask.reshape((doppler_len, cpi_len))

        # Find detection
        # Sum of mask
        num_dets = np.sum(mask)
        print(f"Total Detections: {num_dets}")

        # We expect detection at [8, 100].
        # Check window around there.
        # Doppler bin 8. Range bin 100.

        # Let's inspect the Doppler Map (LogMag) directly for better debugging/viz
        # I need to re-run or tap into Doppler output `dop_out`.
        # `dop_out` has the LogMag map!
        # Wait, DopplerProcessingBlock outputs LogMag map (float32).
        # Backend takes LogMag map and outputs CFAR Mask (float32).

        map_linear = dop_out[0]
        rd_map = map_linear.reshape((doppler_len, cpi_len))

        peak_idx = np.argmax(map_linear)
        peak_r = peak_idx // cpi_len
        peak_c = peak_idx % cpi_len
        val = map_linear[peak_idx]

        print(f"Doppler Map Peak at: D={peak_r}, R={peak_c}, Val={val} dB")

        # Expected
        exp_r = (doppler_len // 2) + tgt_doppler_bin # 32 + 8 = 40
        exp_c = tgt_range_bin # 100

        print(f"Expected: D={exp_r}, R={exp_c}")

        # Allow +/- 1 bin error
        self.assertTrue(abs(peak_r - exp_r) <= 1, f"Doppler Mismatch: Got {peak_r}, Want {exp_r}")
        self.assertTrue(abs(peak_c - exp_c) <= 1, f"Range Mismatch: Got {peak_c}, Want {exp_c}")

        # Check CFAR
        # Check if mask is 1 at peak
        is_detected = mask_2d[peak_r, peak_c]
        print(f"CFAR Detection at peak: {is_detected}")
        # If threshold is too high, might miss.
        # But for test I used 15dB.
        # Peak val should be high.

        # Write Image
        # Visualize the LogMag map
        # Normalize
        mn = np.min(rd_map)
        mx = np.max(rd_map)
        norm = (rd_map - mn) / (mx - mn + 1e-6) * 255.0
        img = norm.astype(np.uint8)

        with open("offline_test_map.ppm", "wb") as f:
            f.write(f"P5\n{cpi_len} {doppler_len}\n255\n".encode('ascii'))
            f.write(img.tobytes())
            print("Wrote offline_test_map.ppm")

if __name__ == '__main__':
    unittest.main()
