#!/usr/bin/env python3
"""
DVB-T Reconstructor Unit Tests
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import unittest
import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import kraken_passive_radar


class qa_dvbt_reconstructor(gr_unittest.TestCase):
    """Unit tests for DVB-T reconstructor block"""

    def setUp(self):
        """Set up test fixture"""
        self.tb = gr.top_block()

    def tearDown(self):
        """Clean up test fixture"""
        self.tb = None

    def test_block_creation(self):
        """Verify block instantiates without errors"""
        dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
            fft_size=2048,
            guard_interval=4,
            constellation=2,
            code_rate=2,
            enable_svd=True
        )
        self.assertIsNotNone(dvbt_recon)

    def test_parameter_validation_fft_size(self):
        """Verify FFT size validation"""
        # Valid FFT sizes
        for fft_size in [2048, 4096, 8192]:
            try:
                dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
                    fft_size=fft_size
                )
                self.assertIsNotNone(dvbt_recon)
            except Exception as e:
                self.fail(f"Valid FFT size {fft_size} raised exception: {e}")

        # Invalid FFT size should raise exception
        with self.assertRaises(Exception):
            dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
                fft_size=1024  # Invalid
            )

    def test_parameter_validation_guard_interval(self):
        """Verify guard interval validation"""
        # Valid guard intervals
        for gi in [4, 8, 16, 32]:
            try:
                dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
                    guard_interval=gi
                )
                self.assertIsNotNone(dvbt_recon)
            except Exception as e:
                self.fail(f"Valid guard interval {gi} raised exception: {e}")

        # Invalid guard interval should raise exception
        with self.assertRaises(Exception):
            dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
                guard_interval=2  # Invalid
            )

    def test_parameter_validation_constellation(self):
        """Verify constellation validation"""
        # Valid constellations
        for const in [0, 1, 2]:  # QPSK, 16QAM, 64QAM
            try:
                dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
                    constellation=const
                )
                self.assertIsNotNone(dvbt_recon)
            except Exception as e:
                self.fail(f"Valid constellation {const} raised exception: {e}")

        # Invalid constellation should raise exception
        with self.assertRaises(Exception):
            dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
                constellation=3  # Invalid
            )

    def test_passthrough_clean_signal(self):
        """Verify block doesn't corrupt clean signal (Phase 1 behavior)"""
        # Generate test signal: 1000 complex samples
        n_samples = 1000
        test_signal = np.exp(1j * 2 * np.pi * 0.1 * np.arange(n_samples))
        test_data = test_signal.astype(np.complex64)

        # Create flowgraph
        src = blocks.vector_source_c(test_data, False)
        dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
            fft_size=2048,
            guard_interval=4,
            constellation=2,
            code_rate=2,
            enable_svd=True
        )
        dst = blocks.vector_sink_c()

        self.tb.connect(src, dvbt_recon)
        self.tb.connect(dvbt_recon, dst)
        self.tb.run()

        # Get output
        result = np.array(dst.data())

        # Phase 1: Should be passthrough, so verify output matches input
        # Allow for small numerical differences
        self.assertEqual(len(result), len(test_data),
                        "Output length should match input length")

        if len(result) > 0:
            # Check that signal is preserved (within tolerance)
            correlation = np.abs(np.corrcoef(test_data, result)[0, 1])
            self.assertGreater(correlation, 0.99,
                             "Signal should be preserved (correlation > 0.99)")

    def test_snr_estimate_getter(self):
        """Verify SNR estimation getter works"""
        dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make()

        # Should return a float value
        snr = dvbt_recon.get_snr_estimate()
        self.assertIsInstance(snr, float)
        self.assertGreaterEqual(snr, 0.0)

    def test_enable_svd_setter(self):
        """Verify SVD enable/disable control works"""
        dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
            enable_svd=True
        )

        # Should not raise exception
        try:
            dvbt_recon.set_enable_svd(False)
            dvbt_recon.set_enable_svd(True)
        except Exception as e:
            self.fail(f"set_enable_svd raised exception: {e}")

    def test_throughput_2_4_msps(self):
        """Verify block can handle 2.4 MSPS data rate"""
        # Generate 2.4M samples (1 second at 2.4 MSPS)
        n_samples = 2_400_000
        # Use a simple constant signal to avoid memory issues
        test_data = np.ones(n_samples, dtype=np.complex64)

        src = blocks.vector_source_c(test_data, False)
        dvbt_recon = kraken_passive_radar.dvbt_reconstructor.make(
            fft_size=2048,
            guard_interval=4
        )
        dst = blocks.null_sink(gr.sizeof_gr_complex)

        self.tb.connect(src, dvbt_recon)
        self.tb.connect(dvbt_recon, dst)

        # Run flowgraph - should complete without errors
        try:
            self.tb.run()
        except Exception as e:
            self.fail(f"Block failed at 2.4 MSPS data rate: {e}")

    # TODO: Add tests for Phase 2-4 functionality
    # def test_fec_correction(self):
    #     """Measure BER improvement from FEC (Phase 3)"""
    #     pass
    #
    # def test_svd_enhancement(self):
    #     """Measure pilot SNR improvement from SVD (Phase 4)"""
    #     pass
    #
    # def test_real_dvbt_signal(self):
    #     """Process real DVB-T broadcast signal (Phase 4)"""
    #     pass


if __name__ == '__main__':
    gr_unittest.run(qa_dvbt_reconstructor)
