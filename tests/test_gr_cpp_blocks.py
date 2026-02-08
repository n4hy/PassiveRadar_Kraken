"""
Tests for GNU Radio C++ pybind11 blocks.

These tests verify that the compiled C++ blocks can be instantiated
and their APIs work correctly. They require the OOT module to be
built and installed (run build_oot.sh first).

Skipped automatically if the pybind11 module is not available.
"""

import unittest
import numpy as np
import sys
import os

# Try to import the compiled C++ blocks.
# Other test files (test_end_to_end, mock_gnuradio) inject MagicMock into
# sys.modules for gnuradio. We must purge those mocks before importing the
# real compiled modules.
from unittest.mock import MagicMock

def _try_import_real_blocks():
    """Import real pybind11 blocks, bypassing any test mocks."""
    # Remove mock gnuradio modules that other test files may have injected
    mock_keys = [k for k in sys.modules
                 if k.startswith('gnuradio') or k == 'osmosdr']
    for k in mock_keys:
        if isinstance(sys.modules[k], MagicMock):
            del sys.modules[k]

    try:
        from gnuradio import gr
        from gnuradio import kraken_passive_radar as kpr
        # Verify it's the real module, not a MagicMock
        if isinstance(kpr, MagicMock):
            return None
        if not hasattr(kpr, 'doppler_processor'):
            return None
        return kpr
    except ImportError:
        return None

kpr = _try_import_real_blocks()
HAS_CPP_BLOCKS = kpr is not None


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestEcaCanceller(unittest.TestCase):
    def test_make_default(self):
        blk = kpr.eca_canceller(num_taps=128, reg_factor=0.001, num_surv=4)
        self.assertIsNotNone(blk)

    def test_make_custom_params(self):
        blk = kpr.eca_canceller(num_taps=64, reg_factor=0.01, num_surv=2)
        self.assertIsNotNone(blk)

    def test_set_num_taps(self):
        blk = kpr.eca_canceller(num_taps=128, reg_factor=0.001, num_surv=4)
        blk.set_num_taps(256)

    def test_set_reg_factor(self):
        blk = kpr.eca_canceller(num_taps=128, reg_factor=0.001, num_surv=4)
        blk.set_reg_factor(0.01)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestDopplerProcessor(unittest.TestCase):
    def test_make_default(self):
        blk = kpr.doppler_processor.make(
            num_range_bins=256,
            num_doppler_bins=64
        )
        self.assertIsNotNone(blk)

    def test_make_custom_window(self):
        for wtype in range(4):  # rect, hamming, hann, blackman
            blk = kpr.doppler_processor.make(
                num_range_bins=256,
                num_doppler_bins=64,
                window_type=wtype
            )
            self.assertIsNotNone(blk)

    def test_make_complex_output(self):
        blk = kpr.doppler_processor.make(
            num_range_bins=256,
            num_doppler_bins=64,
            output_power=False
        )
        self.assertIsNotNone(blk)

    def test_set_num_doppler_bins(self):
        blk = kpr.doppler_processor.make(
            num_range_bins=256,
            num_doppler_bins=64
        )
        blk.set_num_doppler_bins(128)

    def test_set_window_type(self):
        blk = kpr.doppler_processor.make(
            num_range_bins=256,
            num_doppler_bins=64
        )
        blk.set_window_type(2)  # hann


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestCfarDetector(unittest.TestCase):
    def test_make_default(self):
        blk = kpr.cfar_detector.make(
            num_range_bins=256,
            num_doppler_bins=64
        )
        self.assertIsNotNone(blk)

    def test_make_all_types(self):
        for cfar_type in range(4):  # CA, GO, SO, OS
            blk = kpr.cfar_detector.make(
                num_range_bins=256,
                num_doppler_bins=64,
                cfar_type=cfar_type,
                os_k=10 if cfar_type == 3 else 0
            )
            self.assertIsNotNone(blk)

    def test_make_custom_cells(self):
        blk = kpr.cfar_detector.make(
            num_range_bins=512,
            num_doppler_bins=128,
            guard_cells_range=4,
            guard_cells_doppler=4,
            ref_cells_range=16,
            ref_cells_doppler=16,
            pfa=1e-4
        )
        self.assertIsNotNone(blk)

    def test_set_pfa(self):
        blk = kpr.cfar_detector.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_pfa(1e-3)

    def test_set_cfar_type(self):
        blk = kpr.cfar_detector.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_cfar_type(1)  # GO-CFAR

    def test_set_guard_cells(self):
        blk = kpr.cfar_detector.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_guard_cells(4, 4)

    def test_set_ref_cells(self):
        blk = kpr.cfar_detector.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_ref_cells(16, 16)

    def test_get_num_detections_initial(self):
        blk = kpr.cfar_detector.make(num_range_bins=256, num_doppler_bins=64)
        self.assertEqual(blk.get_num_detections(), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestCoherenceMonitor(unittest.TestCase):
    def test_make_default(self):
        blk = kpr.coherence_monitor.make()
        self.assertIsNotNone(blk)

    def test_make_custom(self):
        blk = kpr.coherence_monitor.make(
            num_channels=5,
            sample_rate=2.4e6,
            measure_interval_ms=500.0,
            measure_duration_ms=5.0,
            corr_threshold=0.9,
            phase_threshold_deg=10.0
        )
        self.assertIsNotNone(blk)

    def test_calibration_state(self):
        blk = kpr.coherence_monitor.make()
        self.assertIsInstance(blk.is_calibration_needed(), bool)

    def test_set_measure_interval(self):
        blk = kpr.coherence_monitor.make()
        blk.set_measure_interval(2000.0)

    def test_set_corr_threshold(self):
        blk = kpr.coherence_monitor.make()
        blk.set_corr_threshold(0.85)

    def test_set_phase_threshold(self):
        blk = kpr.coherence_monitor.make()
        blk.set_phase_threshold(15.0)

    def test_manual_calibration(self):
        blk = kpr.coherence_monitor.make()
        blk.request_calibration()
        self.assertTrue(blk.is_calibration_needed())
        blk.acknowledge_calibration()


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestDetectionCluster(unittest.TestCase):
    def test_make_default(self):
        blk = kpr.detection_cluster.make(
            num_range_bins=256,
            num_doppler_bins=64
        )
        self.assertIsNotNone(blk)

    def test_make_custom(self):
        blk = kpr.detection_cluster.make(
            num_range_bins=512,
            num_doppler_bins=128,
            min_cluster_size=3,
            max_cluster_extent=100,
            range_resolution_m=300.0,
            doppler_resolution_hz=2.0,
            max_detections=50
        )
        self.assertIsNotNone(blk)

    def test_set_min_cluster_size(self):
        blk = kpr.detection_cluster.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_min_cluster_size(5)

    def test_set_max_cluster_extent(self):
        blk = kpr.detection_cluster.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_max_cluster_extent(75)

    def test_set_range_resolution(self):
        blk = kpr.detection_cluster.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_range_resolution(1200.0)

    def test_set_doppler_resolution(self):
        blk = kpr.detection_cluster.make(num_range_bins=256, num_doppler_bins=64)
        blk.set_doppler_resolution(5.0)

    def test_get_num_detections_initial(self):
        blk = kpr.detection_cluster.make(num_range_bins=256, num_doppler_bins=64)
        self.assertEqual(blk.get_num_detections(), 0)

    def test_get_detections_initial(self):
        blk = kpr.detection_cluster.make(num_range_bins=256, num_doppler_bins=64)
        dets = blk.get_detections()
        self.assertEqual(len(dets), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestAoaEstimator(unittest.TestCase):
    def test_make_default(self):
        blk = kpr.aoa_estimator.make()
        self.assertIsNotNone(blk)

    def test_make_custom_ula(self):
        blk = kpr.aoa_estimator.make(
            num_elements=4,
            d_lambda=0.5,
            n_angles=361,
            min_angle_deg=-180.0,
            max_angle_deg=180.0,
            array_type=0,  # ULA
            num_range_bins=512,
            num_doppler_bins=128,
            max_detections=50
        )
        self.assertIsNotNone(blk)

    def test_make_uca(self):
        blk = kpr.aoa_estimator.make(
            num_elements=4,
            array_type=1  # UCA
        )
        self.assertIsNotNone(blk)

    def test_set_d_lambda(self):
        blk = kpr.aoa_estimator.make()
        blk.set_d_lambda(0.4)

    def test_set_scan_range(self):
        blk = kpr.aoa_estimator.make()
        blk.set_scan_range(-45.0, 45.0)

    def test_set_array_type(self):
        blk = kpr.aoa_estimator.make()
        blk.set_array_type(1)  # UCA

    def test_get_spectrum_initial(self):
        blk = kpr.aoa_estimator.make(n_angles=181)
        spectrum = blk.get_spectrum()
        self.assertEqual(len(spectrum), 181)

    def test_get_aoa_results_initial(self):
        blk = kpr.aoa_estimator.make()
        results = blk.get_aoa_results()
        self.assertEqual(len(results), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestTracker(unittest.TestCase):
    def test_make_default(self):
        blk = kpr.tracker.make()
        self.assertIsNotNone(blk)

    def test_make_custom(self):
        blk = kpr.tracker.make(
            dt=0.05,
            process_noise_range=25.0,
            process_noise_doppler=2.5,
            meas_noise_range=50.0,
            meas_noise_doppler=1.0,
            gate_threshold=6.63,
            confirm_hits=2,
            delete_misses=3,
            max_tracks=20,
            max_detections=50
        )
        self.assertIsNotNone(blk)

    def test_set_process_noise(self):
        blk = kpr.tracker.make()
        blk.set_process_noise(100.0, 10.0)

    def test_set_measurement_noise(self):
        blk = kpr.tracker.make()
        blk.set_measurement_noise(200.0, 5.0)

    def test_set_gate_threshold(self):
        blk = kpr.tracker.make()
        blk.set_gate_threshold(5.99)

    def test_set_confirm_hits(self):
        blk = kpr.tracker.make()
        blk.set_confirm_hits(5)

    def test_set_delete_misses(self):
        blk = kpr.tracker.make()
        blk.set_delete_misses(10)

    def test_get_num_tracks_initial(self):
        blk = kpr.tracker.make()
        self.assertEqual(blk.get_num_tracks(), 0)

    def test_get_num_confirmed_initial(self):
        blk = kpr.tracker.make()
        self.assertEqual(blk.get_num_confirmed_tracks(), 0)

    def test_get_tracks_initial(self):
        blk = kpr.tracker.make()
        tracks = blk.get_tracks()
        self.assertEqual(len(tracks), 0)

    def test_get_confirmed_tracks_initial(self):
        blk = kpr.tracker.make()
        tracks = blk.get_confirmed_tracks()
        self.assertEqual(len(tracks), 0)

    def test_reset(self):
        blk = kpr.tracker.make()
        blk.reset()
        self.assertEqual(blk.get_num_tracks(), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestTrackStatusEnum(unittest.TestCase):
    def test_tentative(self):
        self.assertEqual(kpr.track_status_t.TENTATIVE.value, 0)

    def test_confirmed(self):
        self.assertEqual(kpr.track_status_t.CONFIRMED.value, 1)

    def test_coasting(self):
        self.assertEqual(kpr.track_status_t.COASTING.value, 2)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestTrackStruct(unittest.TestCase):
    def test_create(self):
        t = kpr.track_t()
        self.assertIsNotNone(t)

    def test_fields(self):
        t = kpr.track_t()
        _ = t.id
        _ = t.status
        _ = t.state
        _ = t.covariance
        _ = t.hits
        _ = t.misses
        _ = t.age
        _ = t.score

    def test_convenience_properties(self):
        t = kpr.track_t()
        _ = t.range_m
        _ = t.doppler_hz
        _ = t.range_rate
        _ = t.doppler_rate


if __name__ == "__main__":
    unittest.main()
