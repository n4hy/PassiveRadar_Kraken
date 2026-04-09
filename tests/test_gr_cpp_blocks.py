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
    """Import real pybind11 blocks, bypassing any test mocks.

    Technique: purge MagicMock entries from sys.modules then attempt real import.
    """
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
    """Test ECA canceller C++ pybind11 block instantiation and parameter setters.

    Technique: verify block construction and runtime parameter changes via API.
    """
    def test_make_default(self):
        """Verify default ECA canceller construction succeeds."""
        blk = kpr.eca_canceller(num_taps=128, reg_factor=0.001, num_surv=4)
        self.assertIsNotNone(blk)

    def test_make_custom_params(self):
        """Verify ECA canceller construction with custom parameters."""
        blk = kpr.eca_canceller(num_taps=64, reg_factor=0.01, num_surv=2)
        self.assertIsNotNone(blk)

    def test_set_num_taps(self):
        """Verify runtime num_taps update does not crash."""
        blk = kpr.eca_canceller(num_taps=128, reg_factor=0.001, num_surv=4)
        blk.set_num_taps(256)

    def test_set_reg_factor(self):
        """Verify runtime regularization factor update does not crash."""
        blk = kpr.eca_canceller(num_taps=128, reg_factor=0.001, num_surv=4)
        blk.set_reg_factor(0.01)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestDopplerProcessor(unittest.TestCase):
    """Test Doppler processor C++ pybind11 block instantiation and parameters.

    Technique: verify construction with various window types and runtime parameter changes.
    """
    def test_make_default(self):
        """Verify default Doppler processor construction succeeds."""
        blk = kpr.doppler_processor(
            num_range_bins=256,
            num_doppler_bins=64
        )
        self.assertIsNotNone(blk)

    def test_make_custom_window(self):
        """Verify construction with each supported window type (rect, hamming, hann, blackman)."""
        for wtype in range(4):  # rect, hamming, hann, blackman
            blk = kpr.doppler_processor(
                num_range_bins=256,
                num_doppler_bins=64,
                window_type=wtype
            )
            self.assertIsNotNone(blk)

    def test_make_complex_output(self):
        """Verify construction with complex (non-power) output mode."""
        blk = kpr.doppler_processor(
            num_range_bins=256,
            num_doppler_bins=64,
            output_power=False
        )
        self.assertIsNotNone(blk)

    def test_set_num_doppler_bins(self):
        """Verify runtime Doppler bin count update does not crash."""
        blk = kpr.doppler_processor(
            num_range_bins=256,
            num_doppler_bins=64
        )
        blk.set_num_doppler_bins(128)

    def test_set_window_type(self):
        """Verify runtime window type update does not crash."""
        blk = kpr.doppler_processor(
            num_range_bins=256,
            num_doppler_bins=64
        )
        blk.set_window_type(2)  # hann


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestCfarDetector(unittest.TestCase):
    """Test CFAR detector C++ pybind11 block instantiation and parameters.

    Technique: verify construction with various CFAR types and runtime parameter changes.
    """
    def test_make_default(self):
        """Verify default CFAR detector construction succeeds."""
        blk = kpr.cfar_detector(
            num_range_bins=256,
            num_doppler_bins=64
        )
        self.assertIsNotNone(blk)

    def test_make_all_types(self):
        """Verify construction with all CFAR variant types (CA, GO, SO, OS)."""
        for cfar_type in range(4):  # CA, GO, SO, OS
            blk = kpr.cfar_detector(
                num_range_bins=256,
                num_doppler_bins=64,
                cfar_type=cfar_type,
                os_k=10 if cfar_type == 3 else 0
            )
            self.assertIsNotNone(blk)

    def test_make_custom_cells(self):
        """Verify construction with custom guard/reference cell sizes and Pfa."""
        blk = kpr.cfar_detector(
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
        """Verify runtime Pfa update does not crash."""
        blk = kpr.cfar_detector(num_range_bins=256, num_doppler_bins=64)
        blk.set_pfa(1e-3)

    def test_set_cfar_type(self):
        """Verify runtime CFAR type switch to GO-CFAR does not crash."""
        blk = kpr.cfar_detector(num_range_bins=256, num_doppler_bins=64)
        blk.set_cfar_type(1)  # GO-CFAR

    def test_set_guard_cells(self):
        """Verify runtime guard cell update does not crash."""
        blk = kpr.cfar_detector(num_range_bins=256, num_doppler_bins=64)
        blk.set_guard_cells(4, 4)

    def test_set_ref_cells(self):
        """Verify runtime reference cell update does not crash."""
        blk = kpr.cfar_detector(num_range_bins=256, num_doppler_bins=64)
        blk.set_ref_cells(16, 16)

    def test_get_num_detections_initial(self):
        """Verify initial detection count is zero before processing."""
        blk = kpr.cfar_detector(num_range_bins=256, num_doppler_bins=64)
        self.assertEqual(blk.get_num_detections(), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestCoherenceMonitor(unittest.TestCase):
    """Test coherence monitor C++ pybind11 block instantiation and parameters.

    Technique: verify construction and calibration state management.
    """
    def test_make_default(self):
        """Verify default coherence monitor construction succeeds."""
        blk = kpr.coherence_monitor()
        self.assertIsNotNone(blk)

    def test_make_custom(self):
        """Verify coherence monitor construction with custom parameters."""
        blk = kpr.coherence_monitor(
            num_channels=5,
            sample_rate=2.4e6,
            measure_interval_ms=500.0,
            measure_duration_ms=5.0,
            corr_threshold=0.9,
            phase_threshold_deg=10.0
        )
        self.assertIsNotNone(blk)

    def test_calibration_state(self):
        """Verify calibration needed flag returns a boolean."""
        blk = kpr.coherence_monitor()
        self.assertIsInstance(blk.is_calibration_needed(), bool)

    def test_set_measure_interval(self):
        """Verify runtime measurement interval update does not crash."""
        blk = kpr.coherence_monitor()
        blk.set_measure_interval(2000.0)

    def test_set_corr_threshold(self):
        """Verify runtime correlation threshold update does not crash."""
        blk = kpr.coherence_monitor()
        blk.set_corr_threshold(0.85)

    def test_set_phase_threshold(self):
        """Verify runtime phase threshold update does not crash."""
        blk = kpr.coherence_monitor()
        blk.set_phase_threshold(15.0)

    def test_manual_calibration(self):
        """Verify manual calibration request/acknowledge cycle works."""
        blk = kpr.coherence_monitor()
        blk.request_calibration()
        self.assertTrue(blk.is_calibration_needed())
        blk.acknowledge_calibration()


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestDetectionCluster(unittest.TestCase):
    """Test detection clustering C++ pybind11 block instantiation and parameters.

    Technique: verify construction and runtime parameter updates via API.
    """
    def test_make_default(self):
        """Verify default detection cluster construction succeeds."""
        blk = kpr.detection_cluster(
            num_range_bins=256,
            num_doppler_bins=64
        )
        self.assertIsNotNone(blk)

    def test_make_custom(self):
        """Verify detection cluster construction with custom parameters."""
        blk = kpr.detection_cluster(
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
        """Verify runtime minimum cluster size update does not crash."""
        blk = kpr.detection_cluster(num_range_bins=256, num_doppler_bins=64)
        blk.set_min_cluster_size(5)

    def test_set_max_cluster_extent(self):
        """Verify runtime max cluster extent update does not crash."""
        blk = kpr.detection_cluster(num_range_bins=256, num_doppler_bins=64)
        blk.set_max_cluster_extent(75)

    def test_set_range_resolution(self):
        """Verify runtime range resolution update does not crash."""
        blk = kpr.detection_cluster(num_range_bins=256, num_doppler_bins=64)
        blk.set_range_resolution(1200.0)

    def test_set_doppler_resolution(self):
        """Verify runtime Doppler resolution update does not crash."""
        blk = kpr.detection_cluster(num_range_bins=256, num_doppler_bins=64)
        blk.set_doppler_resolution(5.0)

    def test_get_num_detections_initial(self):
        """Verify initial detection count is zero before processing."""
        blk = kpr.detection_cluster(num_range_bins=256, num_doppler_bins=64)
        self.assertEqual(blk.get_num_detections(), 0)

    def test_get_detections_initial(self):
        """Verify initial detections list is empty before processing."""
        blk = kpr.detection_cluster(num_range_bins=256, num_doppler_bins=64)
        dets = blk.get_detections()
        self.assertEqual(len(dets), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestAoaEstimator(unittest.TestCase):
    """Test AoA estimator C++ pybind11 block instantiation and parameters.

    Technique: verify construction with ULA/UCA array types and spectrum retrieval.
    """
    def test_make_default(self):
        """Verify default AoA estimator construction succeeds."""
        blk = kpr.aoa_estimator.make()
        self.assertIsNotNone(blk)

    def test_make_custom_ula(self):
        """Verify AoA estimator construction with custom ULA parameters."""
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
        """Verify AoA estimator construction with UCA array type."""
        blk = kpr.aoa_estimator.make(
            num_elements=4,
            array_type=1  # UCA
        )
        self.assertIsNotNone(blk)

    def test_set_d_lambda(self):
        """Verify runtime element spacing update does not crash."""
        blk = kpr.aoa_estimator.make()
        blk.set_d_lambda(0.4)

    def test_set_scan_range(self):
        """Verify runtime scan range update does not crash."""
        blk = kpr.aoa_estimator.make()
        blk.set_scan_range(-45.0, 45.0)

    def test_set_array_type(self):
        """Verify runtime array type switch to UCA does not crash."""
        blk = kpr.aoa_estimator.make()
        blk.set_array_type(1)  # UCA

    def test_get_spectrum_initial(self):
        """Verify initial AoA spectrum has correct length."""
        blk = kpr.aoa_estimator.make(n_angles=181)
        spectrum = blk.get_spectrum()
        self.assertEqual(len(spectrum), 181)

    def test_get_aoa_results_initial(self):
        """Verify initial AoA results are empty before processing."""
        blk = kpr.aoa_estimator.make()
        results = blk.get_aoa_results()
        self.assertEqual(len(results), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestAoaAlgorithmEnum(unittest.TestCase):
    """Test AoA algorithm enum values exported from C++ pybind11.

    Technique: verify enum integer values match expected C++ definitions.
    """
    def test_bartlett_value(self):
        """Verify BARTLETT enum has value 0."""
        self.assertEqual(kpr.aoa_algorithm_t.BARTLETT.value, 0)

    def test_music_value(self):
        """Verify MUSIC enum has value 1."""
        self.assertEqual(kpr.aoa_algorithm_t.MUSIC.value, 1)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestAoaEstimatorMusic(unittest.TestCase):
    """Test MUSIC algorithm support in AoA estimator C++ pybind11 block.

    Technique: verify construction and parameter updates for MUSIC mode.
    """
    def test_make_music(self):
        """Verify AoA estimator construction with MUSIC algorithm."""
        blk = kpr.aoa_estimator.make(
            num_elements=4,
            algorithm=1,  # MUSIC
            n_sources=1,
            n_snapshots=16
        )
        self.assertIsNotNone(blk)

    def test_make_music_custom(self):
        """Verify MUSIC construction with custom sources and snapshots."""
        blk = kpr.aoa_estimator.make(
            num_elements=4,
            d_lambda=0.5,
            n_angles=361,
            algorithm=1,
            n_sources=2,
            n_snapshots=32
        )
        self.assertIsNotNone(blk)

    def test_set_algorithm(self):
        """Verify runtime algorithm switching between Bartlett and MUSIC."""
        blk = kpr.aoa_estimator.make()
        blk.set_algorithm(1)  # Switch to MUSIC
        blk.set_algorithm(0)  # Switch back to Bartlett

    def test_set_n_sources(self):
        """Verify runtime number-of-sources update does not crash."""
        blk = kpr.aoa_estimator.make(num_elements=4)
        blk.set_n_sources(2)

    def test_set_n_snapshots(self):
        """Verify runtime snapshot count update does not crash."""
        blk = kpr.aoa_estimator.make()
        blk.set_n_snapshots(32)

    def test_make_default_backward_compatible(self):
        """Ensure make() without new params still works (Bartlett default)."""
        blk = kpr.aoa_estimator.make(num_elements=4, d_lambda=0.5)
        self.assertIsNotNone(blk)
        spectrum = blk.get_spectrum()
        self.assertEqual(len(spectrum), 181)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestTracker(unittest.TestCase):
    """Test tracker C++ pybind11 block instantiation and parameters.

    Technique: verify construction, parameter setters, and initial state queries.
    """
    def test_make_default(self):
        """Verify default tracker construction succeeds."""
        blk = kpr.tracker()
        self.assertIsNotNone(blk)

    def test_make_custom(self):
        """Verify tracker construction with custom parameters."""
        blk = kpr.tracker(
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
        """Verify runtime process noise update does not crash."""
        blk = kpr.tracker()
        blk.set_process_noise(100.0, 10.0)

    def test_set_measurement_noise(self):
        """Verify runtime measurement noise update does not crash."""
        blk = kpr.tracker()
        blk.set_measurement_noise(200.0, 5.0)

    def test_set_gate_threshold(self):
        """Verify runtime gate threshold update does not crash."""
        blk = kpr.tracker()
        blk.set_gate_threshold(5.99)

    def test_set_confirm_hits(self):
        """Verify runtime confirm hits update does not crash."""
        blk = kpr.tracker()
        blk.set_confirm_hits(5)

    def test_set_delete_misses(self):
        """Verify runtime delete misses update does not crash."""
        blk = kpr.tracker()
        blk.set_delete_misses(10)

    def test_get_num_tracks_initial(self):
        """Verify initial total track count is zero."""
        blk = kpr.tracker()
        self.assertEqual(blk.get_num_tracks(), 0)

    def test_get_num_confirmed_initial(self):
        """Verify initial confirmed track count is zero."""
        blk = kpr.tracker()
        self.assertEqual(blk.get_num_confirmed_tracks(), 0)

    def test_get_tracks_initial(self):
        """Verify initial track list is empty."""
        blk = kpr.tracker()
        tracks = blk.get_tracks()
        self.assertEqual(len(tracks), 0)

    def test_get_confirmed_tracks_initial(self):
        """Verify initial confirmed tracks list is empty."""
        blk = kpr.tracker()
        tracks = blk.get_confirmed_tracks()
        self.assertEqual(len(tracks), 0)

    def test_reset(self):
        """Verify reset clears all tracks."""
        blk = kpr.tracker()
        blk.reset()
        self.assertEqual(blk.get_num_tracks(), 0)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestTrackStatusEnum(unittest.TestCase):
    """Test track status enum values exported from C++ pybind11.

    Technique: verify enum integer values match expected C++ definitions.
    """
    def test_tentative(self):
        """Verify TENTATIVE enum has value 0."""
        self.assertEqual(kpr.track_status_t.TENTATIVE.value, 0)

    def test_confirmed(self):
        """Verify CONFIRMED enum has value 1."""
        self.assertEqual(kpr.track_status_t.CONFIRMED.value, 1)

    def test_coasting(self):
        """Verify COASTING enum has value 2."""
        self.assertEqual(kpr.track_status_t.COASTING.value, 2)


@unittest.skipUnless(HAS_CPP_BLOCKS, "C++ pybind11 blocks not installed (run build_oot.sh)")
class TestTrackStruct(unittest.TestCase):
    """Test track_t struct exported from C++ pybind11.

    Technique: verify struct creation, field access, and convenience properties.
    """
    def test_create(self):
        """Verify track_t struct can be instantiated."""
        t = kpr.track_t()
        self.assertIsNotNone(t)

    def test_fields(self):
        """Verify all track_t fields are accessible without error."""
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
        """Verify track_t convenience properties (range, doppler, rates) are accessible."""
        t = kpr.track_t()
        _ = t.range_m
        _ = t.doppler_hz
        _ = t.range_rate
        _ = t.doppler_rate


if __name__ == "__main__":
    unittest.main()
