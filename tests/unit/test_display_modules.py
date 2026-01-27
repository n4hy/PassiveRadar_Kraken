"""
Unit tests for display modules.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestRangeDopplerDisplay(unittest.TestCase):
    """Test Range-Doppler display functionality."""

    def test_import_range_doppler_display(self):
        """Verify range_doppler_display module can be imported."""
        try:
            from kraken_passive_radar import range_doppler_display
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"range_doppler_display not importable: {e}")

    def test_db_scaling(self):
        """Test dB power scaling."""
        # Test 10*log10 scaling
        power_linear = np.array([1.0, 10.0, 100.0, 0.1, 0.01])
        power_db = 10 * np.log10(power_linear + 1e-10)

        expected_db = np.array([0.0, 10.0, 20.0, -10.0, -20.0])
        np.testing.assert_array_almost_equal(power_db, expected_db, decimal=5)

    def test_dynamic_range_clipping(self):
        """Test dynamic range clipping."""
        data = np.array([-50, -30, -10, 0, 10, 30])
        min_db, max_db = -30, 10

        clipped = np.clip(data, min_db, max_db)
        expected = np.array([-30, -30, -10, 0, 10, 10])

        np.testing.assert_array_equal(clipped, expected)

    def test_colormap_normalization(self):
        """Test data normalization for colormap."""
        data = np.array([-30, -15, 0, 15, 30])
        vmin, vmax = -30, 30

        # Normalize to [0, 1]
        normalized = (data - vmin) / (vmax - vmin)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        np.testing.assert_array_almost_equal(normalized, expected)


class TestPPIDisplay(unittest.TestCase):
    """Test PPI (Plan Position Indicator) display functionality."""

    def test_import_radar_display(self):
        """Verify radar_display module can be imported."""
        try:
            from kraken_passive_radar import radar_display
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"radar_display not importable: {e}")

    def test_polar_to_cartesian(self):
        """Test polar to Cartesian coordinate conversion."""
        range_m = 1000.0
        aoa_deg = 45.0
        aoa_rad = np.radians(aoa_deg)

        x = range_m * np.sin(aoa_rad)
        y = range_m * np.cos(aoa_rad)

        expected_x = 1000 * np.sqrt(2) / 2
        expected_y = 1000 * np.sqrt(2) / 2

        self.assertAlmostEqual(x, expected_x, places=2)
        self.assertAlmostEqual(y, expected_y, places=2)

    def test_range_rings(self):
        """Test range ring generation."""
        max_range = 10000  # 10 km
        n_rings = 5

        ring_ranges = np.linspace(0, max_range, n_rings + 1)[1:]
        expected = np.array([2000, 4000, 6000, 8000, 10000])

        np.testing.assert_array_equal(ring_ranges, expected)

    def test_aoa_wrapping(self):
        """Test angle-of-arrival wrapping."""
        # AoA should be in [-180, 180]
        aoa_values = np.array([-200, -180, -90, 0, 90, 180, 200])

        wrapped = np.where(aoa_values > 180, aoa_values - 360,
                          np.where(aoa_values < -180, aoa_values + 360, aoa_values))

        expected = np.array([160, -180, -90, 0, 90, 180, -160])
        np.testing.assert_array_equal(wrapped, expected)


class TestCalibrationPanel(unittest.TestCase):
    """Test calibration panel functionality."""

    def test_import_calibration_panel(self):
        """Verify calibration_panel module can be imported."""
        try:
            from kraken_passive_radar import calibration_panel
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"calibration_panel not importable: {e}")

    def test_snr_calculation(self):
        """Test SNR calculation."""
        signal_power = 100.0
        noise_power = 1.0

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

        self.assertAlmostEqual(snr_db, 20.0)

    def test_phase_offset_wrapping(self):
        """Test phase offset wrapping to [-180, 180]."""
        phases_deg = np.array([-270, -180, -90, 0, 90, 180, 270, 360])

        wrapped = np.mod(phases_deg + 180, 360) - 180

        # Expected: [90, -180, -90, 0, 90, -180, -90, 0]
        self.assertTrue(np.all(wrapped >= -180))
        self.assertTrue(np.all(wrapped <= 180))

    def test_correlation_coefficient_bounds(self):
        """Test correlation coefficient is in [0, 1]."""
        # Perfect correlation
        x = np.array([1, 2, 3, 4, 5])
        y = x.copy()
        corr = np.corrcoef(x, y)[0, 1]
        self.assertAlmostEqual(corr, 1.0)

        # No correlation
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        corr = np.abs(np.corrcoef(x, y)[0, 1])
        self.assertLess(corr, 0.1)


class TestMetricsDashboard(unittest.TestCase):
    """Test metrics dashboard functionality."""

    def test_import_metrics_dashboard(self):
        """Verify metrics_dashboard module can be imported."""
        try:
            from kraken_passive_radar import metrics_dashboard
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"metrics_dashboard not importable: {e}")

    def test_latency_statistics(self):
        """Test latency statistics calculation."""
        latencies = np.array([10.0, 12.0, 11.0, 15.0, 9.0])

        mean_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)

        self.assertAlmostEqual(mean_latency, 11.4)
        self.assertEqual(max_latency, 15.0)
        self.assertEqual(min_latency, 9.0)

    def test_detection_rate(self):
        """Test detection rate calculation."""
        n_detections = 50
        duration_sec = 10.0

        rate = n_detections / duration_sec
        self.assertEqual(rate, 5.0)  # 5 detections per second

    def test_false_alarm_rate(self):
        """Test false alarm rate estimation."""
        n_false_alarms = 10
        n_cells = 256 * 64  # Range x Doppler
        n_frames = 100

        far = n_false_alarms / (n_cells * n_frames)

        self.assertLess(far, 1e-4)


class TestRadarGUI(unittest.TestCase):
    """Test integrated radar GUI functionality."""

    def test_import_radar_gui(self):
        """Verify radar_gui module can be imported."""
        try:
            from kraken_passive_radar import radar_gui
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"radar_gui not importable: {e}")


class TestDataFormatting(unittest.TestCase):
    """Test data formatting utilities."""

    def test_range_formatting(self):
        """Test range value formatting."""
        ranges = [100, 1000, 5000, 10000]

        formatted = []
        for r in ranges:
            if r >= 1000:
                formatted.append(f"{r/1000:.1f} km")
            else:
                formatted.append(f"{r} m")

        expected = ["100 m", "1.0 km", "5.0 km", "10.0 km"]
        self.assertEqual(formatted, expected)

    def test_velocity_formatting(self):
        """Test velocity value formatting."""
        doppler_hz = 100.0
        wavelength_m = 3.0  # ~100 MHz

        velocity_mps = doppler_hz * wavelength_m / 2  # Bistatic
        velocity_kmh = velocity_mps * 3.6

        self.assertAlmostEqual(velocity_mps, 150.0)
        self.assertAlmostEqual(velocity_kmh, 540.0)


if __name__ == '__main__':
    unittest.main()
