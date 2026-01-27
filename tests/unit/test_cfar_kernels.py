"""
Unit tests for CFAR (Constant False Alarm Rate) detection.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestCFARDetection(unittest.TestCase):
    """Test CFAR detector performance."""

    def cfar_ca_1d(self, data, guard, ref_cells, pfa):
        """
        Reference implementation of Cell-Averaging CFAR.

        Args:
            data: 1D power data
            guard: Number of guard cells on each side
            ref_cells: Number of reference cells on each side
            pfa: Probability of false alarm

        Returns:
            Detection mask
        """
        n = len(data)
        detections = np.zeros(n, dtype=bool)

        # CFAR threshold multiplier for CA-CFAR
        n_ref = 2 * ref_cells
        alpha = n_ref * (pfa ** (-1/n_ref) - 1)

        for i in range(guard + ref_cells, n - guard - ref_cells):
            # Reference cells (excluding guard cells)
            left_ref = data[i - guard - ref_cells:i - guard]
            right_ref = data[i + guard + 1:i + guard + ref_cells + 1]
            ref_sum = np.sum(left_ref) + np.sum(right_ref)

            threshold = alpha * ref_sum / n_ref

            if data[i] > threshold:
                detections[i] = True

        return detections

    def cfar_2d(self, data, guard, ref_cells, pfa):
        """
        Reference implementation of 2D CA-CFAR.

        Args:
            data: 2D power data (range x doppler)
            guard: Guard cells in each dimension
            ref_cells: Reference cells in each dimension
            pfa: Probability of false alarm

        Returns:
            2D detection mask
        """
        n_rows, n_cols = data.shape
        detections = np.zeros_like(data, dtype=bool)

        margin = guard + ref_cells

        for i in range(margin, n_rows - margin):
            for j in range(margin, n_cols - margin):
                # Extract reference window
                window = data[i - margin:i + margin + 1, j - margin:j + margin + 1].copy()

                # Zero out guard region and CUT
                window[ref_cells:ref_cells + 2*guard + 1,
                       ref_cells:ref_cells + 2*guard + 1] = 0

                # Calculate threshold
                n_ref = (2*margin + 1)**2 - (2*guard + 1)**2
                ref_sum = np.sum(window)
                alpha = n_ref * (pfa ** (-1/n_ref) - 1)
                threshold = alpha * ref_sum / n_ref

                if data[i, j] > threshold:
                    detections[i, j] = True

        return detections

    def test_pfa_calibration_noise_only(self):
        """Verify Pfa matches design value in noise-only data."""
        np.random.seed(42)

        target_pfa = 1e-3  # Relaxed for testing
        n_trials = 100
        n_cells = 64

        guard = 2
        ref_cells = 4

        total_cells = 0
        total_detections = 0

        for _ in range(n_trials):
            # Exponential noise (Rayleigh envelope squared)
            noise = np.random.exponential(1.0, n_cells)

            det_mask = self.cfar_ca_1d(noise, guard, ref_cells, target_pfa)

            # Count only interior cells
            interior = slice(guard + ref_cells, n_cells - guard - ref_cells)
            total_cells += len(det_mask[interior])
            total_detections += np.sum(det_mask[interior])

        measured_pfa = total_detections / total_cells

        # Allow 3x tolerance on Pfa (statistical variation)
        self.assertLess(measured_pfa, 3 * target_pfa,
                        f"Pfa too high: {measured_pfa:.2e} vs target {target_pfa:.2e}")
        self.assertGreater(measured_pfa, target_pfa / 5,
                           f"Pfa too low: {measured_pfa:.2e} vs target {target_pfa:.2e}")

    def test_detection_of_strong_target(self):
        """Verify strong target is always detected."""
        np.random.seed(42)

        for snr_db in [15, 20, 30]:  # Start at 15dB for reliable detection
            # Noise background
            noise = np.random.exponential(1.0, 64)
            noise_mean = np.mean(noise)

            # Insert strong target (SNR relative to mean noise)
            target_idx = 32
            noise[target_idx] = noise_mean * 10**(snr_db/10)

            det_mask = self.cfar_ca_1d(noise, guard=2, ref_cells=4, pfa=1e-3)

            self.assertTrue(det_mask[target_idx],
                            f"Failed to detect target at SNR={snr_db}dB")

    def test_pd_increases_with_snr(self):
        """Verify detection probability increases with SNR."""
        np.random.seed(42)

        snr_values = [0, 5, 10, 15, 20]
        pd_measured = []

        for snr_db in snr_values:
            n_trials = 200
            detections = 0

            for _ in range(n_trials):
                noise = np.random.exponential(1.0, 64)
                target_idx = 32
                noise[target_idx] += 10**(snr_db/10)

                det_mask = self.cfar_ca_1d(noise, guard=2, ref_cells=4, pfa=1e-3)

                if det_mask[target_idx]:
                    detections += 1

            pd_measured.append(detections / n_trials)

        # Pd should be monotonically increasing
        for i in range(len(pd_measured) - 1):
            self.assertLessEqual(pd_measured[i], pd_measured[i+1] + 0.05,
                                 f"Pd not increasing: {pd_measured}")

        # At SNR=15dB, Pd should be >80%
        self.assertGreater(pd_measured[3], 0.8,
                           f"Pd too low at SNR=15dB: {pd_measured[3]}")

    def test_2d_cfar_single_target(self):
        """Verify 2D CFAR detects single target."""
        np.random.seed(42)

        # 2D noise field
        data = np.random.exponential(1.0, (64, 64))

        # Insert strong target
        data[30, 40] = 1000.0

        det_mask = self.cfar_2d(data, guard=2, ref_cells=3, pfa=1e-4)

        self.assertTrue(det_mask[30, 40], "Target not detected")

        # Count false alarms (excluding edge region)
        interior = det_mask[5:-5, 5:-5].copy()
        interior[25, 35] = False  # Remove target from count
        n_false = np.sum(interior)

        # Should have few false alarms
        self.assertLess(n_false, 10, f"Too many false alarms: {n_false}")

    def test_edge_handling(self):
        """Verify edge cells don't cause errors."""
        np.random.seed(42)

        data = np.random.exponential(1.0, 64)

        # Should not crash
        det_mask = self.cfar_ca_1d(data, guard=2, ref_cells=4, pfa=1e-4)

        self.assertEqual(len(det_mask), 64)
        self.assertFalse(np.any(np.isnan(det_mask)))

    def test_clustered_targets(self):
        """Verify detection of clustered targets."""
        np.random.seed(42)

        data = np.random.exponential(1.0, 64)

        # Two adjacent strong cells
        data[30] = 100.0
        data[31] = 80.0

        det_mask = self.cfar_ca_1d(data, guard=2, ref_cells=4, pfa=1e-4)

        # At least one should be detected
        self.assertTrue(det_mask[30] or det_mask[31],
                        "Neither clustered target detected")


class TestCFARVariants(unittest.TestCase):
    """Test different CFAR variants."""

    def cfar_go(self, data, guard, ref_cells, pfa):
        """Greatest-Of CFAR (better for clutter edges)."""
        n = len(data)
        detections = np.zeros(n, dtype=bool)

        n_ref = ref_cells
        alpha = n_ref * (pfa ** (-1/n_ref) - 1)

        for i in range(guard + ref_cells, n - guard - ref_cells):
            left_ref = data[i - guard - ref_cells:i - guard]
            right_ref = data[i + guard + 1:i + guard + ref_cells + 1]

            left_mean = np.mean(left_ref)
            right_mean = np.mean(right_ref)

            # Use greater of two estimates
            threshold = alpha * max(left_mean, right_mean)

            if data[i] > threshold:
                detections[i] = True

        return detections

    def cfar_so(self, data, guard, ref_cells, pfa):
        """Smallest-Of CFAR (better for multiple targets)."""
        n = len(data)
        detections = np.zeros(n, dtype=bool)

        n_ref = ref_cells
        alpha = n_ref * (pfa ** (-1/n_ref) - 1)

        for i in range(guard + ref_cells, n - guard - ref_cells):
            left_ref = data[i - guard - ref_cells:i - guard]
            right_ref = data[i + guard + 1:i + guard + ref_cells + 1]

            left_mean = np.mean(left_ref)
            right_mean = np.mean(right_ref)

            # Use smaller of two estimates
            threshold = alpha * min(left_mean, right_mean)

            if data[i] > threshold:
                detections[i] = True

        return detections

    def test_go_cfar_clutter_edge(self):
        """GO-CFAR should handle clutter edge better."""
        np.random.seed(42)

        # Clutter edge: low noise on left, high on right
        data = np.concatenate([
            np.random.exponential(1.0, 32),
            np.random.exponential(10.0, 32)
        ])

        det_go = self.cfar_go(data, guard=2, ref_cells=4, pfa=1e-4)
        det_so = self.cfar_so(data, guard=2, ref_cells=4, pfa=1e-4)

        # GO-CFAR should have fewer false alarms at edge
        edge_region = slice(28, 36)
        go_edge_fa = np.sum(det_go[edge_region])
        so_edge_fa = np.sum(det_so[edge_region])

        self.assertLessEqual(go_edge_fa, so_edge_fa,
                             "GO-CFAR should have fewer edge false alarms")

    def test_so_cfar_close_targets(self):
        """SO-CFAR should better detect close targets."""
        np.random.seed(42)

        data = np.random.exponential(1.0, 64)

        # Two targets close together
        data[28] = 50.0
        data[32] = 50.0

        det_go = self.cfar_go(data, guard=2, ref_cells=4, pfa=1e-4)
        det_so = self.cfar_so(data, guard=2, ref_cells=4, pfa=1e-4)

        go_detections = det_go[28] + det_go[32]
        so_detections = det_so[28] + det_so[32]

        self.assertGreaterEqual(so_detections, go_detections,
                                "SO-CFAR should detect more close targets")


if __name__ == '__main__':
    unittest.main()
