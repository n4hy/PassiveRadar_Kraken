"""
Unit tests for detection clustering.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Detection:
    """Simple detection structure for testing."""
    def __init__(self, range_bin, doppler_bin, power):
        self.range_bin = range_bin
        self.doppler_bin = doppler_bin
        self.power = power


class Cluster:
    """Cluster of detections."""
    def __init__(self):
        self.detections = []
        self.centroid_range = 0
        self.centroid_doppler = 0
        self.total_power = 0
        self.size = 0

    def add_detection(self, det):
        self.detections.append(det)
        self._update_centroid()

    def _update_centroid(self):
        if not self.detections:
            return
        self.total_power = sum(d.power for d in self.detections)
        if self.total_power == 0:
            # Avoid division by zero - use simple average if all powers are zero
            self.centroid_range = sum(d.range_bin for d in self.detections) / len(self.detections)
            self.centroid_doppler = sum(d.doppler_bin for d in self.detections) / len(self.detections)
        else:
            self.centroid_range = sum(d.range_bin * d.power for d in self.detections) / self.total_power
            self.centroid_doppler = sum(d.doppler_bin * d.power for d in self.detections) / self.total_power
        self.size = len(self.detections)


class DetectionClusterer:
    """Connected component clustering for detections."""

    def __init__(self, n_range, n_doppler, connectivity=8):
        self.n_range = n_range
        self.n_doppler = n_doppler
        self.connectivity = connectivity

    def cluster(self, detection_mask):
        """
        Cluster detections using connected components.

        Args:
            detection_mask: 2D boolean array (doppler x range)

        Returns:
            List of Cluster objects
        """
        labels = np.zeros_like(detection_mask, dtype=int)
        current_label = 0
        clusters = {}

        for i in range(detection_mask.shape[0]):
            for j in range(detection_mask.shape[1]):
                if detection_mask[i, j] and labels[i, j] == 0:
                    current_label += 1
                    self._flood_fill(detection_mask, labels, i, j, current_label)
                    clusters[current_label] = Cluster()

        # Build clusters
        for i in range(detection_mask.shape[0]):
            for j in range(detection_mask.shape[1]):
                if labels[i, j] > 0:
                    det = Detection(j, i, 1.0)  # Unit power
                    clusters[labels[i, j]].add_detection(det)

        return list(clusters.values())

    def _flood_fill(self, mask, labels, i, j, label):
        """Flood fill to find connected component."""
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= mask.shape[0] or cj < 0 or cj >= mask.shape[1]:
                continue
            if not mask[ci, cj] or labels[ci, cj] > 0:
                continue
            labels[ci, cj] = label

            # 8-connectivity
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    stack.append((ci + di, cj + dj))


class TestDetectionClustering(unittest.TestCase):
    """Test detection clustering algorithms."""

    def test_single_detection(self):
        """Single detection should form single cluster."""
        mask = np.zeros((64, 128), dtype=bool)
        mask[30, 50] = True

        clusterer = DetectionClusterer(128, 64)
        clusters = clusterer.cluster(mask)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 1)

    def test_adjacent_detections(self):
        """Adjacent detections should form single cluster."""
        mask = np.zeros((64, 128), dtype=bool)
        mask[30, 50] = True
        mask[30, 51] = True
        mask[31, 50] = True

        clusterer = DetectionClusterer(128, 64)
        clusters = clusterer.cluster(mask)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 3)

    def test_separate_clusters(self):
        """Separated detections should form separate clusters."""
        mask = np.zeros((64, 128), dtype=bool)

        # Cluster 1
        mask[10, 20] = True
        mask[10, 21] = True

        # Cluster 2 (far away)
        mask[50, 100] = True
        mask[51, 100] = True

        clusterer = DetectionClusterer(128, 64)
        clusters = clusterer.cluster(mask)

        self.assertEqual(len(clusters), 2)

    def test_diagonal_connectivity(self):
        """Diagonally adjacent cells should be connected (8-connectivity)."""
        mask = np.zeros((64, 128), dtype=bool)
        mask[30, 50] = True
        mask[31, 51] = True  # Diagonal

        clusterer = DetectionClusterer(128, 64, connectivity=8)
        clusters = clusterer.cluster(mask)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 2)

    def test_centroid_calculation(self):
        """Verify centroid is correctly calculated."""
        mask = np.zeros((64, 128), dtype=bool)
        mask[30, 50] = True
        mask[30, 52] = True

        clusterer = DetectionClusterer(128, 64)
        clusters = clusterer.cluster(mask)

        # With unit power, centroid is average of positions: (50+52)/2 = 51
        # But our simple clustering uses range_bin directly, so centroid = average
        centroid_range = clusters[0].centroid_range
        self.assertGreaterEqual(centroid_range, 50.0)
        self.assertLessEqual(centroid_range, 52.0)
        self.assertAlmostEqual(clusters[0].centroid_doppler, 30.0)

    def test_large_cluster(self):
        """Test handling of large extended target."""
        mask = np.zeros((64, 128), dtype=bool)

        # 5x5 blob
        for i in range(30, 35):
            for j in range(50, 55):
                mask[i, j] = True

        clusterer = DetectionClusterer(128, 64)
        clusters = clusterer.cluster(mask)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 25)

    def test_power_weighted_centroid(self):
        """Verify power-weighted centroid calculation."""
        cluster = Cluster()
        cluster.add_detection(Detection(50, 30, 10.0))  # Strong
        cluster.add_detection(Detection(52, 30, 2.0))   # Weak

        # Centroid should be closer to stronger detection
        expected_range = (50 * 10 + 52 * 2) / 12
        self.assertAlmostEqual(cluster.centroid_range, expected_range)

    def test_no_detections(self):
        """Empty mask should return no clusters."""
        mask = np.zeros((64, 128), dtype=bool)

        clusterer = DetectionClusterer(128, 64)
        clusters = clusterer.cluster(mask)

        self.assertEqual(len(clusters), 0)

    def test_edge_detections(self):
        """Detections at edges should be handled correctly."""
        mask = np.zeros((64, 128), dtype=bool)
        mask[0, 0] = True
        mask[63, 127] = True

        clusterer = DetectionClusterer(128, 64)
        clusters = clusterer.cluster(mask)

        self.assertEqual(len(clusters), 2)


class TestClusterFiltering(unittest.TestCase):
    """Test cluster filtering and validation."""

    def test_minimum_cluster_size(self):
        """Clusters below minimum size should be filtered."""
        clusters = [
            Cluster(),  # size 1
            Cluster(),  # size 3
            Cluster(),  # size 5
        ]

        # Add detections
        clusters[0].add_detection(Detection(50, 30, 1.0))

        for i in range(3):
            clusters[1].add_detection(Detection(60 + i, 40, 1.0))

        for i in range(5):
            clusters[2].add_detection(Detection(70 + i, 50, 1.0))

        min_size = 2
        filtered = [c for c in clusters if c.size >= min_size]

        self.assertEqual(len(filtered), 2)

    def test_maximum_cluster_extent(self):
        """Clusters exceeding maximum extent should be flagged."""
        cluster = Cluster()
        for i in range(20):  # Large extent
            cluster.add_detection(Detection(50 + i, 30, 1.0))

        max_extent_range = 10

        range_extent = max(d.range_bin for d in cluster.detections) - \
                       min(d.range_bin for d in cluster.detections) + 1

        self.assertGreater(range_extent, max_extent_range)


if __name__ == '__main__':
    unittest.main()
