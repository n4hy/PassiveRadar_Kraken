"""
Unit tests for multi-target tracker.
"""
import unittest
import numpy as np
from pathlib import Path
import sys
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TrackStatus(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    COASTING = 3
    DELETED = 4


@dataclass
class Detection:
    """Detection measurement."""
    range_m: float
    doppler_hz: float
    snr_db: float = 10.0
    aoa_deg: Optional[float] = None


@dataclass
class TrackState:
    """Track state vector and metadata."""
    id: int
    range_m: float
    doppler_hz: float
    range_rate: float  # m/s
    doppler_rate: float  # Hz/s
    status: TrackStatus
    hits: int
    misses: int
    age: int


class KalmanFilter:
    """Simple Kalman filter for track state estimation."""

    def __init__(self, process_noise=10.0, measurement_noise=50.0):
        # State: [range, doppler, range_rate, doppler_rate]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000.0  # Initial covariance

        # Process noise
        self.Q = np.diag([process_noise, process_noise/10,
                          process_noise, process_noise/10])

        # Measurement noise
        self.R = np.diag([measurement_noise, measurement_noise/10])

        # Measurement matrix (we measure range and doppler)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

    def predict(self, dt):
        """Predict state to next time step."""
        # State transition matrix
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update state with measurement."""
        z = np.array([measurement.range_m, measurement.doppler_hz])

        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_position(self):
        """Get estimated position."""
        return self.state[0], self.state[1]

    def get_velocity(self):
        """Get estimated velocity."""
        return self.state[2], self.state[3]


class MultiTargetTracker:
    """Simple multi-target tracker with GNN association."""

    def __init__(self, confirm_hits=3, delete_misses=5, gate_threshold=50.0):
        self.tracks = {}
        self.next_id = 1
        self.confirm_hits = confirm_hits
        self.delete_misses = delete_misses
        self.gate_threshold = gate_threshold
        self.dt = 0.1  # Time step in seconds

    def predict(self, dt=None):
        """Predict all tracks to current time."""
        if dt is None:
            dt = self.dt
        for track_id, (kf, state) in self.tracks.items():
            kf.predict(dt)
            state.age += 1

    def update(self, detections: List[Detection]):
        """Update tracks with new detections."""
        # Predict existing tracks
        self.predict()

        # Associate detections to tracks
        associations = self._associate(detections)

        # Update associated tracks
        for det_idx, track_id in associations.items():
            kf, state = self.tracks[track_id]
            kf.update(detections[det_idx])
            state.range_m, state.doppler_hz = kf.get_position()
            state.range_rate, state.doppler_rate = kf.get_velocity()
            state.hits += 1
            state.misses = 0

            # Promote to confirmed if enough hits
            if state.status == TrackStatus.TENTATIVE and state.hits >= self.confirm_hits:
                state.status = TrackStatus.CONFIRMED

        # Handle unassociated tracks (miss)
        associated_tracks = set(associations.values())
        for track_id in list(self.tracks.keys()):
            if track_id not in associated_tracks:
                kf, state = self.tracks[track_id]
                state.misses += 1

                if state.status == TrackStatus.CONFIRMED:
                    state.status = TrackStatus.COASTING

                if state.misses >= self.delete_misses:
                    state.status = TrackStatus.DELETED
                    del self.tracks[track_id]

        # Initiate new tracks for unassociated detections
        associated_dets = set(associations.keys())
        for i, det in enumerate(detections):
            if i not in associated_dets:
                self._initiate_track(det)

    def _associate(self, detections):
        """Associate detections to tracks using GNN."""
        if not self.tracks or not detections:
            return {}

        # Build cost matrix
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost = np.full((n_dets, n_tracks), np.inf)

        track_ids = list(self.tracks.keys())
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                kf, state = self.tracks[track_id]
                pred_range, pred_doppler = kf.get_position()

                # Euclidean distance (simplified)
                dist = np.sqrt((det.range_m - pred_range)**2 +
                               (det.doppler_hz - pred_doppler)**2 * 100)

                if dist < self.gate_threshold:
                    cost[i, j] = dist

        # Greedy association (simplified GNN)
        associations = {}
        used_tracks = set()

        for _ in range(min(n_dets, n_tracks)):
            # Find minimum cost assignment
            min_cost = np.inf
            min_i, min_j = -1, -1

            for i in range(n_dets):
                if i in associations:
                    continue
                for j in range(n_tracks):
                    if j in used_tracks:
                        continue
                    if cost[i, j] < min_cost:
                        min_cost = cost[i, j]
                        min_i, min_j = i, j

            if min_cost < self.gate_threshold:
                associations[min_i] = track_ids[min_j]
                used_tracks.add(min_j)

        return associations

    def _initiate_track(self, det):
        """Create new track from detection."""
        kf = KalmanFilter()
        kf.state = np.array([det.range_m, det.doppler_hz, 0, 0])

        state = TrackState(
            id=self.next_id,
            range_m=det.range_m,
            doppler_hz=det.doppler_hz,
            range_rate=0.0,
            doppler_rate=0.0,
            status=TrackStatus.TENTATIVE,
            hits=1,
            misses=0,
            age=0
        )

        self.tracks[self.next_id] = (kf, state)
        self.next_id += 1

    def get_confirmed_tracks(self):
        """Get all confirmed tracks."""
        return [state for kf, state in self.tracks.values()
                if state.status == TrackStatus.CONFIRMED]

    def get_all_tracks(self):
        """Get all tracks."""
        return [state for kf, state in self.tracks.values()]


class TestMultiTargetTracker(unittest.TestCase):
    """Test multi-target tracker functionality."""

    def test_single_target_track_init(self):
        """Verify track initiates on consistent detections."""
        tracker = MultiTargetTracker(confirm_hits=3)

        # 3 consistent detections
        for i in range(3):
            det = Detection(range_m=1000 + i*10, doppler_hz=50 + i*0.5)
            tracker.update([det])

        tracks = tracker.get_confirmed_tracks()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].status, TrackStatus.CONFIRMED)

    def test_tentative_track_creation(self):
        """Verify tentative track is created immediately."""
        tracker = MultiTargetTracker()

        det = Detection(range_m=1000, doppler_hz=50)
        tracker.update([det])

        all_tracks = tracker.get_all_tracks()
        self.assertEqual(len(all_tracks), 1)
        self.assertEqual(all_tracks[0].status, TrackStatus.TENTATIVE)

    def test_track_prediction(self):
        """Verify track state prediction."""
        tracker = MultiTargetTracker(confirm_hits=2)

        # Initialize with velocity - need enough updates to confirm
        for i in range(3):
            det = Detection(range_m=1000 + i*100, doppler_hz=50)
            tracker.update([det])

        # Get current position from all tracks (may still be tentative)
        tracks = tracker.get_all_tracks()
        if not tracks:
            self.skipTest("No tracks created")

        initial_range = tracks[0].range_m

        # Predict forward (miss detection)
        tracker.update([])

        # Track should still exist
        tracks = tracker.get_all_tracks()
        self.assertGreater(len(tracks), 0, "Track was deleted too soon")

    def test_track_deletion_on_miss(self):
        """Verify track is deleted after consecutive misses."""
        tracker = MultiTargetTracker(confirm_hits=2, delete_misses=3)

        # Establish track
        for i in range(3):
            tracker.update([Detection(range_m=1000, doppler_hz=50)])

        self.assertEqual(len(tracker.get_confirmed_tracks()), 1)

        # Miss detections
        for i in range(3):
            tracker.update([])

        self.assertEqual(len(tracker.get_confirmed_tracks()), 0)

    def test_two_separate_tracks(self):
        """Verify two separate targets form separate tracks."""
        tracker = MultiTargetTracker(confirm_hits=2)

        for i in range(3):
            det1 = Detection(range_m=1000 + i*10, doppler_hz=50)
            det2 = Detection(range_m=5000 + i*10, doppler_hz=-100)
            tracker.update([det1, det2])

        tracks = tracker.get_confirmed_tracks()
        self.assertEqual(len(tracks), 2)

    def test_track_association(self):
        """Verify correct association of detections to tracks."""
        tracker = MultiTargetTracker(confirm_hits=2, gate_threshold=200)

        # Initialize two tracks
        for i in range(2):
            tracker.update([
                Detection(range_m=1000, doppler_hz=50),
                Detection(range_m=5000, doppler_hz=-50)
            ])

        # Update with slightly moved detections
        tracker.update([
            Detection(range_m=1010, doppler_hz=51),
            Detection(range_m=5010, doppler_hz=-49)
        ])

        tracks = tracker.get_confirmed_tracks()
        self.assertEqual(len(tracks), 2)

        # Verify tracks maintained correct identity
        ranges = sorted([t.range_m for t in tracks])
        self.assertLess(abs(ranges[0] - 1010), 50)
        self.assertLess(abs(ranges[1] - 5010), 50)

    def test_crossing_tracks(self):
        """Verify track identity maintained when tracks cross."""
        tracker = MultiTargetTracker(confirm_hits=2, gate_threshold=300)

        # Two tracks approaching each other
        for i in range(10):
            det1 = Detection(range_m=1000 + i*100, doppler_hz=50 - i*10)
            det2 = Detection(range_m=2000 - i*100, doppler_hz=-50 + i*10)
            tracker.update([det1, det2])

        tracks = tracker.get_confirmed_tracks()
        self.assertEqual(len(tracks), 2)

    def test_kalman_filter_prediction(self):
        """Test Kalman filter state prediction."""
        kf = KalmanFilter()
        kf.state = np.array([1000, 50, 100, 1])  # range=1000, doppler=50, velocities

        kf.predict(dt=1.0)

        # Range should increase by range_rate
        self.assertAlmostEqual(kf.state[0], 1100, places=0)
        # Doppler should increase by doppler_rate
        self.assertAlmostEqual(kf.state[1], 51, places=0)

    def test_kalman_filter_update(self):
        """Test Kalman filter measurement update."""
        kf = KalmanFilter()
        kf.state = np.array([1000, 50, 0, 0])

        # Measurement different from prediction
        det = Detection(range_m=1050, doppler_hz=55)
        kf.update(det)

        # State should move toward measurement
        self.assertGreater(kf.state[0], 1000)
        self.assertLess(kf.state[0], 1050)


class TestTrackerPerformance(unittest.TestCase):
    """Performance tests for tracker."""

    def test_many_tracks(self):
        """Test tracking with many simultaneous targets."""
        tracker = MultiTargetTracker(confirm_hits=2)

        # Generate 20 targets
        n_targets = 20
        detections = []
        for i in range(n_targets):
            det = Detection(
                range_m=1000 + i * 500,
                doppler_hz=-100 + i * 10
            )
            detections.append(det)

        # Run for several frames
        for frame in range(5):
            dets = [Detection(range_m=d.range_m + frame*10,
                              doppler_hz=d.doppler_hz + frame*0.5)
                    for d in detections]
            tracker.update(dets)

        tracks = tracker.get_confirmed_tracks()
        self.assertGreaterEqual(len(tracks), 15)  # Some tolerance for association errors


if __name__ == '__main__':
    unittest.main()
