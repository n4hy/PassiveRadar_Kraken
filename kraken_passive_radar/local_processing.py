"""
Local Signal Processing Module for Enhanced Remote Radar Display.

Provides CFAR detection, clustering, and multi-target tracking without
GNU Radio dependency. Designed for processing delay-Doppler maps fetched
from remote KrakenSDR passive radar servers.

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import ctypes
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Library Discovery
# ---------------------------------------------------------------------------

def _find_kernel_lib(lib_name: str) -> Optional[Path]:
    """
    Find a kernel library, checking build directory first, then legacy paths.

    Args:
        lib_name: Library name without 'libkraken_' prefix and '.so' suffix.

    Returns:
        Path to the library file, or None if not found.
    """
    # Determine repo root relative to this file
    repo_root = Path(__file__).resolve().parents[1]
    full_name = f"libkraken_{lib_name}.so"

    # Check standard build directory first (build/lib/)
    build_lib = repo_root / "build" / "lib" / full_name
    if build_lib.exists():
        return build_lib

    # Check src/build/lib/ (in-source build of src/ only)
    src_build_lib = repo_root / "src" / "build" / "lib" / full_name
    if src_build_lib.exists():
        return src_build_lib

    # Legacy: check source directory (old in-source builds)
    src_lib = repo_root / "src" / full_name
    if src_lib.exists():
        return src_lib

    return None


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------

class TrackStatus(Enum):
    """Track lifecycle status."""
    TENTATIVE = 1   # New track, not yet confirmed
    CONFIRMED = 2   # Track has enough hits to be confirmed
    COASTING = 3    # Track missed recent detection, predicting forward
    DELETED = 4     # Track scheduled for deletion


@dataclass
class Detection:
    """A single radar detection with physical units."""
    range_m: float           # Bistatic range in meters
    doppler_hz: float        # Doppler shift in Hz
    snr_db: float = 0.0      # Signal-to-noise ratio in dB
    cluster_size: int = 1    # Number of cells in detection cluster


@dataclass
class Track:
    """A radar track with state and history."""
    id: int
    status: TrackStatus
    range_m: float           # Current estimated range (m)
    doppler_hz: float        # Current estimated Doppler (Hz)
    range_rate: float        # Range rate (m/s)
    doppler_rate: float      # Doppler rate (Hz/s)
    history: List[Tuple[float, float]] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    age: int = 0


# ---------------------------------------------------------------------------
#  CFAR Detector (ctypes wrapper)
# ---------------------------------------------------------------------------

class CfarDetector:
    """
    CFAR detector wrapping libkraken_backend.so cfar_2d function.

    Supports CA-CFAR (Cell Averaging), GO-CFAR (Greatest Of),
    and SO-CFAR (Smallest Of) variants.
    """

    def __init__(
        self,
        guard: int = 2,
        train: int = 4,
        threshold_db: float = 12.0,
        cfar_type: str = 'ca',
    ):
        """
        Initialize CFAR detector.

        Args:
            guard: Number of guard cells around cell under test.
            train: Number of training cells for noise estimation.
            threshold_db: Detection threshold above noise floor (dB).
            cfar_type: CFAR variant ('ca', 'go', 'so').
        """
        self.guard = guard
        self.train = train
        self.threshold_db = threshold_db
        self.cfar_type = cfar_type.lower()

        # Try to load the backend library
        self._lib = None
        self._cfar_2d = None

        lib_path = _find_kernel_lib("backend")
        if lib_path is not None and lib_path.exists():
            try:
                self._lib = ctypes.cdll.LoadLibrary(str(lib_path))
                self._cfar_2d = self._lib.cfar_2d
                self._cfar_2d.restype = None
                self._cfar_2d.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # input
                    ctypes.POINTER(ctypes.c_float),  # output
                    ctypes.c_int,                    # rows
                    ctypes.c_int,                    # cols
                    ctypes.c_int,                    # guard
                    ctypes.c_int,                    # train
                    ctypes.c_float,                  # threshold
                ]
            except OSError:
                pass  # Fall back to pure Python

    @property
    def is_native(self) -> bool:
        """Return True if using native C library, False for Python fallback."""
        return self._cfar_2d is not None

    def detect(self, power_db: np.ndarray) -> np.ndarray:
        """
        Run CFAR detection on power map.

        Args:
            power_db: 2D array of power values in dB, shape [n_doppler, n_range].

        Returns:
            Binary detection mask (1.0 = detection, 0.0 = no detection).
        """
        power_db = np.ascontiguousarray(power_db, dtype=np.float32)
        rows, cols = power_db.shape
        output = np.zeros_like(power_db)

        if self._cfar_2d is not None:
            # Use native library
            output = np.ascontiguousarray(output, dtype=np.float32)
            self._cfar_2d(
                power_db.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rows,
                cols,
                self.guard,
                self.train,
                self.threshold_db,
            )
        else:
            # Pure Python fallback (CA-CFAR)
            output = self._cfar_python(power_db)

        return output

    def _cfar_python(self, power_db: np.ndarray) -> np.ndarray:
        """Pure Python CA-CFAR implementation (fallback)."""
        rows, cols = power_db.shape
        output = np.zeros_like(power_db)
        window_size = self.guard + self.train

        for i in range(window_size, rows - window_size):
            for j in range(window_size, cols - window_size):
                # Extract training cells (exclude guard cells)
                cell_under_test = power_db[i, j]

                # Get training window excluding guard and CUT
                train_cells = []
                for di in range(-window_size, window_size + 1):
                    for dj in range(-window_size, window_size + 1):
                        # Skip guard cells and CUT
                        if abs(di) <= self.guard and abs(dj) <= self.guard:
                            continue
                        train_cells.append(power_db[i + di, j + dj])

                if not train_cells:
                    continue

                # Noise estimate based on CFAR type
                train_cells = np.array(train_cells)
                if self.cfar_type == 'ca':
                    noise_estimate = np.mean(train_cells)
                elif self.cfar_type == 'go':
                    # Greatest-of: use maximum of leading/lagging
                    mid = len(train_cells) // 2
                    noise_estimate = max(np.mean(train_cells[:mid]),
                                         np.mean(train_cells[mid:]))
                elif self.cfar_type == 'so':
                    # Smallest-of: use minimum of leading/lagging
                    mid = len(train_cells) // 2
                    noise_estimate = min(np.mean(train_cells[:mid]),
                                         np.mean(train_cells[mid:]))
                else:
                    noise_estimate = np.mean(train_cells)

                # Detection threshold
                threshold = noise_estimate + self.threshold_db
                if cell_under_test > threshold:
                    output[i, j] = 1.0

        return output


# ---------------------------------------------------------------------------
#  Detection Clusterer
# ---------------------------------------------------------------------------

class DetectionClusterer:
    """
    Clusters CFAR detections using connected component analysis.

    Uses scipy.ndimage.label for efficient 8-connectivity clustering,
    then computes power-weighted centroids for each cluster.
    """

    def __init__(
        self,
        range_res_m: float = 150.0,
        doppler_res_hz: float = 1.5,
        min_cluster_size: int = 1,
        max_cluster_size: int = 100,
    ):
        """
        Initialize clusterer.

        Args:
            range_res_m: Range resolution in meters per bin.
            doppler_res_hz: Doppler resolution in Hz per bin.
            min_cluster_size: Minimum cells for valid cluster.
            max_cluster_size: Maximum cells (filter extended clutter).
        """
        self.range_res_m = range_res_m
        self.doppler_res_hz = doppler_res_hz
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        # 8-connectivity structure
        self._structure = np.ones((3, 3), dtype=np.int32)

    def cluster(
        self,
        detection_mask: np.ndarray,
        power_db: np.ndarray,
        range_axis_m: np.ndarray,
        doppler_axis_hz: np.ndarray,
    ) -> List[Detection]:
        """
        Extract detections from CFAR mask with power-weighted centroids.

        Args:
            detection_mask: Binary mask from CFAR detector [n_doppler, n_range].
            power_db: Power map in dB (same shape as mask).
            range_axis_m: Range axis in meters [n_range].
            doppler_axis_hz: Doppler axis in Hz [n_doppler].

        Returns:
            List of Detection objects with physical coordinates.
        """
        if not SCIPY_AVAILABLE:
            return self._cluster_fallback(
                detection_mask, power_db, range_axis_m, doppler_axis_hz
            )

        # Label connected components
        labels, num_features = ndimage.label(detection_mask > 0, structure=self._structure)

        if num_features == 0:
            return []

        detections = []

        for label_id in range(1, num_features + 1):
            # Get indices for this cluster
            cluster_mask = labels == label_id
            indices = np.argwhere(cluster_mask)
            cluster_size = len(indices)

            # Filter by size
            if cluster_size < self.min_cluster_size:
                continue
            if cluster_size > self.max_cluster_size:
                continue

            # Extract power values for this cluster (convert from dB to linear)
            powers_db = power_db[cluster_mask]
            # Shift to positive for weighting (add offset so min is ~1)
            power_offset = powers_db - powers_db.min() + 1.0
            power_linear = 10.0 ** (power_offset / 10.0)
            total_power = np.sum(power_linear)

            if total_power == 0:
                continue

            # Power-weighted centroid in bin coordinates
            doppler_bins = indices[:, 0]
            range_bins = indices[:, 1]

            centroid_doppler_bin = np.sum(doppler_bins * power_linear) / total_power
            centroid_range_bin = np.sum(range_bins * power_linear) / total_power

            # Convert to physical units via interpolation
            range_m = np.interp(centroid_range_bin,
                                np.arange(len(range_axis_m)),
                                range_axis_m)
            doppler_hz = np.interp(centroid_doppler_bin,
                                   np.arange(len(doppler_axis_hz)),
                                   doppler_axis_hz)

            # SNR estimate: peak power minus median background
            peak_power = np.max(powers_db)
            snr_db = peak_power - np.median(power_db)

            detections.append(Detection(
                range_m=float(range_m),
                doppler_hz=float(doppler_hz),
                snr_db=float(snr_db),
                cluster_size=cluster_size,
            ))

        return detections

    def _cluster_fallback(
        self,
        detection_mask: np.ndarray,
        power_db: np.ndarray,
        range_axis_m: np.ndarray,
        doppler_axis_hz: np.ndarray,
    ) -> List[Detection]:
        """Fallback clustering without scipy (simple peak finding)."""
        detections = []
        mask = detection_mask > 0

        # Find local maxima in detection mask
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                if not mask[i, j]:
                    continue

                # Check if this is local maximum in power
                window = power_db[max(0, i-1):i+2, max(0, j-1):j+2]
                if power_db[i, j] < np.max(window):
                    continue

                # Count cluster size (simple 3x3)
                cluster_size = int(np.sum(mask[max(0, i-1):i+2, max(0, j-1):j+2]))

                if cluster_size < self.min_cluster_size:
                    continue
                if cluster_size > self.max_cluster_size:
                    continue

                range_m = range_axis_m[j] if j < len(range_axis_m) else 0
                doppler_hz = doppler_axis_hz[i] if i < len(doppler_axis_hz) else 0
                snr_db = power_db[i, j] - np.median(power_db)

                detections.append(Detection(
                    range_m=float(range_m),
                    doppler_hz=float(doppler_hz),
                    snr_db=float(snr_db),
                    cluster_size=cluster_size,
                ))

        return detections


# ---------------------------------------------------------------------------
#  Kalman Filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Kalman filter for single target tracking.

    State vector: [range, doppler, range_rate, doppler_rate]
    Measurement vector: [range, doppler]

    Uses constant velocity (CV) motion model.
    """

    def __init__(
        self,
        process_noise: float = 10.0,
        measurement_noise: float = 50.0,
    ):
        """
        Initialize Kalman filter.

        Args:
            process_noise: Process noise variance.
            measurement_noise: Measurement noise variance.
        """
        # State: [range, doppler, range_rate, doppler_rate]
        self.state = np.zeros(4, dtype=np.float64)

        # Covariance matrix
        self.P = np.eye(4, dtype=np.float64) * 1000.0

        # Process noise covariance
        self.Q = np.diag([
            process_noise,
            process_noise / 10.0,
            process_noise,
            process_noise / 10.0,
        ])

        # Measurement noise covariance
        self.R = np.diag([
            measurement_noise,
            measurement_noise / 10.0,
        ])

        # Measurement matrix (observe range and doppler)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

    def predict(self, dt: float) -> None:
        """
        Predict state to next time step.

        Args:
            dt: Time step in seconds.
        """
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: Detection) -> None:
        """
        Update state with measurement.

        Args:
            measurement: Detection with range_m and doppler_hz.
        """
        z = np.array([measurement.range_m, measurement.doppler_hz])

        # Innovation
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_position(self) -> Tuple[float, float]:
        """Get estimated position (range, doppler)."""
        return float(self.state[0]), float(self.state[1])

    def get_velocity(self) -> Tuple[float, float]:
        """Get estimated velocity (range_rate, doppler_rate)."""
        return float(self.state[2]), float(self.state[3])

    def mahalanobis_distance(self, measurement: Detection) -> float:
        """
        Compute Mahalanobis distance to measurement.

        Args:
            measurement: Detection to compute distance to.

        Returns:
            Mahalanobis distance (chi-squared distributed with 2 DOF).
        """
        z = np.array([measurement.range_m, measurement.doppler_hz])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        try:
            d2 = y @ np.linalg.inv(S) @ y
            return float(np.sqrt(d2))
        except np.linalg.LinAlgError:
            return float('inf')


# ---------------------------------------------------------------------------
#  Multi-Target Tracker
# ---------------------------------------------------------------------------

class MultiTargetTracker:
    """
    Multi-target tracker using Global Nearest Neighbor (GNN) association.

    Track lifecycle:
    - TENTATIVE: New track from unassociated detection
    - CONFIRMED: Track with confirm_hits consecutive associations
    - COASTING: Confirmed track missing associations (predicting forward)
    - DELETED: Track deleted after delete_misses consecutive misses
    """

    def __init__(
        self,
        dt: float = 1.0,
        confirm_hits: int = 3,
        delete_misses: int = 5,
        gate_threshold: float = 100.0,
        process_noise: float = 10.0,
        measurement_noise: float = 50.0,
        max_history: int = 50,
    ):
        """
        Initialize tracker.

        Args:
            dt: Frame period in seconds.
            confirm_hits: Hits required to confirm track.
            delete_misses: Misses to delete track.
            gate_threshold: Association gate (Mahalanobis distance).
            process_noise: Kalman filter process noise.
            measurement_noise: Kalman filter measurement noise.
            max_history: Maximum track history length.
        """
        self.dt = dt
        self.confirm_hits = confirm_hits
        self.delete_misses = delete_misses
        self.gate_threshold = gate_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_history = max_history

        self._lock = threading.Lock()  # Thread safety for multi-threaded access
        self._tracks: dict = {}  # track_id -> (KalmanFilter, Track)
        self._next_id = 1

    def update(self, detections: List[Detection]) -> None:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects from current frame.
        """
        with self._lock:
            self._update_internal(detections)

    def _update_internal(self, detections: List[Detection]) -> None:
        """Internal update implementation (caller must hold lock)."""
        # Predict all existing tracks
        self._predict_all()

        # Associate detections to tracks
        associations = self._associate(detections)

        # Update associated tracks
        associated_track_ids = set()
        for det_idx, track_id in associations.items():
            kf, track = self._tracks[track_id]
            kf.update(detections[det_idx])

            # Update track state
            track.range_m, track.doppler_hz = kf.get_position()
            track.range_rate, track.doppler_rate = kf.get_velocity()
            track.hits += 1
            track.misses = 0

            # Update history
            track.history.append((track.range_m, track.doppler_hz))
            if len(track.history) > self.max_history:
                track.history.pop(0)

            # Promote to confirmed
            if track.status == TrackStatus.TENTATIVE and track.hits >= self.confirm_hits:
                track.status = TrackStatus.CONFIRMED

            associated_track_ids.add(track_id)

        # Handle unassociated tracks (miss)
        for track_id in list(self._tracks.keys()):
            if track_id in associated_track_ids:
                continue

            kf, track = self._tracks[track_id]
            track.misses += 1

            # Update position from prediction
            track.range_m, track.doppler_hz = kf.get_position()
            track.range_rate, track.doppler_rate = kf.get_velocity()

            # Transition to coasting
            if track.status == TrackStatus.CONFIRMED:
                track.status = TrackStatus.COASTING

            # Delete if too many misses
            if track.misses >= self.delete_misses:
                track.status = TrackStatus.DELETED
                del self._tracks[track_id]

        # Initiate new tracks for unassociated detections
        associated_det_indices = set(associations.keys())
        for i, det in enumerate(detections):
            if i not in associated_det_indices:
                self._initiate_track(det)

    def _predict_all(self) -> None:
        """Predict all tracks to current time."""
        for track_id, (kf, track) in self._tracks.items():
            kf.predict(self.dt)
            track.age += 1

    def _associate(self, detections: List[Detection]) -> dict:
        """
        Associate detections to tracks using GNN.

        Returns:
            Dict mapping detection_index -> track_id.
        """
        if not self._tracks or not detections:
            return {}

        n_dets = len(detections)
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)

        # Build cost matrix (Mahalanobis distances)
        cost = np.full((n_dets, n_tracks), np.inf)

        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                kf, track = self._tracks[track_id]
                dist = kf.mahalanobis_distance(det)
                if dist < self.gate_threshold:
                    cost[i, j] = dist

        # Greedy GNN association (globally optimal for small problems)
        associations = {}
        used_tracks = set()

        for _ in range(min(n_dets, n_tracks)):
            # Find minimum cost
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

            if min_cost < self.gate_threshold and min_i >= 0:
                associations[min_i] = track_ids[min_j]
                used_tracks.add(min_j)

        return associations

    def _initiate_track(self, det: Detection) -> None:
        """Create new track from detection."""
        kf = KalmanFilter(
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        kf.state = np.array([det.range_m, det.doppler_hz, 0.0, 0.0])

        track = Track(
            id=self._next_id,
            status=TrackStatus.TENTATIVE,
            range_m=det.range_m,
            doppler_hz=det.doppler_hz,
            range_rate=0.0,
            doppler_rate=0.0,
            history=[(det.range_m, det.doppler_hz)],
            hits=1,
            misses=0,
            age=0,
        )

        self._tracks[self._next_id] = (kf, track)
        self._next_id += 1

    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks."""
        with self._lock:
            return [track for kf, track in self._tracks.values()
                    if track.status == TrackStatus.CONFIRMED]

    def get_all_tracks(self) -> List[Track]:
        """Get all active tracks (not deleted)."""
        with self._lock:
            return [track for kf, track in self._tracks.values()
                    if track.status != TrackStatus.DELETED]

    def get_coasting_tracks(self) -> List[Track]:
        """Get tracks in coasting mode."""
        with self._lock:
            return [track for kf, track in self._tracks.values()
                    if track.status == TrackStatus.COASTING]

    def reset(self) -> None:
        """Clear all tracks."""
        with self._lock:
            self._tracks.clear()
            self._next_id = 1


# ---------------------------------------------------------------------------
#  Local Processing Pipeline
# ---------------------------------------------------------------------------

class LocalProcessingPipeline:
    """
    Complete local processing pipeline: CFAR -> Clustering -> Tracking.

    Convenience class that combines all processing stages.
    """

    def __init__(
        self,
        cfar_guard: int = 2,
        cfar_train: int = 4,
        cfar_threshold_db: float = 12.0,
        cfar_type: str = 'ca',
        tracker_dt: float = 1.0,
        tracker_confirm: int = 3,
        tracker_delete: int = 5,
        tracker_gate: float = 100.0,
    ):
        """Initialize the processing pipeline."""
        self.cfar = CfarDetector(
            guard=cfar_guard,
            train=cfar_train,
            threshold_db=cfar_threshold_db,
            cfar_type=cfar_type,
        )

        self.clusterer = DetectionClusterer()

        self.tracker = MultiTargetTracker(
            dt=tracker_dt,
            confirm_hits=tracker_confirm,
            delete_misses=tracker_delete,
            gate_threshold=tracker_gate,
        )

    def process(
        self,
        power_db: np.ndarray,
        range_axis_m: np.ndarray,
        doppler_axis_hz: np.ndarray,
    ) -> Tuple[List[Detection], List[Track], np.ndarray]:
        """
        Run full processing pipeline on one frame.

        Args:
            power_db: 2D power map in dB [n_doppler, n_range].
            range_axis_m: Range axis in meters.
            doppler_axis_hz: Doppler axis in Hz.

        Returns:
            Tuple of (detections, tracks, cfar_mask).
        """
        # CFAR detection
        cfar_mask = self.cfar.detect(power_db)

        # Clustering
        detections = self.clusterer.cluster(
            cfar_mask, power_db, range_axis_m, doppler_axis_hz
        )

        # Tracking
        self.tracker.update(detections)
        tracks = self.tracker.get_all_tracks()

        return detections, tracks, cfar_mask

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker.reset()
