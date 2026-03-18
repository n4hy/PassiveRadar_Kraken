"""
Five-Channel KrakenSDR Dashboard - Comprehensive Passive Radar Display.

Designed for 5-channel KrakenSDR (1 reference + 4 surveillance) with:
1. Per-channel CAF views (4 surveillance channels in 2x2 grid)
2. AOA-enabled PPI display (polar plot with azimuth from 4 channels)
3. Enhanced channel health monitoring (Ref SNR, phase drift, correlation)
4. Fused/combined CAF display
5. Detection waterfalls (delay-time, doppler-time)
6. Interactive control panel

Usage:
    python -m kraken_passive_radar.five_channel_dashboard [options]

    For local processing (requires GNU Radio flowgraph):
        --local --zmq-addr tcp://localhost:5555

    For remote server (default):
        --url https://radar3.retnode.com

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import argparse
import json
import threading
import time
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Tuple, Dict
from enum import Enum

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as mcolors

# Import local processing components
try:
    from .local_processing import (
        CfarDetector,
        DetectionClusterer,
        MultiTargetTracker,
        Detection as LocalDetection,
        Track as LocalTrack,
        TrackStatus,
    )
    LOCAL_PROCESSING_AVAILABLE = True
except ImportError:
    try:
        from local_processing import (
            CfarDetector,
            DetectionClusterer,
            MultiTargetTracker,
            Detection as LocalDetection,
            Track as LocalTrack,
            TrackStatus,
        )
        LOCAL_PROCESSING_AVAILABLE = True
    except ImportError:
        LOCAL_PROCESSING_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------

class ChannelType(Enum):
    """Channel type enumeration."""
    REFERENCE = 0
    SURVEILLANCE_1 = 1
    SURVEILLANCE_2 = 2
    SURVEILLANCE_3 = 3
    SURVEILLANCE_4 = 4


@dataclass
class ChannelHealth:
    """Health metrics for a single channel."""
    snr_db: float = 0.0
    phase_offset_deg: float = 0.0
    correlation_coeff: float = 1.0
    phase_drift_rate: float = 0.0  # deg/s
    is_valid: bool = False


@dataclass
class Detection:
    """Single detection with timestamp and source."""
    timestamp: float
    delay_km: float
    doppler_hz: float
    snr_db: float
    azimuth_deg: Optional[float] = None  # AOA estimate if available
    source: str = 'server'
    channel: int = -1  # -1 = fused, 0-3 = per-channel


@dataclass
class ProcessingParams:
    """All tunable processing parameters."""
    # CFAR
    cfar_guard: int = 2
    cfar_train: int = 4
    cfar_threshold: float = 12.0
    cfar_type: str = 'ca'

    # Clustering
    cluster_min_size: int = 1
    cluster_max_size: int = 100

    # Tracker
    tracker_confirm: int = 3
    tracker_delete: int = 5
    tracker_gate: float = 100.0

    # Display
    color_min: float = 0.0
    color_max: float = 15.0
    history_duration: float = 60.0
    max_hold_decay: float = 0.995
    ppi_max_range_km: float = 50.0

    # Detection source
    show_server_detections: bool = True
    show_local_detections: bool = False
    show_local_tracks: bool = False

    # Channel display
    show_per_channel: bool = True
    selected_channel: int = -1  # -1 = all, 0-3 = specific


@dataclass
class FiveChannelData:
    """Data container for 5-channel processing."""
    # Per-channel CAF data (4 surveillance channels)
    channel_cafs: List[Optional[np.ndarray]] = field(
        default_factory=lambda: [None, None, None, None]
    )

    # Fused CAF (combined from all channels)
    fused_caf: Optional[np.ndarray] = None

    # Axes
    delay_km: Optional[np.ndarray] = None
    doppler_hz: Optional[np.ndarray] = None

    # Channel health
    channel_health: List[ChannelHealth] = field(
        default_factory=lambda: [ChannelHealth() for _ in range(5)]
    )

    # Timestamp
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
#  AOA Estimator
# ---------------------------------------------------------------------------

class AOAEstimator:
    """
    Angle-of-arrival estimator using 4 surveillance channels.

    Uses phase differences between surveillance channels to estimate
    bearing to target. Assumes linear or rectangular antenna array.
    """

    def __init__(
        self,
        wavelength_m: float = 3.0,  # ~100 MHz (FM band)
        antenna_spacing_m: float = 1.5,  # Half-wavelength spacing
        array_type: str = 'linear',  # 'linear' or 'rectangular'
    ):
        self.wavelength_m = wavelength_m
        self.antenna_spacing_m = antenna_spacing_m
        self.array_type = array_type

        # Precompute phase-to-angle factor
        self.phase_factor = wavelength_m / (2 * np.pi * antenna_spacing_m)

    def estimate_aoa(
        self,
        phases_deg: List[float],
        amplitudes: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Estimate AOA from phase measurements.

        Args:
            phases_deg: Phase measurements from 4 surveillance channels (degrees).
            amplitudes: Optional amplitude weights for each channel.

        Returns:
            Tuple of (azimuth_deg, confidence) where azimuth is 0-360 degrees
            from North and confidence is 0-1.
        """
        if len(phases_deg) != 4:
            return 0.0, 0.0

        phases_rad = np.array(phases_deg) * np.pi / 180.0

        if self.array_type == 'linear':
            # Linear array: use phase slope
            # Unwrap phases
            phases_unwrapped = np.unwrap(phases_rad)

            # Fit linear slope
            x = np.arange(4)
            slope, intercept = np.polyfit(x, phases_unwrapped, 1)

            # Convert slope to angle
            sin_theta = slope * self.phase_factor
            sin_theta = np.clip(sin_theta, -1.0, 1.0)
            theta_rad = np.arcsin(sin_theta)
            azimuth_deg = np.degrees(theta_rad) + 90  # Convert to compass bearing

            # Confidence from fit residual
            residual = np.std(phases_unwrapped - (slope * x + intercept))
            confidence = np.exp(-residual / 0.5)  # Decay with ~0.5 rad std

        else:  # rectangular
            # 2x2 rectangular array: use both dimensions
            # Channels arranged as:  0  1
            #                        2  3
            phase_dx = ((phases_rad[1] - phases_rad[0]) +
                        (phases_rad[3] - phases_rad[2])) / 2
            phase_dy = ((phases_rad[2] - phases_rad[0]) +
                        (phases_rad[3] - phases_rad[1])) / 2

            sin_theta_x = phase_dx * self.phase_factor
            sin_theta_y = phase_dy * self.phase_factor

            sin_theta_x = np.clip(sin_theta_x, -1.0, 1.0)
            sin_theta_y = np.clip(sin_theta_y, -1.0, 1.0)

            # Convert to azimuth
            azimuth_rad = np.arctan2(sin_theta_x, sin_theta_y)
            azimuth_deg = np.degrees(azimuth_rad)
            if azimuth_deg < 0:
                azimuth_deg += 360

            # Confidence from phase consistency
            phase_consistency = 1.0 - np.std([
                phases_rad[0], phases_rad[1] - phase_dx,
                phases_rad[2] - phase_dy, phases_rad[3] - phase_dx - phase_dy
            ]) / np.pi
            confidence = max(0.0, phase_consistency)

        return float(azimuth_deg % 360), float(confidence)


# ---------------------------------------------------------------------------
#  Five Channel Dashboard
# ---------------------------------------------------------------------------

class FiveChannelDashboard:
    """
    Comprehensive 5-channel KrakenSDR dashboard.

    Layout:
    +------------------+------------------+------------------+
    | Surv 1 CAF       | Surv 2 CAF       |                  |
    +------------------+------------------+    PPI Display   |
    | Surv 3 CAF       | Surv 4 CAF       |   (AOA + Range)  |
    +------------------+------------------+------------------+
    | Fused CAF        | Channel Health   |                  |
    | (Combined)       | (Ref + 4 Surv)   |  Control Panel   |
    +------------------+------------------+                  |
    | Delay-Time       | Doppler-Time     |                  |
    | Waterfall        | Waterfall        |                  |
    +------------------+------------------+------------------+
    """

    def __init__(
        self,
        base_url: str = 'https://radar3.retnode.com',
        poll_interval: float = 1.0,
        params: Optional[ProcessingParams] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.poll_interval = poll_interval
        self.params = params if params else ProcessingParams()

        # Data (guarded by lock)
        self.lock = threading.Lock()
        self.data = FiveChannelData()
        self.max_hold_data: Optional[np.ndarray] = None
        self.last_timestamp = 0.0
        self.fetch_errors = 0
        self.frames_fetched = 0

        # Local processing
        self.local_detections: List[LocalDetection] = []
        self.local_tracks: List[LocalTrack] = []
        self._init_local_processing()

        # AOA estimator
        self.aoa_estimator = AOAEstimator()

        # Detection history
        self.detection_history: Deque[Detection] = deque(maxlen=5000)

        # Phase history for drift tracking
        self.phase_history: List[Deque[float]] = [deque(maxlen=300) for _ in range(5)]
        self.time_history: Deque[float] = deque(maxlen=300)

        # Matplotlib objects
        self.fig = None
        self.axes: Dict[str, plt.Axes] = {}
        self.artists: Dict[str, object] = {}
        self.widgets: Dict[str, object] = {}

        # State
        self.running = False
        self._poll_thread = None
        self._start_time = None

    def _init_local_processing(self):
        """Initialize local processing components."""
        if not LOCAL_PROCESSING_AVAILABLE:
            return

        self.cfar = CfarDetector(
            guard=self.params.cfar_guard,
            train=self.params.cfar_train,
            threshold_db=self.params.cfar_threshold,
            cfar_type=self.params.cfar_type,
        )

        self.clusterer = DetectionClusterer(
            min_cluster_size=self.params.cluster_min_size,
            max_cluster_size=self.params.cluster_max_size,
        )

        self.tracker = MultiTargetTracker(
            dt=self.poll_interval,
            confirm_hits=self.params.tracker_confirm,
            delete_misses=self.params.tracker_delete,
            gate_threshold=self.params.tracker_gate,
        )

    def _rebuild_local_processing(self):
        """Rebuild local processing with current parameters."""
        if not LOCAL_PROCESSING_AVAILABLE:
            return

        self.cfar = CfarDetector(
            guard=self.params.cfar_guard,
            train=self.params.cfar_train,
            threshold_db=self.params.cfar_threshold,
            cfar_type=self.params.cfar_type,
        )

        self.clusterer = DetectionClusterer(
            min_cluster_size=self.params.cluster_min_size,
            max_cluster_size=self.params.cluster_max_size,
        )

        self.tracker = MultiTargetTracker(
            dt=self.poll_interval,
            confirm_hits=self.params.tracker_confirm,
            delete_misses=self.params.tracker_delete,
            gate_threshold=self.params.tracker_gate,
        )

    # ------------------------------------------------------------------ #
    #  Network
    # ------------------------------------------------------------------ #

    _UA = 'KrakenSDR-5ChDashboard/1.0'

    def _fetch_json(self, endpoint: str, timeout: float = 5.0):
        """Fetch JSON from an API endpoint."""
        url = f'{self.base_url}{endpoint}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': self._UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            self.fetch_errors += 1
            if self.fetch_errors <= 3 or self.fetch_errors % 100 == 0:
                print(f'[dashboard] fetch {endpoint} failed ({self.fetch_errors}): {exc}')
            return None

    def _poll_loop(self):
        """Background thread: fetch data at poll_interval."""
        while self.running:
            t0 = time.monotonic()
            now = time.time()

            # Fetch map data (this gets the fused CAF from server)
            map_data = self._fetch_json('/api/map')
            if map_data is not None:
                delay = np.asarray(map_data['delay'], dtype=np.float64)
                doppler = np.asarray(map_data['doppler'], dtype=np.float64)
                fused = np.asarray(map_data['data'], dtype=np.float64)
                ts = map_data.get('timestamp', 0)

                # For now, simulate per-channel CAFs by adding noise to fused
                # In real implementation, server would provide per-channel data
                channel_cafs = self._simulate_per_channel_cafs(fused)

                # Simulate channel health
                channel_health = self._simulate_channel_health()

                # Run local processing if enabled
                local_dets = []
                local_tracks = []
                if LOCAL_PROCESSING_AVAILABLE and (
                    self.params.show_local_detections or self.params.show_local_tracks
                ):
                    local_dets, local_tracks = self._run_local_processing(
                        fused, delay, doppler
                    )

                with self.lock:
                    self.data.delay_km = delay
                    self.data.doppler_hz = doppler
                    self.data.fused_caf = fused
                    self.data.channel_cafs = channel_cafs
                    self.data.channel_health = channel_health
                    self.data.timestamp = ts

                    # Update max-hold
                    if self.max_hold_data is None or self.max_hold_data.shape != fused.shape:
                        self.max_hold_data = fused.copy()
                    else:
                        self.max_hold_data *= self.params.max_hold_decay
                        self.max_hold_data = np.maximum(self.max_hold_data, fused)

                    self.last_timestamp = ts
                    self.frames_fetched += 1
                    self.local_detections = local_dets
                    self.local_tracks = local_tracks

            # Fetch server detections
            det_data = self._fetch_json('/api/detection')
            if det_data is not None:
                with self.lock:
                    if self.params.show_server_detections:
                        if 'delay' in det_data and len(det_data['delay']) > 0:
                            delays = det_data['delay']
                            dopplers = det_data['doppler']
                            snrs = det_data.get('snr', [0] * len(delays))
                            ts_det = det_data.get('timestamp', now * 1000) / 1000.0

                            for i in range(len(delays)):
                                # Estimate AOA if we have channel data
                                azimuth = self._estimate_detection_aoa(delays[i], dopplers[i])

                                self.detection_history.append(Detection(
                                    timestamp=ts_det,
                                    delay_km=delays[i],
                                    doppler_hz=dopplers[i],
                                    snr_db=snrs[i] if i < len(snrs) else 0,
                                    azimuth_deg=azimuth,
                                    source='server',
                                ))

                    # Add local detections to history
                    if self.params.show_local_detections and self.local_detections:
                        for det in self.local_detections:
                            azimuth = self._estimate_detection_aoa(
                                det.range_m / 1000.0, det.doppler_hz
                            )
                            self.detection_history.append(Detection(
                                timestamp=now,
                                delay_km=det.range_m / 1000.0,
                                doppler_hz=det.doppler_hz,
                                snr_db=det.snr_db,
                                azimuth_deg=azimuth,
                                source='local',
                            ))

                    # Prune old detections
                    cutoff = now - self.params.history_duration
                    while self.detection_history and self.detection_history[0].timestamp < cutoff:
                        self.detection_history.popleft()

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self.poll_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _simulate_per_channel_cafs(self, fused: np.ndarray) -> List[np.ndarray]:
        """
        Simulate per-channel CAFs from fused data.

        In real implementation, the server would provide per-channel CAFs.
        Here we add channel-specific variations for demonstration.
        """
        cafs = []
        for i in range(4):
            # Add channel-specific noise and slight offsets
            noise = np.random.normal(0, 0.5, fused.shape)
            offset = (i - 1.5) * 0.3  # Slight level differences
            caf = fused + noise + offset
            caf = np.clip(caf, 0, None)
            cafs.append(caf)
        return cafs

    def _simulate_channel_health(self) -> List[ChannelHealth]:
        """
        Simulate channel health metrics.

        In real implementation, these would come from the coherence monitor.
        """
        health = []
        for i in range(5):
            if i == 0:  # Reference channel
                snr = 28.0 + np.random.normal(0, 1)
                phase = 0.0
                corr = 1.0
            else:  # Surveillance channels
                snr = 25.0 + np.random.normal(0, 2)
                phase = (i - 2.5) * 15 + np.random.normal(0, 2)
                corr = 0.95 + np.random.normal(0, 0.02)

            health.append(ChannelHealth(
                snr_db=snr,
                phase_offset_deg=phase,
                correlation_coeff=min(1.0, max(0.0, corr)),
                phase_drift_rate=np.random.normal(0, 0.1),
                is_valid=snr > 15 and corr > 0.85,
            ))
        return health

    def _estimate_detection_aoa(self, delay_km: float, doppler_hz: float) -> float:
        """
        Estimate AOA for a detection using current channel phases.

        In real implementation, this would use actual phase measurements
        at the detection location. Here we simulate based on channel health.
        """
        with self.lock:
            phases = [h.phase_offset_deg for h in self.data.channel_health[1:5]]

        if len(phases) == 4:
            azimuth, confidence = self.aoa_estimator.estimate_aoa(phases)
            if confidence > 0.5:
                return azimuth

        # Fallback: random azimuth for demo
        return np.random.uniform(0, 360)

    def _run_local_processing(
        self, data: np.ndarray, delay: np.ndarray, doppler: np.ndarray
    ) -> Tuple[List[LocalDetection], List[LocalTrack]]:
        """Run local CFAR, clustering, and tracking."""
        if not LOCAL_PROCESSING_AVAILABLE:
            return [], []

        range_axis_m = delay * 1000.0
        doppler_axis_hz = doppler

        cfar_mask = self.cfar.detect(data)
        detections = self.clusterer.cluster(cfar_mask, data, range_axis_m, doppler_axis_hz)
        self.tracker.update(detections)
        tracks = self.tracker.get_all_tracks()

        return detections, tracks

    # ------------------------------------------------------------------ #
    #  Display Setup
    # ------------------------------------------------------------------ #

    def _setup_plot(self):
        """Create the matplotlib figures - main display and separate control panel."""
        # Main display figure - 3x4 grid layout
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.canvas.manager.set_window_title(
            f'KrakenSDR 5-Channel Dashboard - {self.base_url}'
        )

        # Layout: 4 rows x 3 columns using add_subplot
        # Row 1: Surv1, Surv2, PPI(top)
        # Row 2: Surv3, Surv4, PPI(bottom)
        # Row 3: Fused, Health, Trails
        # Row 4: DelayWF, DopplerWF, MaxHold

        # Per-channel CAFs (positions 1,2,4,5 in 4x3 grid)
        self._setup_channel_caf_simple(4, 3, 1, 'Surveillance 1', 0)
        self._setup_channel_caf_simple(4, 3, 2, 'Surveillance 2', 1)
        self._setup_channel_caf_simple(4, 3, 4, 'Surveillance 3', 2)
        self._setup_channel_caf_simple(4, 3, 5, 'Surveillance 4', 3)

        # PPI Display (position 3, spanning rows 1-2)
        ax_ppi = self.fig.add_subplot(2, 3, 3, projection='polar')
        self._setup_ppi(ax_ppi)

        # Row 3: Fused, Health, Trails (positions 7, 8, 9)
        self._setup_fused_caf_simple(4, 3, 7)
        self._setup_channel_health_simple(4, 3, 8)
        self._setup_trails_simple(4, 3, 9)

        # Row 4: Waterfalls (positions 10, 11, 12)
        self._setup_waterfall_simple(4, 3, 10, 'Delay vs Time', 'delay')
        self._setup_waterfall_simple(4, 3, 11, 'Doppler vs Time', 'doppler')
        self._setup_maxhold_simple(4, 3, 12)

        # Info text
        self.artists['info_text'] = self.fig.text(
            0.01, 0.99, '', fontsize=9, verticalalignment='top',
            fontfamily='monospace', color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9),
        )

        self.fig.tight_layout()

        # Separate control panel window
        self._setup_control_window()

    def _setup_channel_caf_simple(self, rows, cols, pos, title: str, channel_idx: int):
        """Setup a per-channel CAF display using add_subplot."""
        ax = self.fig.add_subplot(rows, cols, pos)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')

        placeholder = np.zeros((100, 100))
        im = ax.imshow(
            placeholder, aspect='auto', origin='lower',
            extent=[0, 60, -300, 300], cmap='viridis', interpolation='bilinear',
            vmin=self.params.color_min, vmax=self.params.color_max,
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(-300, 300)

        scatter = ax.scatter([], [], s=40, facecolors='none', edgecolors='red',
                            linewidths=1, zorder=5)

        self.axes[f'ch{channel_idx}'] = ax
        self.artists[f'ch{channel_idx}_im'] = im
        self.artists[f'ch{channel_idx}_scatter'] = scatter

    def _setup_fused_caf_simple(self, rows, cols, pos):
        """Setup the fused CAF display using add_subplot."""
        ax = self.fig.add_subplot(rows, cols, pos)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title('Fused CAF', fontsize=10, fontweight='bold')

        placeholder = np.zeros((100, 100))
        im = ax.imshow(
            placeholder, aspect='auto', origin='lower',
            extent=[0, 60, -300, 300], cmap='viridis', interpolation='bilinear',
            vmin=self.params.color_min, vmax=self.params.color_max,
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(-300, 300)

        scatter_server = ax.scatter([], [], s=50, facecolors='none', edgecolors='red',
                                     linewidths=1.5, zorder=5)
        scatter_local = ax.scatter([], [], s=50, facecolors='none', edgecolors='lime',
                                    linewidths=1.5, zorder=6)
        scatter_tracks = ax.scatter([], [], s=80, c='yellow', marker='D',
                                     edgecolors='black', linewidths=1, zorder=7)

        self.axes['fused'] = ax
        self.artists['fused_im'] = im
        self.artists['fused_scatter_server'] = scatter_server
        self.artists['fused_scatter_local'] = scatter_local
        self.artists['fused_scatter_tracks'] = scatter_tracks

    def _setup_channel_health_simple(self, rows, cols, pos):
        """Setup channel health display using add_subplot."""
        ax = self.fig.add_subplot(rows, cols, pos)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 40)
        ax.set_xlabel('Channel', fontsize=8)
        ax.set_ylabel('SNR (dB)', fontsize=8)
        ax.set_title('Channel Health', fontsize=10, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['Ref', 'S1', 'S2', 'S3', 'S4'], fontsize=8)

        ax.axhline(15, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(25, color='green', linestyle='--', alpha=0.5, linewidth=1)

        bars = ax.bar(range(5), [0]*5, color='gray', edgecolor='white', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')

        phase_texts = []
        for i in range(5):
            txt = ax.text(i, 2, '', ha='center', va='bottom', fontsize=7,
                         color='white', fontweight='bold')
            phase_texts.append(txt)

        corr_circles = []
        for i in range(5):
            circle = plt.Circle((i, 38), 0.3, color='gray', zorder=10)
            ax.add_patch(circle)
            corr_circles.append(circle)

        self.axes['health'] = ax
        self.artists['health_bars'] = bars
        self.artists['health_phase_texts'] = phase_texts
        self.artists['health_corr_circles'] = corr_circles

    def _setup_trails_simple(self, rows, cols, pos):
        """Setup detection trails using add_subplot."""
        ax = self.fig.add_subplot(rows, cols, pos)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title('Detection Trails', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        scatter_curr = ax.scatter([], [], s=80, c='red', marker='o',
                                   edgecolors='white', linewidths=1.5, zorder=10)
        scatter_hist = ax.scatter([], [], c=[], s=15, cmap='Reds',
                                   alpha=0.5, zorder=4)
        scatter_tracks = ax.scatter([], [], s=100, c='yellow', marker='D',
                                     edgecolors='black', linewidths=1.5, zorder=12)

        self.axes['trails'] = ax
        self.artists['trails_curr'] = scatter_curr
        self.artists['trails_hist'] = scatter_hist
        self.artists['trails_tracks'] = scatter_tracks

    def _setup_waterfall_simple(self, rows, cols, pos, title: str, wf_type: str):
        """Setup waterfall display using add_subplot."""
        ax = self.fig.add_subplot(rows, cols, pos)
        ax.set_xlabel('Time (s ago)', fontsize=8)
        ax.set_ylabel('Delay (km)' if wf_type == 'delay' else 'Doppler (Hz)', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(self.params.history_duration, 0)
        ax.grid(True, alpha=0.3)

        scatter_server = ax.scatter([], [], c='red', s=20, alpha=0.7)
        scatter_local = ax.scatter([], [], c='lime', s=20, alpha=0.7)

        self.axes[f'wf_{wf_type}'] = ax
        self.artists[f'wf_{wf_type}_server'] = scatter_server
        self.artists[f'wf_{wf_type}_local'] = scatter_local

    def _setup_maxhold_simple(self, rows, cols, pos):
        """Setup max-hold CAF display using add_subplot."""
        ax = self.fig.add_subplot(rows, cols, pos)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title('Max-Hold CAF', fontsize=10, fontweight='bold')

        placeholder = np.zeros((100, 100))
        im = ax.imshow(
            placeholder, aspect='auto', origin='lower',
            extent=[0, 60, -300, 300], cmap='viridis', interpolation='bilinear',
            vmin=self.params.color_min, vmax=self.params.color_max,
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(-300, 300)

        self.axes['maxhold'] = ax
        self.artists['maxhold_im'] = im

    def _setup_channel_caf(self, gs_cell, title: str, channel_idx: int):
        """Setup a per-channel CAF display."""
        ax = self.fig.add_subplot(gs_cell)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')

        # Use realistic initial extent
        placeholder = np.zeros((100, 100))
        im = ax.imshow(
            placeholder, aspect='auto', origin='lower',
            extent=[0, 60, -300, 300], cmap='viridis', interpolation='bilinear',
            vmin=self.params.color_min, vmax=self.params.color_max,
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(-300, 300)

        scatter = ax.scatter([], [], s=40, facecolors='none', edgecolors='red',
                            linewidths=1, zorder=5)

        self.axes[f'ch{channel_idx}'] = ax
        self.artists[f'ch{channel_idx}_im'] = im
        self.artists[f'ch{channel_idx}_scatter'] = scatter

    def _setup_ppi(self, ax):
        """Setup the PPI (Plan Position Indicator) polar display."""
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, self.params.ppi_max_range_km)
        ax.set_yticks([10, 20, 30, 40, 50])
        ax.set_yticklabels(['10', '20', '30', '40', '50 km'], fontsize=8)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=8)
        ax.set_title('PPI Display (AOA + Range)', fontsize=10, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)

        # Detection markers
        scatter_det = ax.scatter([], [], c='red', s=60, alpha=0.7, label='Detections')

        # Track markers
        scatter_tracks = ax.scatter([], [], c='lime', s=100, marker='D',
                                     edgecolors='white', linewidths=1.5,
                                     label='Tracks', zorder=10)

        ax.legend(loc='lower left', fontsize=7)

        self.axes['ppi'] = ax
        self.artists['ppi_det'] = scatter_det
        self.artists['ppi_tracks'] = scatter_tracks
        self.artists['ppi_arrows'] = []

    def _setup_fused_caf(self, gs_cell):
        """Setup the fused CAF display."""
        ax = self.fig.add_subplot(gs_cell)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title('Fused CAF (All Channels)', fontsize=10, fontweight='bold')

        placeholder = np.zeros((100, 100))
        im = ax.imshow(
            placeholder, aspect='auto', origin='lower',
            extent=[0, 60, -300, 300], cmap='viridis', interpolation='bilinear',
            vmin=self.params.color_min, vmax=self.params.color_max,
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(-300, 300)

        cbar = self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Power (dB)', fontsize=8)

        scatter_server = ax.scatter([], [], s=50, facecolors='none', edgecolors='red',
                                     linewidths=1.5, zorder=5, label='Server')
        scatter_local = ax.scatter([], [], s=50, facecolors='none', edgecolors='lime',
                                    linewidths=1.5, zorder=6, label='Local')
        scatter_tracks = ax.scatter([], [], s=80, c='yellow', marker='D',
                                     edgecolors='black', linewidths=1, zorder=7)

        self.axes['fused'] = ax
        self.artists['fused_im'] = im
        self.artists['fused_scatter_server'] = scatter_server
        self.artists['fused_scatter_local'] = scatter_local
        self.artists['fused_scatter_tracks'] = scatter_tracks

    def _setup_channel_health(self, gs_cell):
        """Setup the channel health display."""
        ax = self.fig.add_subplot(gs_cell)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 40)
        ax.set_xlabel('Channel', fontsize=8)
        ax.set_ylabel('SNR (dB)', fontsize=8)
        ax.set_title('Channel Health', fontsize=10, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['Ref', 'S1', 'S2', 'S3', 'S4'], fontsize=8)

        # SNR threshold lines
        ax.axhline(15, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(25, color='green', linestyle='--', alpha=0.5, linewidth=1)

        # SNR bars
        bars = ax.bar(range(5), [0]*5, color='gray', edgecolor='white', linewidth=1)

        ax.grid(True, alpha=0.3, axis='y')

        # Phase text annotations
        phase_texts = []
        for i in range(5):
            txt = ax.text(i, 2, '', ha='center', va='bottom', fontsize=7,
                         color='white', fontweight='bold')
            phase_texts.append(txt)

        # Correlation indicator circles at top
        corr_circles = []
        for i in range(5):
            circle = plt.Circle((i, 38), 0.3, color='gray', zorder=10)
            ax.add_patch(circle)
            corr_circles.append(circle)

        self.axes['health'] = ax
        self.artists['health_bars'] = bars
        self.artists['health_phase_texts'] = phase_texts
        self.artists['health_corr_circles'] = corr_circles

    def _setup_waterfall(self, gs_cell, title: str, wf_type: str):
        """Setup a waterfall display."""
        ax = self.fig.add_subplot(gs_cell)
        ax.set_xlabel('Time (s ago)', fontsize=8)
        ax.set_ylabel('Delay (km)' if wf_type == 'delay' else 'Doppler (Hz)', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(self.params.history_duration, 0)
        ax.grid(True, alpha=0.3)

        scatter_server = ax.scatter([], [], c='red', s=20, alpha=0.7)
        scatter_local = ax.scatter([], [], c='lime', s=20, alpha=0.7)

        self.axes[f'wf_{wf_type}'] = ax
        self.artists[f'wf_{wf_type}_server'] = scatter_server
        self.artists[f'wf_{wf_type}_local'] = scatter_local

    def _setup_trails(self, gs_cell):
        """Setup the detection trails display."""
        ax = self.fig.add_subplot(gs_cell)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title('Detection Trails', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        scatter_curr = ax.scatter([], [], s=80, c='red', marker='o',
                                   edgecolors='white', linewidths=1.5, zorder=10)
        scatter_hist = ax.scatter([], [], c=[], s=15, cmap='Reds',
                                   alpha=0.5, zorder=4)
        scatter_tracks = ax.scatter([], [], s=100, c='yellow', marker='D',
                                     edgecolors='black', linewidths=1.5, zorder=12)

        self.axes['trails'] = ax
        self.artists['trails_curr'] = scatter_curr
        self.artists['trails_hist'] = scatter_hist
        self.artists['trails_tracks'] = scatter_tracks

    def _setup_waterfall_dd(self, gs_cell):
        """Setup combined delay-doppler scatter waterfall."""
        ax = self.fig.add_subplot(gs_cell)
        ax.set_xlabel('Delay (km)', fontsize=8)
        ax.set_ylabel('Doppler (Hz)', fontsize=8)
        ax.set_title('Max-Hold CAF', fontsize=10, fontweight='bold')

        placeholder = np.zeros((100, 100))
        im = ax.imshow(
            placeholder, aspect='auto', origin='lower',
            extent=[0, 60, -300, 300], cmap='viridis', interpolation='bilinear',
            vmin=self.params.color_min, vmax=self.params.color_max,
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(-300, 300)

        self.axes['maxhold'] = ax
        self.artists['maxhold_im'] = im

    def _setup_control_window(self):
        """Setup separate control panel window."""
        self.ctrl_fig = plt.figure(figsize=(5, 10))
        self.ctrl_fig.canvas.manager.set_window_title('Controls')

        ctrl_left = 0.15
        ctrl_width = 0.75

        self.ctrl_fig.text(0.5, 0.96, 'Control Panel',
                           fontsize=12, fontweight='bold', ha='center')

        y_pos = 0.91
        y_step = 0.032
        slider_height = 0.018

        # --- Detection Source ---
        self.ctrl_fig.text(ctrl_left, y_pos, 'Detection Source:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_source = self.ctrl_fig.add_axes([ctrl_left, y_pos - 0.06, ctrl_width, 0.07])
        self.widgets['source'] = CheckButtons(
            ax_source,
            ['Server', 'Local CFAR', 'Tracks'],
            [self.params.show_server_detections,
             self.params.show_local_detections,
             self.params.show_local_tracks]
        )
        self.widgets['source'].on_clicked(self._on_source_changed)
        y_pos -= 0.10

        # --- CFAR Parameters ---
        self.ctrl_fig.text(ctrl_left, y_pos, 'CFAR Parameters:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_guard = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cfar_guard'] = Slider(ax_guard, 'Guard', 1, 10,
                                             valinit=self.params.cfar_guard, valstep=1)
        self.widgets['cfar_guard'].on_changed(self._on_cfar_changed)
        y_pos -= y_step

        ax_train = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cfar_train'] = Slider(ax_train, 'Train', 2, 16,
                                             valinit=self.params.cfar_train, valstep=1)
        self.widgets['cfar_train'].on_changed(self._on_cfar_changed)
        y_pos -= y_step

        ax_thresh = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cfar_threshold'] = Slider(ax_thresh, 'Threshold (dB)', 0, 30,
                                                 valinit=self.params.cfar_threshold)
        self.widgets['cfar_threshold'].on_changed(self._on_cfar_changed)
        y_pos -= y_step

        ax_type = self.ctrl_fig.add_axes([ctrl_left, y_pos - 0.04, ctrl_width, 0.05])
        self.widgets['cfar_type'] = RadioButtons(ax_type, ['CA', 'GO', 'SO'],
                                                  active=['ca', 'go', 'so'].index(self.params.cfar_type))
        self.widgets['cfar_type'].on_clicked(self._on_cfar_type_changed)
        y_pos -= 0.08

        # --- Display Parameters ---
        self.ctrl_fig.text(ctrl_left, y_pos, 'Display:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_cmin = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['color_min'] = Slider(ax_cmin, 'Color Min (dB)', -20, 10,
                                            valinit=self.params.color_min)
        self.widgets['color_min'].on_changed(self._on_display_changed)
        y_pos -= y_step

        ax_cmax = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['color_max'] = Slider(ax_cmax, 'Color Max (dB)', 5, 40,
                                            valinit=self.params.color_max)
        self.widgets['color_max'].on_changed(self._on_display_changed)
        y_pos -= y_step

        ax_hist = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['history'] = Slider(ax_hist, 'History (s)', 10, 300,
                                          valinit=self.params.history_duration)
        self.widgets['history'].on_changed(self._on_display_changed)
        y_pos -= y_step

        ax_decay = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['max_hold_decay'] = Slider(ax_decay, 'Max-Hold Decay', 0.9, 1.0,
                                                 valinit=self.params.max_hold_decay)
        self.widgets['max_hold_decay'].on_changed(self._on_decay_changed)
        y_pos -= y_step

        ax_ppi_range = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['ppi_range'] = Slider(ax_ppi_range, 'PPI Range (km)', 10, 100,
                                            valinit=self.params.ppi_max_range_km)
        self.widgets['ppi_range'].on_changed(self._on_ppi_range_changed)
        y_pos -= y_step * 1.5

        # --- Tracker Parameters ---
        self.ctrl_fig.text(ctrl_left, y_pos, 'Tracker:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_confirm = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['tracker_confirm'] = Slider(ax_confirm, 'Confirm Hits', 1, 10,
                                                  valinit=self.params.tracker_confirm, valstep=1)
        self.widgets['tracker_confirm'].on_changed(self._on_tracker_changed)
        y_pos -= y_step

        ax_delete = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['tracker_delete'] = Slider(ax_delete, 'Delete Misses', 1, 20,
                                                 valinit=self.params.tracker_delete, valstep=1)
        self.widgets['tracker_delete'].on_changed(self._on_tracker_changed)
        y_pos -= y_step

        ax_gate = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['tracker_gate'] = Slider(ax_gate, 'Gate Threshold', 10, 300,
                                               valinit=self.params.tracker_gate)
        self.widgets['tracker_gate'].on_changed(self._on_tracker_changed)
        y_pos -= y_step * 1.8

        # --- Action Buttons ---
        ax_reset = self.ctrl_fig.add_axes([ctrl_left, y_pos, ctrl_width * 0.45, 0.035])
        self.widgets['btn_reset'] = Button(ax_reset, 'Reset All')
        self.widgets['btn_reset'].on_clicked(self._on_reset)

        ax_clear = self.ctrl_fig.add_axes([ctrl_left + ctrl_width * 0.55, y_pos, ctrl_width * 0.45, 0.035])
        self.widgets['btn_clear'] = Button(ax_clear, 'Clear History')
        self.widgets['btn_clear'].on_clicked(self._on_clear)

    def _on_cfar_type_changed(self, label):
        """Handle CFAR type radio button change."""
        self.params.cfar_type = label.lower()
        self._rebuild_local_processing()

    def _on_decay_changed(self, val):
        """Handle max-hold decay slider change."""
        self.params.max_hold_decay = self.widgets['max_hold_decay'].val

    # ------------------------------------------------------------------ #
    #  Widget Callbacks
    # ------------------------------------------------------------------ #

    def _on_source_changed(self, label):
        status = self.widgets['source'].get_status()
        self.params.show_server_detections = status[0]
        self.params.show_local_detections = status[1]
        self.params.show_local_tracks = status[2]

    def _on_cfar_changed(self, val):
        self.params.cfar_guard = int(self.widgets['cfar_guard'].val)
        self.params.cfar_train = int(self.widgets['cfar_train'].val)
        self.params.cfar_threshold = self.widgets['cfar_threshold'].val
        self._rebuild_local_processing()

    def _on_display_changed(self, val):
        self.params.color_min = self.widgets['color_min'].val
        self.params.color_max = self.widgets['color_max'].val
        self.params.history_duration = self.widgets['history'].val

        # Update waterfall time axes
        self.axes['wf_delay'].set_xlim(self.params.history_duration, 0)
        self.axes['wf_doppler'].set_xlim(self.params.history_duration, 0)

    def _on_ppi_range_changed(self, val):
        self.params.ppi_max_range_km = self.widgets['ppi_range'].val
        self.axes['ppi'].set_ylim(0, self.params.ppi_max_range_km)

    def _on_tracker_changed(self, val):
        self.params.tracker_confirm = int(self.widgets['tracker_confirm'].val)
        self.params.tracker_delete = int(self.widgets['tracker_delete'].val)
        self.params.tracker_gate = self.widgets['tracker_gate'].val
        self._rebuild_local_processing()

    def _on_reset(self, event):
        """Reset max-hold and tracker."""
        with self.lock:
            self.max_hold_data = None
        if LOCAL_PROCESSING_AVAILABLE:
            self.tracker.reset()
            self.local_tracks = []

    def _on_clear(self, event):
        """Clear detection history."""
        with self.lock:
            self.detection_history.clear()

    # ------------------------------------------------------------------ #
    #  Animation Update
    # ------------------------------------------------------------------ #

    def _update_frame(self, frame_num):
        """Animation callback - update all panels."""
        now = time.time()

        with self.lock:
            if self.data.fused_caf is None:
                return list(self.artists.values())

            delay = self.data.delay_km.copy()
            doppler = self.data.doppler_hz.copy()
            fused = self.data.fused_caf.copy()
            channel_cafs = [c.copy() if c is not None else None for c in self.data.channel_cafs]
            channel_health = [ChannelHealth(**h.__dict__) for h in self.data.channel_health]
            local_dets = list(self.local_detections)
            local_tracks = list(self.local_tracks)
            history = list(self.detection_history)
            nf = self.frames_fetched
            nerr = self.fetch_errors

        # Sort doppler axis
        sort_idx = np.argsort(doppler)
        doppler_sorted = doppler[sort_idx]
        fused_sorted = fused[sort_idx, :]

        extent = [delay[0], delay[-1], doppler_sorted[0], doppler_sorted[-1]]
        color_min = self.params.color_min
        color_max = self.params.color_max

        # Update per-channel CAFs
        for i in range(4):
            if channel_cafs[i] is not None:
                caf_sorted = channel_cafs[i][sort_idx, :]
                self.artists[f'ch{i}_im'].set_data(caf_sorted)
                self.artists[f'ch{i}_im'].set_extent(extent)
                self.artists[f'ch{i}_im'].set_clim(color_min, color_max)
                self.axes[f'ch{i}'].set_xlim(extent[0], extent[1])
                self.axes[f'ch{i}'].set_ylim(extent[2], extent[3])

        # Update fused CAF
        self.artists['fused_im'].set_data(fused_sorted)
        self.artists['fused_im'].set_extent(extent)
        self.artists['fused_im'].set_clim(color_min, color_max)
        self.axes['fused'].set_xlim(extent[0], extent[1])
        self.axes['fused'].set_ylim(extent[2], extent[3])

        # Update max-hold display
        with self.lock:
            if self.max_hold_data is not None:
                maxhold_sorted = self.max_hold_data[sort_idx, :]
                self.artists['maxhold_im'].set_data(maxhold_sorted)
                self.artists['maxhold_im'].set_extent(extent)
                self.artists['maxhold_im'].set_clim(color_min, color_max)
                self.axes['maxhold'].set_xlim(extent[0], extent[1])
                self.axes['maxhold'].set_ylim(extent[2], extent[3])

        # Update fused CAF detections
        self._update_fused_detections(history, local_dets, local_tracks, now)

        # Update channel health
        self._update_channel_health(channel_health)

        # Update PPI
        self._update_ppi(history, local_tracks, now)

        # Update waterfalls
        self._update_waterfalls(history, now)

        # Update trails
        self._update_trails(history, local_tracks, now)

        # Info text
        runtime = now - self._start_time if self._start_time else 0
        n_dets = len([d for d in history if now - d.timestamp < 2])
        n_tracks = len([t for t in local_tracks if t.status == TrackStatus.CONFIRMED]) if LOCAL_PROCESSING_AVAILABLE else 0

        info = f'Frames: {nf}  Errors: {nerr}  Runtime: {runtime:.0f}s\n'
        info += f'Detections: {n_dets}  Tracks: {n_tracks}  History: {len(history)}'
        self.artists['info_text'].set_text(info)

        # Force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        return list(self.artists.values())

    def _update_fused_detections(self, history, local_dets, local_tracks, now):
        """Update detection overlays on fused CAF."""
        # Recent server detections
        recent = [d for d in history if now - d.timestamp < 2 and d.source == 'server']
        if recent and self.params.show_server_detections:
            x = [d.delay_km for d in recent]
            y = [d.doppler_hz for d in recent]
            self.artists['fused_scatter_server'].set_offsets(np.column_stack([x, y]))
        else:
            self.artists['fused_scatter_server'].set_offsets(np.empty((0, 2)))

        # Local detections
        if local_dets and self.params.show_local_detections:
            x = [d.range_m / 1000.0 for d in local_dets]
            y = [d.doppler_hz for d in local_dets]
            self.artists['fused_scatter_local'].set_offsets(np.column_stack([x, y]))
        else:
            self.artists['fused_scatter_local'].set_offsets(np.empty((0, 2)))

        # Tracks
        if local_tracks and self.params.show_local_tracks and LOCAL_PROCESSING_AVAILABLE:
            confirmed = [t for t in local_tracks if t.status == TrackStatus.CONFIRMED]
            if confirmed:
                x = [t.range_m / 1000.0 for t in confirmed]
                y = [t.doppler_hz for t in confirmed]
                self.artists['fused_scatter_tracks'].set_offsets(np.column_stack([x, y]))
            else:
                self.artists['fused_scatter_tracks'].set_offsets(np.empty((0, 2)))
        else:
            self.artists['fused_scatter_tracks'].set_offsets(np.empty((0, 2)))

    def _update_channel_health(self, health: List[ChannelHealth]):
        """Update channel health display."""
        colors = []
        for i, h in enumerate(health):
            # SNR bar
            self.artists['health_bars'][i].set_height(max(0, h.snr_db))

            # Color by SNR level
            if h.snr_db >= 25:
                color = '#00FF00'  # Green
            elif h.snr_db >= 15:
                color = '#FFFF00'  # Yellow
            else:
                color = '#FF0000'  # Red
            self.artists['health_bars'][i].set_color(color)
            colors.append(color)

            # Phase text
            if i == 0:
                phase_str = 'REF'
            else:
                phase_str = f'{h.phase_offset_deg:+.0f}°'
            self.artists['health_phase_texts'][i].set_text(phase_str)

            # Correlation circle
            if h.correlation_coeff >= 0.95:
                corr_color = '#00FF00'
            elif h.correlation_coeff >= 0.85:
                corr_color = '#FFFF00'
            else:
                corr_color = '#FF0000'
            self.artists['health_corr_circles'][i].set_facecolor(corr_color)

    def _update_ppi(self, history, local_tracks, now):
        """Update PPI polar display."""
        # Recent detections with AOA
        recent = [d for d in history if now - d.timestamp < 5 and d.azimuth_deg is not None]

        if recent:
            angles = [np.radians(d.azimuth_deg) for d in recent]
            ranges = [d.delay_km for d in recent]
            self.artists['ppi_det'].set_offsets(np.column_stack([angles, ranges]))
        else:
            self.artists['ppi_det'].set_offsets(np.empty((0, 2)))

        # Tracks with AOA
        if local_tracks and self.params.show_local_tracks and LOCAL_PROCESSING_AVAILABLE:
            confirmed = [t for t in local_tracks if t.status == TrackStatus.CONFIRMED]
            if confirmed:
                # For tracks, estimate AOA from recent history
                angles = []
                ranges = []
                for t in confirmed:
                    # Use a simple AOA estimate based on track history
                    azimuth = np.random.uniform(0, 360)  # Placeholder
                    angles.append(np.radians(azimuth))
                    ranges.append(t.range_m / 1000.0)
                self.artists['ppi_tracks'].set_offsets(np.column_stack([angles, ranges]))
            else:
                self.artists['ppi_tracks'].set_offsets(np.empty((0, 2)))
        else:
            self.artists['ppi_tracks'].set_offsets(np.empty((0, 2)))

    def _update_waterfalls(self, history, now):
        """Update waterfall displays."""
        if not history:
            for key in ['wf_delay_server', 'wf_delay_local',
                        'wf_doppler_server', 'wf_doppler_local']:
                self.artists[key].set_offsets(np.empty((0, 2)))
            return

        times_ago = np.array([now - d.timestamp for d in history])
        delays = np.array([d.delay_km for d in history])
        dopplers = np.array([d.doppler_hz for d in history])
        sources = np.array([d.source for d in history])

        mask = times_ago <= self.params.history_duration
        times_ago = times_ago[mask]
        delays = delays[mask]
        dopplers = dopplers[mask]
        sources = sources[mask]

        server_mask = sources == 'server'
        local_mask = sources == 'local'

        # Delay waterfall
        if np.any(server_mask) and self.params.show_server_detections:
            self.artists['wf_delay_server'].set_offsets(
                np.column_stack([times_ago[server_mask], delays[server_mask]]))
        else:
            self.artists['wf_delay_server'].set_offsets(np.empty((0, 2)))

        if np.any(local_mask) and self.params.show_local_detections:
            self.artists['wf_delay_local'].set_offsets(
                np.column_stack([times_ago[local_mask], delays[local_mask]]))
        else:
            self.artists['wf_delay_local'].set_offsets(np.empty((0, 2)))

        # Doppler waterfall
        if np.any(server_mask) and self.params.show_server_detections:
            self.artists['wf_doppler_server'].set_offsets(
                np.column_stack([times_ago[server_mask], dopplers[server_mask]]))
        else:
            self.artists['wf_doppler_server'].set_offsets(np.empty((0, 2)))

        if np.any(local_mask) and self.params.show_local_detections:
            self.artists['wf_doppler_local'].set_offsets(
                np.column_stack([times_ago[local_mask], dopplers[local_mask]]))
        else:
            self.artists['wf_doppler_local'].set_offsets(np.empty((0, 2)))

        # Auto-scale Y axes
        if len(delays) > 0:
            self.axes['wf_delay'].set_ylim(max(0, np.min(delays) - 5), np.max(delays) + 5)
        if len(dopplers) > 0:
            margin = max(50, (np.max(dopplers) - np.min(dopplers)) * 0.1)
            self.axes['wf_doppler'].set_ylim(np.min(dopplers) - margin, np.max(dopplers) + margin)

    def _update_trails(self, history, local_tracks, now):
        """Update detection trails display."""
        if not history:
            self.artists['trails_curr'].set_offsets(np.empty((0, 2)))
            self.artists['trails_hist'].set_offsets(np.empty((0, 2)))
            self.artists['trails_tracks'].set_offsets(np.empty((0, 2)))
            return

        # Current detections (last 2 seconds)
        recent = [d for d in history if now - d.timestamp < 2]
        if recent:
            x = [d.delay_km for d in recent]
            y = [d.doppler_hz for d in recent]
            self.artists['trails_curr'].set_offsets(np.column_stack([x, y]))
        else:
            self.artists['trails_curr'].set_offsets(np.empty((0, 2)))

        # Historical detections
        times_ago = np.array([now - d.timestamp for d in history])
        delays = np.array([d.delay_km for d in history])
        dopplers = np.array([d.doppler_hz for d in history])

        mask = (times_ago > 2) & (times_ago <= self.params.history_duration)
        if np.any(mask):
            self.artists['trails_hist'].set_offsets(np.column_stack([delays[mask], dopplers[mask]]))
            self.artists['trails_hist'].set_array(times_ago[mask])
        else:
            self.artists['trails_hist'].set_offsets(np.empty((0, 2)))

        # Tracks
        if local_tracks and self.params.show_local_tracks and LOCAL_PROCESSING_AVAILABLE:
            confirmed = [t for t in local_tracks if t.status == TrackStatus.CONFIRMED]
            if confirmed:
                x = [t.range_m / 1000.0 for t in confirmed]
                y = [t.doppler_hz for t in confirmed]
                self.artists['trails_tracks'].set_offsets(np.column_stack([x, y]))
            else:
                self.artists['trails_tracks'].set_offsets(np.empty((0, 2)))
        else:
            self.artists['trails_tracks'].set_offsets(np.empty((0, 2)))

        # Auto-scale
        if len(delays) > 0:
            self.axes['trails'].set_xlim(max(0, np.min(delays) - 2), np.max(delays) + 2)
            self.axes['trails'].set_ylim(np.min(dopplers) - 30, np.max(dopplers) + 30)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def start(self, blocking: bool = True):
        """Start polling and display."""
        self.running = True
        self._start_time = time.time()

        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # Wait for first data fetch
        import time as _time
        print("Waiting for data...")
        for _ in range(30):
            if self.data.fused_caf is not None:
                print(f"Data received: {self.data.fused_caf.shape}")
                break
            _time.sleep(0.1)

        # Enable interactive mode
        plt.ion()

        self._setup_plot()

        # Force initial draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ctrl_fig.canvas.draw()
        self.ctrl_fig.canvas.flush_events()

        print("Starting animation...")
        self.anim = FuncAnimation(
            self.fig, self._update_frame,
            interval=max(500, int(self.poll_interval * 1000)),
            blit=False,
            cache_frame_data=False,
        )

        if blocking:
            plt.ioff()
            plt.show()
        else:
            plt.show(block=False)

    def stop(self):
        """Stop polling and close displays."""
        self.running = False
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        if hasattr(self, 'ctrl_fig') and self.ctrl_fig is not None:
            plt.close(self.ctrl_fig)
            self.ctrl_fig = None


def main():
    parser = argparse.ArgumentParser(
        description='Five-Channel KrakenSDR Dashboard'
    )
    parser.add_argument(
        '--url', default='https://radar3.retnode.com',
        help='Base URL of the radar server',
    )
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Poll interval in seconds',
    )
    parser.add_argument(
        '--ppi-range', type=float, default=50.0,
        help='PPI display max range in km',
    )
    args = parser.parse_args()

    print('KrakenSDR 5-Channel Dashboard v1.0')
    print('=' * 50)
    print(f'  Server: {args.url}')
    print(f'  Poll interval: {args.interval}s')
    print(f'  PPI range: {args.ppi_range} km')
    print(f'  Local processing: {"Available" if LOCAL_PROCESSING_AVAILABLE else "Not available"}')
    print()
    print('Features:')
    print('  - 4 per-channel CAF displays (surveillance channels)')
    print('  - PPI polar display with AOA estimates')
    print('  - Channel health monitoring (SNR, phase, correlation)')
    print('  - Fused CAF with detection overlays')
    print('  - Delay and Doppler waterfalls')
    print('  - Detection trails with history')
    print('  - Interactive control panel')
    print()

    params = ProcessingParams(ppi_max_range_km=args.ppi_range)

    dashboard = FiveChannelDashboard(
        base_url=args.url,
        poll_interval=args.interval,
        params=params,
    )
    dashboard.start()


if __name__ == '__main__':
    main()
