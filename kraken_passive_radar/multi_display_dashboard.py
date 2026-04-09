"""
Multi-Display Dashboard for retnode.com KrakenSDR Passive Radar.

Displays five synchronized panels with interactive control panel:
1. Delay-Doppler Map (live CAF heatmap)
2. Max-Hold Delay-Doppler Map (accumulated maximum power)
3. Detections in Delay over Time (time vs delay waterfall)
4. Detections in Doppler over Time (time vs Doppler waterfall)
5. Detections in Delay-Doppler over Time (with history trails)

Control Panel Features:
- Detection source: Server, Local CFAR, or Both
- CFAR parameters: guard, train, threshold, type
- Clustering parameters: min/max cluster size
- Tracker parameters: confirm hits, delete misses, gate threshold
- Display parameters: color scale, history duration, max-hold decay

Usage:
    python -m kraken_passive_radar.multi_display_dashboard [--url URL] [--interval SEC]

Default server: https://radar3.retnode.com

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
from typing import List, Optional, Deque, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
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


@dataclass
class Detection:
    """Single detection with timestamp and source."""
    timestamp: float  # Unix timestamp in seconds
    delay: float      # km
    doppler: float    # Hz
    snr: float        # dB
    source: str = 'server'  # 'server' or 'local'


@dataclass
class ProcessingParams:
    """All tunable processing parameters."""
    # CFAR
    cfar_guard: int = 2
    cfar_train: int = 4
    cfar_threshold: float = 12.0
    cfar_type: str = 'ca'  # 'ca', 'go', 'so'

    # Clustering
    cluster_min_size: int = 1
    cluster_max_size: int = 100

    # Tracker
    tracker_confirm: int = 3
    tracker_delete: int = 5
    tracker_gate: float = 100.0
    tracker_process_noise: float = 10.0
    tracker_measure_noise: float = 50.0

    # Display
    color_min: float = 0.0
    color_max: float = 15.0
    history_duration: float = 60.0
    max_hold_decay: float = 0.995

    # Detection source
    show_server_detections: bool = True
    show_local_detections: bool = False
    show_local_tracks: bool = False


class MultiDisplayDashboard:
    """
    Multi-panel dashboard with interactive control panel for remote KrakenSDR
    passive radar visualization with optional local processing.
    """

    def __init__(
        self,
        base_url: str = 'https://radar3.retnode.com',
        poll_interval: float = 1.0,
        params: Optional[ProcessingParams] = None,
    ):
        """Initialize dashboard with server connection and local processing pipeline.

        Technique: threaded polling with lock-guarded shared state and deque-based history.
        """
        self.base_url = base_url.rstrip('/')
        self.poll_interval = poll_interval
        self.params = params if params else ProcessingParams()

        # Data (guarded by lock)
        self.lock = threading.Lock()
        self.delay = None            # 1-D array, km
        self.doppler = None          # 1-D array, Hz
        self.map_data = None         # 2-D array (nDoppler x nDelay), dB
        self.max_hold_data = None    # 2-D array, accumulated max
        self.detections_raw = None   # Latest raw detection dict from server
        self.last_timestamp = 0
        self.fetch_errors = 0
        self.frames_fetched = 0

        # Local processing
        self.local_detections: List[LocalDetection] = []
        self.local_tracks: List[LocalTrack] = []
        self._init_local_processing()

        # Detection history (deque for efficient FIFO)
        self.detection_history: Deque[Detection] = deque(maxlen=5000)

        # Matplotlib objects
        self.fig = None
        self.axes = {}
        self.artists = {}
        self.widgets = {}

        # State
        self.running = False
        self._stop_event = threading.Event()
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
            process_noise=self.params.tracker_process_noise,
            measurement_noise=self.params.tracker_measure_noise,
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

        # Reset tracker with new parameters
        self.tracker = MultiTargetTracker(
            dt=self.poll_interval,
            confirm_hits=self.params.tracker_confirm,
            delete_misses=self.params.tracker_delete,
            gate_threshold=self.params.tracker_gate,
            process_noise=self.params.tracker_process_noise,
            measurement_noise=self.params.tracker_measure_noise,
        )

    # ------------------------------------------------------------------ #
    #  Network
    # ------------------------------------------------------------------ #

    _UA = 'KrakenSDR-MultiDisplay/2.0'

    def _fetch_json(self, endpoint: str, timeout: float = 5.0):
        """Fetch JSON from an API endpoint. Returns dict or None on error."""
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
        """Background thread: fetch map + detections at poll_interval."""
        while self.running:
            t0 = time.monotonic()
            now = time.time()

            # Fetch map data
            map_data = self._fetch_json('/api/map')
            if map_data is not None:
                delay = np.asarray(map_data['delay'], dtype=np.float64)
                doppler = np.asarray(map_data['doppler'], dtype=np.float64)
                data = np.asarray(map_data['data'], dtype=np.float64)
                ts = map_data.get('timestamp', 0)

                # Run local processing if enabled
                local_dets = []
                local_tracks = []
                if LOCAL_PROCESSING_AVAILABLE and (
                    self.params.show_local_detections or self.params.show_local_tracks
                ):
                    local_dets, local_tracks = self._run_local_processing(
                        data, delay, doppler
                    )

                with self.lock:
                    self.delay = delay
                    self.doppler = doppler
                    self.map_data = data

                    # Update max-hold with decay
                    if self.max_hold_data is None or self.max_hold_data.shape != data.shape:
                        self.max_hold_data = data.copy()
                    else:
                        self.max_hold_data *= self.params.max_hold_decay
                        self.max_hold_data = np.maximum(self.max_hold_data, data)

                    self.last_timestamp = ts
                    self.frames_fetched += 1

                    self.local_detections = local_dets
                    self.local_tracks = local_tracks

            # Fetch server detections
            det_data = self._fetch_json('/api/detection')
            if det_data is not None:
                with self.lock:
                    self.detections_raw = det_data

                    # Add server detections to history
                    if self.params.show_server_detections:
                        if 'delay' in det_data and len(det_data['delay']) > 0:
                            delays = det_data['delay']
                            dopplers = det_data['doppler']
                            snrs = det_data.get('snr', [0] * len(delays))
                            ts_det = det_data.get('timestamp', now * 1000) / 1000.0

                            for i in range(len(delays)):
                                self.detection_history.append(Detection(
                                    timestamp=ts_det,
                                    delay=delays[i],
                                    doppler=dopplers[i],
                                    snr=snrs[i] if i < len(snrs) else 0,
                                    source='server',
                                ))

                    # Add local detections to history
                    if self.params.show_local_detections and self.local_detections:
                        for det in self.local_detections:
                            self.detection_history.append(Detection(
                                timestamp=now,
                                delay=det.range_m / 1000.0,  # Convert m to km
                                doppler=det.doppler_hz,
                                snr=det.snr_db,
                                source='local',
                            ))

                    # Prune old detections
                    cutoff = now - self.params.history_duration
                    while self.detection_history and self.detection_history[0].timestamp < cutoff:
                        self.detection_history.popleft()

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self.poll_interval - elapsed)
            if sleep_time > 0:
                # Use event wait for interruptible sleep
                self._stop_event.wait(sleep_time)

    def _run_local_processing(
        self, data: np.ndarray, delay: np.ndarray, doppler: np.ndarray
    ) -> Tuple[List[LocalDetection], List[LocalTrack]]:
        """Run local CFAR, clustering, and tracking on a single frame.

        Technique: sequential pipeline of CA-CFAR detection, connected-component clustering, and GNN tracking.
        """
        if not LOCAL_PROCESSING_AVAILABLE:
            return [], []

        # Convert delay from km to meters for processing
        range_axis_m = delay * 1000.0
        doppler_axis_hz = doppler

        # CFAR detection
        cfar_mask = self.cfar.detect(data)

        # Clustering
        detections = self.clusterer.cluster(
            cfar_mask, data, range_axis_m, doppler_axis_hz
        )

        # Tracking
        self.tracker.update(detections)
        tracks = self.tracker.get_all_tracks()

        return detections, tracks

    # ------------------------------------------------------------------ #
    #  Display Setup
    # ------------------------------------------------------------------ #

    def _setup_plot(self):
        """Create the matplotlib figure with 5 display panels and interactive control panel.

        Technique: GridSpec layout with separate display and control regions.
        """
        # Layout with control panel on right:
        #   +---------------+---------------+------------+
        #   | Delay-Doppler | Max-Hold      |            |
        #   | Map           | Map           |  Control   |
        #   +---------------+---------------+   Panel    |
        #   | Det vs Time   | Det vs Time   |            |
        #   | (Delay)       | (Doppler)     |            |
        #   +---------------+---------------+            |
        #   |   Detections in Delay-Doppler |            |
        #   |       with History Trails     |            |
        #   +-------------------------------+------------+

        self.fig = plt.figure(figsize=(20, 12))
        self.fig.canvas.manager.set_window_title(
            f'KrakenSDR Multi-Display Dashboard — {self.base_url}'
        )

        # Main grid: displays on left (75%), controls on right (25%)
        gs_main = GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1], wspace=0.02)

        # Display panels grid
        gs_display = GridSpec(3, 2, figure=self.fig,
                              height_ratios=[1, 0.8, 0.8],
                              left=0.05, right=0.72, bottom=0.05, top=0.95,
                              hspace=0.25, wspace=0.2)

        # Panel 1: Delay-Doppler Map (top-left)
        ax_dd = self.fig.add_subplot(gs_display[0, 0])
        self.axes['delay_doppler'] = ax_dd
        self._setup_delay_doppler_panel(ax_dd, 'Delay-Doppler Map (Live)')

        # Panel 2: Max-Hold Map (top-right)
        ax_mh = self.fig.add_subplot(gs_display[0, 1])
        self.axes['max_hold'] = ax_mh
        self._setup_delay_doppler_panel(ax_mh, 'Max-Hold Delay-Doppler Map')

        # Panel 3: Detections in Delay over Time (middle-left)
        ax_dt = self.fig.add_subplot(gs_display[1, 0])
        self.axes['det_delay_time'] = ax_dt
        self._setup_time_panel(ax_dt, 'Detections: Delay vs Time', 'Delay (km)')

        # Panel 4: Detections in Doppler over Time (middle-right)
        ax_dop = self.fig.add_subplot(gs_display[1, 1])
        self.axes['det_doppler_time'] = ax_dop
        self._setup_time_panel(ax_dop, 'Detections: Doppler vs Time', 'Doppler (Hz)')

        # Panel 5: Detections in Delay-Doppler with trails (bottom)
        ax_trail = self.fig.add_subplot(gs_display[2, :])
        self.axes['det_trails'] = ax_trail
        self._setup_trails_panel(ax_trail)

        # Control Panel
        self._setup_control_panel()

        # Info text
        self.artists['info_text'] = self.fig.text(
            0.05, 0.98, '', fontsize=9, verticalalignment='top',
            fontfamily='monospace', color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9),
        )

        # Handle window close event
        self.fig.canvas.mpl_connect('close_event', lambda evt: self.stop())

    def _setup_delay_doppler_panel(self, ax, title: str):
        """Setup a delay-Doppler heatmap panel with detection overlay scatter plots.

        Technique: imshow heatmap with server, local, and track scatter overlays.
        """
        ax.set_xlabel('Bistatic Delay (km)', fontsize=10)
        ax.set_ylabel('Doppler Shift (Hz)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        placeholder = np.full((10, 10), 0.0)
        im = ax.imshow(
            placeholder,
            aspect='auto',
            origin='lower',
            extent=[0, 1, -1, 1],
            cmap='viridis',
            interpolation='bilinear',
            vmin=self.params.color_min,
            vmax=self.params.color_max,
        )

        cbar = self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Power (dB)', fontsize=9)

        # Server detection markers (red circles)
        scatter_server = ax.scatter(
            [], [], s=50, facecolors='none', edgecolors='red',
            linewidths=1.5, zorder=5, label='Server',
        )

        # Local detection markers (green circles)
        scatter_local = ax.scatter(
            [], [], s=50, facecolors='none', edgecolors='lime',
            linewidths=1.5, zorder=6, label='Local',
        )

        # Local track markers (yellow diamonds)
        scatter_tracks = ax.scatter(
            [], [], s=80, c='yellow', marker='D', edgecolors='black',
            linewidths=1, zorder=7, label='Tracks',
        )

        key_prefix = 'dd' if 'Live' in title else 'mh'
        self.artists[f'{key_prefix}_im'] = im
        self.artists[f'{key_prefix}_scatter_server'] = scatter_server
        self.artists[f'{key_prefix}_scatter_local'] = scatter_local
        self.artists[f'{key_prefix}_scatter_tracks'] = scatter_tracks
        self.artists[f'{key_prefix}_cbar'] = cbar

    def _setup_time_panel(self, ax, title: str, ylabel: str):
        """Setup a time-series scatter panel for detection history waterfall.

        Technique: scatter plot with reversed time axis showing detections over time.
        """
        ax.set_xlabel('Time (seconds ago)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(self.params.history_duration, 0)
        ax.grid(True, alpha=0.3)

        # Server detections (red)
        scatter_server = ax.scatter(
            [], [], c='red', s=30, alpha=0.7, label='Server',
        )

        # Local detections (green)
        scatter_local = ax.scatter(
            [], [], c='lime', s=30, alpha=0.7, label='Local',
        )

        key = 'dt' if 'Delay' in ylabel else 'dop'
        self.artists[f'{key}_scatter_server'] = scatter_server
        self.artists[f'{key}_scatter_local'] = scatter_local

    def _setup_trails_panel(self, ax):
        """Setup the delay-Doppler trails panel with current and historical detection markers.

        Technique: layered scatter plots with age-based colormapping for history trails.
        """
        ax.set_xlabel('Bistatic Delay (km)', fontsize=10)
        ax.set_ylabel('Doppler Shift (Hz)', fontsize=10)
        ax.set_title('Detections in Delay-Doppler (with History Trails)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Server current (red)
        scatter_server_curr = ax.scatter(
            [], [], s=100, c='red', marker='o', edgecolors='white',
            linewidths=1.5, zorder=10, label='Server (current)',
        )

        # Local current (green)
        scatter_local_curr = ax.scatter(
            [], [], s=100, c='lime', marker='o', edgecolors='white',
            linewidths=1.5, zorder=11, label='Local (current)',
        )

        # Tracks (yellow diamonds)
        scatter_tracks = ax.scatter(
            [], [], s=120, c='yellow', marker='D', edgecolors='black',
            linewidths=1.5, zorder=12, label='Tracks',
        )

        # History trails (colored by age)
        scatter_history_server = ax.scatter(
            [], [], c=[], s=15, cmap='Reds', vmin=0, vmax=self.params.history_duration,
            alpha=0.5, zorder=4,
        )
        scatter_history_local = ax.scatter(
            [], [], c=[], s=15, cmap='Greens', vmin=0, vmax=self.params.history_duration,
            alpha=0.5, zorder=5,
        )

        ax.legend(loc='upper right', fontsize=8)

        self.artists['trail_server_curr'] = scatter_server_curr
        self.artists['trail_local_curr'] = scatter_local_curr
        self.artists['trail_tracks'] = scatter_tracks
        self.artists['trail_history_server'] = scatter_history_server
        self.artists['trail_history_local'] = scatter_history_local

    def _setup_control_panel(self):
        """Setup the interactive control panel with sliders, checkboxes, and buttons.

        Technique: matplotlib widgets (Slider, CheckButtons, RadioButtons, Button) for real-time parameter tuning.
        """
        # Control panel area
        ctrl_left = 0.76
        ctrl_width = 0.22

        # Title
        self.fig.text(ctrl_left + ctrl_width/2, 0.96, 'Control Panel',
                      fontsize=12, fontweight='bold', ha='center')

        y_pos = 0.92
        y_step = 0.035
        slider_height = 0.02

        # --- Detection Source Section ---
        self.fig.text(ctrl_left, y_pos, 'Detection Source:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_source = self.fig.add_axes([ctrl_left, y_pos - 0.06, ctrl_width, 0.07])
        self.widgets['source'] = CheckButtons(
            ax_source,
            ['Server Detections', 'Local CFAR', 'Local Tracks'],
            [self.params.show_server_detections,
             self.params.show_local_detections,
             self.params.show_local_tracks]
        )
        self.widgets['source'].on_clicked(self._on_source_changed)
        y_pos -= 0.09

        # --- CFAR Section ---
        self.fig.text(ctrl_left, y_pos, 'CFAR Parameters:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_guard = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cfar_guard'] = Slider(ax_guard, 'Guard', 1, 10,
                                             valinit=self.params.cfar_guard, valstep=1)
        self.widgets['cfar_guard'].on_changed(self._on_cfar_changed)
        y_pos -= y_step

        ax_train = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cfar_train'] = Slider(ax_train, 'Train', 2, 16,
                                             valinit=self.params.cfar_train, valstep=1)
        self.widgets['cfar_train'].on_changed(self._on_cfar_changed)
        y_pos -= y_step

        ax_thresh = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cfar_threshold'] = Slider(ax_thresh, 'Thresh (dB)', 0, 30,
                                                 valinit=self.params.cfar_threshold)
        self.widgets['cfar_threshold'].on_changed(self._on_cfar_changed)
        y_pos -= y_step

        ax_type = self.fig.add_axes([ctrl_left, y_pos - 0.04, ctrl_width, 0.05])
        self.widgets['cfar_type'] = RadioButtons(ax_type, ['CA', 'GO', 'SO'],
                                                  active=['ca', 'go', 'so'].index(self.params.cfar_type))
        self.widgets['cfar_type'].on_clicked(self._on_cfar_type_changed)
        y_pos -= 0.07

        # --- Clustering Section ---
        self.fig.text(ctrl_left, y_pos, 'Clustering:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_cmin = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cluster_min'] = Slider(ax_cmin, 'Min Size', 1, 10,
                                              valinit=self.params.cluster_min_size, valstep=1)
        self.widgets['cluster_min'].on_changed(self._on_cluster_changed)
        y_pos -= y_step

        ax_cmax = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['cluster_max'] = Slider(ax_cmax, 'Max Size', 10, 200,
                                              valinit=self.params.cluster_max_size, valstep=10)
        self.widgets['cluster_max'].on_changed(self._on_cluster_changed)
        y_pos -= y_step * 1.2

        # --- Tracker Section ---
        self.fig.text(ctrl_left, y_pos, 'Tracker:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_confirm = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['tracker_confirm'] = Slider(ax_confirm, 'Confirm', 1, 10,
                                                  valinit=self.params.tracker_confirm, valstep=1)
        self.widgets['tracker_confirm'].on_changed(self._on_tracker_changed)
        y_pos -= y_step

        ax_delete = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['tracker_delete'] = Slider(ax_delete, 'Delete', 1, 20,
                                                 valinit=self.params.tracker_delete, valstep=1)
        self.widgets['tracker_delete'].on_changed(self._on_tracker_changed)
        y_pos -= y_step

        ax_gate = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['tracker_gate'] = Slider(ax_gate, 'Gate', 10, 300,
                                               valinit=self.params.tracker_gate)
        self.widgets['tracker_gate'].on_changed(self._on_tracker_changed)
        y_pos -= y_step * 1.2

        # --- Display Section ---
        self.fig.text(ctrl_left, y_pos, 'Display:', fontsize=10, fontweight='bold')
        y_pos -= y_step

        ax_cmin_d = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['color_min'] = Slider(ax_cmin_d, 'Color Min', -20, 20,
                                            valinit=self.params.color_min)
        self.widgets['color_min'].on_changed(self._on_display_changed)
        y_pos -= y_step

        ax_cmax_d = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['color_max'] = Slider(ax_cmax_d, 'Color Max', 5, 50,
                                            valinit=self.params.color_max)
        self.widgets['color_max'].on_changed(self._on_display_changed)
        y_pos -= y_step

        ax_hist = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['history'] = Slider(ax_hist, 'History (s)', 10, 300,
                                          valinit=self.params.history_duration)
        self.widgets['history'].on_changed(self._on_display_changed)
        y_pos -= y_step

        ax_decay = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, slider_height])
        self.widgets['decay'] = Slider(ax_decay, 'Max-Hold', 0.9, 1.0,
                                        valinit=self.params.max_hold_decay)
        self.widgets['decay'].on_changed(self._on_display_changed)
        y_pos -= y_step * 1.5

        # --- Action Buttons ---
        ax_reset_mh = self.fig.add_axes([ctrl_left, y_pos, ctrl_width * 0.48, 0.03])
        self.widgets['btn_reset_mh'] = Button(ax_reset_mh, 'Reset Max-Hold')
        self.widgets['btn_reset_mh'].on_clicked(self._on_reset_maxhold)

        ax_clear_hist = self.fig.add_axes([ctrl_left + ctrl_width * 0.52, y_pos, ctrl_width * 0.48, 0.03])
        self.widgets['btn_clear_hist'] = Button(ax_clear_hist, 'Clear History')
        self.widgets['btn_clear_hist'].on_clicked(self._on_clear_history)
        y_pos -= y_step * 1.2

        ax_reset_tracker = self.fig.add_axes([ctrl_left, y_pos, ctrl_width, 0.03])
        self.widgets['btn_reset_tracker'] = Button(ax_reset_tracker, 'Reset Tracker')
        self.widgets['btn_reset_tracker'].on_clicked(self._on_reset_tracker)

    # ------------------------------------------------------------------ #
    #  Widget Callbacks
    # ------------------------------------------------------------------ #

    def _on_source_changed(self, label):
        """Handle detection source checkbox changes."""
        status = self.widgets['source'].get_status()
        self.params.show_server_detections = status[0]
        self.params.show_local_detections = status[1]
        self.params.show_local_tracks = status[2]

    def _on_cfar_changed(self, val):
        """Handle CFAR parameter changes."""
        self.params.cfar_guard = int(self.widgets['cfar_guard'].val)
        self.params.cfar_train = int(self.widgets['cfar_train'].val)
        self.params.cfar_threshold = self.widgets['cfar_threshold'].val
        self._rebuild_local_processing()

    def _on_cfar_type_changed(self, label):
        """Handle CFAR type changes."""
        self.params.cfar_type = label.lower()
        self._rebuild_local_processing()

    def _on_cluster_changed(self, val):
        """Handle clustering parameter changes."""
        self.params.cluster_min_size = int(self.widgets['cluster_min'].val)
        self.params.cluster_max_size = int(self.widgets['cluster_max'].val)
        self._rebuild_local_processing()

    def _on_tracker_changed(self, val):
        """Handle tracker parameter changes."""
        self.params.tracker_confirm = int(self.widgets['tracker_confirm'].val)
        self.params.tracker_delete = int(self.widgets['tracker_delete'].val)
        self.params.tracker_gate = self.widgets['tracker_gate'].val
        self._rebuild_local_processing()

    def _on_display_changed(self, val):
        """Handle display parameter changes."""
        self.params.color_min = self.widgets['color_min'].val
        self.params.color_max = self.widgets['color_max'].val
        self.params.history_duration = self.widgets['history'].val
        self.params.max_hold_decay = self.widgets['decay'].val

        # Update time axis limits
        self.axes['det_delay_time'].set_xlim(self.params.history_duration, 0)
        self.axes['det_doppler_time'].set_xlim(self.params.history_duration, 0)

    def _on_reset_maxhold(self, event):
        """Reset max-hold accumulator."""
        self.reset_max_hold()

    def _on_clear_history(self, event):
        """Clear detection history."""
        self.clear_history()

    def _on_reset_tracker(self, event):
        """Reset tracker."""
        if LOCAL_PROCESSING_AVAILABLE:
            self.tracker.reset()
            with self.lock:
                self.local_tracks = []

    # ------------------------------------------------------------------ #
    #  Animation Update
    # ------------------------------------------------------------------ #

    def _update_frame(self, frame_num):
        """Animation callback — update all panels."""
        now = time.time()

        with self.lock:
            if self.map_data is None:
                return list(self.artists.values())

            delay = self.delay.copy()
            doppler = self.doppler.copy()
            data = self.map_data.copy()
            max_hold = self.max_hold_data.copy() if self.max_hold_data is not None else data
            detections_raw = dict(self.detections_raw) if self.detections_raw else {}
            local_dets = list(self.local_detections)
            local_tracks = list(self.local_tracks)
            history = list(self.detection_history)
            nf = self.frames_fetched
            nerr = self.fetch_errors

        # Sort doppler axis
        sort_idx = np.argsort(doppler)
        doppler_sorted = doppler[sort_idx]
        data_sorted = data[sort_idx, :]
        max_hold_sorted = max_hold[sort_idx, :]

        extent = [delay[0], delay[-1], doppler_sorted[0], doppler_sorted[-1]]

        # Get current display parameters
        color_min = self.params.color_min
        color_max = self.params.color_max

        # ---- Panel 1: Live Delay-Doppler ----
        self._update_dd_panel('dd', data_sorted, extent, color_min, color_max,
                              detections_raw, local_dets, local_tracks)

        # ---- Panel 2: Max-Hold ----
        self._update_dd_panel('mh', max_hold_sorted, extent, color_min, color_max,
                              detections_raw, local_dets, local_tracks)

        # ---- Panels 3 & 4: Detections over Time ----
        self._update_time_panels(history, now)

        # ---- Panel 5: Trails ----
        self._update_trails_panel(history, detections_raw, local_dets, local_tracks, now)

        # ---- Info Text ----
        runtime = now - self._start_time if self._start_time else 0
        n_server = len(detections_raw.get('delay', []))
        n_local = len(local_dets)
        n_tracks = len([t for t in local_tracks if t.status == TrackStatus.CONFIRMED]) if LOCAL_PROCESSING_AVAILABLE else 0

        info_lines = [
            f'Frames: {nf}  Errors: {nerr}  Runtime: {runtime:.0f}s',
            f'Server: {n_server}  Local: {n_local}  Tracks: {n_tracks}  History: {len(history)}',
        ]
        if LOCAL_PROCESSING_AVAILABLE:
            info_lines.append(f'CFAR: {self.params.cfar_type.upper()} g={self.params.cfar_guard} t={self.params.cfar_train} th={self.params.cfar_threshold:.1f}dB')

        self.artists['info_text'].set_text('\n'.join(info_lines))

        return list(self.artists.values())

    def _update_dd_panel(self, prefix, data, extent, vmin, vmax,
                         server_dets, local_dets, local_tracks):
        """Update a delay-Doppler panel with new data, detection markers, and track overlays.

        Technique: set_data/set_extent/set_clim on imshow artist, set_offsets on scatter artists.
        """
        im = self.artists[f'{prefix}_im']
        im.set_data(data)
        im.set_extent(extent)
        im.set_clim(vmin, vmax)

        ax = self.axes['delay_doppler' if prefix == 'dd' else 'max_hold']
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Server detections
        if self.params.show_server_detections and server_dets and 'delay' in server_dets:
            det_x = np.asarray(server_dets['delay'])
            det_y = np.asarray(server_dets['doppler'])
            self.artists[f'{prefix}_scatter_server'].set_offsets(np.column_stack([det_x, det_y]))
        else:
            self.artists[f'{prefix}_scatter_server'].set_offsets(np.empty((0, 2)))

        # Local detections
        if self.params.show_local_detections and local_dets:
            det_x = np.array([d.range_m / 1000.0 for d in local_dets])
            det_y = np.array([d.doppler_hz for d in local_dets])
            self.artists[f'{prefix}_scatter_local'].set_offsets(np.column_stack([det_x, det_y]))
        else:
            self.artists[f'{prefix}_scatter_local'].set_offsets(np.empty((0, 2)))

        # Local tracks
        if self.params.show_local_tracks and local_tracks and LOCAL_PROCESSING_AVAILABLE:
            confirmed = [t for t in local_tracks if t.status == TrackStatus.CONFIRMED]
            if confirmed:
                track_x = np.array([t.range_m / 1000.0 for t in confirmed])
                track_y = np.array([t.doppler_hz for t in confirmed])
                self.artists[f'{prefix}_scatter_tracks'].set_offsets(np.column_stack([track_x, track_y]))
            else:
                self.artists[f'{prefix}_scatter_tracks'].set_offsets(np.empty((0, 2)))
        else:
            self.artists[f'{prefix}_scatter_tracks'].set_offsets(np.empty((0, 2)))

    def _update_time_panels(self, history, now):
        """Update the delay-vs-time and Doppler-vs-time waterfall panels.

        Technique: vectorized source-mask filtering with auto-scaling Y axes.
        """
        if not history:
            for key in ['dt_scatter_server', 'dt_scatter_local',
                        'dop_scatter_server', 'dop_scatter_local']:
                self.artists[key].set_offsets(np.empty((0, 2)))
            return

        times_ago = np.array([now - d.timestamp for d in history])
        delays = np.array([d.delay for d in history])
        dopplers = np.array([d.doppler for d in history])
        sources = np.array([d.source for d in history])

        mask = times_ago <= self.params.history_duration
        times_ago = times_ago[mask]
        delays = delays[mask]
        dopplers = dopplers[mask]
        sources = sources[mask]

        server_mask = sources == 'server'
        local_mask = sources == 'local'

        # Delay vs Time
        if np.any(server_mask) and self.params.show_server_detections:
            self.artists['dt_scatter_server'].set_offsets(
                np.column_stack([times_ago[server_mask], delays[server_mask]]))
        else:
            self.artists['dt_scatter_server'].set_offsets(np.empty((0, 2)))

        if np.any(local_mask) and self.params.show_local_detections:
            self.artists['dt_scatter_local'].set_offsets(
                np.column_stack([times_ago[local_mask], delays[local_mask]]))
        else:
            self.artists['dt_scatter_local'].set_offsets(np.empty((0, 2)))

        # Doppler vs Time
        if np.any(server_mask) and self.params.show_server_detections:
            self.artists['dop_scatter_server'].set_offsets(
                np.column_stack([times_ago[server_mask], dopplers[server_mask]]))
        else:
            self.artists['dop_scatter_server'].set_offsets(np.empty((0, 2)))

        if np.any(local_mask) and self.params.show_local_detections:
            self.artists['dop_scatter_local'].set_offsets(
                np.column_stack([times_ago[local_mask], dopplers[local_mask]]))
        else:
            self.artists['dop_scatter_local'].set_offsets(np.empty((0, 2)))

        # Auto-scale Y axes
        if len(delays) > 0:
            self.axes['det_delay_time'].set_ylim(
                max(0, np.min(delays) - 5), np.max(delays) + 5)
        if len(dopplers) > 0:
            margin = max(50, (np.max(dopplers) - np.min(dopplers)) * 0.1)
            self.axes['det_doppler_time'].set_ylim(
                np.min(dopplers) - margin, np.max(dopplers) + margin)

    def _update_trails_panel(self, history, server_dets, local_dets, local_tracks, now):
        """Update the trails panel with current detections, history, and track markers.

        Technique: age-based colormap on scatter history with auto-scaling axes.
        """
        # Current server detections
        if self.params.show_server_detections and server_dets and 'delay' in server_dets:
            curr_x = np.asarray(server_dets['delay'])
            curr_y = np.asarray(server_dets['doppler'])
            self.artists['trail_server_curr'].set_offsets(np.column_stack([curr_x, curr_y]))
        else:
            self.artists['trail_server_curr'].set_offsets(np.empty((0, 2)))

        # Current local detections
        if self.params.show_local_detections and local_dets:
            curr_x = np.array([d.range_m / 1000.0 for d in local_dets])
            curr_y = np.array([d.doppler_hz for d in local_dets])
            self.artists['trail_local_curr'].set_offsets(np.column_stack([curr_x, curr_y]))
        else:
            self.artists['trail_local_curr'].set_offsets(np.empty((0, 2)))

        # Tracks
        if self.params.show_local_tracks and local_tracks and LOCAL_PROCESSING_AVAILABLE:
            confirmed = [t for t in local_tracks if t.status == TrackStatus.CONFIRMED]
            if confirmed:
                track_x = np.array([t.range_m / 1000.0 for t in confirmed])
                track_y = np.array([t.doppler_hz for t in confirmed])
                self.artists['trail_tracks'].set_offsets(np.column_stack([track_x, track_y]))
            else:
                self.artists['trail_tracks'].set_offsets(np.empty((0, 2)))
        else:
            self.artists['trail_tracks'].set_offsets(np.empty((0, 2)))

        # History trails
        if history:
            times_ago = np.array([now - d.timestamp for d in history])
            delays = np.array([d.delay for d in history])
            dopplers = np.array([d.doppler for d in history])
            sources = np.array([d.source for d in history])

            mask = times_ago <= self.params.history_duration

            server_mask = mask & (sources == 'server')
            local_mask = mask & (sources == 'local')

            if np.any(server_mask) and self.params.show_server_detections:
                self.artists['trail_history_server'].set_offsets(
                    np.column_stack([delays[server_mask], dopplers[server_mask]]))
                self.artists['trail_history_server'].set_array(times_ago[server_mask])
            else:
                self.artists['trail_history_server'].set_offsets(np.empty((0, 2)))

            if np.any(local_mask) and self.params.show_local_detections:
                self.artists['trail_history_local'].set_offsets(
                    np.column_stack([delays[local_mask], dopplers[local_mask]]))
                self.artists['trail_history_local'].set_array(times_ago[local_mask])
            else:
                self.artists['trail_history_local'].set_offsets(np.empty((0, 2)))

            # Auto-scale
            all_delays = delays[mask]
            all_dopplers = dopplers[mask]
            if len(all_delays) > 0:
                self.axes['det_trails'].set_xlim(
                    max(0, np.min(all_delays) - 2), np.max(all_delays) + 2)
                self.axes['det_trails'].set_ylim(
                    np.min(all_dopplers) - 30, np.max(all_dopplers) + 30)
        else:
            self.artists['trail_history_server'].set_offsets(np.empty((0, 2)))
            self.artists['trail_history_local'].set_offsets(np.empty((0, 2)))

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def start(self, blocking: bool = True):
        """Start polling and display."""
        self.running = True
        self._start_time = time.time()

        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        self._setup_plot()
        self.anim = FuncAnimation(
            self.fig, self._update_frame,
            interval=max(200, int(self.poll_interval * 1000)),
            blit=False,
            cache_frame_data=False,
        )

        if blocking:
            plt.show()
        else:
            plt.show(block=False)

    def stop(self):
        """Stop polling and close display."""
        self.running = False
        self._stop_event.set()  # Interrupt any blocking wait
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def reset_max_hold(self):
        """Reset the max-hold accumulator."""
        with self.lock:
            if self.map_data is not None:
                self.max_hold_data = self.map_data.copy()

    def clear_history(self):
        """Clear detection history."""
        with self.lock:
            self.detection_history.clear()


def main():
    """Parse command-line arguments and launch the multi-display dashboard.

    Technique: argparse CLI with server URL and poll interval options.
    """
    parser = argparse.ArgumentParser(
        description='Multi-Display Dashboard for KrakenSDR Passive Radar'
    )
    parser.add_argument(
        '--url', default='https://radar3.retnode.com',
        help='Base URL of the radar server (default: https://radar3.retnode.com)',
    )
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Poll interval in seconds (default: 1.0)',
    )
    args = parser.parse_args()

    print(f'KrakenSDR Multi-Display Dashboard v2.0')
    print(f'  Server: {args.url}')
    print(f'  Poll interval: {args.interval}s')
    print(f'  Local processing: {"Available" if LOCAL_PROCESSING_AVAILABLE else "Not available"}')
    print()
    print('Controls:')
    print('  - Use checkboxes to toggle detection sources')
    print('  - Adjust sliders to tune CFAR/Tracker parameters')
    print('  - Changes take effect immediately')
    print()

    dashboard = MultiDisplayDashboard(
        base_url=args.url,
        poll_interval=args.interval,
    )
    dashboard.start()


if __name__ == '__main__':
    main()
