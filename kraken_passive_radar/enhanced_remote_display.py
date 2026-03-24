"""
Enhanced Remote Radar Display with Local Processing.

Extends RemoteRadarDisplay to run local CFAR detection, clustering, and
multi-target tracking on delay-Doppler maps fetched from remote servers.
Displays both server detections and locally computed detections/tracks.

Usage:
    # Server detections only (default)
    python -m kraken_passive_radar.enhanced_remote_display

    # With local processing
    python -m kraken_passive_radar.enhanced_remote_display --local

    # Custom CFAR parameters
    python -m kraken_passive_radar.enhanced_remote_display --local \\
        --cfar-guard 2 --cfar-train 8 --cfar-threshold 10

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import argparse
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .remote_display import RemoteRadarDisplay
    from .local_processing import (
        CfarDetector,
        DetectionClusterer,
        MultiTargetTracker,
        Detection,
        Track,
        TrackStatus,
    )
except ImportError:
    from remote_display import RemoteRadarDisplay
    from local_processing import (
        CfarDetector,
        DetectionClusterer,
        MultiTargetTracker,
        Detection,
        Track,
        TrackStatus,
    )


class EnhancedRemoteRadarDisplay(RemoteRadarDisplay):
    """
    Enhanced remote radar display with local CFAR, clustering, and tracking.

    Extends RemoteRadarDisplay to overlay:
    - Server detections (red circles)
    - Local CFAR detections (green circles)
    - Confirmed tracks (yellow diamonds with history trails)

    The local processing pipeline runs independently of the server's
    detection algorithms, allowing comparison and potentially better
    detection/tracking performance with tuned parameters.
    """

    def __init__(
        self,
        base_url: str = 'https://radar3.retnode.com',
        poll_interval: float = 1.0,
        enable_local: bool = False,
        cfar_guard: int = 2,
        cfar_train: int = 4,
        cfar_threshold: float = 12.0,
        cfar_type: str = 'ca',
        tracker_dt: Optional[float] = None,
        tracker_confirm: int = 3,
        tracker_delete: int = 5,
        tracker_gate: float = 100.0,
    ):
        """
        Initialize enhanced display.

        Args:
            base_url: Base URL of radar server.
            poll_interval: Poll interval in seconds.
            enable_local: Enable local processing pipeline.
            cfar_guard: CFAR guard cells.
            cfar_train: CFAR training cells.
            cfar_threshold: CFAR threshold in dB.
            cfar_type: CFAR variant ('ca', 'go', 'so').
            tracker_dt: Tracker time step (defaults to poll_interval).
            tracker_confirm: Hits to confirm track.
            tracker_delete: Misses to delete track.
            tracker_gate: Association gate threshold.
        """
        super().__init__(base_url=base_url, poll_interval=poll_interval)

        self.enable_local = enable_local

        # Local processing components
        self.cfar: Optional[CfarDetector] = None
        self.clusterer: Optional[DetectionClusterer] = None
        self.tracker: Optional[MultiTargetTracker] = None

        if enable_local:
            self.cfar = CfarDetector(
                guard=cfar_guard,
                train=cfar_train,
                threshold_db=cfar_threshold,
                cfar_type=cfar_type,
            )

            self.clusterer = DetectionClusterer()

            self.tracker = MultiTargetTracker(
                dt=tracker_dt if tracker_dt is not None else poll_interval,
                confirm_hits=tracker_confirm,
                delete_misses=tracker_delete,
                gate_threshold=tracker_gate,
            )

        # Local processing results (guarded by parent's lock)
        self.local_detections: List[Detection] = []
        self.tracks: List[Track] = []
        self.cfar_mask: Optional[np.ndarray] = None
        self.local_processing_time_ms: float = 0.0

        # Display objects
        self.local_det_scatter = None
        self.track_scatter = None
        self.track_lines: List[Line2D] = []

    def _poll_loop(self):
        """Background thread: fetch map + detections and run local processing."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()

            # Fetch map data
            map_data = self._fetch_json('/api/map')
            if map_data is not None:
                delay = np.asarray(map_data['delay'], dtype=np.float64)
                doppler = np.asarray(map_data['doppler'], dtype=np.float64)
                data = np.asarray(map_data['data'], dtype=np.float64)
                ts = map_data.get('timestamp', 0)

                # Run local processing if enabled
                local_dets = []
                tracks = []
                cfar_mask = None
                local_time_ms = 0.0

                if self.enable_local and self.cfar is not None:
                    local_t0 = time.monotonic()

                    # Convert delay from km to m for processing
                    range_axis_m = delay * 1000.0
                    doppler_axis_hz = doppler

                    # CFAR detection
                    cfar_mask = self.cfar.detect(data.astype(np.float32))

                    # Clustering
                    local_dets = self.clusterer.cluster(
                        cfar_mask, data, range_axis_m, doppler_axis_hz
                    )

                    # Tracking
                    self.tracker.update(local_dets)
                    tracks = self.tracker.get_all_tracks()

                    local_time_ms = (time.monotonic() - local_t0) * 1000.0

                with self.lock:
                    self.delay = delay
                    self.doppler = doppler
                    self.map_data = data
                    self.last_timestamp = ts
                    self.frames_fetched += 1
                    self.local_detections = local_dets
                    self.tracks = tracks
                    self.cfar_mask = cfar_mask
                    self.local_processing_time_ms = local_time_ms

            # Fetch server detections
            det_data = self._fetch_json('/api/detection')
            if det_data is not None:
                with self.lock:
                    self.detections = det_data

            # Fetch timing
            timing = self._fetch_json('/api/timing')
            if timing is not None:
                with self.lock:
                    self.timing = timing

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self.poll_interval - elapsed)
            if sleep_time > 0:
                # Use event wait for interruptible sleep (matches parent class)
                self._stop_event.wait(sleep_time)

    def _setup_plot(self):
        """Create the matplotlib figure with enhanced overlays."""
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(
            f'Enhanced Remote Passive Radar — {self.base_url}'
        )

        # Placeholder image
        placeholder = np.full((10, 10), -60.0)
        self.im = self.ax.imshow(
            placeholder,
            aspect='auto',
            origin='lower',
            extent=[0, 1, -1, 1],
            cmap='viridis',
            interpolation='bilinear',
        )
        self.colorbar = self.fig.colorbar(
            self.im, ax=self.ax, label='Power (dB)',
            fraction=0.03, pad=0.02,
        )

        self.ax.set_xlabel('Bistatic Delay (km)', fontsize=12)
        self.ax.set_ylabel('Doppler Shift (Hz)', fontsize=12)
        self.ax.set_title('Enhanced Delay-Doppler Map (connecting…)', fontsize=13)

        # Server detection markers (red circles)
        self.det_scatter = self.ax.scatter(
            [], [], s=60, facecolors='none', edgecolors='red',
            linewidths=1.5, zorder=5, label='Server Detections',
        )

        # Local detection markers (green circles)
        if self.enable_local:
            self.local_det_scatter = self.ax.scatter(
                [], [], s=50, facecolors='none', edgecolors='lime',
                linewidths=1.5, zorder=6, label='Local Detections',
            )

            # Track markers (yellow diamonds)
            self.track_scatter = self.ax.scatter(
                [], [], s=80, marker='D', facecolors='none', edgecolors='yellow',
                linewidths=2.0, zorder=7, label='Tracks',
            )

        # Info text (top-left)
        self.info_text = self.ax.text(
            0.01, 0.99, '', transform=self.ax.transAxes,
            fontsize=9, verticalalignment='top', color='white',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
        )

        # Cursor readout (bottom-left)
        self.cursor_text = self.ax.text(
            0.01, 0.01, '', transform=self.ax.transAxes,
            fontsize=9, verticalalignment='bottom', color='white',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
        )

        # Legend
        if self.enable_local:
            self.ax.legend(loc='upper right', fontsize=8)

        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('close_event', lambda evt: self.stop())
        self.fig.tight_layout()

    def _update_frame(self, frame_num):
        """Animation callback — refresh plot from latest fetched data."""
        with self.lock:
            if self.map_data is None:
                return [self.im]
            delay = self.delay.copy()
            doppler = self.doppler.copy()
            data = self.map_data.copy()
            detections = self.detections.copy() if self.detections else None
            timing = dict(self.timing) if self.timing else None
            ts = self.last_timestamp
            nf = self.frames_fetched
            nerr = self.fetch_errors
            local_dets = list(self.local_detections)
            tracks = list(self.tracks)
            local_time_ms = self.local_processing_time_ms

        n_doppler, n_delay = data.shape

        # Sort doppler axis
        sort_idx = np.argsort(doppler)
        doppler_sorted = doppler[sort_idx]
        data_sorted = data[sort_idx, :]

        extent = [delay[0], delay[-1], doppler_sorted[0], doppler_sorted[-1]]

        self.im.set_data(data_sorted)
        self.im.set_extent(extent)

        # Fixed 0-15 dB scale for bluer background
        self.im.set_clim(0, 15)

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])

        # Server detections (red)
        if detections and 'delay' in detections and len(detections['delay']) > 0:
            det_x = np.asarray(detections['delay'])
            det_y = np.asarray(detections['doppler'])
            snr = np.asarray(detections['snr'])
            self.det_scatter.set_offsets(np.column_stack([det_x, det_y]))
            sizes = 30 + snr * 5
            self.det_scatter.set_sizes(sizes)
            server_det_info = f'  Server: {len(det_x)}'
        else:
            self.det_scatter.set_offsets(np.empty((0, 2)))
            server_det_info = '  Server: 0'

        # Local processing overlays
        local_det_info = ''
        track_info = ''

        if self.enable_local:
            # Local detections (green)
            if local_dets:
                # Convert from m back to km for display
                local_x = np.array([d.range_m / 1000.0 for d in local_dets])
                local_y = np.array([d.doppler_hz for d in local_dets])
                local_snr = np.array([d.snr_db for d in local_dets])
                self.local_det_scatter.set_offsets(np.column_stack([local_x, local_y]))
                sizes = 30 + np.clip(local_snr, 0, 20) * 3
                self.local_det_scatter.set_sizes(sizes)
                local_det_info = f'  Local: {len(local_dets)}'
            else:
                self.local_det_scatter.set_offsets(np.empty((0, 2)))
                local_det_info = '  Local: 0'

            # Tracks (yellow)
            confirmed = [t for t in tracks if t.status == TrackStatus.CONFIRMED]
            coasting = [t for t in tracks if t.status == TrackStatus.COASTING]
            tentative = [t for t in tracks if t.status == TrackStatus.TENTATIVE]

            # Remove old track lines
            for line in self.track_lines:
                line.remove()
            self.track_lines.clear()

            # Plot confirmed tracks with history trails
            if confirmed or coasting:
                track_x = []
                track_y = []

                for track in confirmed + coasting:
                    # Current position (convert m to km)
                    track_x.append(track.range_m / 1000.0)
                    track_y.append(track.doppler_hz)

                    # History trail
                    if len(track.history) > 1:
                        hist_x = [h[0] / 1000.0 for h in track.history]
                        hist_y = [h[1] for h in track.history]

                        # Color based on status
                        color = 'yellow' if track.status == TrackStatus.CONFIRMED else 'orange'
                        line, = self.ax.plot(
                            hist_x, hist_y,
                            color=color, linewidth=1.0, alpha=0.6, zorder=4
                        )
                        self.track_lines.append(line)

                self.track_scatter.set_offsets(np.column_stack([track_x, track_y]))
                self.track_scatter.set_sizes([80] * len(track_x))
            else:
                self.track_scatter.set_offsets(np.empty((0, 2)))

            n_confirmed = len(confirmed)
            n_coasting = len(coasting)
            n_tentative = len(tentative)
            track_info = f'  Tracks: {n_confirmed}C/{n_coasting}O/{n_tentative}T'

        # Timing info
        cpi_ms = ''
        if timing:
            cpi_val = timing.get('cpi', 0)
            uptime_d = timing.get('uptime_days', 0)
            n_cpi = timing.get('nCpi', 0)
            cpi_ms = f'  CPI: {cpi_val:.0f}ms  Up: {uptime_d:.1f}d'

        # Title
        title_suffix = ' [Local Processing Active]' if self.enable_local else ''
        self.ax.set_title(
            f'Enhanced Delay-Doppler Map — {n_delay}×{n_doppler}{title_suffix}',
            fontsize=13,
        )

        # Info text
        local_time_str = f'  Proc: {local_time_ms:.1f}ms' if self.enable_local else ''
        using_native = ''
        if self.enable_local and self.cfar is not None:
            using_native = ' [Native]' if self.cfar.is_native else ' [Python]'

        self.info_text.set_text(
            f'Frame {nf}{server_det_info}{local_det_info}{track_info}\n'
            f'Errors: {nerr}  Poll: {self.poll_interval:.1f}s{cpi_ms}{local_time_str}{using_native}'
        )

        # Return updated artists
        artists = [self.im, self.det_scatter, self.info_text]
        if self.enable_local:
            artists.extend([self.local_det_scatter, self.track_scatter])
            artists.extend(self.track_lines)
        return artists


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Remote Delay-Doppler Display with Local Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Server detections only (default)
  python -m kraken_passive_radar.enhanced_remote_display

  # Enable local CFAR/tracking pipeline
  python -m kraken_passive_radar.enhanced_remote_display --local

  # Custom CFAR parameters
  python -m kraken_passive_radar.enhanced_remote_display --local \\
      --cfar-guard 2 --cfar-train 8 --cfar-threshold 10

  # Custom tracker parameters
  python -m kraken_passive_radar.enhanced_remote_display --local \\
      --track-confirm 2 --track-delete 3 --track-gate 150

Display Legend:
  Red circles   - Server detections
  Green circles - Local CFAR detections
  Yellow ◆      - Confirmed tracks
  Orange trail  - Coasting track history
"""
    )

    # Connection options
    parser.add_argument(
        '--url', default='https://radar3.retnode.com',
        help='Base URL of the radar server (default: https://radar3.retnode.com)',
    )
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Poll interval in seconds (default: 1.0)',
    )

    # Local processing toggle
    parser.add_argument(
        '--local', action='store_true',
        help='Enable local CFAR/clustering/tracking pipeline',
    )

    # CFAR parameters
    cfar_group = parser.add_argument_group('CFAR Parameters')
    cfar_group.add_argument(
        '--cfar-guard', type=int, default=2,
        help='Guard cells around CUT (default: 2)',
    )
    cfar_group.add_argument(
        '--cfar-train', type=int, default=4,
        help='Training cells for noise estimate (default: 4)',
    )
    cfar_group.add_argument(
        '--cfar-threshold', type=float, default=12.0,
        help='Detection threshold in dB above noise (default: 12.0)',
    )
    cfar_group.add_argument(
        '--cfar-type', choices=['ca', 'go', 'so'], default='ca',
        help='CFAR variant: ca=cell-averaging, go=greatest-of, so=smallest-of (default: ca)',
    )

    # Tracker parameters
    track_group = parser.add_argument_group('Tracker Parameters')
    track_group.add_argument(
        '--track-dt', type=float, default=None,
        help='Tracker time step in seconds (default: poll interval)',
    )
    track_group.add_argument(
        '--track-confirm', type=int, default=3,
        help='Consecutive hits to confirm track (default: 3)',
    )
    track_group.add_argument(
        '--track-delete', type=int, default=5,
        help='Consecutive misses to delete track (default: 5)',
    )
    track_group.add_argument(
        '--track-gate', type=float, default=100.0,
        help='Association gate threshold (default: 100.0)',
    )

    args = parser.parse_args()

    mode_str = 'local processing' if args.local else 'server detections only'
    print(f'Connecting to {args.url} (poll every {args.interval}s, {mode_str})...')

    if args.local:
        print(f'CFAR: guard={args.cfar_guard}, train={args.cfar_train}, '
              f'threshold={args.cfar_threshold}dB, type={args.cfar_type}')
        print(f'Tracker: confirm={args.track_confirm}, delete={args.track_delete}, '
              f'gate={args.track_gate}')

    display = EnhancedRemoteRadarDisplay(
        base_url=args.url,
        poll_interval=args.interval,
        enable_local=args.local,
        cfar_guard=args.cfar_guard,
        cfar_train=args.cfar_train,
        cfar_threshold=args.cfar_threshold,
        cfar_type=args.cfar_type,
        tracker_dt=args.track_dt,
        tracker_confirm=args.track_confirm,
        tracker_delete=args.track_delete,
        tracker_gate=args.track_gate,
    )
    display.start()


if __name__ == '__main__':
    main()
