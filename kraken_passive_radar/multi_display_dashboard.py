"""
Multi-Display Dashboard for retnode.com KrakenSDR Passive Radar.

Displays five synchronized panels:
1. Delay-Doppler Map (live CAF heatmap)
2. Max-Hold Delay-Doppler Map (accumulated maximum power)
3. Detections in Delay over Time (time vs delay waterfall)
4. Detections in Doppler over Time (time vs Doppler waterfall)
5. Detections in Delay-Doppler over Time (with history trails)

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
from typing import List, Optional, Deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


@dataclass
class Detection:
    """Single detection with timestamp."""
    timestamp: float  # Unix timestamp in seconds
    delay: float      # km
    doppler: float    # Hz
    snr: float        # dB


class MultiDisplayDashboard:
    """
    Multi-panel dashboard for remote KrakenSDR passive radar visualization.

    Fetches delay-Doppler map and detection data from a remote server and
    displays five synchronized views:
    - Live delay-Doppler map
    - Max-hold delay-Doppler map
    - Detections in delay over time
    - Detections in Doppler over time
    - Detections in delay-Doppler with history trails
    """

    def __init__(
        self,
        base_url: str = 'https://radar3.retnode.com',
        poll_interval: float = 1.0,
        history_duration: float = 60.0,  # seconds of detection history
        max_hold_decay: float = 0.995,   # decay factor per frame (0=instant, 1=infinite)
    ):
        self.base_url = base_url.rstrip('/')
        self.poll_interval = poll_interval
        self.history_duration = history_duration
        self.max_hold_decay = max_hold_decay

        # Data (guarded by lock)
        self.lock = threading.Lock()
        self.delay = None            # 1-D array, km
        self.doppler = None          # 1-D array, Hz
        self.map_data = None         # 2-D array (nDoppler x nDelay), dB
        self.max_hold_data = None    # 2-D array, accumulated max
        self.detections_raw = None   # Latest raw detection dict
        self.last_timestamp = 0
        self.fetch_errors = 0
        self.frames_fetched = 0

        # Detection history (deque for efficient FIFO)
        self.detection_history: Deque[Detection] = deque(maxlen=2000)

        # Matplotlib objects
        self.fig = None
        self.axes = {}
        self.artists = {}

        # State
        self.running = False
        self._poll_thread = None
        self._start_time = None

    # ------------------------------------------------------------------ #
    #  Network
    # ------------------------------------------------------------------ #

    _UA = 'KrakenSDR-MultiDisplay/1.0'

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

                with self.lock:
                    self.delay = delay
                    self.doppler = doppler
                    self.map_data = data

                    # Update max-hold with decay
                    if self.max_hold_data is None or self.max_hold_data.shape != data.shape:
                        self.max_hold_data = data.copy()
                    else:
                        # Apply decay then take max
                        self.max_hold_data *= self.max_hold_decay
                        self.max_hold_data = np.maximum(self.max_hold_data, data)

                    self.last_timestamp = ts
                    self.frames_fetched += 1

            # Fetch detections
            det_data = self._fetch_json('/api/detection')
            if det_data is not None:
                with self.lock:
                    self.detections_raw = det_data

                    # Add to history
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
                            ))

                    # Prune old detections
                    cutoff = now - self.history_duration
                    while self.detection_history and self.detection_history[0].timestamp < cutoff:
                        self.detection_history.popleft()

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self.poll_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------ #
    #  Display Setup
    # ------------------------------------------------------------------ #

    def _setup_plot(self):
        """Create the matplotlib figure with 5 panels."""
        # Create figure with GridSpec layout
        # Layout:
        #   +---------------+---------------+
        #   | Delay-Doppler | Max-Hold      |
        #   | Map           | Map           |
        #   +---------------+---------------+
        #   | Det vs Time   | Det vs Time   |
        #   | (Delay)       | (Doppler)     |
        #   +---------------+---------------+
        #   |   Detections in Delay-Doppler |
        #   |       with History Trails     |
        #   +-------------------------------+

        self.fig = plt.figure(figsize=(16, 12))
        self.fig.canvas.manager.set_window_title(
            f'KrakenSDR Multi-Display Dashboard — {self.base_url}'
        )

        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[1, 0.8, 0.8],
                      hspace=0.3, wspace=0.25)

        # Panel 1: Delay-Doppler Map (top-left)
        ax_dd = self.fig.add_subplot(gs[0, 0])
        self.axes['delay_doppler'] = ax_dd
        self._setup_delay_doppler_panel(ax_dd, 'Delay-Doppler Map (Live)')

        # Panel 2: Max-Hold Map (top-right)
        ax_mh = self.fig.add_subplot(gs[0, 1])
        self.axes['max_hold'] = ax_mh
        self._setup_delay_doppler_panel(ax_mh, 'Max-Hold Delay-Doppler Map')

        # Panel 3: Detections in Delay over Time (middle-left)
        ax_dt = self.fig.add_subplot(gs[1, 0])
        self.axes['det_delay_time'] = ax_dt
        self._setup_time_panel(ax_dt, 'Detections: Delay vs Time', 'Delay (km)')

        # Panel 4: Detections in Doppler over Time (middle-right)
        ax_dop = self.fig.add_subplot(gs[1, 1])
        self.axes['det_doppler_time'] = ax_dop
        self._setup_time_panel(ax_dop, 'Detections: Doppler vs Time', 'Doppler (Hz)')

        # Panel 5: Detections in Delay-Doppler with trails (bottom, spans both columns)
        ax_trail = self.fig.add_subplot(gs[2, :])
        self.axes['det_trails'] = ax_trail
        self._setup_trails_panel(ax_trail)

        # Info text
        self.artists['info_text'] = self.fig.text(
            0.01, 0.99, '', fontsize=9, verticalalignment='top',
            fontfamily='monospace', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            transform=self.fig.transFigure,
        )

        self.fig.tight_layout(rect=[0, 0, 1, 0.98])

    def _setup_delay_doppler_panel(self, ax, title: str):
        """Setup a delay-Doppler heatmap panel."""
        ax.set_xlabel('Bistatic Delay (km)', fontsize=10)
        ax.set_ylabel('Doppler Shift (Hz)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Placeholder image
        placeholder = np.full((10, 10), 0.0)
        im = ax.imshow(
            placeholder,
            aspect='auto',
            origin='lower',
            extent=[0, 1, -1, 1],
            cmap='viridis',
            interpolation='bilinear',
            vmin=0, vmax=15,
        )

        # Colorbar
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Power (dB)', fontsize=9)

        # Detection markers
        scatter = ax.scatter(
            [], [], s=50, facecolors='none', edgecolors='red',
            linewidths=1.5, zorder=5,
        )

        # Store artists
        key_prefix = 'dd' if 'Live' in title else 'mh'
        self.artists[f'{key_prefix}_im'] = im
        self.artists[f'{key_prefix}_scatter'] = scatter
        self.artists[f'{key_prefix}_cbar'] = cbar

    def _setup_time_panel(self, ax, title: str, ylabel: str):
        """Setup a time-series scatter panel."""
        ax.set_xlabel('Time (seconds ago)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(self.history_duration, 0)  # Reversed so recent is on right
        ax.grid(True, alpha=0.3)

        # Scatter plot with colormap for SNR
        scatter = ax.scatter(
            [], [], c=[], s=30, cmap='hot', vmin=0, vmax=20, alpha=0.7,
        )

        key = 'dt_scatter' if 'Delay' in ylabel else 'dop_scatter'
        self.artists[key] = scatter

    def _setup_trails_panel(self, ax):
        """Setup the delay-Doppler trails panel."""
        ax.set_xlabel('Bistatic Delay (km)', fontsize=10)
        ax.set_ylabel('Doppler Shift (Hz)', fontsize=10)
        ax.set_title('Detections in Delay-Doppler (with History Trails)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Current detections (large markers)
        scatter_current = ax.scatter(
            [], [], s=100, c='red', marker='o', edgecolors='white',
            linewidths=1.5, zorder=10, label='Current',
        )

        # History trail (smaller, fading markers)
        scatter_history = ax.scatter(
            [], [], c=[], s=20, cmap='plasma', vmin=0, vmax=self.history_duration,
            alpha=0.6, zorder=5,
        )

        # Legend
        ax.legend(loc='upper right', fontsize=9)

        self.artists['trail_current'] = scatter_current
        self.artists['trail_history'] = scatter_history

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
            history = list(self.detection_history)
            nf = self.frames_fetched
            nerr = self.fetch_errors

        # Sort doppler axis
        sort_idx = np.argsort(doppler)
        doppler_sorted = doppler[sort_idx]
        data_sorted = data[sort_idx, :]
        max_hold_sorted = max_hold[sort_idx, :]

        extent = [delay[0], delay[-1], doppler_sorted[0], doppler_sorted[-1]]

        # ---- Panel 1: Live Delay-Doppler ----
        im_dd = self.artists['dd_im']
        im_dd.set_data(data_sorted)
        im_dd.set_extent(extent)
        im_dd.set_clim(0, 15)
        self.axes['delay_doppler'].set_xlim(extent[0], extent[1])
        self.axes['delay_doppler'].set_ylim(extent[2], extent[3])

        # Current detections
        if detections_raw and 'delay' in detections_raw and len(detections_raw['delay']) > 0:
            det_x = np.asarray(detections_raw['delay'])
            det_y = np.asarray(detections_raw['doppler'])
            self.artists['dd_scatter'].set_offsets(np.column_stack([det_x, det_y]))
            n_det = len(det_x)
        else:
            self.artists['dd_scatter'].set_offsets(np.empty((0, 2)))
            n_det = 0

        # ---- Panel 2: Max-Hold ----
        im_mh = self.artists['mh_im']
        im_mh.set_data(max_hold_sorted)
        im_mh.set_extent(extent)
        im_mh.set_clim(0, 15)
        self.axes['max_hold'].set_xlim(extent[0], extent[1])
        self.axes['max_hold'].set_ylim(extent[2], extent[3])

        # Show same detections on max-hold
        if detections_raw and 'delay' in detections_raw and len(detections_raw['delay']) > 0:
            det_x = np.asarray(detections_raw['delay'])
            det_y = np.asarray(detections_raw['doppler'])
            self.artists['mh_scatter'].set_offsets(np.column_stack([det_x, det_y]))
        else:
            self.artists['mh_scatter'].set_offsets(np.empty((0, 2)))

        # ---- Panels 3 & 4: Detections over Time ----
        if history:
            times_ago = np.array([now - d.timestamp for d in history])
            delays = np.array([d.delay for d in history])
            dopplers = np.array([d.doppler for d in history])
            snrs = np.array([d.snr for d in history])

            # Filter to visible history
            mask = times_ago <= self.history_duration
            times_ago = times_ago[mask]
            delays = delays[mask]
            dopplers = dopplers[mask]
            snrs = snrs[mask]

            # Panel 3: Delay vs Time
            self.artists['dt_scatter'].set_offsets(np.column_stack([times_ago, delays]))
            self.artists['dt_scatter'].set_array(snrs)
            if len(delays) > 0:
                self.axes['det_delay_time'].set_ylim(
                    max(0, np.min(delays) - 5),
                    np.max(delays) + 5
                )

            # Panel 4: Doppler vs Time
            self.artists['dop_scatter'].set_offsets(np.column_stack([times_ago, dopplers]))
            self.artists['dop_scatter'].set_array(snrs)
            if len(dopplers) > 0:
                dop_margin = max(50, (np.max(dopplers) - np.min(dopplers)) * 0.1)
                self.axes['det_doppler_time'].set_ylim(
                    np.min(dopplers) - dop_margin,
                    np.max(dopplers) + dop_margin
                )

            # ---- Panel 5: Delay-Doppler Trails ----
            # History (older, fading)
            self.artists['trail_history'].set_offsets(np.column_stack([delays, dopplers]))
            self.artists['trail_history'].set_array(times_ago)  # Color by age

            # Current (most recent)
            if detections_raw and 'delay' in detections_raw and len(detections_raw['delay']) > 0:
                curr_x = np.asarray(detections_raw['delay'])
                curr_y = np.asarray(detections_raw['doppler'])
                self.artists['trail_current'].set_offsets(np.column_stack([curr_x, curr_y]))
            else:
                self.artists['trail_current'].set_offsets(np.empty((0, 2)))

            # Set axis limits for trails panel
            if len(delays) > 0:
                self.axes['det_trails'].set_xlim(
                    max(0, np.min(delays) - 2),
                    np.max(delays) + 2
                )
                self.axes['det_trails'].set_ylim(
                    np.min(dopplers) - 30,
                    np.max(dopplers) + 30
                )
        else:
            # No history
            self.artists['dt_scatter'].set_offsets(np.empty((0, 2)))
            self.artists['dop_scatter'].set_offsets(np.empty((0, 2)))
            self.artists['trail_history'].set_offsets(np.empty((0, 2)))
            self.artists['trail_current'].set_offsets(np.empty((0, 2)))

        # ---- Info Text ----
        runtime = now - self._start_time if self._start_time else 0
        self.artists['info_text'].set_text(
            f'Frames: {nf}  Errors: {nerr}  '
            f'Detections: {n_det}  History: {len(history)}  '
            f'Runtime: {runtime:.0f}s  Poll: {self.poll_interval:.1f}s'
        )

        return list(self.artists.values())

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def start(self, blocking: bool = True):
        """Start polling and display."""
        self.running = True
        self._start_time = time.time()

        # Start background fetch thread
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # Setup and run display
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
    parser.add_argument(
        '--history', type=float, default=60.0,
        help='Detection history duration in seconds (default: 60.0)',
    )
    parser.add_argument(
        '--decay', type=float, default=0.995,
        help='Max-hold decay factor per frame (default: 0.995)',
    )
    args = parser.parse_args()

    print(f'KrakenSDR Multi-Display Dashboard')
    print(f'  Server: {args.url}')
    print(f'  Poll interval: {args.interval}s')
    print(f'  History duration: {args.history}s')
    print(f'  Max-hold decay: {args.decay}')
    print()

    dashboard = MultiDisplayDashboard(
        base_url=args.url,
        poll_interval=args.interval,
        history_duration=args.history,
        max_hold_decay=args.decay,
    )
    dashboard.start()


if __name__ == '__main__':
    main()
