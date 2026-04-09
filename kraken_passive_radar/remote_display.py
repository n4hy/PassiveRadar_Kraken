"""
Remote Delay-Doppler Display Client for retnode.com KrakenSDR Passive Radar.

Fetches delay-Doppler map data from a remote KrakenSDR passive radar server
and displays it as a real-time heatmap.

Usage:
    python -m kraken_passive_radar.remote_display [--url URL] [--interval SEC]

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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RemoteRadarDisplay:
    """
    Fetches delay-Doppler map from a remote KrakenSDR passive radar API
    and displays it as a real-time heatmap.

    API endpoints used:
        /api/map       - delay-Doppler CAF matrix (JSON)
        /api/detection - CFAR detections (JSON)
        /api/timing    - processing timing info (JSON)
    """

    def __init__(self, base_url: str = 'https://radar3.retnode.com',
                 poll_interval: float = 1.0):
        """Initialize remote radar display with server URL and polling configuration.

        Technique: threaded HTTP polling with lock-guarded shared state for display updates.
        """
        self.base_url = base_url.rstrip('/')
        self.poll_interval = poll_interval

        # Data (guarded by lock)
        self.lock = threading.Lock()
        self.delay = None        # 1-D array, km
        self.doppler = None      # 1-D array, Hz
        self.map_data = None     # 2-D array (nRows x nCols), dB
        self.detections = None   # dict with delay/doppler/snr arrays
        self.timing = None       # dict with processing times
        self.last_timestamp = 0
        self.fetch_errors = 0
        self.frames_fetched = 0

        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.im = None
        self.colorbar = None
        self.det_scatter = None
        self.info_text = None
        self.cursor_text = None

        # State
        self._stop_event = threading.Event()
        self._poll_thread = None

    # ------------------------------------------------------------------ #
    #  Network
    # ------------------------------------------------------------------ #

    _UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'

    def _fetch_json(self, endpoint: str, timeout: float = 5.0):
        """Fetch JSON from an API endpoint. Returns dict or None on error."""
        url = f'{self.base_url}{endpoint}'
        # Exponential backoff on repeated failures (max 3s delay)
        if self.fetch_errors > 0:
            backoff = min(0.1 * (2 ** min(self.fetch_errors - 1, 5)), 3.0)
            time.sleep(backoff)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': self._UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            self.fetch_errors += 1
            if self.fetch_errors <= 3 or self.fetch_errors % 50 == 0:
                print(f'[remote_display] fetch {endpoint} failed ({self.fetch_errors}): {exc}')
            return None

    def _poll_loop(self):
        """Background thread: fetch map + detections at poll_interval."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()

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
                    self.last_timestamp = ts
                    self.frames_fetched += 1

            det_data = self._fetch_json('/api/detection')
            if det_data is not None:
                with self.lock:
                    self.detections = det_data

            timing = self._fetch_json('/api/timing')
            if timing is not None:
                with self.lock:
                    self.timing = timing

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self.poll_interval - elapsed)
            if sleep_time > 0:
                # Use event wait for interruptible sleep
                self._stop_event.wait(sleep_time)

    # ------------------------------------------------------------------ #
    #  Display
    # ------------------------------------------------------------------ #

    def _setup_plot(self):
        """Create the matplotlib figure."""
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(
            f'Remote Passive Radar — {self.base_url}'
        )

        # Placeholder image — will be replaced on first data frame
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
        self.ax.set_title('Delay-Doppler Map (connecting…)', fontsize=13)

        # Detection markers (empty initially)
        self.det_scatter = self.ax.scatter(
            [], [], s=60, facecolors='none', edgecolors='red',
            linewidths=1.5, zorder=5, label='Detections',
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

        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('close_event', lambda evt: self.stop())
        self.fig.tight_layout()

    def _on_mouse_move(self, event):
        """Update cursor readout with delay, Doppler, and power at mouse position.

        Technique: nearest-bin lookup via searchsorted on sorted axes.
        """
        if event.inaxes != self.ax:
            self.cursor_text.set_text('')
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        with self.lock:
            if self.delay is None or self.doppler is None or self.map_data is None:
                self.cursor_text.set_text('')
                return
            # Copy data inside lock to prevent race conditions
            delay = self.delay.copy()
            doppler = np.sort(self.doppler)
            data = self.map_data[np.argsort(self.doppler), :].copy()

        # Find nearest bin
        ri = np.searchsorted(delay, x)
        ri = np.clip(ri, 0, len(delay) - 1)
        di = np.searchsorted(doppler, y)
        di = np.clip(di, 0, data.shape[0] - 1)
        power = data[di, ri]

        self.cursor_text.set_text(
            f'Delay: {x:.2f} km  Doppler: {y:.1f} Hz  Power: {power:.1f} dB'
        )

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

        n_doppler, n_delay = data.shape

        # Sort doppler axis so origin='lower' works correctly
        sort_idx = np.argsort(doppler)
        doppler_sorted = doppler[sort_idx]
        # data rows correspond to doppler values
        data_sorted = data[sort_idx, :]

        extent = [delay[0], delay[-1], doppler_sorted[0], doppler_sorted[-1]]

        self.im.set_data(data_sorted)
        self.im.set_extent(extent)

        # Fixed 0-15 dB scale for bluer background
        self.im.set_clim(0, 15)

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])

        # Overlay detections
        if detections and 'delay' in detections and len(detections['delay']) > 0:
            det_x = np.asarray(detections['delay'])
            det_y = np.asarray(detections['doppler'])
            snr = np.asarray(detections['snr'])
            self.det_scatter.set_offsets(np.column_stack([det_x, det_y]))
            # Scale marker size by SNR
            sizes = 30 + snr * 5
            self.det_scatter.set_sizes(sizes)
            det_info = f'  Dets: {len(det_x)}'
        else:
            self.det_scatter.set_offsets(np.empty((0, 2)))
            det_info = '  Dets: 0'

        # Timing info
        cpi_ms = ''
        if timing:
            cpi_val = timing.get('cpi', 0)
            uptime_d = timing.get('uptime_days', 0)
            n_cpi = timing.get('nCpi', 0)
            cpi_ms = f'  CPI: {cpi_val:.0f}ms  Up: {uptime_d:.1f}d  #CPI: {n_cpi}'

        self.ax.set_title(
            f'Remote Delay-Doppler Map — {n_delay}×{n_doppler}',
            fontsize=13,
        )
        self.info_text.set_text(
            f'Frame {nf}{det_info}{cpi_ms}\n'
            f'Errors: {nerr}  Poll: {self.poll_interval:.1f}s'
        )

        return [self.im, self.det_scatter, self.info_text]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def start(self, blocking: bool = True):
        """Start polling and display."""
        self._stop_event.clear()

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
        self._stop_event.set()
        # Wait for polling thread to finish
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def main():
    """Parse command-line arguments and launch the remote radar display client.

    Technique: argparse CLI with server URL and poll interval options.
    """
    parser = argparse.ArgumentParser(
        description='Remote Delay-Doppler Display for KrakenSDR Passive Radar'
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

    print(f'Connecting to {args.url} (poll every {args.interval}s)...')
    display = RemoteRadarDisplay(base_url=args.url, poll_interval=args.interval)
    display.start()


if __name__ == '__main__':
    main()
