"""
Range-Doppler Map Display for Passive Radar
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Real-time visualization of Cross-Ambiguity Function (CAF) output
with detection and track overlays.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time


@dataclass
class Detection:
    """Single detection from CFAR."""
    range_bin: int
    doppler_bin: int
    snr_db: float
    range_m: float = 0.0
    doppler_hz: float = 0.0


@dataclass
class Track:
    """Track state for visualization."""
    id: int
    range_m: float
    doppler_hz: float
    range_rate: float = 0.0
    doppler_rate: float = 0.0
    status: str = 'tentative'  # 'tentative', 'confirmed', 'coasting'
    history: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class RDDisplayParams:
    """Range-Doppler display parameters."""
    n_range_bins: int = 256
    n_doppler_bins: int = 64
    range_resolution_m: float = 600.0  # ~600m for 250 kHz BW
    doppler_resolution_hz: float = 3.9  # ~3.9 Hz for 64 CPIs at 250 kHz
    max_range_km: float = 15.0
    min_doppler_hz: float = -125.0
    max_doppler_hz: float = 125.0
    dynamic_range_db: float = 60.0
    cmap: str = 'viridis'


class RangeDopplerDisplay:
    """
    Real-time Range-Doppler Map Display.

    Features:
    - Heatmap of CAF output in dB scale
    - Detection overlays (circles)
    - Track overlays (lines with history)
    - Cursor readout (range, velocity)
    - Adjustable dynamic range
    """

    def __init__(self, params: Optional[RDDisplayParams] = None, update_interval_ms: int = 100):
        self.params = params if params else RDDisplayParams()
        self.interval = update_interval_ms

        # Data storage (thread-safe)
        self.lock = threading.Lock()
        self.caf_data_db = np.zeros((self.params.n_doppler_bins, self.params.n_range_bins))
        self.detections: List[Detection] = []
        self.tracks: List[Track] = []
        self.timestamp = 0.0

        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.im = None
        self.detection_scatter = None
        self.track_lines = {}
        self.track_markers = {}
        self.colorbar = None
        self.cursor_text = None

        # Display state
        self.running = False
        self.vmin = -30.0
        self.vmax = 30.0

    def _compute_axis_values(self):
        """Compute range and Doppler axis values."""
        # Range axis (km)
        range_bins = np.arange(self.params.n_range_bins)
        self.range_km = range_bins * self.params.range_resolution_m / 1000.0

        # Doppler axis (Hz) - centered at zero
        doppler_bins = np.arange(self.params.n_doppler_bins)
        center = self.params.n_doppler_bins // 2
        self.doppler_hz = (doppler_bins - center) * self.params.doppler_resolution_hz

        # Velocity axis (m/s) assuming FM band (~100 MHz)
        # v = f_d * c / (2 * f_c), but for bistatic: v = f_d * lambda
        # Approximate: v ~= f_d / 3 for FM band
        self.velocity_ms = self.doppler_hz / 3.0

    def _setup_plot(self):
        """Initialize the matplotlib figure."""
        self._compute_axis_values()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Range-Doppler Map')

        # Create extent for proper axis labeling
        extent = [
            0, self.range_km[-1],
            self.doppler_hz[0], self.doppler_hz[-1]
        ]

        # Initialize heatmap
        self.im = self.ax.imshow(
            self.caf_data_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=self.params.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation='nearest'
        )

        # Colorbar
        self.colorbar = self.fig.colorbar(self.im, ax=self.ax, label='Power (dB)')

        # Labels
        self.ax.set_xlabel('Bistatic Range (km)')
        self.ax.set_ylabel('Doppler Shift (Hz)')
        self.ax.set_title('Passive Radar Range-Doppler Map')

        # Secondary y-axis for velocity
        self.ax2 = self.ax.secondary_yaxis('right',
            functions=(lambda x: x / 3.0, lambda x: x * 3.0))
        self.ax2.set_ylabel('Radial Velocity (m/s)')

        # Detection overlay (scatter)
        self.detection_scatter = self.ax.scatter(
            [], [], c='red', s=100, marker='o', alpha=0.8,
            edgecolors='white', linewidths=1.5, label='Detections'
        )

        # Cursor readout text
        self.cursor_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Connect mouse motion for cursor readout
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        # Legend
        self.ax.legend(loc='upper right')

        # Grid
        self.ax.grid(True, alpha=0.3)

    def _on_mouse_move(self, event):
        """Update cursor readout on mouse move."""
        if event.inaxes == self.ax:
            range_km = event.xdata
            doppler_hz = event.ydata
            if range_km is not None and doppler_hz is not None:
                velocity_ms = doppler_hz / 3.0
                # Find nearest bin
                range_bin = int(range_km * 1000 / self.params.range_resolution_m)
                doppler_bin = int(doppler_hz / self.params.doppler_resolution_hz + self.params.n_doppler_bins // 2)
                if 0 <= range_bin < self.params.n_range_bins and 0 <= doppler_bin < self.params.n_doppler_bins:
                    power_db = self.caf_data_db[doppler_bin, range_bin]
                    self.cursor_text.set_text(
                        f'Range: {range_km:.2f} km\n'
                        f'Doppler: {doppler_hz:.1f} Hz\n'
                        f'Velocity: {velocity_ms:.1f} m/s\n'
                        f'Power: {power_db:.1f} dB'
                    )
                else:
                    self.cursor_text.set_text('')
        else:
            self.cursor_text.set_text('')

    def _update(self, frame):
        """Animation update callback."""
        with self.lock:
            caf_data = self.caf_data_db.copy()
            detections = self.detections.copy()
            tracks = self.tracks.copy()

        # Update heatmap
        self.im.set_data(caf_data)

        # Update detection markers
        if detections:
            det_ranges = [d.range_m / 1000.0 for d in detections]  # Convert to km
            det_dopplers = [d.doppler_hz for d in detections]
            self.detection_scatter.set_offsets(np.column_stack([det_ranges, det_dopplers]))
        else:
            self.detection_scatter.set_offsets(np.empty((0, 2)))

        # Update track visualization
        self._update_tracks(tracks)

        return [self.im, self.detection_scatter]

    def _update_tracks(self, tracks: List[Track]):
        """Update track lines and markers."""
        active_ids = set()

        for track in tracks:
            active_ids.add(track.id)

            # Color based on status
            if track.status == 'confirmed':
                color = 'lime'
                marker = 'D'
            elif track.status == 'tentative':
                color = 'yellow'
                marker = 's'
            else:  # coasting
                color = 'orange'
                marker = 'x'

            range_km = track.range_m / 1000.0

            # Update or create track marker
            if track.id not in self.track_markers:
                self.track_markers[track.id], = self.ax.plot(
                    [range_km], [track.doppler_hz],
                    color=color, marker=marker, markersize=10, linestyle='none'
                )
            else:
                self.track_markers[track.id].set_data([range_km], [track.doppler_hz])
                self.track_markers[track.id].set_color(color)
                self.track_markers[track.id].set_marker(marker)

            # Update or create track history line
            if track.history and len(track.history) > 1:
                hist_ranges = [h[0] / 1000.0 for h in track.history]
                hist_dopplers = [h[1] for h in track.history]

                if track.id not in self.track_lines:
                    self.track_lines[track.id], = self.ax.plot(
                        hist_ranges, hist_dopplers,
                        color=color, linewidth=1.5, alpha=0.6
                    )
                else:
                    self.track_lines[track.id].set_data(hist_ranges, hist_dopplers)
                    self.track_lines[track.id].set_color(color)

        # Remove stale tracks
        for track_id in list(self.track_markers.keys()):
            if track_id not in active_ids:
                self.track_markers[track_id].remove()
                del self.track_markers[track_id]
                if track_id in self.track_lines:
                    self.track_lines[track_id].remove()
                    del self.track_lines[track_id]

    def update_caf(self, caf_data_db: np.ndarray):
        """
        Thread-safe update of CAF data.

        Args:
            caf_data_db: 2D array of CAF power in dB, shape (n_doppler, n_range)
        """
        with self.lock:
            if caf_data_db.shape == self.caf_data_db.shape:
                self.caf_data_db = caf_data_db.copy()
            else:
                # Reshape if needed
                self.params.n_doppler_bins, self.params.n_range_bins = caf_data_db.shape
                self.caf_data_db = caf_data_db.copy()
                self._compute_axis_values()

    def update_detections(self, detections: List[Detection]):
        """
        Thread-safe update of detections.

        Args:
            detections: List of Detection objects
        """
        with self.lock:
            self.detections = detections.copy()

    def update_tracks(self, tracks: List[Track]):
        """
        Thread-safe update of tracks.

        Args:
            tracks: List of Track objects
        """
        with self.lock:
            self.tracks = tracks.copy()

    def set_dynamic_range(self, vmin: float, vmax: float):
        """Set the display dynamic range in dB."""
        self.vmin = vmin
        self.vmax = vmax
        if self.im is not None:
            self.im.set_clim(vmin, vmax)

    def start(self, blocking: bool = True):
        """
        Start the display.

        Args:
            blocking: If True, blocks until window is closed
        """
        self.running = True
        self._setup_plot()

        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=self.interval,
            blit=False,
            cache_frame_data=False
        )

        if blocking:
            plt.show()
        else:
            plt.show(block=False)

    def stop(self):
        """Stop the display."""
        self.running = False
        plt.close(self.fig)


def demo_range_doppler_display():
    """Demo with simulated data."""
    params = RDDisplayParams(
        n_range_bins=256,
        n_doppler_bins=64,
        range_resolution_m=600.0,
        doppler_resolution_hz=3.9
    )

    display = RangeDopplerDisplay(params, update_interval_ms=100)

    def data_generator():
        """Generate simulated data."""
        import random

        frame = 0
        target_range = 5000.0  # meters
        target_doppler = 50.0  # Hz

        while True:
            # Simulated CAF with noise and target
            caf = np.random.randn(64, 256) * 5 - 20  # Noise floor at -20 dB

            # Add target peak
            target_range_bin = int(target_range / 600.0)
            target_doppler_bin = int(target_doppler / 3.9 + 32)
            if 0 <= target_range_bin < 256 and 0 <= target_doppler_bin < 64:
                caf[target_doppler_bin-1:target_doppler_bin+2,
                    target_range_bin-1:target_range_bin+2] = 20.0

            display.update_caf(caf)

            # Simulated detection
            detections = [
                Detection(
                    range_bin=target_range_bin,
                    doppler_bin=target_doppler_bin,
                    snr_db=25.0,
                    range_m=target_range,
                    doppler_hz=target_doppler
                )
            ]
            display.update_detections(detections)

            # Simulated track
            track = Track(
                id=1,
                range_m=target_range,
                doppler_hz=target_doppler,
                status='confirmed',
                history=[(target_range - i*50, target_doppler - i*0.5) for i in range(20)]
            )
            display.update_tracks([track])

            # Move target
            target_range += 50  # 50 m/frame
            target_doppler -= 0.5  # Slight Doppler change

            if target_range > 12000:
                target_range = 3000
                target_doppler = random.uniform(20, 80)

            frame += 1
            time.sleep(0.1)

    # Start data generator thread
    import threading
    gen_thread = threading.Thread(target=data_generator, daemon=True)
    gen_thread.start()

    # Start display (blocking)
    display.start()


if __name__ == "__main__":
    demo_range_doppler_display()
