"""
Enhanced PPI (Plan Position Indicator) Display for Passive Radar
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Polar display showing range vs azimuth with:
- Detection markers
- Track visualization with history trails
- Velocity vectors
- Track status coloring
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow
from matplotlib.collections import LineCollection
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class PPIDetection:
    """Detection for PPI display."""
    azimuth_deg: float      # Degrees from North (0-360)
    range_m: float          # Range in meters
    power_db: float         # Signal power in dB
    doppler_hz: float = 0.0 # Doppler shift


@dataclass
class PPITrack:
    """Track state for PPI display."""
    id: int
    azimuth_deg: float      # Current azimuth (degrees)
    range_m: float          # Current range (meters)
    velocity_ms: float = 0.0      # Radial velocity (m/s)
    heading_deg: float = 0.0      # Heading (degrees, for velocity arrow)
    status: str = 'tentative'     # 'tentative', 'confirmed', 'coasting'
    score: float = 1.0            # Track quality (0-1)
    history: List[Tuple[float, float]] = field(default_factory=list)  # [(az, range), ...]


@dataclass
class PPIDisplayParams:
    """PPI display parameters."""
    max_range_km: float = 15.0
    range_rings_km: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])
    update_interval_ms: int = 100
    history_fade: bool = True
    history_max_points: int = 50
    show_velocity_arrows: bool = True
    arrow_scale: float = 0.1      # Arrow length scale (km per m/s)
    show_track_ids: bool = True
    show_grid: bool = True
    detection_marker_size: int = 80
    track_marker_size: int = 120


class PPIDisplay:
    """
    Enhanced Polar (PPI) Display for Passive Radar.

    Features:
    - Real-time detection markers (color by power)
    - Track markers with status coloring
    - Track history trails with fading
    - Velocity arrows showing heading
    - Track ID labels
    - Range rings and azimuth grid
    """

    def __init__(self, params: Optional[PPIDisplayParams] = None):
        self.params = params if params else PPIDisplayParams()

        # Data storage (thread-safe)
        self.lock = threading.Lock()
        self.detections: List[PPIDetection] = []
        self.tracks: List[PPITrack] = []

        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.detection_scatter = None
        self.track_scatter = None
        self.track_lines = {}       # track_id -> Line2D
        self.track_arrows = {}      # track_id -> FancyArrow
        self.track_labels = {}      # track_id -> Text
        self.colorbar = None
        self.info_text = None

        # Animation
        self.anim = None
        self.running = False

        # Track status colors
        self.status_colors = {
            'tentative': '#FFFF00',   # Yellow
            'confirmed': '#00FF00',   # Green
            'coasting': '#FFA500'     # Orange
        }

    def _setup_plot(self):
        """Initialize the matplotlib figure."""
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Passive Radar PPI Display')

        self.ax = self.fig.add_subplot(111, projection='polar')

        # North at top, clockwise
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)

        # Range limits
        self.ax.set_ylim(0, self.params.max_range_km)

        # Range rings
        self.ax.set_yticks(self.params.range_rings_km)
        self.ax.set_yticklabels([f'{r:.0f} km' for r in self.params.range_rings_km])

        # Azimuth labels
        self.ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        self.ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

        # Grid
        if self.params.show_grid:
            self.ax.grid(True, alpha=0.3)

        # Title
        self.ax.set_title('Passive Radar PPI Display', pad=20)

        # Detection scatter (color by power)
        self.detection_scatter = self.ax.scatter(
            [], [], c=[], cmap='hot', s=self.params.detection_marker_size,
            marker='o', alpha=0.6, vmin=-10, vmax=30,
            label='Detections', edgecolors='white', linewidths=0.5
        )

        # Track scatter (color by status, set later)
        self.track_scatter = self.ax.scatter(
            [], [], c='lime', s=self.params.track_marker_size,
            marker='D', alpha=0.9, edgecolors='white', linewidths=1.5,
            label='Tracks'
        )

        # Info text in corner
        self.info_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white', family='monospace'
        )

        # Legend
        self.ax.legend(loc='lower left', fontsize=8)

    def _update(self, frame):
        """Animation update callback."""
        with self.lock:
            detections = self.detections.copy()
            tracks = self.tracks.copy()

        # Update detections
        if detections:
            angles = [np.radians(d.azimuth_deg) for d in detections]
            ranges = [d.range_m / 1000.0 for d in detections]  # Convert to km
            powers = [d.power_db for d in detections]

            self.detection_scatter.set_offsets(np.column_stack([angles, ranges]))
            self.detection_scatter.set_array(np.array(powers))
        else:
            self.detection_scatter.set_offsets(np.empty((0, 2)))

        # Update tracks
        self._update_tracks(tracks)

        # Update info text
        num_dets = len(detections)
        num_tracks = len(tracks)
        num_confirmed = sum(1 for t in tracks if t.status == 'confirmed')
        self.info_text.set_text(
            f'Detections: {num_dets}\n'
            f'Tracks: {num_tracks} ({num_confirmed} confirmed)'
        )

        return [self.detection_scatter, self.track_scatter]

    def _update_tracks(self, tracks: List[PPITrack]):
        """Update track visualization."""
        active_ids = set()

        # Current track positions and colors
        track_angles = []
        track_ranges = []
        track_colors = []

        for track in tracks:
            active_ids.add(track.id)

            angle_rad = np.radians(track.azimuth_deg)
            range_km = track.range_m / 1000.0

            track_angles.append(angle_rad)
            track_ranges.append(range_km)
            track_colors.append(self.status_colors.get(track.status, '#FFFFFF'))

            # Track history trail
            if track.history and len(track.history) > 1:
                hist_angles = [np.radians(h[0]) for h in track.history]
                hist_ranges = [h[1] / 1000.0 for h in track.history]

                if track.id in self.track_lines:
                    self.track_lines[track.id].set_data(hist_angles, hist_ranges)
                    self.track_lines[track.id].set_color(self.status_colors.get(track.status, '#FFFFFF'))
                else:
                    line, = self.ax.plot(
                        hist_angles, hist_ranges,
                        color=self.status_colors.get(track.status, '#FFFFFF'),
                        linewidth=1.5, alpha=0.5, linestyle='-'
                    )
                    self.track_lines[track.id] = line

            # Velocity arrow
            if self.params.show_velocity_arrows and abs(track.velocity_ms) > 1.0:
                # Arrow in direction of heading
                arrow_len = abs(track.velocity_ms) * self.params.arrow_scale
                heading_rad = np.radians(track.heading_deg)

                # Calculate arrow end point
                end_angle = angle_rad + 0.1 * np.sign(track.velocity_ms)
                end_range = range_km + arrow_len

                if track.id in self.track_arrows:
                    # Update existing arrow (remove and recreate)
                    self.track_arrows[track.id].remove()

                arrow = self.ax.annotate(
                    '', xy=(end_angle, end_range), xytext=(angle_rad, range_km),
                    arrowprops=dict(
                        arrowstyle='->', color=self.status_colors.get(track.status, '#FFFFFF'),
                        lw=1.5, mutation_scale=10
                    )
                )
                self.track_arrows[track.id] = arrow

            # Track ID label
            if self.params.show_track_ids:
                label_text = f'T{track.id}'
                if track.id in self.track_labels:
                    self.track_labels[track.id].set_position((angle_rad, range_km))
                    self.track_labels[track.id].set_text(label_text)
                else:
                    text = self.ax.text(
                        angle_rad, range_km + 0.3, label_text,
                        fontsize=8, color='white', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
                    )
                    self.track_labels[track.id] = text

        # Update track scatter
        if track_angles:
            self.track_scatter.set_offsets(np.column_stack([track_angles, track_ranges]))
            self.track_scatter.set_facecolors(track_colors)
        else:
            self.track_scatter.set_offsets(np.empty((0, 2)))

        # Remove stale track elements
        for track_id in list(self.track_lines.keys()):
            if track_id not in active_ids:
                self.track_lines[track_id].remove()
                del self.track_lines[track_id]

        for track_id in list(self.track_arrows.keys()):
            if track_id not in active_ids:
                self.track_arrows[track_id].remove()
                del self.track_arrows[track_id]

        for track_id in list(self.track_labels.keys()):
            if track_id not in active_ids:
                self.track_labels[track_id].remove()
                del self.track_labels[track_id]

    def update_detections(self, detections: List[PPIDetection]):
        """Thread-safe update of detections."""
        with self.lock:
            self.detections = detections.copy()

    def update_tracks(self, tracks: List[PPITrack]):
        """Thread-safe update of tracks."""
        with self.lock:
            self.tracks = tracks.copy()

    def start(self, blocking: bool = True):
        """Start the display."""
        self.running = True
        self._setup_plot()

        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=self.params.update_interval_ms,
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


def demo_ppi_display():
    """Demo with simulated tracks."""
    params = PPIDisplayParams(
        max_range_km=20.0,
        range_rings_km=[5.0, 10.0, 15.0, 20.0],
        update_interval_ms=100
    )

    display = PPIDisplay(params)

    def data_generator():
        """Generate simulated track data."""
        import random

        # Simulated tracks
        track_states = [
            {'id': 1, 'az': 45, 'range': 8000, 'vel': 50, 'status': 'confirmed'},
            {'id': 2, 'az': 180, 'range': 12000, 'vel': -30, 'status': 'confirmed'},
            {'id': 3, 'az': 270, 'range': 5000, 'vel': 20, 'status': 'tentative'},
        ]

        histories = {t['id']: [] for t in track_states}

        while True:
            # Update track positions
            for t in track_states:
                # Move along azimuth
                t['az'] = (t['az'] + random.uniform(-1, 2)) % 360
                t['range'] += t['vel'] * 0.1  # 0.1 second update

                # Wrap range
                if t['range'] > 18000:
                    t['range'] = 5000
                if t['range'] < 3000:
                    t['range'] = 15000

                # Update history
                histories[t['id']].append((t['az'], t['range']))
                if len(histories[t['id']]) > 30:
                    histories[t['id']] = histories[t['id']][-30:]

            # Create track objects
            tracks = []
            for t in track_states:
                track = PPITrack(
                    id=t['id'],
                    azimuth_deg=t['az'],
                    range_m=t['range'],
                    velocity_ms=t['vel'],
                    heading_deg=t['az'] + 90,
                    status=t['status'],
                    history=histories[t['id']].copy()
                )
                tracks.append(track)

            display.update_tracks(tracks)

            # Random detections
            detections = []
            for _ in range(random.randint(3, 8)):
                det = PPIDetection(
                    azimuth_deg=random.uniform(0, 360),
                    range_m=random.uniform(3000, 18000),
                    power_db=random.uniform(5, 25),
                    doppler_hz=random.uniform(-100, 100)
                )
                detections.append(det)

            display.update_detections(detections)

            time.sleep(0.1)

    # Start data generator thread
    gen_thread = threading.Thread(target=data_generator, daemon=True)
    gen_thread.start()

    # Start display (blocking)
    display.start()


if __name__ == "__main__":
    demo_ppi_display()
