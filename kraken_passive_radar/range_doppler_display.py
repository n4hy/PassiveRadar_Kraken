"""
Range-Doppler Map Display for Passive Radar
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Real-time visualization of Cross-Ambiguity Function (CAF) output
as a Delay-Doppler heatmap matching the standard passive radar display style.
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
        """Initialize default history list if not provided.

        Technique: post-init processing for mutable default field.
        """
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
    Real-time Delay-Doppler Map Display.

    Clean full-frame heatmap with viridis colormap and vertical intensity
    colorbar. Matches standard passive radar delay-Doppler display style.
    """

    def __init__(self, params: Optional[RDDisplayParams] = None, update_interval_ms: int = 100):
        """Initialize range-Doppler display with configurable parameters.

        Technique: thread-safe data storage with lock-guarded CAF array and detection lists.
        """
        self.params = params if params else RDDisplayParams()
        self.interval = update_interval_ms

        # Data storage (thread-safe)
        self.lock = threading.Lock()
        self.caf_data_db = np.full(
            (self.params.n_doppler_bins, self.params.n_range_bins), -60.0
        )
        self.detections: List[Detection] = []
        self.tracks: List[Track] = []
        self.timestamp = 0.0

        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.im = None
        self.colorbar = None
        self.cursor_text = None

        # Display state
        self.running = False
        self.auto_scale = True

    def _compute_axis_values(self):
        """Compute range and Doppler axis values from bin indices and resolution parameters.

        Technique: linear mapping from bin index to physical units (km, Hz).
        """
        range_bins = np.arange(self.params.n_range_bins)
        self.range_km = range_bins * self.params.range_resolution_m / 1000.0

        doppler_bins = np.arange(self.params.n_doppler_bins)
        center = self.params.n_doppler_bins // 2
        self.doppler_hz = (doppler_bins - center) * self.params.doppler_resolution_hz

    def _setup_plot(self):
        """Initialize the matplotlib figure with viridis heatmap and cursor readout.

        Technique: imshow with bilinear interpolation and vertical colorbar.
        """
        self._compute_axis_values()

        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('Passive Radar Delay-Doppler Map')

        extent = [
            0, self.range_km[-1],
            self.doppler_hz[0], self.doppler_hz[-1]
        ]

        # Full-frame heatmap, viridis colormap
        self.im = self.ax.imshow(
            self.caf_data_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=self.params.cmap,
            interpolation='bilinear'
        )

        # Vertical colorbar on right - intensity scale
        self.colorbar = self.fig.colorbar(
            self.im, ax=self.ax, label='Intensity (dB)',
            fraction=0.03, pad=0.02
        )

        # Clean axis labels
        self.ax.set_xlabel('Bistatic Range (km)', fontsize=12)
        self.ax.set_ylabel('Doppler Shift (Hz)', fontsize=12)

        # Cursor readout - subtle dark background
        self.cursor_text = self.ax.text(
            0.01, 0.99, '', transform=self.ax.transAxes,
            fontsize=9, verticalalignment='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
        )

        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self.fig.tight_layout()

    def _on_mouse_move(self, event):
        """Update cursor readout on mouse move with range, Doppler, and power values.

        Technique: bin lookup from mouse coordinates with real-time text overlay.
        """
        if event.inaxes == self.ax:
            range_km = event.xdata
            doppler_hz = event.ydata
            if range_km is not None and doppler_hz is not None:
                range_bin = int(range_km * 1000 / self.params.range_resolution_m)
                doppler_bin = int(doppler_hz / self.params.doppler_resolution_hz + self.params.n_doppler_bins // 2)
                if 0 <= range_bin < self.params.n_range_bins and 0 <= doppler_bin < self.params.n_doppler_bins:
                    power_db = self.caf_data_db[doppler_bin, range_bin]
                    self.cursor_text.set_text(
                        f'R: {range_km:.1f} km  D: {doppler_hz:.1f} Hz  P: {power_db:.1f} dB'
                    )
                else:
                    self.cursor_text.set_text('')
        else:
            self.cursor_text.set_text('')

    def _update(self, frame):
        """Animation update callback that refreshes the heatmap with auto-scaling.

        Technique: percentile-based auto-scaling (5th to 99.5th percentile) for dynamic range.
        """
        with self.lock:
            caf_data = self.caf_data_db.copy()

        self.im.set_data(caf_data)

        # Auto-scale using percentile so the direct-path peak doesn't
        # crush everything else into the noise floor
        if self.auto_scale:
            vmax = np.percentile(caf_data, 99.5)
            vmin = np.percentile(caf_data, 5)
            # Ensure at least some dynamic range
            if vmax - vmin < 10.0:
                vmin = vmax - 10.0
            self.im.set_clim(vmin, vmax)

        return [self.im]

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
                self.params.n_doppler_bins, self.params.n_range_bins = caf_data_db.shape
                self.caf_data_db = caf_data_db.copy()
                self._compute_axis_values()

    def update_detections(self, detections: List[Detection]):
        """Thread-safe update of detections."""
        with self.lock:
            self.detections = detections.copy()

    def update_tracks(self, tracks: List[Track]):
        """Thread-safe update of tracks."""
        with self.lock:
            self.tracks = tracks.copy()

    def set_dynamic_range(self, vmin: float, vmax: float):
        """Set the display dynamic range in dB and disable auto-scale.

        Technique: direct clim update on imshow artist.
        """
        self.auto_scale = False
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
        """Stop the display and clean up resources."""
        self.running = False
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def _generate_target_response(caf_db, r_bin, d_bin, peak_db, n_range, n_doppler,
                               range_spread=6, doppler_spread=3):
    """
    Paint a realistic target response into the delay-Doppler map.

    Real targets produce a sinc-like spread in range (from matched filter)
    and some Doppler spread (from CPI windowing). This creates visible
    cross-shaped responses like real passive radar data.
    """
    # Range sidelobe pattern (sinc-like decay)
    for dr in range(-range_spread, range_spread + 1):
        ri = r_bin + dr
        if 0 <= ri < n_range:
            if dr == 0:
                r_atten = 0.0
            else:
                r_atten = 13.0 + 5.0 * np.log10(abs(dr))  # sinc sidelobes
            for dd in range(-doppler_spread, doppler_spread + 1):
                di = d_bin + dd
                if 0 <= di < n_doppler:
                    if dd == 0:
                        d_atten = 0.0
                    else:
                        d_atten = 10.0 + 4.0 * np.log10(abs(dd))
                    level = peak_db - r_atten - d_atten
                    if level > caf_db[di, ri]:
                        caf_db[di, ri] = level

    # Main peak (bright center)
    if 0 <= r_bin < n_range and 0 <= d_bin < n_doppler:
        caf_db[d_bin, r_bin] = max(caf_db[d_bin, r_bin], peak_db)


def demo_delay_doppler():
    """
    Demo with realistic simulated passive radar data.

    Generates a delay-Doppler map matching real FM passive radar output:
    - Direct-path signal at (0 range, 0 Doppler) with range/Doppler spreading
    - Noise floor
    - Multiple moving targets with sinc-like sidelobe responses
    - Multipath clutter at near ranges
    """
    # RSPduo parameters: 2 MHz sample rate, 50 km max range
    range_res = 75.0      # m/bin at 2 MHz
    max_range_km = 50.0
    n_range = int(max_range_km * 1000 / range_res) + 1  # 667 bins
    n_doppler = 256
    doppler_res = 2e6 / (4096 * n_doppler)  # 1.91 Hz/bin
    max_doppler = doppler_res * (n_doppler // 2)

    params = RDDisplayParams(
        n_range_bins=n_range,
        n_doppler_bins=n_doppler,
        range_resolution_m=range_res,
        doppler_resolution_hz=doppler_res,
        max_range_km=max_range_km,
        min_doppler_hz=-max_doppler,
        max_doppler_hz=max_doppler,
        dynamic_range_db=40.0,
    )

    display = RangeDopplerDisplay(params, update_interval_ms=100)

    def data_generator():
        """Generate realistic passive radar delay-Doppler data."""
        rng = np.random.default_rng(42)
        center_d = n_doppler // 2

        # Target state: [range_m, doppler_hz, peak_db, range_rate_m_per_frame]
        targets = [
            [6000.0,    45.0,  20.0,   12.0],   # car on highway ~55 km/h
            [12000.0,  -38.0,  16.0,  -10.0],   # approaching vehicle
            [8500.0,    70.0,  14.0,   18.0],    # faster vehicle
            [22000.0,  110.0,  18.0,   30.0],    # aircraft
            [30000.0, -150.0,  15.0,  -40.0],    # fast aircraft approaching
            [4000.0,    22.0,  12.0,    5.0],    # slow vehicle, weak
            [18000.0,  -80.0,  13.0,  -15.0],    # moderate target
            [40000.0,  180.0,  10.0,   50.0],    # distant fast aircraft
        ]

        frame = 0
        while True:
            # Noise floor: exponential (Rayleigh power) -> dB
            noise_linear = rng.exponential(scale=1.0, size=(n_doppler, n_range))
            caf_db = 10.0 * np.log10(noise_linear + 1e-20)
            # Noise floor is now centered around 0 dB with spread ~[-15, +8]

            # Direct-path: strong peak at (0 range, 0 Doppler) with
            # sidelobe ridge along range axis at 0 Doppler
            # Peak is ~40 dB above noise
            dp_peak = 40.0
            # Range sidelobe ridge at zero-Doppler
            for r in range(min(200, n_range)):
                decay = dp_peak - 10.0 * np.log10(max(r, 1)) - 2.0
                for dd in range(-2, 3):
                    di = center_d + dd
                    if 0 <= di < n_doppler:
                        d_atten = 8.0 * abs(dd)
                        level = decay - d_atten
                        if level > caf_db[di, r]:
                            caf_db[di, r] = level
            # Doppler sidelobe ridge at zero-range
            for d in range(n_doppler):
                d_dist = abs(d - center_d)
                if d_dist == 0:
                    continue
                level = dp_peak - 10.0 * np.log10(d_dist) - 6.0
                for dr in range(min(5, n_range)):
                    r_atten = 8.0 * dr
                    val = level - r_atten
                    if val > caf_db[d, dr]:
                        caf_db[d, dr] = val

            # Multipath clutter: faint returns at near ranges, various Dopplers
            n_clutter = 12 + rng.integers(0, 5)
            for _ in range(n_clutter):
                cr = rng.integers(20, 150)
                cd = rng.integers(center_d - 8, center_d + 8)
                clevel = rng.uniform(4.0, 10.0)
                _generate_target_response(caf_db, cr, cd, clevel,
                                          n_range, n_doppler,
                                          range_spread=3, doppler_spread=2)

            # Add targets with realistic sidelobe responses
            for t in targets:
                t_range, t_doppler, t_peak, t_rate = t
                r_bin = int(t_range / range_res)
                d_bin = int(t_doppler / doppler_res + center_d)

                # Add per-frame SNR fluctuation (Swerling I)
                fluct = t_peak + rng.standard_normal() * 2.0

                _generate_target_response(caf_db, r_bin, d_bin, fluct,
                                          n_range, n_doppler,
                                          range_spread=8, doppler_spread=3)

                # Move target
                t[0] += t_rate
                if t[0] > max_range_km * 1000:
                    t[0] = 3000.0 + rng.uniform(0, 2000)
                elif t[0] < 1000:
                    t[0] = max_range_km * 800 + rng.uniform(0, 2000)

            display.update_caf(caf_db)
            frame += 1
            time.sleep(0.1)

    gen_thread = threading.Thread(target=data_generator, daemon=True)
    gen_thread.start()

    display.start()


if __name__ == "__main__":
    demo_delay_doppler()
