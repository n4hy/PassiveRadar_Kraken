"""
Five-Channel KrakenSDR Dashboard DEMO - Fully Simulated Animated Display.

Simulates realistic passive radar data for demonstration:
- Moving targets in CAF displays
- AOA estimates on PPI
- Channel health variations
- Detection history and tracks

Usage:
    python -m kraken_passive_radar.five_channel_demo

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons
from collections import deque
from dataclasses import dataclass
from typing import List, Deque


@dataclass
class SimulatedTarget:
    """A simulated moving target."""
    delay_km: float
    doppler_hz: float
    delay_rate: float  # km/s
    doppler_rate: float  # Hz/s
    snr_db: float
    azimuth_deg: float
    azimuth_rate: float  # deg/s


class FiveChannelDemo:
    """Fully simulated 5-channel dashboard for demonstration."""

    def __init__(self):
        """Initialize demo with simulated targets and display configuration.

        Technique: pre-configured target list with kinematic state vectors.
        """
        self.running = False
        self._start_time = None

        # Simulated targets
        self.targets: List[SimulatedTarget] = [
            SimulatedTarget(15, 50, 0.1, -0.5, 12, 45, 0.5),
            SimulatedTarget(25, -80, -0.05, 0.3, 15, 120, -0.3),
            SimulatedTarget(35, 120, 0.08, -0.2, 10, 220, 0.2),
            SimulatedTarget(8, -30, 0.15, 0.1, 18, 300, -0.1),
        ]

        # Detection history
        self.detection_history: Deque = deque(maxlen=500)

        # Display ranges
        self.delay_range = (0, 60)  # km
        self.doppler_range = (-300, 300)  # Hz
        self.ppi_range = 50  # km

        # Channel health (simulated)
        self.channel_snrs = [28, 25, 24, 26, 23]
        self.channel_phases = [0, 5, -3, 8, -2]
        self.channel_corr = [1.0, 0.96, 0.95, 0.97, 0.94]

        # Matplotlib objects
        self.fig = None
        self.ctrl_fig = None
        self.axes = {}
        self.artists = {}
        self.anim = None

    def _generate_caf(self, t: float, channel_offset: float = 0) -> np.ndarray:
        """Generate simulated CAF with targets.

        Technique: Gaussian blobs in delay-Doppler space over exponential noise floor.
        """
        delay_bins = np.linspace(self.delay_range[0], self.delay_range[1], 200)
        doppler_bins = np.linspace(self.doppler_range[0], self.doppler_range[1], 150)
        D, R = np.meshgrid(doppler_bins, delay_bins, indexing='ij')

        # Background noise
        caf = np.random.exponential(1.0, (150, 200)) * 0.5

        # Add targets
        for target in self.targets:
            # Current position
            delay = target.delay_km + target.delay_rate * t
            doppler = target.doppler_hz + target.doppler_rate * t

            # Wrap delay
            delay = delay % self.delay_range[1]

            # Gaussian blob for target
            sigma_r = 1.5
            sigma_d = 15
            snr = target.snr_db + channel_offset + np.random.normal(0, 1)
            amplitude = 10 ** (snr / 10)

            blob = amplitude * np.exp(-((R - delay)**2 / (2*sigma_r**2) +
                                        (D - doppler)**2 / (2*sigma_d**2)))
            caf += blob

        # Convert to dB
        caf_db = 10 * np.log10(caf + 1e-10)
        return caf_db, delay_bins, doppler_bins

    def _setup_main_figure(self):
        """Setup main display figure with 4x3 grid of subplot panels.

        Technique: matplotlib subplot grid with imshow heatmaps, scatter overlays, and bar charts.
        """
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.canvas.manager.set_window_title('KrakenSDR 5-Channel Demo')

        # Row 1: Surv 1, Surv 2, PPI
        ax1 = self.fig.add_subplot(4, 3, 1)
        ax1.set_title('Surveillance 1', fontweight='bold')
        im1 = ax1.imshow(np.zeros((150, 200)), aspect='auto', origin='lower',
                         extent=[0, 60, -300, 300], cmap='viridis', vmin=0, vmax=15)
        self.axes['ch0'] = ax1
        self.artists['ch0_im'] = im1

        ax2 = self.fig.add_subplot(4, 3, 2)
        ax2.set_title('Surveillance 2', fontweight='bold')
        im2 = ax2.imshow(np.zeros((150, 200)), aspect='auto', origin='lower',
                         extent=[0, 60, -300, 300], cmap='viridis', vmin=0, vmax=15)
        self.axes['ch1'] = ax2
        self.artists['ch1_im'] = im2

        # PPI (polar)
        ax_ppi = self.fig.add_subplot(2, 3, 3, projection='polar')
        ax_ppi.set_theta_zero_location('N')
        ax_ppi.set_theta_direction(-1)
        ax_ppi.set_ylim(0, self.ppi_range)
        ax_ppi.set_title('PPI Display', fontweight='bold', pad=10)
        scatter_ppi = ax_ppi.scatter([], [], c='red', s=100, alpha=0.8)
        self.axes['ppi'] = ax_ppi
        self.artists['ppi_scatter'] = scatter_ppi

        # Row 2: Surv 3, Surv 4
        ax3 = self.fig.add_subplot(4, 3, 4)
        ax3.set_title('Surveillance 3', fontweight='bold')
        im3 = ax3.imshow(np.zeros((150, 200)), aspect='auto', origin='lower',
                         extent=[0, 60, -300, 300], cmap='viridis', vmin=0, vmax=15)
        self.axes['ch2'] = ax3
        self.artists['ch2_im'] = im3

        ax4 = self.fig.add_subplot(4, 3, 5)
        ax4.set_title('Surveillance 4', fontweight='bold')
        im4 = ax4.imshow(np.zeros((150, 200)), aspect='auto', origin='lower',
                         extent=[0, 60, -300, 300], cmap='viridis', vmin=0, vmax=15)
        self.axes['ch3'] = ax4
        self.artists['ch3_im'] = im4

        # Row 3: Fused, Health, Trails
        ax_fused = self.fig.add_subplot(4, 3, 7)
        ax_fused.set_title('Fused CAF', fontweight='bold')
        im_fused = ax_fused.imshow(np.zeros((150, 200)), aspect='auto', origin='lower',
                                    extent=[0, 60, -300, 300], cmap='viridis', vmin=0, vmax=15)
        scatter_det = ax_fused.scatter([], [], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
        self.axes['fused'] = ax_fused
        self.artists['fused_im'] = im_fused
        self.artists['fused_det'] = scatter_det

        ax_health = self.fig.add_subplot(4, 3, 8)
        ax_health.set_title('Channel Health', fontweight='bold')
        ax_health.set_xlim(-0.5, 4.5)
        ax_health.set_ylim(0, 40)
        ax_health.set_xticks(range(5))
        ax_health.set_xticklabels(['Ref', 'S1', 'S2', 'S3', 'S4'])
        ax_health.axhline(15, color='yellow', linestyle='--', alpha=0.5)
        ax_health.axhline(25, color='green', linestyle='--', alpha=0.5)
        bars = ax_health.bar(range(5), self.channel_snrs, color='green', edgecolor='white')
        ax_health.grid(True, alpha=0.3, axis='y')
        self.axes['health'] = ax_health
        self.artists['health_bars'] = bars

        ax_trails = self.fig.add_subplot(4, 3, 9)
        ax_trails.set_title('Detection Trails', fontweight='bold')
        ax_trails.set_xlabel('Delay (km)')
        ax_trails.set_ylabel('Doppler (Hz)')
        ax_trails.set_xlim(0, 60)
        ax_trails.set_ylim(-300, 300)
        ax_trails.grid(True, alpha=0.3)
        scatter_curr = ax_trails.scatter([], [], s=100, c='red', marker='o', edgecolors='white', linewidths=2, zorder=10)
        scatter_hist = ax_trails.scatter([], [], s=20, c='orange', alpha=0.5, zorder=5)
        self.axes['trails'] = ax_trails
        self.artists['trails_curr'] = scatter_curr
        self.artists['trails_hist'] = scatter_hist

        # Row 4: Waterfalls
        ax_wf_delay = self.fig.add_subplot(4, 3, 10)
        ax_wf_delay.set_title('Delay vs Time', fontweight='bold')
        ax_wf_delay.set_xlabel('Time (s ago)')
        ax_wf_delay.set_ylabel('Delay (km)')
        ax_wf_delay.set_xlim(60, 0)
        ax_wf_delay.set_ylim(0, 60)
        ax_wf_delay.grid(True, alpha=0.3)
        scatter_wf_d = ax_wf_delay.scatter([], [], c='red', s=30, alpha=0.7)
        self.axes['wf_delay'] = ax_wf_delay
        self.artists['wf_delay_scatter'] = scatter_wf_d

        ax_wf_doppler = self.fig.add_subplot(4, 3, 11)
        ax_wf_doppler.set_title('Doppler vs Time', fontweight='bold')
        ax_wf_doppler.set_xlabel('Time (s ago)')
        ax_wf_doppler.set_ylabel('Doppler (Hz)')
        ax_wf_doppler.set_xlim(60, 0)
        ax_wf_doppler.set_ylim(-300, 300)
        ax_wf_doppler.grid(True, alpha=0.3)
        scatter_wf_dop = ax_wf_doppler.scatter([], [], c='red', s=30, alpha=0.7)
        self.axes['wf_doppler'] = ax_wf_doppler
        self.artists['wf_doppler_scatter'] = scatter_wf_dop

        ax_maxhold = self.fig.add_subplot(4, 3, 12)
        ax_maxhold.set_title('Max-Hold CAF', fontweight='bold')
        im_maxhold = ax_maxhold.imshow(np.zeros((150, 200)), aspect='auto', origin='lower',
                                        extent=[0, 60, -300, 300], cmap='viridis', vmin=0, vmax=15)
        self.axes['maxhold'] = ax_maxhold
        self.artists['maxhold_im'] = im_maxhold
        self.maxhold_data = None

        # Info text
        self.artists['info'] = self.fig.text(0.01, 0.99, '', fontsize=9, va='top',
                                              fontfamily='monospace',
                                              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        self.fig.tight_layout()

    def _setup_control_figure(self):
        """Setup control panel figure with target management buttons and speed slider.

        Technique: matplotlib widget-based GUI with Button and Slider controls.
        """
        self.ctrl_fig = plt.figure(figsize=(4, 8))
        self.ctrl_fig.canvas.manager.set_window_title('Controls')

        self.ctrl_fig.text(0.5, 0.95, 'Demo Controls', fontsize=12, fontweight='bold', ha='center')

        # Add target button
        ax_add = self.ctrl_fig.add_axes([0.2, 0.85, 0.6, 0.05])
        btn_add = Button(ax_add, 'Add Random Target')
        btn_add.on_clicked(self._add_target)
        self.artists['btn_add'] = btn_add

        # Clear button
        ax_clear = self.ctrl_fig.add_axes([0.2, 0.78, 0.6, 0.05])
        btn_clear = Button(ax_clear, 'Clear Targets')
        btn_clear.on_clicked(self._clear_targets)
        self.artists['btn_clear'] = btn_clear

        # Speed slider
        ax_speed = self.ctrl_fig.add_axes([0.2, 0.68, 0.6, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0)

        self.ctrl_fig.text(0.5, 0.55, 'Simulating 4 targets\nmoving in delay-Doppler space\n\n'
                           'PPI shows AOA estimates\nChannel health varies\nDetection trails accumulate',
                           fontsize=10, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def _add_target(self, event):
        """Add a random target with uniformly distributed kinematic parameters.

        Technique: random sampling of delay, Doppler, SNR, and azimuth state.
        """
        target = SimulatedTarget(
            delay_km=np.random.uniform(5, 50),
            doppler_hz=np.random.uniform(-200, 200),
            delay_rate=np.random.uniform(-0.2, 0.2),
            doppler_rate=np.random.uniform(-1, 1),
            snr_db=np.random.uniform(8, 18),
            azimuth_deg=np.random.uniform(0, 360),
            azimuth_rate=np.random.uniform(-0.5, 0.5),
        )
        self.targets.append(target)

    def _clear_targets(self, event):
        """Clear all targets."""
        self.targets.clear()
        self.detection_history.clear()
        self.maxhold_data = None

    def _on_close(self, event):
        """Handle figure close event."""
        self.running = False
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def _update(self, frame):
        """Animation update for all panels including CAFs, PPI, health, trails, and waterfalls.

        Technique: FuncAnimation callback updating matplotlib artists from simulated target state.
        """
        if not self.running or self.fig is None:
            return []
        t = (time.time() - self._start_time) * self.speed_slider.val
        now = time.time()

        # Update targets positions
        for target in self.targets:
            # Wrap delay
            target.delay_km = (target.delay_km + target.delay_rate * 0.1) % self.delay_range[1]
            # Wrap doppler
            if target.doppler_hz > self.doppler_range[1]:
                target.doppler_hz = self.doppler_range[0]
            elif target.doppler_hz < self.doppler_range[0]:
                target.doppler_hz = self.doppler_range[1]
            else:
                target.doppler_hz += target.doppler_rate * 0.1
            # Update azimuth
            target.azimuth_deg = (target.azimuth_deg + target.azimuth_rate) % 360

        # Generate CAFs for each channel
        channel_offsets = [0, -1, -2, 1, -1.5]
        for i in range(4):
            caf, delay, doppler = self._generate_caf(t, channel_offsets[i+1])
            self.artists[f'ch{i}_im'].set_data(caf)

        # Fused CAF (average of channels)
        fused_caf, delay, doppler = self._generate_caf(t, 0)
        self.artists['fused_im'].set_data(fused_caf)

        # Max-hold
        if self.maxhold_data is None:
            self.maxhold_data = fused_caf.copy()
        else:
            self.maxhold_data = np.maximum(self.maxhold_data * 0.995, fused_caf)
        self.artists['maxhold_im'].set_data(self.maxhold_data)

        # Detection scatter on fused
        if self.targets:
            det_x = [t.delay_km for t in self.targets]
            det_y = [t.doppler_hz for t in self.targets]
            self.artists['fused_det'].set_offsets(np.column_stack([det_x, det_y]))

            # Add to history
            for target in self.targets:
                self.detection_history.append({
                    'time': now,
                    'delay': target.delay_km,
                    'doppler': target.doppler_hz,
                })
        else:
            self.artists['fused_det'].set_offsets(np.empty((0, 2)))

        # PPI display
        if self.targets:
            angles = [np.radians(t.azimuth_deg) for t in self.targets]
            ranges = [t.delay_km for t in self.targets]
            self.artists['ppi_scatter'].set_offsets(np.column_stack([angles, ranges]))
        else:
            self.artists['ppi_scatter'].set_offsets(np.empty((0, 2)))

        # Channel health (with small variations)
        for i, bar in enumerate(self.artists['health_bars']):
            snr = self.channel_snrs[i] + np.random.normal(0, 0.5)
            bar.set_height(snr)
            if snr >= 25:
                bar.set_color('green')
            elif snr >= 15:
                bar.set_color('yellow')
            else:
                bar.set_color('red')

        # Detection trails
        if self.targets:
            curr_x = [t.delay_km for t in self.targets]
            curr_y = [t.doppler_hz for t in self.targets]
            self.artists['trails_curr'].set_offsets(np.column_stack([curr_x, curr_y]))
        else:
            self.artists['trails_curr'].set_offsets(np.empty((0, 2)))

        # History trails
        if self.detection_history:
            hist_x = [d['delay'] for d in self.detection_history]
            hist_y = [d['doppler'] for d in self.detection_history]
            self.artists['trails_hist'].set_offsets(np.column_stack([hist_x, hist_y]))
        else:
            self.artists['trails_hist'].set_offsets(np.empty((0, 2)))

        # Prune old history
        cutoff = now - 60
        while self.detection_history and self.detection_history[0]['time'] < cutoff:
            self.detection_history.popleft()

        # Waterfalls
        if self.detection_history:
            times_ago = [now - d['time'] for d in self.detection_history]
            delays = [d['delay'] for d in self.detection_history]
            dopplers = [d['doppler'] for d in self.detection_history]
            self.artists['wf_delay_scatter'].set_offsets(np.column_stack([times_ago, delays]))
            self.artists['wf_doppler_scatter'].set_offsets(np.column_stack([times_ago, dopplers]))
        else:
            self.artists['wf_delay_scatter'].set_offsets(np.empty((0, 2)))
            self.artists['wf_doppler_scatter'].set_offsets(np.empty((0, 2)))

        # Info text
        runtime = now - self._start_time
        self.artists['info'].set_text(
            f'Runtime: {runtime:.0f}s  Targets: {len(self.targets)}  History: {len(self.detection_history)}'
        )

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

        return list(self.artists.values())

    def start(self):
        """Start the demo with interactive matplotlib animation loop.

        Technique: FuncAnimation at 10 fps with interactive mode toggling for blocking display.
        """
        self.running = True
        self._start_time = time.time()

        plt.ion()

        self._setup_main_figure()
        self._setup_control_figure()

        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.ctrl_fig.canvas.mpl_connect('close_event', self._on_close)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ctrl_fig.canvas.draw()
        self.ctrl_fig.canvas.flush_events()

        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=100,  # 10 fps
            blit=False,
            cache_frame_data=False,
        )

        plt.ioff()
        plt.show()


def main():
    """Launch the five-channel demo with simulated moving targets.

    Technique: instantiates FiveChannelDemo and starts interactive display.
    """
    print("=" * 60)
    print("KrakenSDR 5-Channel Dashboard DEMO")
    print("=" * 60)
    print()
    print("Simulating 4 moving targets in delay-Doppler space")
    print("- All 4 surveillance channel CAFs with independent noise")
    print("- PPI display with AOA estimates")
    print("- Channel health monitoring")
    print("- Detection trails and waterfalls")
    print()
    print("Use the Controls window to add/remove targets")
    print()

    demo = FiveChannelDemo()
    demo.start()


if __name__ == '__main__':
    main()
