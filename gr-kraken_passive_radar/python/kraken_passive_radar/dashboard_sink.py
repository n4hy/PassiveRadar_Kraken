"""
Dashboard Sink Block for GNU Radio - Five Channel KrakenSDR Display.

Receives CAF data directly from the GNU Radio flowgraph and displays
it in real-time. No network connection needed.

Usage in GNU Radio Companion or Python flowgraph:
    from kraken_passive_radar import dashboard_sink

    sink = dashboard_sink.dashboard_sink(
        fft_len=1024,
        doppler_len=256,
        num_channels=4,
        sample_rate=2.4e6,
    )

    # Connect each surveillance channel's CAF output
    self.connect((doppler_proc_ch1, 0), (sink, 0))
    self.connect((doppler_proc_ch2, 0), (sink, 1))
    self.connect((doppler_proc_ch3, 0), (sink, 2))
    self.connect((doppler_proc_ch4, 0), (sink, 3))

Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT
"""

import numpy as np
from gnuradio import gr
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque

# Import matplotlib with TkAgg backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons


@dataclass
class ChannelHealth:
    """Health metrics for a channel."""
    snr_db: float = 0.0
    phase_offset_deg: float = 0.0
    correlation_coeff: float = 1.0


@dataclass
class Detection:
    """A radar detection."""
    timestamp: float
    delay_km: float
    doppler_hz: float
    snr_db: float
    channel: int = -1


class dashboard_sink(gr.sync_block):
    """
    GNU Radio sink block that displays 5-channel passive radar data.

    Inputs: 4 surveillance channel CAF images (flattened float32 arrays)
    Each input is fft_len * doppler_len floats representing the CAF in dB.
    """

    def __init__(
        self,
        fft_len: int = 1024,
        doppler_len: int = 256,
        num_channels: int = 4,
        sample_rate: float = 2.4e6,
        center_freq: float = 100e6,
        update_rate: float = 10.0,
    ):
        """
        Initialize the dashboard sink.

        Args:
            fft_len: FFT size (number of range bins)
            doppler_len: Number of Doppler bins
            num_channels: Number of surveillance channels (default 4)
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            update_rate: Display update rate in Hz
        """
        # Input signature: num_channels inputs, each fft_len*doppler_len floats
        in_sig = [(np.float32, fft_len * doppler_len)] * num_channels

        gr.sync_block.__init__(
            self,
            name="Dashboard Sink",
            in_sig=in_sig,
            out_sig=None,
        )

        self.fft_len = fft_len
        self.doppler_len = doppler_len
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.update_rate = update_rate

        # Compute physical axes
        # Range: c * tau / 2, but for bistatic we use delay directly
        # Delay bins to km (assuming 150m resolution per bin at 2.4MHz)
        self.range_res_m = 3e8 / (2 * sample_rate)  # meters per bin
        self.delay_axis_km = np.arange(fft_len) * self.range_res_m / 1000.0

        # Doppler axis in Hz
        doppler_res = sample_rate / fft_len / doppler_len
        self.doppler_axis_hz = (np.arange(doppler_len) - doppler_len // 2) * doppler_res

        # Data buffers (thread-safe)
        self.lock = threading.Lock()
        self.channel_cafs: List[Optional[np.ndarray]] = [None] * num_channels
        self.fused_caf: Optional[np.ndarray] = None
        self.max_hold_caf: Optional[np.ndarray] = None
        self.channel_health: List[ChannelHealth] = [ChannelHealth() for _ in range(5)]
        self.detection_history: Deque[Detection] = deque(maxlen=1000)
        self.frames_received = 0
        self.last_update_time = 0.0

        # Display parameters
        self.color_min = 0.0
        self.color_max = 15.0
        self.max_hold_decay = 0.995
        self.history_duration = 60.0
        self.ppi_range_km = 50.0
        self.cfar_threshold = 12.0

        # Matplotlib objects
        self.fig = None
        self.ctrl_fig = None
        self.axes = {}
        self.artists = {}
        self.widgets = {}
        self.anim = None

        # Start display thread
        self._display_thread = None
        self._running = False

    def start(self):
        """Called when flowgraph starts."""
        self._running = True
        self._display_thread = threading.Thread(target=self._run_display, daemon=True)
        self._display_thread.start()
        return True

    def stop(self):
        """Called when flowgraph stops."""
        self._running = False
        if self.fig is not None:
            plt.close(self.fig)
        if self.ctrl_fig is not None:
            plt.close(self.ctrl_fig)
        return True

    def work(self, input_items, output_items):
        """Process incoming CAF data."""
        now = time.time()

        with self.lock:
            # Store each channel's CAF
            for i in range(self.num_channels):
                if len(input_items[i]) > 0:
                    # Reshape from flattened to 2D (doppler x range)
                    caf = input_items[i][0].reshape(self.doppler_len, self.fft_len)
                    self.channel_cafs[i] = caf.copy()

            # Compute fused CAF (average of all channels)
            valid_cafs = [c for c in self.channel_cafs if c is not None]
            if valid_cafs:
                self.fused_caf = np.mean(valid_cafs, axis=0)

                # Update max-hold
                if self.max_hold_caf is None:
                    self.max_hold_caf = self.fused_caf.copy()
                else:
                    self.max_hold_caf *= self.max_hold_decay
                    self.max_hold_caf = np.maximum(self.max_hold_caf, self.fused_caf)

                # Simple peak detection for detections
                self._detect_peaks(self.fused_caf, now)

                # Estimate channel health from CAF statistics
                self._update_channel_health()

            self.frames_received += 1
            self.last_update_time = now

        return 1  # Consumed 1 item from each input

    def _detect_peaks(self, caf: np.ndarray, timestamp: float):
        """Simple peak detection on CAF."""
        threshold = np.median(caf) + self.cfar_threshold
        peaks = np.argwhere(caf > threshold)

        for peak in peaks[:10]:  # Limit to 10 detections per frame
            doppler_idx, range_idx = peak
            snr = caf[doppler_idx, range_idx] - np.median(caf)

            self.detection_history.append(Detection(
                timestamp=timestamp,
                delay_km=self.delay_axis_km[range_idx],
                doppler_hz=self.doppler_axis_hz[doppler_idx],
                snr_db=snr,
            ))

    def _update_channel_health(self):
        """Update channel health estimates from CAF data."""
        # Reference channel (index 0)
        self.channel_health[0].snr_db = 30.0  # Assumed good
        self.channel_health[0].correlation_coeff = 1.0

        # Surveillance channels
        for i in range(self.num_channels):
            if self.channel_cafs[i] is not None:
                caf = self.channel_cafs[i]
                peak = np.max(caf)
                noise = np.median(caf)
                snr = peak - noise

                self.channel_health[i + 1].snr_db = max(0, min(40, snr + 15))
                self.channel_health[i + 1].correlation_coeff = min(1.0, 0.9 + snr / 100)
                self.channel_health[i + 1].phase_offset_deg = (i - 1.5) * 10  # Placeholder

    def _run_display(self):
        """Display thread - runs matplotlib."""
        plt.ion()

        self._setup_main_figure()
        self._setup_control_figure()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ctrl_fig.canvas.draw()
        self.ctrl_fig.canvas.flush_events()

        self.anim = FuncAnimation(
            self.fig, self._update_display,
            interval=int(1000 / self.update_rate),
            blit=False,
            cache_frame_data=False,
        )

        plt.ioff()
        plt.show()

    def _setup_main_figure(self):
        """Setup main display figure."""
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.canvas.manager.set_window_title('KrakenSDR 5-Channel Dashboard (Live)')

        # Layout: 4 rows x 3 columns
        # Row 1: Surv1, Surv2, PPI
        # Row 2: Surv3, Surv4, (PPI continues)
        # Row 3: Fused, Health, Trails
        # Row 4: Delay WF, Doppler WF, Max-Hold

        # Surveillance channel CAFs
        for i in range(4):
            row = i // 2
            col = i % 2
            pos = row * 3 + col + 1
            ax = self.fig.add_subplot(4, 3, pos)
            ax.set_title(f'Surveillance {i+1}', fontweight='bold')
            ax.set_xlabel('Delay (km)')
            ax.set_ylabel('Doppler (Hz)')
            im = ax.imshow(
                np.zeros((self.doppler_len, self.fft_len)),
                aspect='auto', origin='lower',
                extent=[self.delay_axis_km[0], self.delay_axis_km[-1],
                        self.doppler_axis_hz[0], self.doppler_axis_hz[-1]],
                cmap='viridis', vmin=self.color_min, vmax=self.color_max,
            )
            self.axes[f'ch{i}'] = ax
            self.artists[f'ch{i}_im'] = im

        # PPI display
        ax_ppi = self.fig.add_subplot(2, 3, 3, projection='polar')
        ax_ppi.set_theta_zero_location('N')
        ax_ppi.set_theta_direction(-1)
        ax_ppi.set_ylim(0, self.ppi_range_km)
        ax_ppi.set_title('PPI Display', fontweight='bold', pad=10)
        scatter_ppi = ax_ppi.scatter([], [], c='red', s=100, alpha=0.8)
        self.axes['ppi'] = ax_ppi
        self.artists['ppi_scatter'] = scatter_ppi

        # Fused CAF
        ax_fused = self.fig.add_subplot(4, 3, 7)
        ax_fused.set_title('Fused CAF', fontweight='bold')
        ax_fused.set_xlabel('Delay (km)')
        ax_fused.set_ylabel('Doppler (Hz)')
        im_fused = ax_fused.imshow(
            np.zeros((self.doppler_len, self.fft_len)),
            aspect='auto', origin='lower',
            extent=[self.delay_axis_km[0], self.delay_axis_km[-1],
                    self.doppler_axis_hz[0], self.doppler_axis_hz[-1]],
            cmap='viridis', vmin=self.color_min, vmax=self.color_max,
        )
        scatter_det = ax_fused.scatter([], [], s=80, facecolors='none',
                                        edgecolors='red', linewidths=2, zorder=5)
        self.axes['fused'] = ax_fused
        self.artists['fused_im'] = im_fused
        self.artists['fused_det'] = scatter_det

        # Channel health
        ax_health = self.fig.add_subplot(4, 3, 8)
        ax_health.set_title('Channel Health', fontweight='bold')
        ax_health.set_xlim(-0.5, 4.5)
        ax_health.set_ylim(0, 40)
        ax_health.set_xticks(range(5))
        ax_health.set_xticklabels(['Ref', 'S1', 'S2', 'S3', 'S4'])
        ax_health.set_ylabel('SNR (dB)')
        ax_health.axhline(15, color='yellow', linestyle='--', alpha=0.5)
        ax_health.axhline(25, color='green', linestyle='--', alpha=0.5)
        bars = ax_health.bar(range(5), [0]*5, color='gray', edgecolor='white')
        ax_health.grid(True, alpha=0.3, axis='y')
        self.axes['health'] = ax_health
        self.artists['health_bars'] = bars

        # Detection trails
        ax_trails = self.fig.add_subplot(4, 3, 9)
        ax_trails.set_title('Detection Trails', fontweight='bold')
        ax_trails.set_xlabel('Delay (km)')
        ax_trails.set_ylabel('Doppler (Hz)')
        ax_trails.grid(True, alpha=0.3)
        scatter_curr = ax_trails.scatter([], [], s=80, c='red', marker='o',
                                          edgecolors='white', linewidths=1.5, zorder=10)
        scatter_hist = ax_trails.scatter([], [], s=20, c='orange', alpha=0.5, zorder=5)
        self.axes['trails'] = ax_trails
        self.artists['trails_curr'] = scatter_curr
        self.artists['trails_hist'] = scatter_hist

        # Waterfalls
        ax_wf_delay = self.fig.add_subplot(4, 3, 10)
        ax_wf_delay.set_title('Delay vs Time', fontweight='bold')
        ax_wf_delay.set_xlabel('Time (s ago)')
        ax_wf_delay.set_ylabel('Delay (km)')
        ax_wf_delay.set_xlim(self.history_duration, 0)
        ax_wf_delay.grid(True, alpha=0.3)
        scatter_wf_d = ax_wf_delay.scatter([], [], c='red', s=20, alpha=0.7)
        self.axes['wf_delay'] = ax_wf_delay
        self.artists['wf_delay_scatter'] = scatter_wf_d

        ax_wf_doppler = self.fig.add_subplot(4, 3, 11)
        ax_wf_doppler.set_title('Doppler vs Time', fontweight='bold')
        ax_wf_doppler.set_xlabel('Time (s ago)')
        ax_wf_doppler.set_ylabel('Doppler (Hz)')
        ax_wf_doppler.set_xlim(self.history_duration, 0)
        ax_wf_doppler.grid(True, alpha=0.3)
        scatter_wf_dop = ax_wf_doppler.scatter([], [], c='red', s=20, alpha=0.7)
        self.axes['wf_doppler'] = ax_wf_doppler
        self.artists['wf_doppler_scatter'] = scatter_wf_dop

        # Max-hold
        ax_maxhold = self.fig.add_subplot(4, 3, 12)
        ax_maxhold.set_title('Max-Hold CAF', fontweight='bold')
        ax_maxhold.set_xlabel('Delay (km)')
        ax_maxhold.set_ylabel('Doppler (Hz)')
        im_maxhold = ax_maxhold.imshow(
            np.zeros((self.doppler_len, self.fft_len)),
            aspect='auto', origin='lower',
            extent=[self.delay_axis_km[0], self.delay_axis_km[-1],
                    self.doppler_axis_hz[0], self.doppler_axis_hz[-1]],
            cmap='viridis', vmin=self.color_min, vmax=self.color_max,
        )
        self.axes['maxhold'] = ax_maxhold
        self.artists['maxhold_im'] = im_maxhold

        # Info text
        self.artists['info'] = self.fig.text(
            0.01, 0.99, '', fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        )

        self.fig.tight_layout()

    def _setup_control_figure(self):
        """Setup control panel figure."""
        self.ctrl_fig = plt.figure(figsize=(5, 10))
        self.ctrl_fig.canvas.manager.set_window_title('Controls')

        self.ctrl_fig.text(0.5, 0.96, 'Live Radar Controls', fontsize=12,
                           fontweight='bold', ha='center')

        y = 0.90
        dy = 0.035
        slider_h = 0.02
        left = 0.15
        width = 0.7

        # CFAR threshold
        self.ctrl_fig.text(left, y, 'CFAR:', fontsize=10, fontweight='bold')
        y -= dy
        ax_thresh = self.ctrl_fig.add_axes([left, y, width, slider_h])
        self.widgets['cfar_thresh'] = Slider(ax_thresh, 'Threshold (dB)', 5, 25,
                                              valinit=self.cfar_threshold)
        self.widgets['cfar_thresh'].on_changed(self._on_cfar_changed)
        y -= dy * 1.5

        # Display settings
        self.ctrl_fig.text(left, y, 'Display:', fontsize=10, fontweight='bold')
        y -= dy
        ax_cmin = self.ctrl_fig.add_axes([left, y, width, slider_h])
        self.widgets['color_min'] = Slider(ax_cmin, 'Color Min', -10, 10,
                                            valinit=self.color_min)
        self.widgets['color_min'].on_changed(self._on_display_changed)
        y -= dy

        ax_cmax = self.ctrl_fig.add_axes([left, y, width, slider_h])
        self.widgets['color_max'] = Slider(ax_cmax, 'Color Max', 5, 30,
                                            valinit=self.color_max)
        self.widgets['color_max'].on_changed(self._on_display_changed)
        y -= dy

        ax_decay = self.ctrl_fig.add_axes([left, y, width, slider_h])
        self.widgets['decay'] = Slider(ax_decay, 'Max-Hold Decay', 0.9, 1.0,
                                        valinit=self.max_hold_decay)
        self.widgets['decay'].on_changed(self._on_decay_changed)
        y -= dy

        ax_ppi = self.ctrl_fig.add_axes([left, y, width, slider_h])
        self.widgets['ppi_range'] = Slider(ax_ppi, 'PPI Range (km)', 10, 100,
                                            valinit=self.ppi_range_km)
        self.widgets['ppi_range'].on_changed(self._on_ppi_changed)
        y -= dy * 1.5

        # Buttons
        ax_reset = self.ctrl_fig.add_axes([left, y, width * 0.45, 0.04])
        self.widgets['btn_reset'] = Button(ax_reset, 'Reset Max-Hold')
        self.widgets['btn_reset'].on_clicked(self._on_reset)

        ax_clear = self.ctrl_fig.add_axes([left + width * 0.55, y, width * 0.45, 0.04])
        self.widgets['btn_clear'] = Button(ax_clear, 'Clear History')
        self.widgets['btn_clear'].on_clicked(self._on_clear)

    def _on_cfar_changed(self, val):
        self.cfar_threshold = val

    def _on_display_changed(self, val):
        self.color_min = self.widgets['color_min'].val
        self.color_max = self.widgets['color_max'].val

    def _on_decay_changed(self, val):
        self.max_hold_decay = val

    def _on_ppi_changed(self, val):
        self.ppi_range_km = val
        self.axes['ppi'].set_ylim(0, val)

    def _on_reset(self, event):
        with self.lock:
            self.max_hold_caf = None

    def _on_clear(self, event):
        with self.lock:
            self.detection_history.clear()

    def _update_display(self, frame):
        """Animation update callback."""
        now = time.time()

        with self.lock:
            channel_cafs = [c.copy() if c is not None else None for c in self.channel_cafs]
            fused = self.fused_caf.copy() if self.fused_caf is not None else None
            maxhold = self.max_hold_caf.copy() if self.max_hold_caf is not None else None
            health = [ChannelHealth(h.snr_db, h.phase_offset_deg, h.correlation_coeff)
                      for h in self.channel_health]
            history = list(self.detection_history)
            frames = self.frames_received

        # Update channel CAFs
        for i in range(self.num_channels):
            if channel_cafs[i] is not None:
                self.artists[f'ch{i}_im'].set_data(channel_cafs[i])
                self.artists[f'ch{i}_im'].set_clim(self.color_min, self.color_max)

        # Update fused CAF
        if fused is not None:
            self.artists['fused_im'].set_data(fused)
            self.artists['fused_im'].set_clim(self.color_min, self.color_max)

        # Update max-hold
        if maxhold is not None:
            self.artists['maxhold_im'].set_data(maxhold)
            self.artists['maxhold_im'].set_clim(self.color_min, self.color_max)

        # Update detections on fused
        recent = [d for d in history if now - d.timestamp < 2]
        if recent:
            det_x = [d.delay_km for d in recent]
            det_y = [d.doppler_hz for d in recent]
            self.artists['fused_det'].set_offsets(np.column_stack([det_x, det_y]))
        else:
            self.artists['fused_det'].set_offsets(np.empty((0, 2)))

        # Update PPI (use azimuth placeholder for now)
        if recent:
            angles = [np.radians(i * 30 + d.delay_km) for i, d in enumerate(recent)]
            ranges = [d.delay_km for d in recent]
            self.artists['ppi_scatter'].set_offsets(np.column_stack([angles, ranges]))
        else:
            self.artists['ppi_scatter'].set_offsets(np.empty((0, 2)))

        # Update channel health
        for i, bar in enumerate(self.artists['health_bars']):
            snr = health[i].snr_db
            bar.set_height(snr)
            bar.set_color('green' if snr >= 25 else 'yellow' if snr >= 15 else 'red')

        # Update trails
        if recent:
            self.artists['trails_curr'].set_offsets(
                np.column_stack([[d.delay_km for d in recent],
                                 [d.doppler_hz for d in recent]])
            )
        else:
            self.artists['trails_curr'].set_offsets(np.empty((0, 2)))

        older = [d for d in history if 2 <= now - d.timestamp <= self.history_duration]
        if older:
            self.artists['trails_hist'].set_offsets(
                np.column_stack([[d.delay_km for d in older],
                                 [d.doppler_hz for d in older]])
            )
        else:
            self.artists['trails_hist'].set_offsets(np.empty((0, 2)))

        # Update waterfalls
        valid_hist = [d for d in history if now - d.timestamp <= self.history_duration]
        if valid_hist:
            times = [now - d.timestamp for d in valid_hist]
            delays = [d.delay_km for d in valid_hist]
            dopplers = [d.doppler_hz for d in valid_hist]
            self.artists['wf_delay_scatter'].set_offsets(np.column_stack([times, delays]))
            self.artists['wf_doppler_scatter'].set_offsets(np.column_stack([times, dopplers]))
        else:
            self.artists['wf_delay_scatter'].set_offsets(np.empty((0, 2)))
            self.artists['wf_doppler_scatter'].set_offsets(np.empty((0, 2)))

        # Info text
        self.artists['info'].set_text(
            f'Frames: {frames}  Detections: {len(recent)}  History: {len(history)}'
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        return list(self.artists.values())
