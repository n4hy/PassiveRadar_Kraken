"""
Calibration Status Panel for Passive Radar
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Real-time display of KrakenSDR calibration status including:
- Per-channel SNR meters
- Phase offset displays
- Correlation coefficients
- Calibration validity indicator
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque
import time


@dataclass
class CalibrationStatus:
    """Calibration status from coherence monitor."""
    channel_snrs_db: List[float] = field(default_factory=lambda: [0.0] * 5)
    phase_offsets_deg: List[float] = field(default_factory=lambda: [0.0] * 5)
    correlation_coeffs: List[float] = field(default_factory=lambda: [1.0] * 5)
    is_valid: bool = False
    timestamp: float = 0.0
    phase_drift_rates: List[float] = field(default_factory=lambda: [0.0] * 5)


@dataclass
class CalibrationPanelParams:
    """Calibration panel parameters."""
    num_channels: int = 5
    snr_min_db: float = 0.0
    snr_max_db: float = 40.0
    snr_warning_db: float = 15.0
    snr_good_db: float = 25.0
    correlation_warning: float = 0.9
    correlation_good: float = 0.95
    phase_drift_warning: float = 1.0  # deg/s
    update_interval_ms: int = 200
    history_seconds: float = 60.0


class CalibrationPanel:
    """
    Real-time Calibration Status Display.

    Features:
    - SNR bar meters for each channel
    - Phase offset indicators (-180 to +180 deg)
    - Correlation coefficient bars
    - Overall calibration status indicator
    - Phase drift history
    - Recalibration trigger
    """

    def __init__(self, params: Optional[CalibrationPanelParams] = None):
        self.params = params if params else CalibrationPanelParams()

        # Data storage (thread-safe)
        self.lock = threading.Lock()
        self.status = CalibrationStatus()
        self.last_cal_time = time.time()

        # History for drift tracking (using deque for O(1) popleft)
        max_history = int(self.params.history_seconds * 1000 / self.params.update_interval_ms)
        self.phase_history = [deque(maxlen=max_history) for _ in range(self.params.num_channels)]
        self.time_history: Deque[float] = deque(maxlen=max_history)

        # Matplotlib objects
        self.fig = None
        self.axes = {}
        self.snr_bars = []
        self.phase_indicators = []
        self.corr_bars = []
        self.status_indicator = None
        self.drift_lines = []
        self.info_text = None

        # Animation
        self.anim = None
        self.running = False

        # Colors
        self.color_good = '#00FF00'
        self.color_warning = '#FFFF00'
        self.color_bad = '#FF0000'
        self.color_inactive = '#404040'

    def _setup_plot(self):
        """Initialize the matplotlib figure."""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('KrakenSDR Calibration Status')

        # Create subplot grid
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # SNR meters (top row)
        self.axes['snr'] = self.fig.add_subplot(gs[0, :3])
        self._setup_snr_meters()

        # Overall status (top right)
        self.axes['status'] = self.fig.add_subplot(gs[0, 3])
        self._setup_status_indicator()

        # Phase offsets (middle row left)
        self.axes['phase'] = self.fig.add_subplot(gs[1, :2])
        self._setup_phase_display()

        # Correlation coefficients (middle row right)
        self.axes['corr'] = self.fig.add_subplot(gs[1, 2:])
        self._setup_correlation_display()

        # Phase drift history (bottom row)
        self.axes['drift'] = self.fig.add_subplot(gs[2, :])
        self._setup_drift_history()

        self.fig.suptitle('KrakenSDR Calibration Status', fontsize=14, fontweight='bold')

    def _setup_snr_meters(self):
        """Setup SNR bar meters."""
        ax = self.axes['snr']
        ax.set_xlim(-0.5, self.params.num_channels - 0.5)
        ax.set_ylim(self.params.snr_min_db, self.params.snr_max_db)
        ax.set_xlabel('Channel')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('Channel SNR')
        ax.set_xticks(range(self.params.num_channels))
        ax.set_xticklabels(['Ref', 'Surv 1', 'Surv 2', 'Surv 3', 'Surv 4'])

        # Add threshold lines
        ax.axhline(self.params.snr_warning_db, color='yellow', linestyle='--', alpha=0.5, label='Warning')
        ax.axhline(self.params.snr_good_db, color='green', linestyle='--', alpha=0.5, label='Good')

        # Create bars
        self.snr_bars = ax.bar(
            range(self.params.num_channels),
            [0] * self.params.num_channels,
            color=[self.color_inactive] * self.params.num_channels,
            edgecolor='white',
            linewidth=1
        )

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _setup_status_indicator(self):
        """Setup overall status indicator."""
        ax = self.axes['status']
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Calibration Status')

        # Main status circle
        self.status_indicator = Circle((0, 0), 0.6, facecolor=self.color_inactive,
                                         edgecolor='white', linewidth=3)
        ax.add_patch(self.status_indicator)

        # Status text
        self.status_text = ax.text(0, 0, 'INIT', ha='center', va='center',
                                    fontsize=12, fontweight='bold', color='white')

        # Time since calibration
        self.cal_time_text = ax.text(0, -0.85, 'Last cal: --',
                                      ha='center', va='center', fontsize=9, color='white')

    def _setup_phase_display(self):
        """Setup phase offset display."""
        ax = self.axes['phase']
        ax.set_xlim(-0.5, self.params.num_channels - 0.5)
        ax.set_ylim(-180, 180)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Phase Offset (deg)')
        ax.set_title('Phase Offsets (relative to Ref)')
        ax.set_xticks(range(self.params.num_channels))
        ax.set_xticklabels(['Ref', 'Surv 1', 'Surv 2', 'Surv 3', 'Surv 4'])

        # Zero line
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)

        # Phase markers
        self.phase_indicators = ax.scatter(
            range(self.params.num_channels),
            [0] * self.params.num_channels,
            c=[self.color_inactive] * self.params.num_channels,
            s=200, marker='d', edgecolors='white', linewidths=1.5
        )

        ax.grid(True, alpha=0.3)

    def _setup_correlation_display(self):
        """Setup correlation coefficient display."""
        ax = self.axes['corr']
        ax.set_xlim(-0.5, self.params.num_channels - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation with Reference')
        ax.set_xticks(range(self.params.num_channels))
        ax.set_xticklabels(['Ref', 'Surv 1', 'Surv 2', 'Surv 3', 'Surv 4'])

        # Threshold lines
        ax.axhline(self.params.correlation_warning, color='yellow', linestyle='--', alpha=0.5)
        ax.axhline(self.params.correlation_good, color='green', linestyle='--', alpha=0.5)

        # Correlation bars
        self.corr_bars = ax.bar(
            range(self.params.num_channels),
            [0] * self.params.num_channels,
            color=[self.color_inactive] * self.params.num_channels,
            edgecolor='white',
            linewidth=1
        )

        ax.grid(True, alpha=0.3, axis='y')

    def _setup_drift_history(self):
        """Setup phase drift history plot."""
        ax = self.axes['drift']
        ax.set_xlim(-self.params.history_seconds, 0)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Phase Drift (deg)')
        ax.set_title('Phase Drift History (relative to initial calibration)')

        # Create lines for each surveillance channel
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        self.drift_lines = []
        for i in range(4):
            line, = ax.plot([], [], color=colors[i], linewidth=1.5,
                           label=f'Surv {i+1}', alpha=0.8)
            self.drift_lines.append(line)

        ax.legend(loc='upper left', fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)

    def _get_color_for_snr(self, snr_db: float) -> str:
        """Get color based on SNR value."""
        if snr_db >= self.params.snr_good_db:
            return self.color_good
        elif snr_db >= self.params.snr_warning_db:
            return self.color_warning
        else:
            return self.color_bad

    def _get_color_for_corr(self, corr: float) -> str:
        """Get color based on correlation value."""
        if corr >= self.params.correlation_good:
            return self.color_good
        elif corr >= self.params.correlation_warning:
            return self.color_warning
        else:
            return self.color_bad

    def _update(self, frame):
        """Animation update callback."""
        with self.lock:
            status = CalibrationStatus(
                channel_snrs_db=self.status.channel_snrs_db.copy(),
                phase_offsets_deg=self.status.phase_offsets_deg.copy(),
                correlation_coeffs=self.status.correlation_coeffs.copy(),
                is_valid=self.status.is_valid,
                timestamp=self.status.timestamp,
                phase_drift_rates=self.status.phase_drift_rates.copy()
            )

        current_time = time.time()

        # Update SNR bars
        for i, bar in enumerate(self.snr_bars):
            snr = status.channel_snrs_db[i]
            bar.set_height(max(0, snr))
            bar.set_color(self._get_color_for_snr(snr))

        # Update phase indicators
        phase_colors = []
        for i, phase in enumerate(status.phase_offsets_deg):
            # Color based on drift rate for surveillance channels
            if i > 0 and abs(status.phase_drift_rates[i]) > self.params.phase_drift_warning:
                phase_colors.append(self.color_warning)
            else:
                phase_colors.append(self.color_good if status.is_valid else self.color_inactive)

        self.phase_indicators.set_offsets(
            np.column_stack([range(self.params.num_channels), status.phase_offsets_deg])
        )
        self.phase_indicators.set_facecolors(phase_colors)

        # Update correlation bars
        for i, bar in enumerate(self.corr_bars):
            corr = status.correlation_coeffs[i]
            bar.set_height(corr)
            bar.set_color(self._get_color_for_corr(corr))

        # Update overall status
        if status.is_valid:
            # Check if all channels are good
            all_snr_good = all(s >= self.params.snr_warning_db for s in status.channel_snrs_db)
            all_corr_good = all(c >= self.params.correlation_warning for c in status.correlation_coeffs[1:])

            if all_snr_good and all_corr_good:
                self.status_indicator.set_facecolor(self.color_good)
                self.status_text.set_text('VALID')
            else:
                self.status_indicator.set_facecolor(self.color_warning)
                self.status_text.set_text('WARN')
        else:
            self.status_indicator.set_facecolor(self.color_bad)
            self.status_text.set_text('INVALID')

        # Update calibration time
        elapsed = current_time - self.last_cal_time
        if elapsed < 60:
            time_str = f'{elapsed:.0f}s ago'
        elif elapsed < 3600:
            time_str = f'{elapsed/60:.1f}m ago'
        else:
            time_str = f'{elapsed/3600:.1f}h ago'
        self.cal_time_text.set_text(f'Last cal: {time_str}')

        # Update drift history
        self._update_drift_history(status, current_time)

        return list(self.snr_bars) + list(self.corr_bars) + [self.phase_indicators]

    def _update_drift_history(self, status: CalibrationStatus, current_time: float):
        """Update phase drift history plot."""
        # Add current phase offsets to history (deque handles maxlen automatically)
        self.time_history.append(current_time)
        for i in range(self.params.num_channels):
            self.phase_history[i].append(status.phase_offsets_deg[i])

        # Note: Trimming is handled automatically by deque's maxlen

        # Update drift lines (surveillance channels only)
        if len(self.time_history) > 1:
            times_rel = [t - current_time for t in self.time_history]

            for i in range(4):
                if self.phase_history[i+1]:
                    # Compute drift relative to initial
                    initial_phase = self.phase_history[i+1][0] if self.phase_history[i+1] else 0
                    drift = [p - initial_phase for p in self.phase_history[i+1]]
                    # Unwrap phase jumps
                    drift = np.unwrap(np.array(drift) * np.pi / 180) * 180 / np.pi
                    self.drift_lines[i].set_data(times_rel[:len(drift)], drift)

            # Adjust y-axis if needed
            all_drift = []
            for i in range(4):
                if self.phase_history[i+1]:
                    initial_phase = self.phase_history[i+1][0]
                    drift = [p - initial_phase for p in self.phase_history[i+1]]
                    all_drift.extend(drift)

            if all_drift:
                max_drift = max(abs(min(all_drift)), abs(max(all_drift)), 5)
                self.axes['drift'].set_ylim(-max_drift * 1.2, max_drift * 1.2)

    def update_status(self, status: CalibrationStatus):
        """Thread-safe update of calibration status."""
        with self.lock:
            self.status = status
            if status.is_valid and not self.status.is_valid:
                # New valid calibration
                self.last_cal_time = time.time()

    def trigger_recalibration(self):
        """Request recalibration (sends message to coherence_monitor)."""
        # This would typically send a message to the GNU Radio flowgraph
        print("Recalibration requested")

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
        """Stop the display and clean up resources."""
        self.running = False
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def demo_calibration_panel():
    """Demo with simulated calibration data."""
    params = CalibrationPanelParams(
        update_interval_ms=200,
        history_seconds=30.0
    )

    panel = CalibrationPanel(params)

    def data_generator():
        """Generate simulated calibration data."""
        import random

        base_phases = [0.0, 5.0, -3.0, 2.0, -1.0]
        drift_rates = [0.0, 0.1, -0.05, 0.08, -0.12]  # deg/s

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            # Simulate SNR with some variation
            snrs = [
                25 + random.gauss(0, 2),  # Ref
                24 + random.gauss(0, 2),
                23 + random.gauss(0, 3),
                25 + random.gauss(0, 2),
                22 + random.gauss(0, 3),
            ]

            # Simulate phase drift
            phases = [base_phases[i] + drift_rates[i] * elapsed + random.gauss(0, 0.5)
                      for i in range(5)]

            # Wrap to -180 to 180
            phases = [(p + 180) % 360 - 180 for p in phases]

            # Simulate correlation
            corrs = [1.0] + [0.96 + random.gauss(0, 0.02) for _ in range(4)]
            corrs = [min(1.0, max(0.0, c)) for c in corrs]

            status = CalibrationStatus(
                channel_snrs_db=snrs,
                phase_offsets_deg=phases,
                correlation_coeffs=corrs,
                is_valid=True,
                timestamp=time.time(),
                phase_drift_rates=drift_rates
            )

            panel.update_status(status)
            time.sleep(0.2)

    # Start data generator thread
    gen_thread = threading.Thread(target=data_generator, daemon=True)
    gen_thread.start()

    # Start display (blocking)
    panel.start()


if __name__ == "__main__":
    demo_calibration_panel()
