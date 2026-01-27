"""
Integrated Passive Radar GUI
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Multi-panel display combining:
- Range-Doppler Map
- PPI Display
- Calibration Panel
- Metrics Dashboard
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure Tk backend for GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import sys
import os

# Import display components (support both package and direct execution)
try:
    from .range_doppler_display import (
        RangeDopplerDisplay, RDDisplayParams, Detection, Track as RDTrack
    )
    from .radar_display import PPIDisplay, PPIDisplayParams, PPIDetection, PPITrack
    from .calibration_panel import CalibrationPanel, CalibrationPanelParams, CalibrationStatus
    from .metrics_dashboard import MetricsDashboard, MetricsDashboardParams, ProcessingMetrics
except ImportError:
    from range_doppler_display import (
        RangeDopplerDisplay, RDDisplayParams, Detection, Track as RDTrack
    )
    from radar_display import PPIDisplay, PPIDisplayParams, PPIDetection, PPITrack
    from calibration_panel import CalibrationPanel, CalibrationPanelParams, CalibrationStatus
    from metrics_dashboard import MetricsDashboard, MetricsDashboardParams, ProcessingMetrics


@dataclass
class RadarGUIParams:
    """Integrated GUI parameters."""
    window_title: str = "KrakenSDR Passive Radar"
    update_interval_ms: int = 100
    max_range_km: float = 15.0
    n_range_bins: int = 256
    n_doppler_bins: int = 64


class RadarGUI:
    """
    Integrated Passive Radar GUI.

    Layout:
    +-------------------+-------------------+
    |   Range-Doppler   |       PPI         |
    |      Map          |     Display       |
    +-------------------+-------------------+
    |   Calibration     |     Metrics       |
    |     Panel         |    Dashboard      |
    +-------------------+-------------------+

    Plus control panel with:
    - Start/Stop buttons
    - Parameter adjustment
    - Recording controls
    """

    def __init__(self, params: Optional[RadarGUIParams] = None):
        self.params = params if params else RadarGUIParams()

        # Create main window
        self.root = tk.Tk()
        self.root.title(self.params.window_title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1600, screen_width - 100)
        window_height = min(900, screen_height - 100)
        self.root.geometry(f"{window_width}x{window_height}")

        # State
        self.running = False
        self.lock = threading.Lock()

        # Data storage
        self.caf_data = np.zeros((self.params.n_doppler_bins, self.params.n_range_bins))
        self.detections: List[Detection] = []
        self.tracks: List[RDTrack] = []
        self.calibration = CalibrationStatus()
        self.metrics = ProcessingMetrics()

        # Create GUI components
        self._create_widgets()
        self._setup_figures()

        # Animation
        self.anim = None

    def _create_widgets(self):
        """Create GUI widgets."""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top row: Range-Doppler + PPI
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Range-Doppler
        self.rd_frame = ttk.LabelFrame(self.top_frame, text="Range-Doppler Map")
        self.rd_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Right: PPI
        self.ppi_frame = ttk.LabelFrame(self.top_frame, text="PPI Display")
        self.ppi_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Bottom row: Calibration + Metrics
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Calibration
        self.cal_frame = ttk.LabelFrame(self.bottom_frame, text="Calibration Status")
        self.cal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Right: Metrics
        self.metrics_frame = ttk.LabelFrame(self.bottom_frame, text="System Metrics")
        self.metrics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Control panel at bottom
        self._create_control_panel()

    def _create_control_panel(self):
        """Create control panel with buttons."""
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=5)

        # Start/Stop button
        self.start_button = ttk.Button(
            self.control_frame, text="Start",
            command=self._toggle_running
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Reset button
        self.reset_button = ttk.Button(
            self.control_frame, text="Reset",
            command=self._reset
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Recalibrate button
        self.recal_button = ttk.Button(
            self.control_frame, text="Recalibrate",
            command=self._request_recalibration
        )
        self.recal_button.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(self.control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        # Dynamic range slider
        ttk.Label(self.control_frame, text="Dynamic Range:").pack(side=tk.LEFT)
        self.dr_slider = ttk.Scale(
            self.control_frame, from_=20, to=80,
            orient=tk.HORIZONTAL, length=150,
            command=self._on_dr_change
        )
        self.dr_slider.set(60)
        self.dr_slider.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Status: Stopped")
        self.status_label = ttk.Label(
            self.control_frame, textvariable=self.status_var
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def _setup_figures(self):
        """Setup matplotlib figures for each panel."""
        # Range-Doppler figure
        self.rd_fig = Figure(figsize=(6, 4), dpi=100)
        self.rd_ax = self.rd_fig.add_subplot(111)
        self._setup_rd_plot()

        self.rd_canvas = FigureCanvasTkAgg(self.rd_fig, master=self.rd_frame)
        self.rd_canvas.draw()
        self.rd_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # PPI figure
        self.ppi_fig = Figure(figsize=(6, 4), dpi=100)
        self.ppi_ax = self.ppi_fig.add_subplot(111, projection='polar')
        self._setup_ppi_plot()

        self.ppi_canvas = FigureCanvasTkAgg(self.ppi_fig, master=self.ppi_frame)
        self.ppi_canvas.draw()
        self.ppi_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Calibration figure
        self.cal_fig = Figure(figsize=(6, 3), dpi=100)
        self._setup_cal_plot()

        self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=self.cal_frame)
        self.cal_canvas.draw()
        self.cal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Metrics figure
        self.metrics_fig = Figure(figsize=(6, 3), dpi=100)
        self._setup_metrics_plot()

        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=self.metrics_frame)
        self.metrics_canvas.draw()
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_rd_plot(self):
        """Setup Range-Doppler plot."""
        self.rd_ax.set_xlabel('Range (km)')
        self.rd_ax.set_ylabel('Doppler (Hz)')
        self.rd_ax.set_title('Range-Doppler Map')

        # Initial heatmap
        self.rd_im = self.rd_ax.imshow(
            self.caf_data,
            aspect='auto',
            origin='lower',
            extent=[0, self.params.max_range_km, -125, 125],
            cmap='viridis',
            vmin=-30, vmax=30
        )
        self.rd_fig.colorbar(self.rd_im, ax=self.rd_ax, label='Power (dB)')

        # Detection scatter
        self.rd_det_scatter = self.rd_ax.scatter(
            [], [], c='red', s=80, marker='o', alpha=0.8,
            edgecolors='white', linewidths=1
        )

        self.rd_ax.grid(True, alpha=0.3)

    def _setup_ppi_plot(self):
        """Setup PPI plot."""
        self.ppi_ax.set_theta_zero_location('N')
        self.ppi_ax.set_theta_direction(-1)
        self.ppi_ax.set_ylim(0, self.params.max_range_km)
        self.ppi_ax.set_title('PPI Display')

        # Track scatter
        self.ppi_track_scatter = self.ppi_ax.scatter(
            [], [], c='lime', s=100, marker='D',
            edgecolors='white', linewidths=1.5
        )

        # Detection scatter
        self.ppi_det_scatter = self.ppi_ax.scatter(
            [], [], c='red', s=50, marker='o', alpha=0.6
        )

        self.ppi_ax.grid(True, alpha=0.3)

    def _setup_cal_plot(self):
        """Setup calibration plot."""
        gs = self.cal_fig.add_gridspec(1, 3, wspace=0.3)

        # SNR bars
        self.cal_snr_ax = self.cal_fig.add_subplot(gs[0, 0])
        self.cal_snr_ax.set_title('SNR (dB)', fontsize=9)
        self.cal_snr_ax.set_ylim(0, 40)
        self.cal_snr_bars = self.cal_snr_ax.bar(
            range(5), [0]*5, color='#404040'
        )
        self.cal_snr_ax.set_xticks(range(5))
        self.cal_snr_ax.set_xticklabels(['R', '1', '2', '3', '4'], fontsize=8)

        # Phase plot
        self.cal_phase_ax = self.cal_fig.add_subplot(gs[0, 1])
        self.cal_phase_ax.set_title('Phase (deg)', fontsize=9)
        self.cal_phase_ax.set_ylim(-180, 180)
        self.cal_phase_scatter = self.cal_phase_ax.scatter(
            range(5), [0]*5, c='#404040', s=100, marker='d'
        )
        self.cal_phase_ax.axhline(0, color='white', alpha=0.3)
        self.cal_phase_ax.set_xticks(range(5))
        self.cal_phase_ax.set_xticklabels(['R', '1', '2', '3', '4'], fontsize=8)

        # Correlation bars
        self.cal_corr_ax = self.cal_fig.add_subplot(gs[0, 2])
        self.cal_corr_ax.set_title('Correlation', fontsize=9)
        self.cal_corr_ax.set_ylim(0, 1)
        self.cal_corr_bars = self.cal_corr_ax.bar(
            range(5), [0]*5, color='#404040'
        )
        self.cal_corr_ax.set_xticks(range(5))
        self.cal_corr_ax.set_xticklabels(['R', '1', '2', '3', '4'], fontsize=8)
        self.cal_corr_ax.axhline(0.95, color='green', linestyle='--', alpha=0.5)

    def _setup_metrics_plot(self):
        """Setup metrics plot."""
        gs = self.metrics_fig.add_gridspec(1, 2, wspace=0.3)

        # Latency bars
        self.metrics_lat_ax = self.metrics_fig.add_subplot(gs[0, 0])
        self.metrics_lat_ax.set_title('Latency (ms)', fontsize=9)
        self.metrics_lat_ax.set_xlim(0, 100)

        labels = ['ECA', 'CAF', 'CFAR', 'Track', 'Total']
        self.metrics_lat_bars = self.metrics_lat_ax.barh(
            range(5), [0]*5, color='#00FF00'
        )
        self.metrics_lat_ax.set_yticks(range(5))
        self.metrics_lat_ax.set_yticklabels(labels, fontsize=8)
        self.metrics_lat_ax.axvline(80, color='yellow', linestyle='--', alpha=0.5)
        self.metrics_lat_ax.axvline(100, color='red', linestyle='--', alpha=0.5)

        # Counts text
        self.metrics_counts_ax = self.metrics_fig.add_subplot(gs[0, 1])
        self.metrics_counts_ax.axis('off')
        self.metrics_counts_ax.set_title('Status', fontsize=9)

        self.metrics_det_text = self.metrics_counts_ax.text(
            0.1, 0.7, 'Detections: 0', fontsize=10,
            transform=self.metrics_counts_ax.transAxes
        )
        self.metrics_track_text = self.metrics_counts_ax.text(
            0.1, 0.4, 'Tracks: 0', fontsize=10,
            transform=self.metrics_counts_ax.transAxes
        )
        self.metrics_frame_text = self.metrics_counts_ax.text(
            0.1, 0.1, 'Frames: 0', fontsize=10,
            transform=self.metrics_counts_ax.transAxes
        )

    def _update(self, frame):
        """Animation update callback."""
        with self.lock:
            caf_data = self.caf_data.copy()
            detections = self.detections.copy()
            tracks = self.tracks.copy()
            cal = CalibrationStatus(
                channel_snrs_db=self.calibration.channel_snrs_db.copy(),
                phase_offsets_deg=self.calibration.phase_offsets_deg.copy(),
                correlation_coeffs=self.calibration.correlation_coeffs.copy(),
                is_valid=self.calibration.is_valid
            )
            metrics = ProcessingMetrics(
                eca_latency_ms=self.metrics.eca_latency_ms,
                caf_latency_ms=self.metrics.caf_latency_ms,
                cfar_latency_ms=self.metrics.cfar_latency_ms,
                tracker_latency_ms=self.metrics.tracker_latency_ms,
                total_latency_ms=self.metrics.total_latency_ms,
                num_detections=self.metrics.num_detections,
                num_tracks=self.metrics.num_tracks,
                frames_processed=self.metrics.frames_processed
            )

        # Update Range-Doppler
        self.rd_im.set_data(caf_data)

        if detections:
            det_ranges = [d.range_m / 1000.0 for d in detections]
            det_dopplers = [d.doppler_hz for d in detections]
            self.rd_det_scatter.set_offsets(np.column_stack([det_ranges, det_dopplers]))
        else:
            self.rd_det_scatter.set_offsets(np.empty((0, 2)))

        # Update PPI
        if tracks:
            # Use actual AoA if available, otherwise use track ID for demo display
            track_angles = []
            for t in tracks:
                if hasattr(t, 'aoa_deg') and t.aoa_deg is not None:
                    track_angles.append(np.radians(t.aoa_deg))
                else:
                    # Fallback: distribute tracks around display for visibility
                    track_angles.append(np.radians(45 * (t.id % 8)) if hasattr(t, 'id') else 0)
            track_ranges = [t.range_m / 1000.0 for t in tracks]
            self.ppi_track_scatter.set_offsets(np.column_stack([track_angles, track_ranges]))
        else:
            self.ppi_track_scatter.set_offsets(np.empty((0, 2)))

        # Update calibration
        for i, bar in enumerate(self.cal_snr_bars):
            snr = cal.channel_snrs_db[i]
            bar.set_height(max(0, snr))
            bar.set_color('#00FF00' if snr > 20 else '#FFFF00' if snr > 10 else '#FF0000')

        self.cal_phase_scatter.set_offsets(
            np.column_stack([range(5), cal.phase_offsets_deg])
        )

        for i, bar in enumerate(self.cal_corr_bars):
            corr = cal.correlation_coeffs[i]
            bar.set_height(corr)
            bar.set_color('#00FF00' if corr > 0.95 else '#FFFF00' if corr > 0.9 else '#FF0000')

        # Update metrics
        latencies = [
            metrics.eca_latency_ms,
            metrics.caf_latency_ms,
            metrics.cfar_latency_ms,
            metrics.tracker_latency_ms,
            metrics.total_latency_ms
        ]
        for bar, lat in zip(self.metrics_lat_bars, latencies):
            bar.set_width(lat)
            if lat > 100:
                bar.set_color('#FF0000')
            elif lat > 80:
                bar.set_color('#FFFF00')
            else:
                bar.set_color('#00FF00')

        self.metrics_det_text.set_text(f'Detections: {metrics.num_detections}')
        self.metrics_track_text.set_text(f'Tracks: {metrics.num_tracks}')
        self.metrics_frame_text.set_text(f'Frames: {metrics.frames_processed}')

        # Redraw canvases
        self.rd_canvas.draw()
        self.ppi_canvas.draw()
        self.cal_canvas.draw()
        self.metrics_canvas.draw()

    def _toggle_running(self):
        """Toggle running state."""
        self.running = not self.running
        if self.running:
            self.start_button.config(text="Stop")
            self.status_var.set("Status: Running")
            self._start_animation()
        else:
            self.start_button.config(text="Start")
            self.status_var.set("Status: Stopped")
            self._stop_animation()

    def _start_animation(self):
        """Start the animation."""
        if self.anim is None:
            self.anim = FuncAnimation(
                self.rd_fig, self._update,
                interval=self.params.update_interval_ms,
                blit=False,
                cache_frame_data=False
            )

    def _stop_animation(self):
        """Stop the animation and clean up resources."""
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def _reset(self):
        """Reset all data."""
        with self.lock:
            self.caf_data = np.zeros((self.params.n_doppler_bins, self.params.n_range_bins))
            self.detections.clear()
            self.tracks.clear()
            self.metrics = ProcessingMetrics()

    def _request_recalibration(self):
        """Request system recalibration."""
        print("Recalibration requested")
        self.status_var.set("Status: Recalibrating...")

    def _on_dr_change(self, value):
        """Handle dynamic range slider change."""
        if not hasattr(self, 'rd_im') or self.rd_im is None:
            return
        dr = float(value)
        self.rd_im.set_clim(-dr/2, dr/2)
        self.rd_canvas.draw()

    def _on_close(self):
        """Handle window close."""
        self.running = False
        if self.anim is not None:
            self.anim.event_source.stop()
        self.root.quit()
        self.root.destroy()

    # Public update methods (thread-safe)
    def update_caf(self, caf_data: np.ndarray):
        """Update CAF data."""
        with self.lock:
            if caf_data.shape == self.caf_data.shape:
                self.caf_data = caf_data.copy()

    def update_detections(self, detections: List[Detection]):
        """Update detections."""
        with self.lock:
            self.detections = detections.copy()

    def update_tracks(self, tracks: List[RDTrack]):
        """Update tracks."""
        with self.lock:
            self.tracks = tracks.copy()

    def update_calibration(self, calibration: CalibrationStatus):
        """Update calibration status."""
        with self.lock:
            self.calibration = calibration

    def update_metrics(self, metrics: ProcessingMetrics):
        """Update processing metrics."""
        with self.lock:
            self.metrics = metrics

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def demo_radar_gui():
    """Demo with simulated data."""
    gui = RadarGUI()

    def data_generator():
        """Generate simulated data."""
        import random

        frame = 0
        target_range = 5000.0
        target_doppler = 50.0

        while True:
            if not gui.running:
                time.sleep(0.1)
                continue

            frame += 1

            # Generate CAF data
            caf = np.random.randn(64, 256) * 5 - 20
            target_r = int(target_range / 600.0)
            target_d = int(target_doppler / 3.9 + 32)
            if 0 <= target_r < 256 and 0 <= target_d < 64:
                caf[max(0,target_d-1):target_d+2, max(0,target_r-1):target_r+2] = 20.0

            gui.update_caf(caf)

            # Detections
            detections = [Detection(
                range_bin=target_r, doppler_bin=target_d,
                snr_db=25.0, range_m=target_range, doppler_hz=target_doppler
            )]
            gui.update_detections(detections)

            # Tracks
            tracks = [RDTrack(
                id=1, range_m=target_range, doppler_hz=target_doppler,
                status='confirmed'
            )]
            gui.update_tracks(tracks)

            # Calibration
            cal = CalibrationStatus(
                channel_snrs_db=[25 + random.gauss(0, 1) for _ in range(5)],
                phase_offsets_deg=[random.gauss(0, 5) for _ in range(5)],
                correlation_coeffs=[0.96 + random.gauss(0, 0.02) for _ in range(5)],
                is_valid=True
            )
            gui.update_calibration(cal)

            # Metrics
            metrics = ProcessingMetrics(
                eca_latency_ms=5 + random.gauss(0, 1),
                caf_latency_ms=25 + random.gauss(0, 3),
                cfar_latency_ms=8 + random.gauss(0, 1),
                tracker_latency_ms=3 + random.gauss(0, 0.5),
                total_latency_ms=41 + random.gauss(0, 3),
                num_detections=len(detections),
                num_tracks=len(tracks),
                frames_processed=frame
            )
            gui.update_metrics(metrics)

            # Move target
            target_range += 50
            target_doppler -= 0.5
            if target_range > 12000:
                target_range = 3000
                target_doppler = random.uniform(20, 80)

            time.sleep(0.1)

    # Start data generator
    gen_thread = threading.Thread(target=data_generator, daemon=True)
    gen_thread.start()

    # Run GUI
    gui.run()


if __name__ == "__main__":
    demo_radar_gui()
