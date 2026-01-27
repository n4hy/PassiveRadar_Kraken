"""
System Metrics Dashboard for Passive Radar
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Real-time display of processing metrics:
- Latencies with sparklines
- Detection and track counts
- System resource usage
- Backend status indicators
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import deque
import time
import os


@dataclass
class ProcessingMetrics:
    """Processing metrics from radar pipeline."""
    # Latencies in milliseconds
    eca_latency_ms: float = 0.0
    caf_latency_ms: float = 0.0
    cfar_latency_ms: float = 0.0
    clustering_latency_ms: float = 0.0
    tracker_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Frame timing
    frame_period_ms: float = 100.0
    frames_processed: int = 0
    dropped_frames: int = 0

    # Detection/tracking
    num_detections: int = 0
    num_tracks: int = 0
    num_confirmed_tracks: int = 0
    detection_rate: float = 0.0  # detections per second
    false_alarm_rate: float = 0.0  # estimated

    # System resources
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    sample_rate_actual: float = 0.0
    sample_rate_expected: float = 250000.0

    # Backend status
    neon_enabled: bool = False
    vulkan_enabled: bool = False
    vulkan_device: str = ""

    timestamp: float = 0.0


@dataclass
class MetricsDashboardParams:
    """Metrics dashboard parameters."""
    update_interval_ms: int = 200
    history_length: int = 100  # Number of samples for sparklines
    latency_warning_ms: float = 80.0  # Frame period threshold
    latency_critical_ms: float = 100.0


class MetricsDashboard:
    """
    System Metrics Dashboard.

    Features:
    - Processing latency breakdown with sparklines
    - Detection and track counts
    - Frame drop indicator
    - CPU/Memory usage
    - Backend acceleration status
    - Sample rate verification
    """

    def __init__(self, params: Optional[MetricsDashboardParams] = None):
        self.params = params if params else MetricsDashboardParams()

        # Data storage (thread-safe)
        self.lock = threading.Lock()
        self.metrics = ProcessingMetrics()

        # History for sparklines
        self.latency_history = {
            'eca': deque(maxlen=self.params.history_length),
            'caf': deque(maxlen=self.params.history_length),
            'cfar': deque(maxlen=self.params.history_length),
            'cluster': deque(maxlen=self.params.history_length),
            'tracker': deque(maxlen=self.params.history_length),
            'total': deque(maxlen=self.params.history_length),
        }

        self.detection_history = deque(maxlen=self.params.history_length)
        self.track_history = deque(maxlen=self.params.history_length)

        # Matplotlib objects
        self.fig = None
        self.axes = {}
        self.sparklines = {}
        self.status_indicators = {}
        self.text_elements = {}

        # Animation
        self.anim = None
        self.running = False

        # Colors
        self.color_good = '#00FF00'
        self.color_warning = '#FFFF00'
        self.color_bad = '#FF0000'
        self.color_inactive = '#404040'
        self.color_bg = '#1a1a2e'

    def _setup_plot(self):
        """Initialize the matplotlib figure."""
        self.fig = plt.figure(figsize=(16, 9), facecolor=self.color_bg)
        self.fig.canvas.manager.set_window_title('Passive Radar Metrics Dashboard')

        # Create subplot grid
        gs = self.fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

        # Latency breakdown (top left, spans 2 columns)
        self.axes['latency'] = self.fig.add_subplot(gs[0:2, 0:2], facecolor=self.color_bg)
        self._setup_latency_display()

        # Detection/Track counts (top right)
        self.axes['counts'] = self.fig.add_subplot(gs[0, 2:], facecolor=self.color_bg)
        self._setup_counts_display()

        # Detection sparkline (row 2, cols 2-3)
        self.axes['det_spark'] = self.fig.add_subplot(gs[1, 2:], facecolor=self.color_bg)
        self._setup_detection_sparkline()

        # System resources (row 3, cols 0-1)
        self.axes['resources'] = self.fig.add_subplot(gs[2, 0:2], facecolor=self.color_bg)
        self._setup_resources_display()

        # Backend status (row 3, cols 2-3)
        self.axes['backend'] = self.fig.add_subplot(gs[2, 2:], facecolor=self.color_bg)
        self._setup_backend_display()

        # Frame timing (bottom row)
        self.axes['timing'] = self.fig.add_subplot(gs[3, :], facecolor=self.color_bg)
        self._setup_timing_display()

        self.fig.suptitle('Passive Radar System Metrics', fontsize=14,
                          fontweight='bold', color='white')

    def _setup_latency_display(self):
        """Setup latency breakdown with sparklines."""
        ax = self.axes['latency']
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 5.5)
        ax.set_xlabel('Latency (ms)', color='white')
        ax.set_title('Processing Latency Breakdown', color='white')
        ax.tick_params(colors='white')

        labels = ['ECA', 'CAF', 'CFAR', 'Cluster', 'Tracker', 'Total']
        keys = ['eca', 'caf', 'cfar', 'cluster', 'tracker', 'total']

        ax.set_yticks(range(6))
        ax.set_yticklabels(labels)

        # Create horizontal bars and sparklines
        self.latency_bars = []
        for i, (label, key) in enumerate(zip(labels, keys)):
            bar = ax.barh(i, 0, height=0.6, color=self.color_good,
                          edgecolor='white', linewidth=0.5)
            self.latency_bars.append(bar[0])

            # Value text
            text = ax.text(2, i, '0.0 ms', va='center', ha='left',
                          color='white', fontsize=9)
            self.text_elements[f'latency_{key}'] = text

        # Threshold lines
        ax.axvline(self.params.latency_warning_ms, color=self.color_warning,
                   linestyle='--', alpha=0.5, label='Warning')
        ax.axvline(self.params.latency_critical_ms, color=self.color_bad,
                   linestyle='--', alpha=0.5, label='Critical')

        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.2, axis='x')

        for spine in ax.spines.values():
            spine.set_color('white')

    def _setup_counts_display(self):
        """Setup detection/track count display."""
        ax = self.axes['counts']
        ax.axis('off')
        ax.set_title('Detection & Tracking', color='white')

        # Create text elements
        self.text_elements['detections'] = ax.text(
            0.1, 0.7, 'Detections: 0', transform=ax.transAxes,
            fontsize=14, color='white', fontweight='bold'
        )
        self.text_elements['tracks'] = ax.text(
            0.1, 0.4, 'Tracks: 0 (0 confirmed)', transform=ax.transAxes,
            fontsize=14, color='white', fontweight='bold'
        )
        self.text_elements['det_rate'] = ax.text(
            0.1, 0.1, 'Det Rate: 0.0 /s', transform=ax.transAxes,
            fontsize=12, color='#AAAAAA'
        )

    def _setup_detection_sparkline(self):
        """Setup detection count sparkline."""
        ax = self.axes['det_spark']
        ax.set_xlim(0, self.params.history_length)
        ax.set_ylim(0, 20)
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Count', color='white')
        ax.set_title('Detection/Track History', color='white')
        ax.tick_params(colors='white')

        self.sparklines['detections'], = ax.plot([], [], color='#FF6B6B',
                                                   linewidth=1.5, label='Detections')
        self.sparklines['tracks'], = ax.plot([], [], color='#4ECDC4',
                                               linewidth=1.5, label='Tracks')

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        for spine in ax.spines.values():
            spine.set_color('white')

    def _setup_resources_display(self):
        """Setup system resources display."""
        ax = self.axes['resources']
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('Usage (%)', color='white')
        ax.set_title('System Resources', color='white')
        ax.tick_params(colors='white')

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['CPU', 'Memory'])

        # Resource bars
        self.resource_bars = []
        for i, label in enumerate(['CPU', 'Memory']):
            bar = ax.barh(i, 0, height=0.6, color=self.color_good,
                          edgecolor='white', linewidth=0.5)
            self.resource_bars.append(bar[0])

            text = ax.text(2, i, '0%', va='center', ha='left',
                          color='white', fontsize=9)
            self.text_elements[f'resource_{label.lower()}'] = text

        # Threshold lines
        ax.axvline(70, color=self.color_warning, linestyle='--', alpha=0.5)
        ax.axvline(90, color=self.color_bad, linestyle='--', alpha=0.5)

        ax.grid(True, alpha=0.2, axis='x')

        for spine in ax.spines.values():
            spine.set_color('white')

    def _setup_backend_display(self):
        """Setup backend acceleration status display."""
        ax = self.axes['backend']
        ax.axis('off')
        ax.set_title('Acceleration Backend', color='white')

        # NEON status
        self.status_indicators['neon'] = Circle((0.2, 0.6), 0.08,
                                                  transform=ax.transAxes,
                                                  facecolor=self.color_inactive,
                                                  edgecolor='white', linewidth=2)
        ax.add_patch(self.status_indicators['neon'])
        ax.text(0.35, 0.6, 'NEON SIMD', transform=ax.transAxes,
                fontsize=12, color='white', va='center')

        # Vulkan status
        self.status_indicators['vulkan'] = Circle((0.2, 0.3), 0.08,
                                                    transform=ax.transAxes,
                                                    facecolor=self.color_inactive,
                                                    edgecolor='white', linewidth=2)
        ax.add_patch(self.status_indicators['vulkan'])
        self.text_elements['vulkan_label'] = ax.text(
            0.35, 0.3, 'Vulkan GPU', transform=ax.transAxes,
            fontsize=12, color='white', va='center'
        )

    def _setup_timing_display(self):
        """Setup frame timing display."""
        ax = self.axes['timing']
        ax.axis('off')
        ax.set_title('Frame Timing', color='white')

        # Create text elements
        self.text_elements['frame_period'] = ax.text(
            0.05, 0.6, 'Frame Period: -- ms', transform=ax.transAxes,
            fontsize=11, color='white'
        )
        self.text_elements['frames_processed'] = ax.text(
            0.3, 0.6, 'Processed: 0', transform=ax.transAxes,
            fontsize=11, color='white'
        )
        self.text_elements['dropped_frames'] = ax.text(
            0.55, 0.6, 'Dropped: 0', transform=ax.transAxes,
            fontsize=11, color='white'
        )
        self.text_elements['sample_rate'] = ax.text(
            0.05, 0.2, 'Sample Rate: -- / -- Hz', transform=ax.transAxes,
            fontsize=11, color='white'
        )

        # Sample rate status indicator
        self.status_indicators['sample_rate'] = Circle((0.85, 0.4), 0.05,
                                                         transform=ax.transAxes,
                                                         facecolor=self.color_inactive,
                                                         edgecolor='white', linewidth=2)
        ax.add_patch(self.status_indicators['sample_rate'])
        ax.text(0.92, 0.4, 'Sample Rate OK', transform=ax.transAxes,
                fontsize=9, color='white', va='center')

    def _get_latency_color(self, latency_ms: float) -> str:
        """Get color based on latency value."""
        if latency_ms >= self.params.latency_critical_ms:
            return self.color_bad
        elif latency_ms >= self.params.latency_warning_ms:
            return self.color_warning
        else:
            return self.color_good

    def _get_resource_color(self, percent: float) -> str:
        """Get color based on resource usage."""
        if percent >= 90:
            return self.color_bad
        elif percent >= 70:
            return self.color_warning
        else:
            return self.color_good

    def _update(self, frame):
        """Animation update callback."""
        with self.lock:
            m = ProcessingMetrics(
                eca_latency_ms=self.metrics.eca_latency_ms,
                caf_latency_ms=self.metrics.caf_latency_ms,
                cfar_latency_ms=self.metrics.cfar_latency_ms,
                clustering_latency_ms=self.metrics.clustering_latency_ms,
                tracker_latency_ms=self.metrics.tracker_latency_ms,
                total_latency_ms=self.metrics.total_latency_ms,
                frame_period_ms=self.metrics.frame_period_ms,
                frames_processed=self.metrics.frames_processed,
                dropped_frames=self.metrics.dropped_frames,
                num_detections=self.metrics.num_detections,
                num_tracks=self.metrics.num_tracks,
                num_confirmed_tracks=self.metrics.num_confirmed_tracks,
                detection_rate=self.metrics.detection_rate,
                cpu_percent=self.metrics.cpu_percent,
                memory_percent=self.metrics.memory_percent,
                sample_rate_actual=self.metrics.sample_rate_actual,
                sample_rate_expected=self.metrics.sample_rate_expected,
                neon_enabled=self.metrics.neon_enabled,
                vulkan_enabled=self.metrics.vulkan_enabled,
                vulkan_device=self.metrics.vulkan_device,
            )

        # Update latency bars
        latencies = [m.eca_latency_ms, m.caf_latency_ms, m.cfar_latency_ms,
                     m.clustering_latency_ms, m.tracker_latency_ms, m.total_latency_ms]
        keys = ['eca', 'caf', 'cfar', 'cluster', 'tracker', 'total']

        for i, (bar, lat, key) in enumerate(zip(self.latency_bars, latencies, keys)):
            bar.set_width(lat)
            bar.set_color(self._get_latency_color(lat))
            self.text_elements[f'latency_{key}'].set_text(f'{lat:.1f} ms')
            self.text_elements[f'latency_{key}'].set_x(max(lat + 2, 2))

            self.latency_history[key].append(lat)

        # Update counts
        self.text_elements['detections'].set_text(f'Detections: {m.num_detections}')
        self.text_elements['tracks'].set_text(
            f'Tracks: {m.num_tracks} ({m.num_confirmed_tracks} confirmed)'
        )
        self.text_elements['det_rate'].set_text(f'Det Rate: {m.detection_rate:.1f} /s')

        # Update sparklines
        self.detection_history.append(m.num_detections)
        self.track_history.append(m.num_tracks)

        x = list(range(len(self.detection_history)))
        self.sparklines['detections'].set_data(x, list(self.detection_history))
        self.sparklines['tracks'].set_data(x, list(self.track_history))

        # Adjust y-axis
        max_val = max(max(self.detection_history, default=1),
                      max(self.track_history, default=1), 5)
        self.axes['det_spark'].set_ylim(0, max_val * 1.2)

        # Update resources
        resources = [m.cpu_percent, m.memory_percent]
        for i, (bar, val) in enumerate(zip(self.resource_bars, resources)):
            bar.set_width(val)
            bar.set_color(self._get_resource_color(val))

        self.text_elements['resource_cpu'].set_text(f'{m.cpu_percent:.1f}%')
        self.text_elements['resource_cpu'].set_x(max(m.cpu_percent + 2, 2))
        self.text_elements['resource_memory'].set_text(f'{m.memory_percent:.1f}%')
        self.text_elements['resource_memory'].set_x(max(m.memory_percent + 2, 2))

        # Update backend status
        self.status_indicators['neon'].set_facecolor(
            self.color_good if m.neon_enabled else self.color_inactive
        )
        self.status_indicators['vulkan'].set_facecolor(
            self.color_good if m.vulkan_enabled else self.color_inactive
        )

        if m.vulkan_device:
            self.text_elements['vulkan_label'].set_text(f'Vulkan: {m.vulkan_device}')

        # Update timing
        self.text_elements['frame_period'].set_text(
            f'Frame Period: {m.frame_period_ms:.1f} ms'
        )
        self.text_elements['frames_processed'].set_text(
            f'Processed: {m.frames_processed}'
        )
        self.text_elements['dropped_frames'].set_text(
            f'Dropped: {m.dropped_frames}'
        )
        self.text_elements['dropped_frames'].set_color(
            self.color_bad if m.dropped_frames > 0 else 'white'
        )

        # Sample rate check
        rate_error = abs(m.sample_rate_actual - m.sample_rate_expected) / m.sample_rate_expected
        if rate_error < 0.01:
            sr_color = self.color_good
        elif rate_error < 0.05:
            sr_color = self.color_warning
        else:
            sr_color = self.color_bad

        self.status_indicators['sample_rate'].set_facecolor(sr_color)
        self.text_elements['sample_rate'].set_text(
            f'Sample Rate: {m.sample_rate_actual/1000:.1f} / {m.sample_rate_expected/1000:.1f} kHz'
        )

        return list(self.latency_bars) + list(self.resource_bars)

    def update_metrics(self, metrics: ProcessingMetrics):
        """Thread-safe update of metrics."""
        with self.lock:
            self.metrics = metrics

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


def demo_metrics_dashboard():
    """Demo with simulated metrics."""
    params = MetricsDashboardParams(update_interval_ms=200)
    dashboard = MetricsDashboard(params)

    def data_generator():
        """Generate simulated metrics."""
        import random

        frame_count = 0
        dropped = 0
        start_time = time.time()

        while True:
            frame_count += 1
            elapsed = time.time() - start_time

            # Simulate processing latencies
            eca_lat = 5 + random.gauss(0, 1)
            caf_lat = 25 + random.gauss(0, 3)
            cfar_lat = 8 + random.gauss(0, 1)
            cluster_lat = 2 + random.gauss(0, 0.5)
            tracker_lat = 3 + random.gauss(0, 0.5)
            total_lat = eca_lat + caf_lat + cfar_lat + cluster_lat + tracker_lat

            # Occasional spike
            if random.random() < 0.02:
                caf_lat += 50
                total_lat += 50
                dropped += 1

            # Simulate detections/tracks
            num_dets = int(5 + 3 * np.sin(elapsed * 0.5) + random.gauss(0, 1))
            num_dets = max(0, num_dets)
            num_tracks = int(3 + 2 * np.sin(elapsed * 0.3) + random.gauss(0, 0.5))
            num_tracks = max(0, num_tracks)

            metrics = ProcessingMetrics(
                eca_latency_ms=eca_lat,
                caf_latency_ms=caf_lat,
                cfar_latency_ms=cfar_lat,
                clustering_latency_ms=cluster_lat,
                tracker_latency_ms=tracker_lat,
                total_latency_ms=total_lat,
                frame_period_ms=100.0,
                frames_processed=frame_count,
                dropped_frames=dropped,
                num_detections=num_dets,
                num_tracks=num_tracks,
                num_confirmed_tracks=max(0, num_tracks - 1),
                detection_rate=num_dets * 10.0,
                cpu_percent=30 + random.gauss(0, 5),
                memory_percent=45 + random.gauss(0, 2),
                sample_rate_actual=250000 + random.gauss(0, 500),
                sample_rate_expected=250000,
                neon_enabled=True,
                vulkan_enabled=True,
                vulkan_device="VideoCore VII",
                timestamp=time.time()
            )

            dashboard.update_metrics(metrics)
            time.sleep(0.1)

    # Start data generator thread
    gen_thread = threading.Thread(target=data_generator, daemon=True)
    gen_thread.start()

    # Start dashboard
    dashboard.start()


if __name__ == "__main__":
    demo_metrics_dashboard()
