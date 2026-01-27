"""
Calibration Controller for KrakenSDR Phase Coherence
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Handles automatic phase calibration when coherence drift is detected.

The KrakenSDR has an internal noise source with a high-isolation silicon switch.
When the noise source is enabled:
  - The silicon switch DISCONNECTS all antennas from the signal path
  - ONLY the internal noise source feeds all 5 channels
  - This provides a common reference for measuring inter-channel phase offsets

Calibration Flow:
  1. Coherence monitor detects phase drift exceeding threshold
  2. Coherence monitor emits cal_request message
  3. CalibrationController receives cal_request
  4. CalibrationController enables noise source (antennas isolated by hardware)
  5. CalibrationController captures calibration samples
  6. CalibrationController computes phase correction phasors
  7. CalibrationController applies corrections to signal path
  8. CalibrationController disables noise source (antennas reconnected)
  9. CalibrationController emits cal_complete message
  10. Normal operation resumes with corrected phase

This MUST occur BEFORE ECA processing uses the data, as ECA requires
phase-coherent inputs for proper clutter cancellation.
"""

import numpy as np
import threading
import time
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum


class CalibrationState(Enum):
    """Calibration controller state machine."""
    IDLE = 0                    # Normal operation, monitoring coherence
    CALIBRATING = 1             # Noise source ON, capturing samples
    COMPUTING = 2               # Computing correction phasors
    APPLYING = 3                # Applying corrections


@dataclass
class CalibrationResult:
    """Result of a calibration cycle."""
    success: bool
    phase_offsets: np.ndarray   # Phase correction for each channel (radians)
    correlation: np.ndarray     # Correlation coefficient with reference
    timestamp: float
    reason: str                 # Why calibration was triggered


class CalibrationController:
    """
    Manages automatic phase calibration for KrakenSDR.

    Integrates with:
    - krakensdr_source: Controls noise source enable/disable
    - coherence_monitor: Receives cal_request, sends cal_complete
    - Signal path: Applies phase corrections to incoming samples

    IMPORTANT: When noise source is enabled, the KrakenSDR hardware
    silicon switch DISCONNECTS all antennas. Only the internal noise
    source feeds the receivers. This is a hardware feature, not software.
    """

    def __init__(self,
                 source,                    # krakensdr_source instance
                 num_channels: int = 5,
                 cal_samples: int = 24000,  # 10ms at 2.4 MHz
                 settle_time_ms: float = 50.0,
                 on_calibration_complete: Optional[Callable] = None):
        """
        Initialize calibration controller.

        Args:
            source: krakensdr_source instance with set_noise_source() method
            num_channels: Number of coherent channels (default 5)
            cal_samples: Number of samples to capture for calibration
            settle_time_ms: Time to wait after noise source enable (for switch settling)
            on_calibration_complete: Callback when calibration finishes
        """
        self.source = source
        self.num_channels = num_channels
        self.cal_samples = cal_samples
        self.settle_time_sec = settle_time_ms / 1000.0
        self.callback = on_calibration_complete

        # State
        self.state = CalibrationState.IDLE
        self.state_lock = threading.Lock()

        # Current phase corrections (radians, applied to channels 1-4)
        # Channel 0 is reference, always 0 correction
        self.phase_corrections = np.zeros(num_channels, dtype=np.float32)

        # Calibration buffers
        self.cal_buffers: List[np.ndarray] = [None] * num_channels
        self.cal_sample_count = 0

        # Statistics
        self.calibration_count = 0
        self.last_calibration_time = 0.0
        self.last_result: Optional[CalibrationResult] = None

    def handle_cal_request(self, reason: str = "coherence_degraded"):
        """
        Handle calibration request from coherence monitor.

        This triggers the calibration sequence:
        1. Enable noise source (hardware isolates antennas)
        2. Wait for switch settling
        3. Capture calibration samples
        4. Compute phase corrections
        5. Apply corrections
        6. Disable noise source (hardware reconnects antennas)

        Args:
            reason: Why calibration was requested
        """
        with self.state_lock:
            if self.state != CalibrationState.IDLE:
                print(f"CalibrationController: Already calibrating, ignoring request")
                return
            self.state = CalibrationState.CALIBRATING

        print(f"CalibrationController: Starting calibration (reason: {reason})")

        # Step 1: Enable noise source
        # CRITICAL: This activates the hardware silicon switch that
        # DISCONNECTS all antennas and routes ONLY the noise source
        # to all 5 receiver channels
        try:
            self.source.set_noise_source(True)
            print("CalibrationController: Noise source ENABLED (antennas isolated)")
        except Exception as e:
            print(f"CalibrationController: Failed to enable noise source: {e}")
            with self.state_lock:
                self.state = CalibrationState.IDLE
            return

        # Step 2: Wait for switch settling
        time.sleep(self.settle_time_sec)

        # Step 3: Reset calibration buffers
        self.cal_buffers = [np.zeros(self.cal_samples, dtype=np.complex64)
                           for _ in range(self.num_channels)]
        self.cal_sample_count = 0

        # Samples will be collected via process_calibration_samples()
        # which should be called by the signal path during CALIBRATING state

    def process_calibration_samples(self, channel: int, samples: np.ndarray):
        """
        Process incoming samples during calibration.

        Called by the signal path when state is CALIBRATING.
        When noise source is ON, these samples contain ONLY noise
        (antennas are isolated by hardware switch).

        Args:
            channel: Channel index (0-4)
            samples: Complex samples from this channel
        """
        with self.state_lock:
            if self.state != CalibrationState.CALIBRATING:
                return

        # Append samples to buffer
        n_needed = self.cal_samples - self.cal_sample_count
        n_copy = min(len(samples), n_needed)

        if n_copy > 0:
            start = self.cal_sample_count
            self.cal_buffers[channel][start:start + n_copy] = samples[:n_copy]

        # Check if we have enough samples on channel 0 (reference)
        if channel == 0:
            self.cal_sample_count += n_copy

            if self.cal_sample_count >= self.cal_samples:
                # We have enough samples, compute corrections
                self._compute_corrections()

    def _compute_corrections(self):
        """Compute phase corrections from calibration samples."""
        with self.state_lock:
            self.state = CalibrationState.COMPUTING

        print("CalibrationController: Computing phase corrections...")

        reference = self.cal_buffers[0]
        correlations = np.zeros(self.num_channels)
        phase_offsets = np.zeros(self.num_channels)

        for ch in range(self.num_channels):
            if ch == 0:
                correlations[ch] = 1.0
                phase_offsets[ch] = 0.0
                continue

            surv = self.cal_buffers[ch]

            # Cross-correlation to find phase offset
            # Since noise source is common, signals should be highly correlated
            # with only phase difference
            xcorr = np.vdot(reference, surv)  # Conjugate dot product

            # Correlation coefficient
            ref_power = np.sqrt(np.vdot(reference, reference).real)
            surv_power = np.sqrt(np.vdot(surv, surv).real)

            if ref_power > 0 and surv_power > 0:
                correlations[ch] = np.abs(xcorr) / (ref_power * surv_power)
            else:
                correlations[ch] = 0.0

            # Phase offset (negative because we want to correct it)
            phase_offsets[ch] = -np.angle(xcorr)

        # Apply corrections
        self._apply_corrections(phase_offsets, correlations)

    def _apply_corrections(self, phase_offsets: np.ndarray, correlations: np.ndarray):
        """Apply computed phase corrections and finish calibration."""
        with self.state_lock:
            self.state = CalibrationState.APPLYING

        # Update corrections
        self.phase_corrections = phase_offsets.astype(np.float32)

        print(f"CalibrationController: Phase corrections (deg): "
              f"{np.degrees(self.phase_corrections)}")
        print(f"CalibrationController: Correlations: {correlations}")

        # Disable noise source
        # CRITICAL: This deactivates the hardware silicon switch,
        # RECONNECTING all antennas to their respective receivers
        try:
            self.source.set_noise_source(False)
            print("CalibrationController: Noise source DISABLED (antennas reconnected)")
        except Exception as e:
            print(f"CalibrationController: Failed to disable noise source: {e}")

        # Record result
        self.last_result = CalibrationResult(
            success=True,
            phase_offsets=self.phase_corrections.copy(),
            correlation=correlations,
            timestamp=time.time(),
            reason="coherence_degraded"
        )

        self.calibration_count += 1
        self.last_calibration_time = time.time()

        # Return to idle
        with self.state_lock:
            self.state = CalibrationState.IDLE

        print(f"CalibrationController: Calibration complete (#{self.calibration_count})")

        # Notify callback
        if self.callback:
            self.callback(self.last_result)

    def apply_phase_correction(self, channel: int, samples: np.ndarray) -> np.ndarray:
        """
        Apply phase correction to samples from a channel.

        This should be called in the signal path BEFORE ECA processing.

        Args:
            channel: Channel index (0-4)
            samples: Complex samples to correct

        Returns:
            Phase-corrected samples
        """
        if channel == 0 or self.phase_corrections[channel] == 0:
            return samples

        # Apply phase rotation
        correction = np.exp(1j * self.phase_corrections[channel])
        return samples * correction

    def get_correction_phasors(self) -> np.ndarray:
        """
        Get current correction phasors for all channels.

        Returns:
            Complex phasors to multiply with each channel's samples
        """
        return np.exp(1j * self.phase_corrections).astype(np.complex64)

    @property
    def is_calibrating(self) -> bool:
        """Check if calibration is in progress."""
        with self.state_lock:
            return self.state != CalibrationState.IDLE

    def get_status(self) -> dict:
        """Get current calibration status."""
        return {
            'state': self.state.name,
            'calibration_count': self.calibration_count,
            'last_calibration_time': self.last_calibration_time,
            'phase_corrections_deg': np.degrees(self.phase_corrections).tolist(),
            'is_calibrating': self.is_calibrating
        }


class PhaseCorrector:
    """
    GNU Radio-compatible block that applies phase corrections.

    Place this AFTER the source and BEFORE the ECA canceller.

    Signal flow:
        KrakenSDR Source -> PhaseCorrector -> ECA Canceller -> ...

    The PhaseCorrector:
    - During normal operation: applies stored phase corrections
    - During calibration: passes samples to CalibrationController
    """

    def __init__(self, calibration_controller: CalibrationController):
        """
        Initialize phase corrector.

        Args:
            calibration_controller: CalibrationController instance
        """
        self.controller = calibration_controller

    def process(self, channel: int, samples: np.ndarray) -> np.ndarray:
        """
        Process samples through phase correction.

        Args:
            channel: Channel index
            samples: Input samples

        Returns:
            Corrected samples (or original if calibrating)
        """
        # If calibrating, feed samples to controller
        if self.controller.is_calibrating:
            self.controller.process_calibration_samples(channel, samples)
            # During calibration, don't process antenna signals
            # (they're disconnected anyway - hardware switch)
            return np.zeros_like(samples)

        # Normal operation: apply phase correction
        return self.controller.apply_phase_correction(channel, samples)
