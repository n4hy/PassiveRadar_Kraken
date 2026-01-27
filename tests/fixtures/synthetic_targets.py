"""
Synthetic target generation for passive radar testing.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TargetSpec:
    """Specification for a synthetic radar target."""
    range_m: float          # Bistatic range in meters
    doppler_hz: float       # Doppler shift in Hz
    snr_db: float           # Signal-to-noise ratio in dB
    rcs_db: float = 0.0     # Radar cross section (relative)
    aoa_deg: float = 0.0    # Angle of arrival in degrees


@dataclass
class SystemParams:
    """Passive radar system parameters."""
    sample_rate: float = 2.4e6      # Sample rate in Hz
    center_freq: float = 100e6       # Center frequency in Hz
    range_resolution: float = 125.0  # Range resolution in meters
    doppler_resolution: float = 1.0  # Doppler resolution in Hz
    num_channels: int = 5            # Number of array channels
    element_spacing: float = 0.5     # Element spacing in wavelengths


@dataclass
class ScenarioData:
    """Container for a complete test scenario."""
    ref_signal: np.ndarray      # Reference channel signal
    surv_signals: np.ndarray    # Surveillance channel signals (n_channels x n_samples)
    targets: List[TargetSpec]   # Target specifications
    ground_truth: dict          # Expected detection results
    params: SystemParams        # System parameters


class BistaticTargetGenerator:
    """Generate bistatic radar target returns for testing."""

    def __init__(self, params: Optional[SystemParams] = None):
        self.params = params or SystemParams()

    def generate_fm_waveform(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate FM broadcast-like waveform.

        Args:
            n_samples: Number of samples
            seed: Random seed for reproducibility

        Returns:
            Complex FM signal
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.arange(n_samples) / self.params.sample_rate

        # Audio-like modulation (multiple tones)
        audio = (np.sin(2 * np.pi * 1000 * t) +
                 0.7 * np.sin(2 * np.pi * 2500 * t) +
                 0.5 * np.sin(2 * np.pi * 4000 * t) +
                 0.3 * np.random.randn(n_samples))  # Add some noise for realism

        # FM modulation with 75 kHz deviation
        deviation = 75e3
        phase = 2 * np.pi * deviation * np.cumsum(audio) / self.params.sample_rate

        return np.exp(1j * phase).astype(np.complex64)

    def apply_delay(self, signal: np.ndarray, delay_samples: float) -> np.ndarray:
        """
        Apply fractional sample delay using sinc interpolation.

        Args:
            signal: Input signal
            delay_samples: Delay in samples (can be fractional)

        Returns:
            Delayed signal
        """
        n = len(signal)
        int_delay = int(np.floor(delay_samples))
        frac_delay = delay_samples - int_delay

        # Integer delay via roll
        delayed = np.roll(signal, int_delay)

        # Fractional delay via FFT phase shift
        if abs(frac_delay) > 1e-6:
            freq = np.fft.fftfreq(n)
            phase_shift = np.exp(-2j * np.pi * freq * frac_delay)
            delayed = np.fft.ifft(np.fft.fft(delayed) * phase_shift).astype(np.complex64)

        # Zero out wrapped samples
        if int_delay > 0:
            delayed[:int_delay] = 0

        return delayed

    def generate_target(self,
                        ref_signal: np.ndarray,
                        bistatic_range_m: float,
                        doppler_hz: float,
                        snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complex IQ samples for a point target.

        Args:
            ref_signal: Reference channel signal
            bistatic_range_m: Bistatic range in meters
            doppler_hz: Doppler shift in Hz
            snr_db: Signal-to-noise ratio in dB

        Returns:
            target_signal: Target component only
            surv_signal: Surveillance signal (direct path + target + noise)
        """
        n_samples = len(ref_signal)
        t = np.arange(n_samples) / self.params.sample_rate

        # Convert range to delay
        c = 3e8  # Speed of light
        delay_samples = (bistatic_range_m / c) * self.params.sample_rate

        # Apply delay
        target_signal = self.apply_delay(ref_signal, delay_samples)

        # Apply Doppler shift
        doppler_shift = np.exp(2j * np.pi * doppler_hz * t).astype(np.complex64)
        target_signal = target_signal * doppler_shift

        # Scale for SNR relative to reference
        ref_power = np.mean(np.abs(ref_signal)**2)
        target_amplitude = np.sqrt(ref_power * 10**(snr_db/10))
        target_signal = target_signal * (target_amplitude / (np.std(target_signal) + 1e-10))

        # Direct path (40 dB stronger than target)
        direct_path = ref_signal * 10**((snr_db + 40) / 20)

        # Noise floor
        noise_power = ref_power * 10**(-20/10)  # -20 dB below reference
        noise = np.sqrt(noise_power/2) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))

        surv_signal = (direct_path + target_signal + noise).astype(np.complex64)

        return target_signal.astype(np.complex64), surv_signal

    def generate_multipath(self, ref_signal: np.ndarray,
                           n_paths: int = 5,
                           max_delay_samples: int = 100) -> np.ndarray:
        """
        Generate multipath clutter.

        Args:
            ref_signal: Reference signal
            n_paths: Number of multipath reflections
            max_delay_samples: Maximum delay for multipath

        Returns:
            Multipath signal component
        """
        multipath = np.zeros_like(ref_signal)

        for i in range(n_paths):
            delay = np.random.randint(1, max_delay_samples)
            amplitude = 10**(-np.random.uniform(10, 30)/20)  # -10 to -30 dB
            phase = np.random.uniform(0, 2*np.pi)

            delayed = self.apply_delay(ref_signal, delay)
            multipath += amplitude * np.exp(1j * phase) * delayed

        return multipath.astype(np.complex64)

    def generate_scenario(self, targets: List[TargetSpec],
                          duration_sec: float = 0.1,
                          seed: Optional[int] = None) -> ScenarioData:
        """
        Generate multi-target scenario with specified targets.

        Args:
            targets: List of target specifications
            duration_sec: Duration in seconds
            seed: Random seed

        Returns:
            ScenarioData containing all signals and ground truth
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = int(duration_sec * self.params.sample_rate)

        # Generate reference
        ref_signal = self.generate_fm_waveform(n_samples)

        # Initialize surveillance (4 surveillance channels)
        n_surv = self.params.num_channels - 1
        surv_signals = np.zeros((n_surv, n_samples), dtype=np.complex64)

        # Direct path for each surveillance channel
        for ch in range(n_surv):
            surv_signals[ch] = ref_signal * 10**(40/20)  # 40 dB direct path

        # Add targets
        for target in targets:
            target_sig, _ = self.generate_target(
                ref_signal,
                target.range_m,
                target.doppler_hz,
                target.snr_db
            )

            # Add to all surveillance channels with phase based on AoA
            for ch in range(n_surv):
                phase_offset = 2 * np.pi * self.params.element_spacing * ch * \
                              np.sin(np.radians(target.aoa_deg))
                surv_signals[ch] += target_sig * np.exp(1j * phase_offset)

        # Add multipath and noise to each channel
        for ch in range(n_surv):
            surv_signals[ch] += self.generate_multipath(ref_signal)
            noise_power = np.mean(np.abs(ref_signal)**2) * 10**(-20/10)
            noise = np.sqrt(noise_power/2) * (np.random.randn(n_samples) +
                                               1j * np.random.randn(n_samples))
            surv_signals[ch] += noise

        # Build ground truth
        ground_truth = {
            'targets': [
                {
                    'range_m': t.range_m,
                    'doppler_hz': t.doppler_hz,
                    'snr_db': t.snr_db,
                    'aoa_deg': t.aoa_deg
                }
                for t in targets
            ],
            'expected_detections': len(targets),
        }

        return ScenarioData(
            ref_signal=ref_signal,
            surv_signals=surv_signals.astype(np.complex64),
            targets=targets,
            ground_truth=ground_truth,
            params=self.params
        )


# Predefined test scenarios
def single_target_snr15() -> ScenarioData:
    """Single target at SNR=15dB."""
    gen = BistaticTargetGenerator()
    return gen.generate_scenario(
        [TargetSpec(range_m=5000, doppler_hz=100, snr_db=15)],
        seed=42
    )


def two_close_targets() -> ScenarioData:
    """Two targets 2 bins apart for resolution testing."""
    gen = BistaticTargetGenerator()
    return gen.generate_scenario([
        TargetSpec(range_m=5000, doppler_hz=100, snr_db=15),
        TargetSpec(range_m=5250, doppler_hz=102, snr_db=15),  # 2 range bins, 2 Doppler bins
    ], seed=42)


def crossing_targets() -> ScenarioData:
    """Two targets on crossing paths."""
    gen = BistaticTargetGenerator()
    return gen.generate_scenario([
        TargetSpec(range_m=5000, doppler_hz=100, snr_db=15, aoa_deg=-30),
        TargetSpec(range_m=8000, doppler_hz=-50, snr_db=12, aoa_deg=30),
    ], seed=42)


def weak_target() -> ScenarioData:
    """Single weak target at detection threshold."""
    gen = BistaticTargetGenerator()
    return gen.generate_scenario(
        [TargetSpec(range_m=5000, doppler_hz=50, snr_db=-3)],
        seed=42
    )


def multi_target_dense() -> ScenarioData:
    """10 targets for scale testing."""
    gen = BistaticTargetGenerator()
    targets = [
        TargetSpec(
            range_m=3000 + i * 1000,
            doppler_hz=-100 + i * 25,
            snr_db=10 + np.random.uniform(-3, 3),
            aoa_deg=-40 + i * 10
        )
        for i in range(10)
    ]
    return gen.generate_scenario(targets, seed=42)
