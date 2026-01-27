"""
Clutter models for passive radar testing.
"""
import numpy as np
from typing import Optional, Tuple


class ClutterGenerator:
    """Generate realistic clutter for passive radar testing."""

    def __init__(self, sample_rate: float = 2.4e6):
        self.sample_rate = sample_rate

    def generate_direct_path(self, ref_signal: np.ndarray,
                              power_db: float = 40.0) -> np.ndarray:
        """
        Generate direct path interference.

        Args:
            ref_signal: Reference signal
            power_db: Direct path power relative to target (dB)

        Returns:
            Direct path signal
        """
        amplitude = 10**(power_db / 20)
        return (ref_signal * amplitude).astype(np.complex64)

    def generate_multipath(self, ref_signal: np.ndarray,
                           n_paths: int = 10,
                           max_delay_m: float = 15000.0,
                           min_power_db: float = -10.0,
                           max_power_db: float = -40.0,
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Generate multipath clutter with exponential power decay.

        Args:
            ref_signal: Reference signal
            n_paths: Number of multipath components
            max_delay_m: Maximum multipath delay in meters
            min_power_db: Power of shortest multipath (dB below direct)
            max_power_db: Power of longest multipath (dB below direct)
            seed: Random seed

        Returns:
            Multipath clutter signal
        """
        if seed is not None:
            np.random.seed(seed)

        c = 3e8
        max_delay_samples = int((max_delay_m / c) * self.sample_rate)

        multipath = np.zeros_like(ref_signal)

        for i in range(n_paths):
            # Exponential distribution for delays (more short paths)
            delay = int(np.random.exponential(max_delay_samples / 3))
            delay = min(delay, max_delay_samples)

            # Power decays with delay
            power_db = min_power_db + (max_power_db - min_power_db) * (delay / max_delay_samples)
            amplitude = 10**(power_db / 20)

            # Random phase
            phase = np.random.uniform(0, 2 * np.pi)

            # Apply delay
            delayed = np.roll(ref_signal, delay)
            if delay > 0:
                delayed[:delay] = 0

            multipath += amplitude * np.exp(1j * phase) * delayed

        return multipath.astype(np.complex64)

    def generate_ground_clutter(self, ref_signal: np.ndarray,
                                clutter_bandwidth_hz: float = 5.0,
                                power_db: float = -20.0,
                                seed: Optional[int] = None) -> np.ndarray:
        """
        Generate ground clutter (near-zero Doppler spread).

        Args:
            ref_signal: Reference signal
            clutter_bandwidth_hz: Doppler spread of clutter
            power_db: Clutter power relative to reference
            seed: Random seed

        Returns:
            Ground clutter signal
        """
        if seed is not None:
            np.random.seed(seed)

        n = len(ref_signal)
        t = np.arange(n) / self.sample_rate

        # Multiple scatterers at different ranges with small Doppler spread
        clutter = np.zeros(n, dtype=np.complex64)

        for _ in range(20):  # 20 clutter scatterers
            delay = np.random.randint(10, 200)  # 10-200 sample delays
            doppler = np.random.normal(0, clutter_bandwidth_hz / 3)  # Small Doppler spread
            amplitude = 10**((power_db + np.random.uniform(-10, 0)) / 20)
            phase = np.random.uniform(0, 2 * np.pi)

            delayed = np.roll(ref_signal, delay)
            if delay > 0:
                delayed[:delay] = 0

            doppler_shift = np.exp(2j * np.pi * doppler * t)
            clutter += amplitude * np.exp(1j * phase) * delayed * doppler_shift

        return clutter.astype(np.complex64)

    def generate_interference(self, n_samples: int,
                              center_offset_hz: float = 50e3,
                              bandwidth_hz: float = 10e3,
                              power_db: float = -10.0,
                              seed: Optional[int] = None) -> np.ndarray:
        """
        Generate narrowband interference.

        Args:
            n_samples: Number of samples
            center_offset_hz: Frequency offset from center
            bandwidth_hz: Interference bandwidth
            power_db: Interference power
            seed: Random seed

        Returns:
            Interference signal
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.arange(n_samples) / self.sample_rate
        amplitude = 10**(power_db / 20)

        # Narrowband noise
        noise = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

        # Filter to bandwidth
        fft_noise = np.fft.fft(noise)
        freqs = np.fft.fftfreq(n_samples, 1/self.sample_rate)

        # Bandpass around center_offset_hz
        mask = np.abs(freqs - center_offset_hz) < bandwidth_hz / 2
        mask |= np.abs(freqs + center_offset_hz) < bandwidth_hz / 2
        fft_noise[~mask] = 0

        interference = np.fft.ifft(fft_noise)
        interference = amplitude * interference / (np.std(interference) + 1e-10)

        return interference.astype(np.complex64)

    def generate_combined_clutter(self, ref_signal: np.ndarray,
                                  direct_path_db: float = 40.0,
                                  n_multipath: int = 10,
                                  include_ground: bool = True,
                                  seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Generate combined clutter scenario.

        Args:
            ref_signal: Reference signal
            direct_path_db: Direct path power (dB)
            n_multipath: Number of multipath components
            include_ground: Include ground clutter
            seed: Random seed

        Returns:
            Combined clutter signal, clutter parameters dict
        """
        if seed is not None:
            np.random.seed(seed)

        clutter = np.zeros_like(ref_signal)

        # Direct path
        direct = self.generate_direct_path(ref_signal, direct_path_db)
        clutter += direct

        # Multipath
        multipath = self.generate_multipath(ref_signal, n_multipath)
        clutter += multipath

        # Ground clutter
        if include_ground:
            ground = self.generate_ground_clutter(ref_signal)
            clutter += ground

        params = {
            'direct_path_db': direct_path_db,
            'n_multipath': n_multipath,
            'include_ground': include_ground,
            'direct_power': np.mean(np.abs(direct)**2),
            'multipath_power': np.mean(np.abs(multipath)**2),
        }

        return clutter.astype(np.complex64), params
