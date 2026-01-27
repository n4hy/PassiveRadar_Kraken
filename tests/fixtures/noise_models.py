"""
Noise models for passive radar testing.
"""
import numpy as np
from typing import Optional, Tuple


class NoiseGenerator:
    """Generate various noise types for radar testing."""

    def __init__(self, sample_rate: float = 2.4e6):
        self.sample_rate = sample_rate

    def generate_awgn(self, n_samples: int,
                      power_dbm: float = -100.0,
                      seed: Optional[int] = None) -> np.ndarray:
        """
        Generate Additive White Gaussian Noise.

        Args:
            n_samples: Number of samples
            power_dbm: Noise power in dBm
            seed: Random seed

        Returns:
            Complex AWGN samples
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert dBm to linear power
        power_linear = 10**((power_dbm - 30) / 10)  # Convert to Watts
        std = np.sqrt(power_linear / 2)  # Split between I and Q

        noise = std * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        return noise.astype(np.complex64)

    def generate_colored_noise(self, n_samples: int,
                               power_dbm: float = -100.0,
                               alpha: float = 1.0,
                               seed: Optional[int] = None) -> np.ndarray:
        """
        Generate colored (1/f^alpha) noise.

        Args:
            n_samples: Number of samples
            power_dbm: Noise power in dBm
            alpha: Spectral exponent (0=white, 1=pink, 2=brown)
            seed: Random seed

        Returns:
            Complex colored noise samples
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate white noise
        white = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

        # Apply 1/f^alpha shaping
        freqs = np.fft.fftfreq(n_samples)
        freqs[0] = 1e-10  # Avoid division by zero

        # Create filter
        filt = 1.0 / (np.abs(freqs) ** (alpha / 2))
        filt[0] = 0  # Remove DC

        # Apply filter
        colored = np.fft.ifft(np.fft.fft(white) * filt)

        # Normalize to desired power
        power_linear = 10**((power_dbm - 30) / 10)
        current_power = np.mean(np.abs(colored)**2)
        colored = colored * np.sqrt(power_linear / current_power)

        return colored.astype(np.complex64)

    def generate_impulsive_noise(self, n_samples: int,
                                 background_power_dbm: float = -100.0,
                                 impulse_rate: float = 0.001,
                                 impulse_power_dbm: float = -60.0,
                                 seed: Optional[int] = None) -> np.ndarray:
        """
        Generate impulsive noise (Gaussian background + impulses).

        Args:
            n_samples: Number of samples
            background_power_dbm: Background noise power
            impulse_rate: Probability of impulse per sample
            impulse_power_dbm: Peak impulse power
            seed: Random seed

        Returns:
            Complex impulsive noise samples
        """
        if seed is not None:
            np.random.seed(seed)

        # Background AWGN
        noise = self.generate_awgn(n_samples, background_power_dbm)

        # Add impulses
        impulse_locations = np.random.random(n_samples) < impulse_rate
        impulse_power = 10**((impulse_power_dbm - 30) / 10)
        impulse_std = np.sqrt(impulse_power / 2)

        impulses = impulse_std * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        impulses[~impulse_locations] = 0

        return (noise + impulses).astype(np.complex64)

    def generate_phase_noise(self, n_samples: int,
                             pn_level_dbc_hz: float = -100.0,
                             corner_freq_hz: float = 1000.0,
                             seed: Optional[int] = None) -> np.ndarray:
        """
        Generate phase noise for LO simulation.

        Args:
            n_samples: Number of samples
            pn_level_dbc_hz: Phase noise level at 1 Hz offset (dBc/Hz)
            corner_freq_hz: 1/f corner frequency
            seed: Random seed

        Returns:
            Complex phase noise multiplier (multiply with signal)
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate 1/f + white phase noise
        freqs = np.fft.fftfreq(n_samples, 1/self.sample_rate)
        freqs[0] = 1e-10

        # Phase noise PSD: flat at low freq, 1/f at high freq
        psd = np.ones(n_samples)
        high_freq = np.abs(freqs) > corner_freq_hz
        psd[high_freq] = corner_freq_hz / np.abs(freqs[high_freq])

        # Scale to desired level
        psd *= 10**(pn_level_dbc_hz / 10)

        # Generate random phase
        phase_spectrum = np.sqrt(psd) * np.exp(2j * np.pi * np.random.random(n_samples))
        phase = np.real(np.fft.ifft(phase_spectrum))

        # Convert to complex multiplier
        return np.exp(1j * phase).astype(np.complex64)

    def add_receiver_noise(self, signal: np.ndarray,
                           noise_figure_db: float = 3.0,
                           bandwidth_hz: float = 2.4e6,
                           temperature_k: float = 290.0) -> Tuple[np.ndarray, float]:
        """
        Add thermal noise based on receiver noise figure.

        Args:
            signal: Input signal
            noise_figure_db: Receiver noise figure in dB
            bandwidth_hz: Receiver bandwidth
            temperature_k: Temperature in Kelvin

        Returns:
            Noisy signal, noise floor in dBm
        """
        # Thermal noise power
        k_b = 1.38e-23  # Boltzmann constant
        thermal_noise_w = k_b * temperature_k * bandwidth_hz
        thermal_noise_dbm = 10 * np.log10(thermal_noise_w * 1000)

        # Add noise figure
        noise_floor_dbm = thermal_noise_dbm + noise_figure_db

        # Generate and add noise
        noise = self.generate_awgn(len(signal), noise_floor_dbm)

        return (signal + noise).astype(np.complex64), noise_floor_dbm

    def generate_quantization_noise(self, signal: np.ndarray,
                                    n_bits: int = 12) -> np.ndarray:
        """
        Simulate ADC quantization.

        Args:
            signal: Input signal
            n_bits: ADC resolution in bits

        Returns:
            Quantized signal
        """
        # Find signal range
        max_val = np.max(np.abs(signal)) * 1.1  # 10% headroom

        # Quantize
        n_levels = 2**n_bits
        step = 2 * max_val / n_levels

        real_q = np.round(np.real(signal) / step) * step
        imag_q = np.round(np.imag(signal) / step) * step

        return (real_q + 1j * imag_q).astype(np.complex64)
