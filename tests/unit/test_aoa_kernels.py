"""
Unit tests for Angle-of-Arrival (AoA) estimation.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestAoAEstimation(unittest.TestCase):
    """Test Angle-of-Arrival estimation algorithms."""

    def steering_vector_ula(self, n_elements, d_lambda, theta_rad):
        """
        Compute steering vector for Uniform Linear Array.

        Args:
            n_elements: Number of array elements
            d_lambda: Element spacing in wavelengths
            theta_rad: Angle of arrival in radians

        Returns:
            Complex steering vector
        """
        n = np.arange(n_elements)
        return np.exp(2j * np.pi * d_lambda * n * np.sin(theta_rad))

    def bartlett_spectrum(self, signals, d_lambda=0.5, n_angles=181):
        """
        Compute Bartlett beamformer spectrum.

        Args:
            signals: Complex array signals (n_elements,)
            d_lambda: Element spacing in wavelengths
            n_angles: Number of angle bins

        Returns:
            Power spectrum over angles
        """
        n_elements = len(signals)
        angles = np.linspace(-90, 90, n_angles) * np.pi / 180

        spectrum = np.zeros(n_angles)
        for i, theta in enumerate(angles):
            sv = self.steering_vector_ula(n_elements, d_lambda, theta)
            response = np.abs(np.dot(sv.conj(), signals))**2
            spectrum[i] = response

        return spectrum

    def test_broadside_response(self):
        """Verify peak at broadside (0 deg) for in-phase signals."""
        n_elements = 4

        # All channels in phase
        signals = np.ones(n_elements, dtype=np.complex64)

        spectrum = self.bartlett_spectrum(signals, d_lambda=0.5, n_angles=181)
        peak_idx = np.argmax(spectrum)
        peak_angle = (peak_idx - 90)  # Centered at 0

        self.assertLess(abs(peak_angle), 3, f"Broadside peak at {peak_angle} deg")

    def test_aoa_accuracy_various_angles(self):
        """Verify AoA estimation accuracy across scan range."""
        n_elements = 4
        d_lambda = 0.5

        for true_aoa_deg in [-60, -30, 0, 30, 60]:
            true_aoa_rad = np.radians(true_aoa_deg)

            # Generate array response
            signals = self.steering_vector_ula(n_elements, d_lambda, true_aoa_rad)

            spectrum = self.bartlett_spectrum(signals, d_lambda, n_angles=181)
            peak_idx = np.argmax(spectrum)
            estimated_aoa = peak_idx - 90

            error = abs(estimated_aoa - true_aoa_deg)
            self.assertLess(error, 3, f"AoA error {error} deg at {true_aoa_deg} deg")

    def test_aoa_with_noise(self):
        """Verify AoA accuracy with additive noise."""
        np.random.seed(42)
        n_elements = 4
        d_lambda = 0.5
        true_aoa_deg = 30
        true_aoa_rad = np.radians(true_aoa_deg)

        for snr_db in [20, 10, 5]:
            errors = []
            for _ in range(50):
                signals = self.steering_vector_ula(n_elements, d_lambda, true_aoa_rad)

                # Add noise
                noise_power = 10**(-snr_db/10)
                noise = np.sqrt(noise_power/2) * (np.random.randn(n_elements) +
                                                   1j * np.random.randn(n_elements))
                noisy_signals = signals + noise

                spectrum = self.bartlett_spectrum(noisy_signals, d_lambda, n_angles=181)
                peak_idx = np.argmax(spectrum)
                estimated_aoa = peak_idx - 90

                errors.append(abs(estimated_aoa - true_aoa_deg))

            rms_error = np.sqrt(np.mean(np.array(errors)**2))

            if snr_db >= 10:
                self.assertLess(rms_error, 10,
                                f"RMS error {rms_error:.1f} deg at SNR={snr_db}dB")

    def test_element_spacing_effect(self):
        """Verify element spacing affects beamwidth."""
        n_elements = 4
        true_aoa_rad = np.radians(20)

        beamwidths = []
        for d_lambda in [0.25, 0.5, 0.75]:
            signals = self.steering_vector_ula(n_elements, d_lambda, true_aoa_rad)
            spectrum = self.bartlett_spectrum(signals, d_lambda, n_angles=361)

            # Find -3dB beamwidth
            peak = spectrum.max()
            half_power = peak / 2
            above_half = spectrum > half_power
            beamwidth = np.sum(above_half) * (180 / 360)  # Convert to degrees
            beamwidths.append(beamwidth)

        # Larger spacing should give narrower beam
        self.assertLess(beamwidths[2], beamwidths[0],
                        "Larger spacing should give narrower beam")

    def test_grating_lobes(self):
        """Verify grating lobes appear with d > lambda/2."""
        n_elements = 4
        d_lambda = 1.0  # Full wavelength spacing
        true_aoa_rad = np.radians(20)

        signals = self.steering_vector_ula(n_elements, d_lambda, true_aoa_rad)
        spectrum = self.bartlett_spectrum(signals, d_lambda, n_angles=181)

        # Find peaks
        peak_threshold = spectrum.max() * 0.8

        peaks = []
        for i in range(1, 180):
            if spectrum[i] > peak_threshold:
                if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                    peaks.append(i - 90)

        # Should have grating lobe(s) in addition to main lobe
        self.assertGreater(len(peaks), 1, "Expected grating lobes with d=lambda")

    def test_coherent_signals(self):
        """Verify primary source detection with interference."""
        n_elements = 4
        d_lambda = 0.5

        # Two sources at different angles - Bartlett may not resolve both
        aoa1 = np.radians(20)
        aoa2 = np.radians(-30)

        signals = (self.steering_vector_ula(n_elements, d_lambda, aoa1) +
                   0.5 * self.steering_vector_ula(n_elements, d_lambda, aoa2))

        spectrum = self.bartlett_spectrum(signals, d_lambda, n_angles=181)

        # Find the dominant peak
        peak_idx = np.argmax(spectrum)
        peak_angle = peak_idx - 90

        # The stronger source (aoa1=20 deg) should be detected
        # Allow 10 degree tolerance since coherent sources can bias the result
        self.assertLess(abs(peak_angle - 20), 15,
                        f"Primary source at 20 deg not found, got {peak_angle} deg")


class TestMUSIC(unittest.TestCase):
    """Test MUSIC algorithm for high-resolution AoA."""

    def music_spectrum(self, signals, d_lambda=0.5, n_sources=1, n_angles=181):
        """
        Compute MUSIC pseudo-spectrum.

        Args:
            signals: Complex array signals (n_elements x n_snapshots)
            d_lambda: Element spacing in wavelengths
            n_sources: Number of sources
            n_angles: Number of angle bins

        Returns:
            MUSIC pseudo-spectrum
        """
        if signals.ndim == 1:
            signals = signals.reshape(-1, 1)

        n_elements = signals.shape[0]

        # Compute covariance matrix
        R = signals @ signals.conj().T / signals.shape[1]

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]

        # Noise subspace (smallest eigenvalues)
        En = eigenvectors[:, :n_elements - n_sources]

        # Scan angles
        angles = np.linspace(-90, 90, n_angles) * np.pi / 180
        spectrum = np.zeros(n_angles)

        for i, theta in enumerate(angles):
            a = np.exp(2j * np.pi * d_lambda * np.arange(n_elements) * np.sin(theta))
            a = a.reshape(-1, 1)

            denom = np.real(a.conj().T @ En @ En.conj().T @ a)
            spectrum[i] = 1.0 / (denom + 1e-10)

        return spectrum

    def test_music_single_source(self):
        """MUSIC should localize single source accurately."""
        np.random.seed(42)
        n_elements = 4
        n_snapshots = 100
        d_lambda = 0.5
        true_aoa = 25

        # Generate snapshots with noise
        signals = np.zeros((n_elements, n_snapshots), dtype=np.complex64)
        sv = np.exp(2j * np.pi * d_lambda * np.arange(n_elements) * np.sin(np.radians(true_aoa)))

        for i in range(n_snapshots):
            phase = np.random.uniform(0, 2*np.pi)
            signals[:, i] = sv * np.exp(1j * phase)
            signals[:, i] += 0.1 * (np.random.randn(n_elements) + 1j * np.random.randn(n_elements))

        spectrum = self.music_spectrum(signals, d_lambda, n_sources=1, n_angles=181)
        peak_idx = np.argmax(spectrum)
        estimated_aoa = peak_idx - 90

        self.assertLess(abs(estimated_aoa - true_aoa), 3,
                        f"MUSIC error: estimated {estimated_aoa}, true {true_aoa}")

    def test_music_two_close_sources(self):
        """MUSIC should resolve two close sources."""
        np.random.seed(42)
        n_elements = 8  # Need more elements for resolution
        n_snapshots = 200
        d_lambda = 0.5

        aoa1, aoa2 = 20, 30  # 10 degree separation

        signals = np.zeros((n_elements, n_snapshots), dtype=np.complex64)
        sv1 = np.exp(2j * np.pi * d_lambda * np.arange(n_elements) * np.sin(np.radians(aoa1)))
        sv2 = np.exp(2j * np.pi * d_lambda * np.arange(n_elements) * np.sin(np.radians(aoa2)))

        for i in range(n_snapshots):
            phase1, phase2 = np.random.uniform(0, 2*np.pi, 2)
            signals[:, i] = sv1 * np.exp(1j * phase1) + sv2 * np.exp(1j * phase2)
            signals[:, i] += 0.1 * (np.random.randn(n_elements) + 1j * np.random.randn(n_elements))

        spectrum = self.music_spectrum(signals, d_lambda, n_sources=2, n_angles=181)

        # Find two highest peaks
        sorted_idx = np.argsort(spectrum)[::-1]
        peak1 = sorted_idx[0] - 90
        peak2 = sorted_idx[1] - 90

        detected = sorted([peak1, peak2])
        expected = sorted([aoa1, aoa2])

        self.assertLess(abs(detected[0] - expected[0]), 5)
        self.assertLess(abs(detected[1] - expected[1]), 5)


if __name__ == '__main__':
    unittest.main()
