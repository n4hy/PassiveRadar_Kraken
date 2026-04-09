"""
Pytest configuration and shared fixtures for PassiveRadar_Kraken tests.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "kraken_passive_radar"))
sys.path.insert(0, str(PROJECT_ROOT / "gr-kraken_passive_radar" / "python"))


def find_kernel_lib(lib_name: str, repo_root: Path = None) -> Path:
    """
    Find a kernel library, checking build directory first, then legacy source directory.

    Args:
        lib_name: Library name without 'libkraken_' prefix and '.so' suffix.
                  e.g., 'caf_processing', 'eca_b_clutter_canceller'
        repo_root: Repository root path. Defaults to PROJECT_ROOT.

    Returns:
        Path to the library file, or expected path if not found.
    """
    if repo_root is None:
        repo_root = PROJECT_ROOT

    full_name = f"libkraken_{lib_name}.so"

    # Check standard build directory first (build/lib/)
    build_lib = repo_root / "build" / "lib" / full_name
    if build_lib.exists():
        return build_lib

    # Check src/build/lib/ (in-source build of src/ only)
    src_build_lib = repo_root / "src" / "build" / "lib" / full_name
    if src_build_lib.exists():
        return src_build_lib

    # Legacy: check source directory (old in-source builds)
    src_lib = repo_root / "src" / full_name
    if src_lib.exists():
        return src_lib

    # Return expected path for better error messages
    return build_lib


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 2.4e6  # 2.4 MHz


@pytest.fixture
def cpi_samples():
    """Coherent Processing Interval in samples."""
    return 4096


@pytest.fixture
def num_channels():
    """Number of KrakenSDR channels."""
    return 5


@pytest.fixture
def random_seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def complex_noise(cpi_samples, random_seed):
    """Generate complex Gaussian noise."""
    np.random.seed(random_seed)
    return (np.random.randn(cpi_samples) + 1j * np.random.randn(cpi_samples)).astype(np.complex64)


@pytest.fixture
def fm_reference_signal(cpi_samples, sample_rate):
    """Generate FM broadcast-like reference signal."""
    t = np.arange(cpi_samples) / sample_rate
    # FM modulated signal with audio-like modulation
    modulation = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
    phase = 2 * np.pi * 75e3 * np.cumsum(modulation) / sample_rate  # 75 kHz deviation
    signal = np.exp(1j * phase).astype(np.complex64)
    return signal


@pytest.fixture
def target_with_delay_doppler(fm_reference_signal, sample_rate):
    """Factory fixture for creating target signals with delay and Doppler."""
    def _create_target(delay_samples, doppler_hz, snr_db=20):
        """Create a target signal with specified delay, Doppler shift, and SNR.

        Technique: circular shift for delay, complex exponential for Doppler, additive noise for SNR.
        """
        n = len(fm_reference_signal)
        t = np.arange(n) / sample_rate

        # Apply delay (circular shift for simplicity)
        delayed = np.roll(fm_reference_signal, delay_samples)

        # Apply Doppler shift
        doppler_shift = np.exp(2j * np.pi * doppler_hz * t).astype(np.complex64)
        target = delayed * doppler_shift

        # Scale for SNR
        signal_power = np.mean(np.abs(target)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(n) + 1j * np.random.randn(n))

        return (target + noise).astype(np.complex64)

    return _create_target


@pytest.fixture
def lib_paths():
    """Return dictionary of library paths."""
    return {
        'caf': find_kernel_lib("caf_processing"),
        'eca': find_kernel_lib("eca_b_clutter_canceller"),
        'doppler': find_kernel_lib("doppler_processing"),
        'backend': find_kernel_lib("backend"),
        'conditioning': find_kernel_lib("conditioning"),
        'time_alignment': find_kernel_lib("time_alignment"),
    }


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "hardware: marks tests requiring KrakenSDR hardware")
    config.addinivalue_line("markers", "visual: marks visual validation tests")
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")
