"""
GPU Runtime Tests - Device Detection and Backend Selection
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Tests GPU runtime functionality:
- Device detection
- Backend selection
- Memory management
- Python API bindings
"""

import pytest
import os
import sys
from pathlib import Path

# Add repo root to path (contains the display/GUI kraken_passive_radar package)
# Clear any cached OOT version that test_end_to_end may have loaded
_repo_root = str(Path(__file__).parents[2])
sys.path.insert(0, _repo_root)
if 'kraken_passive_radar' in sys.modules:
    del sys.modules['kraken_passive_radar']

from kraken_passive_radar import (
    is_gpu_available,
    set_processing_backend,
    get_active_backend,
    get_gpu_info,
    GPU_AVAILABLE,
)

pytestmark = pytest.mark.gpu


class TestGPUDeviceDetection:
    """Test GPU device detection and information queries."""

    def test_gpu_availability_flag(self):
        """Test GPU_AVAILABLE module-level flag."""
        assert isinstance(GPU_AVAILABLE, bool)
        if not GPU_AVAILABLE:
            pytest.skip("GPU libraries not installed (CPU-only mode)")

    def test_is_gpu_available(self):
        """Test GPU availability check."""
        result = is_gpu_available()
        assert isinstance(result, bool)

        if result:
            # If GPU is available, should have device info
            info = get_gpu_info()
            assert isinstance(info, dict)
            assert 'name' in info
            assert 'compute_capability' in info
            assert len(info['name']) > 0
        else:
            pytest.skip("No GPU hardware detected")

    def test_gpu_info_format(self):
        """Test GPU device information structure."""
        if not is_gpu_available():
            pytest.skip("No GPU available")

        info = get_gpu_info()
        assert 'name' in info
        assert 'compute_capability' in info
        assert 'device_id' in info

        assert isinstance(info['name'], str)
        assert isinstance(info['compute_capability'], int)
        assert isinstance(info['device_id'], int)

        # Compute capability should be reasonable (5.0 to 12.0 for modern GPUs, including Blackwell)
        assert 50 <= info['compute_capability'] <= 120, \
            f"Unexpected compute capability: {info['compute_capability']}"

        print(f"\nDetected GPU: {info['name']} (compute {info['compute_capability']/10:.1f})")


class TestBackendSelection:
    """Test runtime backend selection API."""

    def test_set_backend_auto(self):
        """Test setting backend to 'auto' mode."""
        set_processing_backend('auto')
        backend = get_active_backend()
        assert backend in ['cpu', 'gpu']

        # Check environment variable
        assert os.environ.get('KRAKEN_GPU_BACKEND') == 'auto'

    def test_set_backend_cpu(self):
        """Test forcing CPU backend."""
        set_processing_backend('cpu')
        backend = get_active_backend()
        assert backend == 'cpu'

        # Even if GPU available, should return 'cpu'
        assert os.environ.get('KRAKEN_GPU_BACKEND') == 'cpu'

    def test_set_backend_gpu(self):
        """Test setting GPU backend."""
        set_processing_backend('gpu')
        assert os.environ.get('KRAKEN_GPU_BACKEND') == 'gpu'

        # Note: get_active_backend() will return 'cpu' if GPU not actually available
        backend = get_active_backend()
        if is_gpu_available():
            assert backend == 'gpu'
        else:
            assert backend == 'cpu'

    def test_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            set_processing_backend('invalid')

        with pytest.raises(ValueError):
            set_processing_backend('cuda')  # Should be 'gpu' not 'cuda'

    def test_backend_persistence(self):
        """Test backend setting persists across calls."""
        set_processing_backend('cpu')
        assert get_active_backend() == 'cpu'
        assert get_active_backend() == 'cpu'  # Should be consistent

        set_processing_backend('auto')
        backend1 = get_active_backend()
        backend2 = get_active_backend()
        assert backend1 == backend2


class TestBackendFallback:
    """Test graceful fallback behavior."""

    def test_cpu_only_fallback(self):
        """Test that CPU-only mode works even when GPU requested."""
        # This test is mainly for RPi5 CPU-only builds
        set_processing_backend('gpu')

        if not is_gpu_available():
            # Should gracefully fall back to CPU
            assert get_active_backend() == 'cpu'

            # get_gpu_info should return empty dict
            info = get_gpu_info()
            assert info == {}

    def test_auto_mode_chooses_correctly(self):
        """Test that 'auto' mode makes correct choice."""
        set_processing_backend('auto')
        backend = get_active_backend()

        if is_gpu_available():
            assert backend == 'gpu', "Should use GPU when available in auto mode"
        else:
            assert backend == 'cpu', "Should use CPU when GPU unavailable in auto mode"


class TestPythonAPIConsistency:
    """Test Python API remains consistent across platforms."""

    @staticmethod
    def _reload_display_package():
        """Ensure we load the display package (not OOT module)."""
        # The OOT module (gr-kraken_passive_radar/python/kraken_passive_radar)
        # may be cached in sys.modules from other tests. Clear it so we can
        # import the display/GUI package from the repo root.
        if 'kraken_passive_radar' in sys.modules:
            mod = sys.modules['kraken_passive_radar']
            mod_file = getattr(mod, '__file__', '') or ''
            if 'gr-kraken_passive_radar' in mod_file:
                del sys.modules['kraken_passive_radar']

    def test_all_functions_exist(self):
        """Test that all GPU API functions exist (even on CPU-only)."""
        self._reload_display_package()
        # These should exist on both GPU and CPU-only builds
        from kraken_passive_radar import (
            is_gpu_available,
            set_processing_backend,
            get_active_backend,
            get_gpu_info,
            GPUBackend,
        )

        assert callable(is_gpu_available)
        assert callable(set_processing_backend)
        assert callable(get_active_backend)
        assert callable(get_gpu_info)
        assert hasattr(GPUBackend, 'is_available')

    def test_cpu_only_stubs(self):
        """Test that CPU-only mode provides working stubs."""
        # Even on CPU-only build, functions should not crash
        result = is_gpu_available()
        assert isinstance(result, bool)

        set_processing_backend('cpu')  # Should not crash

        backend = get_active_backend()
        assert backend == 'cpu'  # Always CPU on CPU-only build

        info = get_gpu_info()
        assert isinstance(info, dict)  # Empty dict on CPU-only

    def test_no_imports_fail_on_rpi5(self):
        """Test that imports work even without GPU libraries (RPi5 mode)."""
        self._reload_display_package()
        # This import should never fail, even on RPi5 CPU-only
        from kraken_passive_radar import is_gpu_available

        # Should return False on RPi5, True on GPU systems
        result = is_gpu_available()
        assert result in [True, False]


class TestEnvironmentVariableControl:
    """Test KRAKEN_GPU_BACKEND environment variable."""

    def test_env_var_override(self):
        """Test that environment variable controls backend."""
        # Set via API
        set_processing_backend('cpu')
        assert os.environ.get('KRAKEN_GPU_BACKEND') == 'cpu'
        assert get_active_backend() == 'cpu'

        # Change via environment variable
        os.environ['KRAKEN_GPU_BACKEND'] = 'auto'
        backend = get_active_backend()
        assert backend in ['cpu', 'gpu']

    def test_env_var_precedence(self):
        """Test environment variable takes precedence."""
        # Set environment variable directly
        os.environ['KRAKEN_GPU_BACKEND'] = 'cpu'
        assert get_active_backend() == 'cpu'

        # Setting via API should update env var
        set_processing_backend('auto')
        assert os.environ.get('KRAKEN_GPU_BACKEND') == 'auto'


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
