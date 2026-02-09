"""
GPU Acceleration Backend for PassiveRadar_Kraken
Copyright (c) 2026 Dr Robert W McGwier, PhD
SPDX-License-Identifier: MIT

Provides runtime GPU backend selection and device management.
"""

import os
import ctypes
from pathlib import Path
from typing import Optional, Literal, Dict

# Backend types
BackendType = Literal['auto', 'gpu', 'cpu']

class GPUBackend:
    """
    GPU acceleration utilities and configuration.

    Manages GPU device detection, backend selection, and provides
    Python bindings to the GPU runtime library.
    """

    _backend: Optional[BackendType] = None
    _gpu_runtime_lib = None
    _initialized = False

    @classmethod
    def initialize(cls):
        """Initialize GPU runtime library (loads libkraken_gpu_runtime.so)."""
        if cls._initialized:
            return

        try:
            # Try to find GPU runtime library
            # First check in the same directory as this module
            module_dir = Path(__file__).parent
            lib_path = module_dir / "libkraken_gpu_runtime.so"

            if lib_path.exists():
                cls._gpu_runtime_lib = ctypes.cdll.LoadLibrary(str(lib_path))
            else:
                # Try src directory (for development builds)
                lib_path = module_dir.parent / "src" / "libkraken_gpu_runtime.so"
                if lib_path.exists():
                    cls._gpu_runtime_lib = ctypes.cdll.LoadLibrary(str(lib_path))
                else:
                    # Try system-wide install
                    cls._gpu_runtime_lib = ctypes.cdll.LoadLibrary("libkraken_gpu_runtime.so")

            # Setup function signatures
            cls._gpu_runtime_lib.kraken_gpu_device_count.restype = ctypes.c_int
            cls._gpu_runtime_lib.kraken_gpu_device_count.argtypes = []

            cls._gpu_runtime_lib.kraken_gpu_is_available.restype = ctypes.c_int
            cls._gpu_runtime_lib.kraken_gpu_is_available.argtypes = []

            cls._gpu_runtime_lib.kraken_gpu_get_device_info.restype = None
            cls._gpu_runtime_lib.kraken_gpu_get_device_info.argtypes = [
                ctypes.c_int,           # device_id
                ctypes.c_char_p,        # name buffer
                ctypes.POINTER(ctypes.c_int)  # compute_capability
            ]

            cls._gpu_runtime_lib.kraken_gpu_init.restype = ctypes.c_int
            cls._gpu_runtime_lib.kraken_gpu_init.argtypes = [ctypes.c_int]  # device_id

            cls._gpu_runtime_lib.kraken_gpu_cleanup.restype = None
            cls._gpu_runtime_lib.kraken_gpu_cleanup.argtypes = []

            cls._gpu_runtime_lib.kraken_should_use_gpu.restype = ctypes.c_int
            cls._gpu_runtime_lib.kraken_should_use_gpu.argtypes = []

            # Initialize GPU runtime (device 0 by default)
            if cls._gpu_runtime_lib.kraken_gpu_init(0) == 0:
                cls._initialized = True
            else:
                cls._gpu_runtime_lib = None

        except (OSError, AttributeError) as e:
            # GPU library not available (RPi5 CPU-only mode)
            cls._gpu_runtime_lib = None
            cls._initialized = False

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if GPU acceleration is available.

        Returns:
            True if GPU hardware and libraries are available, False otherwise
        """
        cls.initialize()
        if cls._gpu_runtime_lib is None:
            return False
        return cls._gpu_runtime_lib.kraken_gpu_is_available() > 0

    @classmethod
    def device_count(cls) -> int:
        """
        Get number of CUDA-capable GPU devices.

        Returns:
            Number of GPUs available (0 if no GPU or library not available)
        """
        cls.initialize()
        if cls._gpu_runtime_lib is None:
            return 0
        return cls._gpu_runtime_lib.kraken_gpu_device_count()

    @classmethod
    def get_device_info(cls, device_id: int = 0) -> Dict[str, any]:
        """
        Get information about a specific GPU device.

        Args:
            device_id: GPU device ID (0-based index)

        Returns:
            Dictionary containing:
                - 'name': GPU device name (str)
                - 'compute_capability': CUDA compute capability version (int)
                - 'device_id': Device ID (int)
            Returns empty dict if GPU not available or device_id invalid
        """
        cls.initialize()
        if cls._gpu_runtime_lib is None:
            return {}

        if device_id < 0 or device_id >= cls.device_count():
            return {}

        name_buffer = ctypes.create_string_buffer(256)
        compute_cap = ctypes.c_int()

        cls._gpu_runtime_lib.kraken_gpu_get_device_info(
            device_id, name_buffer, ctypes.byref(compute_cap)
        )

        return {
            'name': name_buffer.value.decode('utf-8'),
            'compute_capability': compute_cap.value,
            'device_id': device_id
        }

    @classmethod
    def cleanup(cls):
        """Cleanup GPU runtime resources."""
        if cls._gpu_runtime_lib is not None and cls._initialized:
            cls._gpu_runtime_lib.kraken_gpu_cleanup()
            cls._initialized = False


def set_processing_backend(backend: BackendType):
    """
    Set global processing backend for all GPU-capable kernels.

    This controls which implementation (CPU or GPU) is used for CAF,
    Doppler, and CFAR processing. The setting is process-wide.

    Args:
        backend: Processing backend selection:
            - 'auto': Use GPU if available, fallback to CPU (default)
            - 'gpu': Require GPU, fail if unavailable
            - 'cpu': Force CPU even if GPU is present

    Example:
        >>> set_processing_backend('gpu')  # Require GPU
        >>> set_processing_backend('auto') # Auto-detect (default)
    """
    if backend not in ('auto', 'gpu', 'cpu'):
        raise ValueError(f"Invalid backend '{backend}'. Must be 'auto', 'gpu', or 'cpu'")

    os.environ['KRAKEN_GPU_BACKEND'] = backend
    GPUBackend._backend = backend


def get_active_backend() -> str:
    """
    Get the currently active processing backend.

    Takes into account KRAKEN_GPU_BACKEND environment variable
    and actual GPU availability.

    Returns:
        'gpu' if GPU is being used, 'cpu' otherwise

    Example:
        >>> print(get_active_backend())
        'gpu'
    """
    backend_setting = os.environ.get('KRAKEN_GPU_BACKEND', 'auto')

    if backend_setting == 'cpu':
        return 'cpu'
    elif backend_setting == 'gpu':
        return 'gpu' if GPUBackend.is_available() else 'cpu'
    else:  # 'auto'
        return 'gpu' if GPUBackend.is_available() else 'cpu'


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns:
        True if GPU hardware and libraries are available

    Example:
        >>> if is_gpu_available():
        ...     print(f"GPU detected: {GPUBackend.get_device_info()['name']}")
        ... else:
        ...     print("Running in CPU-only mode (RPi5)")
    """
    return GPUBackend.is_available()


def get_gpu_info() -> Dict[str, any]:
    """
    Get information about the primary GPU device.

    Returns:
        Dictionary with GPU information, or empty dict if no GPU

    Example:
        >>> info = get_gpu_info()
        >>> if info:
        ...     print(f"Using {info['name']} (compute {info['compute_capability']})")
    """
    return GPUBackend.get_device_info(0)


# Auto-initialize on module import (graceful failure on RPi5)
GPUBackend.initialize()

# Set default backend if not already configured
if 'KRAKEN_GPU_BACKEND' not in os.environ:
    set_processing_backend('auto')
