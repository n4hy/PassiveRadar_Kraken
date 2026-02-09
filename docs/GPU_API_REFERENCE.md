# GPU API Reference
## PassiveRadar_Kraken CUDA Acceleration API

**Version:** 0.2.0
**Date:** 2026-02-08
**Language:** Python (ctypes bindings to C/CUDA libraries)

---

## Table of Contents

- [Overview](#overview)
- [Python High-Level API](#python-high-level-api)
- [GPU Runtime Library](#gpu-runtime-library)
- [Doppler GPU Kernel](#doppler-gpu-kernel)
- [CFAR GPU Kernel](#cfar-gpu-kernel)
- [CAF GPU Kernel](#caf-gpu-kernel)
- [Error Handling](#error-handling)
- [Memory Management](#memory-management)
- [Advanced Usage](#advanced-usage)

---

## Overview

PassiveRadar_Kraken GPU acceleration is exposed through two API layers:

1. **High-Level Python API** (`kraken_passive_radar.gpu_backend`)
   - Backend selection (auto/gpu/cpu)
   - GPU detection and device info
   - Simplified interface for most users

2. **Low-Level C/CUDA API** (ctypes bindings)
   - Direct kernel access
   - Fine-grained control
   - Advanced performance tuning

---

## Python High-Level API

### Module Import

```python
from kraken_passive_radar import (
    is_gpu_available,      # GPU detection
    get_gpu_info,          # Device information
    set_processing_backend,  # Backend selection
    get_active_backend,    # Query active backend
    GPUBackend             # Advanced: Direct GPU runtime access
)
```

### GPU Detection

#### `is_gpu_available() -> bool`

Check if GPU hardware and CUDA libraries are available.

**Returns:**
- `bool`: `True` if GPU available, `False` otherwise

**Example:**
```python
if is_gpu_available():
    print("GPU acceleration available")
else:
    print("Running in CPU-only mode")
```

**Implementation:**
- Checks for CUDA-capable GPU via `cudaGetDeviceCount()`
- Verifies GPU runtime library loaded successfully
- Returns `False` gracefully if CUDA not installed

---

#### `get_gpu_info(device_id=0) -> dict`

Get information about a GPU device.

**Parameters:**
- `device_id` (int, optional): CUDA device ID (default: 0)

**Returns:**
- `dict` with keys:
  - `'name'` (str): GPU device name (e.g., "NVIDIA GeForce RTX 5090")
  - `'compute_capability'` (int): Compute capability × 10 (e.g., 120 for sm_12.0)
  - `'device_id'` (int): CUDA device ID

**Returns empty dict if:**
- GPU not available
- Invalid `device_id`

**Example:**
```python
info = get_gpu_info()
if info:
    print(f"GPU: {info['name']}")
    print(f"Compute Capability: {info['compute_capability'] / 10.0:.1f}")
else:
    print("No GPU available")
```

---

### Backend Selection

#### `set_processing_backend(backend: str) -> None`

Set global processing backend for all GPU-capable kernels.

**Parameters:**
- `backend` (str): One of:
  - `'auto'`: Use GPU if available, fallback to CPU (default)
  - `'gpu'`: Require GPU, fail if unavailable
  - `'cpu'`: Force CPU even if GPU present

**Raises:**
- `ValueError`: If `backend` not in `('auto', 'gpu', 'cpu')`

**Side Effects:**
- Sets `KRAKEN_GPU_BACKEND` environment variable
- Affects all kernels in current process

**Example:**
```python
# Auto-detect (recommended)
set_processing_backend('auto')

# Require GPU for production workloads
set_processing_backend('gpu')

# Force CPU for debugging/testing
set_processing_backend('cpu')
```

---

#### `get_active_backend() -> str`

Get currently active processing backend.

**Returns:**
- `str`: `'gpu'` if GPU being used, `'cpu'` otherwise

**Example:**
```python
backend = get_active_backend()
print(f"Active backend: {backend}")

if backend == 'gpu':
    print("Using GPU acceleration")
else:
    print("Using CPU fallback")
```

**Logic:**
- If `KRAKEN_GPU_BACKEND='cpu'` → returns `'cpu'`
- If `KRAKEN_GPU_BACKEND='gpu'` → returns `'gpu'` (if available) or `'cpu'` (fallback)
- If `KRAKEN_GPU_BACKEND='auto'` → returns `'gpu'` (if available) or `'cpu'`

---

### Advanced: GPUBackend Class

Direct access to GPU runtime library.

```python
from kraken_passive_radar import GPUBackend

# Device count
n = GPUBackend.device_count()
print(f"Found {n} CUDA devices")

# Device info (all devices)
for i in range(n):
    info = GPUBackend.get_device_info(i)
    print(f"Device {i}: {info['name']}")

# Cleanup (usually automatic)
GPUBackend.cleanup()
```

**Methods:**
- `device_count() -> int`: Number of CUDA devices
- `get_device_info(device_id) -> dict`: Device information
- `is_available() -> bool`: GPU availability check
- `cleanup() -> None`: Release GPU resources

---

## GPU Runtime Library

Low-level C API (`libkraken_gpu_runtime.so`) accessed via ctypes.

### Load Library

```python
import ctypes
from pathlib import Path

lib_path = Path("/home/n4hy/PassiveRadar_Kraken/src/libkraken_gpu_runtime.so")
gpu_runtime = ctypes.cdll.LoadLibrary(str(lib_path))
```

### C API Functions

#### `kraken_gpu_device_count() -> int`

**Signature:**
```c
int kraken_gpu_device_count(void);
```

**Returns:** Number of CUDA-capable GPUs (0 if none)

**Python:**
```python
gpu_runtime.kraken_gpu_device_count.restype = ctypes.c_int
n = gpu_runtime.kraken_gpu_device_count()
```

---

#### `kraken_gpu_is_available() -> int`

**Signature:**
```c
int kraken_gpu_is_available(void);
```

**Returns:** 1 if GPU available, 0 otherwise

**Python:**
```python
gpu_runtime.kraken_gpu_is_available.restype = ctypes.c_int
available = gpu_runtime.kraken_gpu_is_available() > 0
```

---

#### `kraken_gpu_get_device_info(device_id, name, compute_cap)`

**Signature:**
```c
void kraken_gpu_get_device_info(int device_id, char* name, int* compute_capability);
```

**Parameters:**
- `device_id`: CUDA device ID (0-based)
- `name`: Output buffer for device name (min 256 bytes)
- `compute_capability`: Output pointer for compute capability × 10

**Python:**
```python
gpu_runtime.kraken_gpu_get_device_info.restype = None
gpu_runtime.kraken_gpu_get_device_info.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int)
]

name_buf = ctypes.create_string_buffer(256)
compute_cap = ctypes.c_int()

gpu_runtime.kraken_gpu_get_device_info(0, name_buf, ctypes.byref(compute_cap))

print(f"GPU: {name_buf.value.decode()}")
print(f"Compute: {compute_cap.value / 10.0}")
```

---

#### `kraken_gpu_init(device_id) -> int`

**Signature:**
```c
int kraken_gpu_init(int device_id);
```

**Parameters:**
- `device_id`: CUDA device to initialize (usually 0)

**Returns:** 0 on success, -1 on failure

**Python:**
```python
gpu_runtime.kraken_gpu_init.restype = ctypes.c_int
gpu_runtime.kraken_gpu_init.argtypes = [ctypes.c_int]

if gpu_runtime.kraken_gpu_init(0) == 0:
    print("GPU initialized successfully")
else:
    print("GPU initialization failed")
```

---

#### `kraken_gpu_cleanup()`

**Signature:**
```c
void kraken_gpu_cleanup(void);
```

**Python:**
```python
gpu_runtime.kraken_gpu_cleanup.restype = None
gpu_runtime.kraken_gpu_cleanup()
```

---

## Doppler GPU Kernel

Batched 2D FFT Doppler processing (`libkraken_doppler_gpu.so`).

### Load Library

```python
import ctypes
import numpy as np

doppler_lib = ctypes.cdll.LoadLibrary("libkraken_doppler_gpu.so")

# Setup function signatures
doppler_lib.doppler_gpu_create.restype = ctypes.c_void_p
doppler_lib.doppler_gpu_create.argtypes = [ctypes.c_int, ctypes.c_int]

doppler_lib.doppler_gpu_destroy.restype = None
doppler_lib.doppler_gpu_destroy.argtypes = [ctypes.c_void_p]

doppler_lib.doppler_gpu_process.restype = None
doppler_lib.doppler_gpu_process.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]
```

### `doppler_gpu_create(fft_len, doppler_len) -> void*`

Create Doppler GPU processing context.

**Parameters:**
- `fft_len` (int): FFT length (range bins)
- `doppler_len` (int): Number of Doppler bins

**Returns:** Opaque handle (void pointer), or NULL on error

**Example:**
```python
fft_len = 2048
doppler_len = 512

handle = doppler_lib.doppler_gpu_create(fft_len, doppler_len)
if handle == 0:
    raise RuntimeError("Failed to create Doppler GPU context")
```

**Allocates:**
- Device memory: ~16 MB (for 2048×512)
- Pinned host memory for async transfers
- cuFFT plan for batched 1D FFTs

---

### `doppler_gpu_destroy(handle)`

Release Doppler GPU resources.

**Parameters:**
- `handle` (void*): Context from `doppler_gpu_create()`

**Example:**
```python
doppler_lib.doppler_gpu_destroy(handle)
```

---

### `doppler_gpu_process(handle, input, output)`

Process Doppler FFT with log magnitude output.

**Parameters:**
- `handle` (void*): Context from `doppler_gpu_create()`
- `input` (float*): Input data, interleaved complex (I,Q,I,Q,...)
  - Shape: `(doppler_len * fft_len * 2,)` float32
  - Layout: Row-major, `[doppler_bin_0_range_0_I, doppler_bin_0_range_0_Q, ...]`
- `output` (float*): Output log magnitude in dB
  - Shape: `(doppler_len * fft_len,)` float32
  - Layout: Row-major, `[doppler_bin, range_bin]`

**Example:**
```python
# Generate complex input (doppler_len rows, fft_len cols)
input_complex = np.random.randn(doppler_len, fft_len) + \
                1j * np.random.randn(doppler_len, fft_len)
input_complex = input_complex.astype(np.complex64)

# Convert to interleaved format
input_flat = np.zeros(doppler_len * fft_len * 2, dtype=np.float32)
input_flat[0::2] = input_complex.real.flatten()  # I components
input_flat[1::2] = input_complex.imag.flatten()  # Q components

# Allocate output
output_flat = np.zeros(doppler_len * fft_len, dtype=np.float32)

# Create ctypes pointers
input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Process
doppler_lib.doppler_gpu_process(handle, input_ptr, output_ptr)

# Reshape output
output_db = output_flat.reshape((doppler_len, fft_len))
```

**Processing:**
1. Transfer input to GPU (async)
2. Apply Hamming window
3. Batched 1D FFT along Doppler dimension (rows)
4. FFT shift
5. Compute log magnitude: `10 * log10(|X|^2 + epsilon)`
6. Transfer output to CPU (async)

**Performance:**
- RTX 5090: 1.27 ms (2048×512)
- Throughput: 786 Hz

---

### `doppler_gpu_process_complex(handle, input, output)`

Process Doppler FFT with complex output (advanced).

**Parameters:**
- Same as `doppler_gpu_process()`, except:
- `output` (float*): Complex interleaved output (I,Q,I,Q,...)
  - Shape: `(doppler_len * fft_len * 2,)` float32

**Example:**
```python
output_complex_flat = np.zeros(doppler_len * fft_len * 2, dtype=np.float32)
output_ptr = output_complex_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

doppler_lib.doppler_gpu_process_complex(handle, input_ptr, output_ptr)

# Convert back to complex
output_complex = (output_complex_flat[0::2] +
                  1j * output_complex_flat[1::2]).reshape((doppler_len, fft_len))
```

---

## CFAR GPU Kernel

Parallel 2D CFAR detection (`libkraken_cfar_gpu.so`).

### Load Library

```python
import ctypes

cfar_lib = ctypes.cdll.LoadLibrary("libkraken_cfar_gpu.so")

cfar_lib.cfar_gpu_2d.restype = None
cfar_lib.cfar_gpu_2d.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                     # rows
    ctypes.c_int,                     # cols
    ctypes.c_int,                     # guard
    ctypes.c_int,                     # train
    ctypes.c_float                    # threshold
]
```

### `cfar_gpu_2d(input, output, rows, cols, guard, train, threshold)`

2D CA-CFAR detection.

**Parameters:**
- `input` (float*): Input magnitude data (row-major)
  - Shape: `(rows * cols,)` float32
- `output` (float*): Binary detection mask
  - Shape: `(rows * cols,)` float32
  - Values: 1.0 = detection, 0.0 = no detection
- `rows` (int): Number of rows (Doppler bins)
- `cols` (int): Number of columns (range bins)
- `guard` (int): Guard cell size (both dimensions)
- `train` (int): Training cell size (both dimensions)
- `threshold` (float): Detection threshold (linear, not dB)

**Example:**
```python
rows, cols = 256, 512
guard, train = 2, 4
threshold = 3.0  # 3x noise level

# Input data (e.g., from Doppler output)
input_data = np.abs(np.random.randn(rows, cols)).astype(np.float32)
input_flat = input_data.flatten()

# Output detection mask
output_flat = np.zeros_like(input_flat)

# Create ctypes pointers
input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Process
cfar_lib.cfar_gpu_2d(input_ptr, output_ptr, rows, cols, guard, train, threshold)

# Reshape and find detections
output_mask = output_flat.reshape((rows, cols))
detections = np.argwhere(output_mask > 0)
print(f"Found {len(detections)} detections")
```

**Algorithm:**
- Cell-Averaging CFAR (CA-CFAR)
- Each thread processes one cell
- 2D sliding window:
  - Guard region: `[cell - guard, cell + guard]`
  - Training region: `[cell - guard - train, cell + guard + train]`
- Detection: `cell_value > threshold * mean(training_cells)`

**Performance:**
- RTX 5090: 1.94 ms (256×512), **305x faster than CPU**
- Throughput: 515 Hz (medium), 2411 Hz (large)

---

## CAF GPU Kernel

Batched cuFFT CAF processing (`libkraken_caf_gpu.so`).

**⚠️ Note:** CAF GPU has excellent performance (23x speedup) but correctness issue in progress. API documented for reference.

### Load Library

```python
caf_lib = ctypes.cdll.LoadLibrary("libkraken_caf_gpu.so")

caf_lib.caf_gpu_create_full.restype = ctypes.c_void_p
caf_lib.caf_gpu_create_full.argtypes = [
    ctypes.c_int,    # n_samples
    ctypes.c_int,    # n_doppler
    ctypes.c_int,    # n_range
    ctypes.c_float,  # doppler_start_hz
    ctypes.c_float,  # doppler_step_hz
    ctypes.c_float   # sample_rate_hz
]

caf_lib.caf_gpu_destroy.restype = None
caf_lib.caf_gpu_destroy.argtypes = [ctypes.c_void_p]

caf_lib.caf_gpu_process_full.restype = None
caf_lib.caf_gpu_process_full.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),  # reference
    ctypes.POINTER(ctypes.c_float),  # surveillance
    ctypes.POINTER(ctypes.c_float)   # output
]
```

### `caf_gpu_create_full(...) -> void*`

Create CAF GPU processing context.

**Parameters:**
- `n_samples` (int): Number of range samples
- `n_doppler` (int): Number of Doppler bins
- `n_range` (int): Number of range bins (correlation lags)
- `doppler_start_hz` (float): Starting Doppler frequency
- `doppler_step_hz` (float): Doppler bin spacing
- `sample_rate_hz` (float): Sample rate

**Returns:** Opaque handle, or NULL on error

---

### `caf_gpu_process_full(handle, ref, surv, output)`

Process CAF with full Doppler-range grid.

**Parameters:**
- `handle` (void*): Context from `caf_gpu_create_full()`
- `ref` (float*): Reference signal (interleaved I/Q)
  - Shape: `(n_samples * 2,)` float32
- `surv` (float*): Surveillance signal (interleaved I/Q)
  - Shape: `(n_samples * 2,)` float32
- `output` (float*): CAF magnitude
  - Shape: `(n_doppler * n_range,)` float32
  - Layout: Row-major `[doppler_bin, range_bin]`

**Performance:**
- RTX 5090: 2.03 ms (8K samples, 512 Doppler)
- Throughput: 492 Hz

---

## Error Handling

### GPU Errors

**Detection:**
```python
from kraken_passive_radar import is_gpu_available

if not is_gpu_available():
    print("GPU not available - check CUDA installation")
    # Fallback to CPU
```

**Handling kernel failures:**
```python
handle = doppler_lib.doppler_gpu_create(fft_len, doppler_len)
if handle == 0:
    raise RuntimeError("Failed to create GPU context - out of memory?")
```

**Check for NaN/Inf in output:**
```python
output_data = ...  # from GPU kernel

if np.any(np.isnan(output_data)):
    print("ERROR: GPU output contains NaN")
elif np.any(np.isinf(output_data)):
    print("ERROR: GPU output contains Inf")
else:
    print("GPU output valid")
```

---

## Memory Management

### Lifecycle

1. **Create context:** `*_gpu_create()` allocates GPU memory
2. **Process data:** `*_gpu_process()` reuses allocated buffers
3. **Destroy context:** `*_gpu_destroy()` releases memory

**Best Practice:** Create context once, process multiple CPIs, then destroy.

```python
# GOOD: Create once, process many
handle = doppler_lib.doppler_gpu_create(fft_len, doppler_len)

for _ in range(1000):
    doppler_lib.doppler_gpu_process(handle, input_ptr, output_ptr)
    # ... use output

doppler_lib.doppler_gpu_destroy(handle)
```

```python
# BAD: Create/destroy every iteration (slow!)
for _ in range(1000):
    handle = doppler_lib.doppler_gpu_create(fft_len, doppler_len)
    doppler_lib.doppler_gpu_process(handle, input_ptr, output_ptr)
    doppler_lib.doppler_gpu_destroy(handle)  # Overhead!
```

---

## Advanced Usage

### Warmup for Accurate Benchmarks

First kernel call includes CUDA initialization overhead. Warmup before benchmarking:

```python
# Warmup (exclude from timing)
for _ in range(10):
    doppler_lib.doppler_gpu_process(handle, input_ptr, output_ptr)

# Benchmark
import time
n_iter = 100
start = time.time()
for _ in range(n_iter):
    doppler_lib.doppler_gpu_process(handle, input_ptr, output_ptr)
elapsed = time.time() - start

avg_time_ms = (elapsed / n_iter) * 1000
print(f"Average time: {avg_time_ms:.2f} ms")
```

### Multi-Device Support (Future)

Currently only device 0 is used. Future API:

```python
# Create context on specific device
handle = doppler_lib.doppler_gpu_create_ex(
    fft_len, doppler_len, device_id=1
)
```

---

**Document Version:** 1.0
**Last Updated:** 2026-02-08
**Author:** Dr. Robert W McGwier, PhD, N4HY
**GPU Implementation:** Claude (Anthropic)
