# GPU Performance Profiling Guide
## PassiveRadar_Kraken CUDA Optimization

**Date:** 2026-02-08
**Target:** NVIDIA RTX 5090 (Blackwell)
**Status:** All kernels production-ready, profiling for optimization

---

## Table of Contents

- [Overview](#overview)
- [Profiling Tools](#profiling-tools)
- [Quick Start](#quick-start)
- [Kernel-Specific Profiling](#kernel-specific-profiling)
- [Optimization Opportunities](#optimization-opportunities)
- [Advanced Topics](#advanced-topics)

---

## Overview

All three GPU kernels (Doppler, CFAR, CAF) achieve 1.0 correlation with CPU and provide significant speedups. This guide shows how to profile for further optimization.

**Current Performance (RTX 5090):**
- CAF: 10 ms (16K samples), 2 ms (8K samples) - **23-29x speedup**
- Doppler: 1.27 ms (2048×512) - **10-15x speedup**
- CFAR: 1.94 ms (256×512) - **305x speedup**

---

## Profiling Tools

### NVIDIA Nsight Systems
**Purpose:** Timeline profiling, CPU-GPU interaction, API overhead

```bash
# Install (usually bundled with CUDA Toolkit)
sudo apt install nvidia-nsight-systems

# Profile CAF GPU test
nsys profile -o caf_profile python3 tests/gpu/test_gpu_caf.py

# View results
nsys-ui caf_profile.nsys-rep
```

**Key Metrics:**
- Kernel execution time
- Memory transfers (H2D, D2H)
- CUDA API overhead
- GPU utilization

### NVIDIA Nsight Compute
**Purpose:** Kernel-level profiling, warp efficiency, memory bandwidth

```bash
# Install
sudo apt install nvidia-nsight-compute

# Profile specific kernel
ncu --set full -o caf_kernel python3 tests/gpu/test_gpu_caf.py

# View results
ncu-ui caf_kernel.ncu-rep
```

**Key Metrics:**
- SM (Streaming Multiprocessor) efficiency
- Memory throughput (global, shared, L1/L2)
- Warp execution efficiency
- Register/shared memory usage
- Occupancy

### nvidia-smi
**Purpose:** Real-time monitoring during development

```bash
# Continuous monitoring (1 sec intervals)
watch -n 1 nvidia-smi

# Log GPU stats
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used \
           --format=csv -l 1 > gpu_monitor.csv
```

---

## Quick Start

### Step 1: Baseline Performance

```bash
# Run integration test to get baseline
python3 test_gpu_integration.py

# Expected output:
#   CAF:      10 ms  (16K samples)
#   Doppler:   1.5 ms
#   CFAR:      4 ms
#   Total:    15.5 ms (64 Hz)
```

### Step 2: Profile with Nsight Systems

```bash
# Profile full pipeline
nsys profile -t cuda,nvtx -o pipeline_profile \
    --stats=true \
    python3 test_gpu_integration.py

# Check summary
nsys stats pipeline_profile.nsys-rep
```

**Look for:**
- Long kernel execution times
- Excessive H2D/D2H transfers
- Idle GPU time
- CUDA API call overhead

### Step 3: Profile Specific Kernel

```bash
# Profile CAF kernel in detail
ncu --set full \
    --target-processes all \
    -o caf_detailed \
    python3 -c "
from tests.gpu.test_gpu_caf import *
import pytest
pytest.main(['-k', 'test_gpu_throughput'])
"

# View kernel metrics
ncu-ui caf_detailed.ncu-rep
```

---

## Kernel-Specific Profiling

### CAF GPU Profiling

**File:** `src/gpu/caf_gpu.cu`
**Critical Kernels:**
1. `interleaved_to_complex_kernel`
2. `apply_doppler_shift_to_reference_kernel`
3. `cross_correlation_multiply_kernel`
4. `extract_magnitude_kernel`

**Profile CAF:**
```bash
# Detailed CAF profiling
ncu --set roofline \
    --kernel-name "apply_doppler_shift|cross_correlation|extract_magnitude" \
    -o caf_kernels \
    python3 tests/gpu/test_gpu_caf.py::TestGPUCAFPerformance::test_gpu_throughput
```

**Expected Bottlenecks:**
1. **cuFFT calls** (forward/inverse) - Library-optimized, hard to improve
2. **Memory bandwidth** - Large data transfers (16K × 1K × 2)
3. **Zero-padding overhead** - Copy n_samples, zero rest to fft_len

**Optimization Opportunities:**
- Use streams for overlap (H2D during compute)
- Persistent memory allocations (already done ✅)
- Batch multiple CPIs together
- Investigate Tensor Cores for complex multiply (Blackwell)

### Doppler GPU Profiling

**File:** `src/gpu/doppler_gpu.cu`

**Profile Doppler:**
```bash
ncu --set full \
    --kernel-name doppler \
    -o doppler_kernel \
    python3 tests/gpu/test_gpu_doppler.py
```

**Key Metrics:**
- Memory coalescing efficiency
- FFT kernel performance (cuFFT library)
- Window application bandwidth

### CFAR GPU Profiling

**File:** `src/gpu/cfar_gpu.cu`

**Profile CFAR:**
```bash
ncu --set full \
    --kernel-name cfar \
    -o cfar_kernel \
    python3 tests/gpu/test_gpu_cfar.py
```

**Key Metrics:**
- Shared memory usage (for reference cell averaging)
- Warp divergence (edge cases)
- Global memory bandwidth

**Already Optimized:**
- ✅ Shared memory for neighborhoods
- ✅ Coalesced memory access
- ✅ Minimal warp divergence

---

## Optimization Opportunities

### 1. Multi-Stream Execution

**Current:** Sequential execution of kernels
**Proposed:** Overlap H2D, compute, D2H using streams

```cuda
// Create streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Pipeline: While processing CPI N, transfer CPI N+1
cudaMemcpyAsync(d_data1, h_data1, size, H2D, stream1);
process_kernel<<<grid, block, 0, stream1>>>(d_data1, ...);

cudaMemcpyAsync(d_data2, h_data2, size, H2D, stream2);  // Overlap!
process_kernel<<<grid, block, 0, stream2>>>(d_data2, ...);
```

**Expected Gain:** 10-20% reduction in end-to-end latency

### 2. Tensor Core Utilization (Blackwell)

**RTX 5090 has 5th-gen Tensor Cores**

Current complex multiply in `cross_correlation_multiply_kernel`:
```cuda
xcorr.x = (surv.x * ref.x + surv.y * ref.y) * scale;
xcorr.y = (surv.y * ref.x - surv.x * ref.y) * scale;
```

**Potential:** Use `wmma` (Warp Matrix Multiply-Accumulate) for batched operations

**Requirements:**
- Matrices must be 16×16 or larger
- FP16/BF16/TF32 precision
- Blackwell supports FP8 for even faster compute

**Investigation Needed:** Check if cuFFT + Tensor Core path exists

### 3. Multi-GPU Scaling

**Current:** Single GPU

**Proposed:** Distribute Doppler bins across GPUs
```python
# GPU 0: Doppler bins 0-511
# GPU 1: Doppler bins 512-1023
```

**Expected Gain:** Near-linear scaling (2 GPUs → 2× throughput)

**Complexity:** Moderate (requires NCCL for data distribution)

### 4. Reduced Precision

**Current:** FP32 throughout

**Options:**
- **FP16:** 2× memory bandwidth, 2× compute throughput
- **TF32:** Tensor Core friendly (19-bit mantissa)
- **FP8:** Blackwell exclusive, 4× bandwidth

**Validation Required:** Check if FP16 maintains 1.0 correlation

### 5. Kernel Fusion

**Current:** Separate kernels for each operation

**Proposed:** Fuse related operations to reduce memory traffic

Example:
```cuda
// Before: interleaved_to_complex + apply_doppler_shift (2 kernels)
// After: interleaved_to_complex_and_doppler_shift (1 kernel)
```

**Expected Gain:** 5-10% reduction in memory bandwidth

---

## Advanced Topics

### Memory Bandwidth Analysis

**Theoretical Peak (RTX 5090):** 1,792 GB/s

**Measure actual bandwidth:**
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -o bandwidth_test \
    python3 tests/gpu/test_gpu_caf.py
```

**Goal:** >80% of peak

### Occupancy Optimization

**Check kernel occupancy:**
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    -o occupancy \
    python3 tests/gpu/test_gpu_caf.py
```

**Occupancy Calculator:**
```bash
# Use CUDA Occupancy Calculator
cuda-occupancy-calculator
```

**Typical targets:**
- 50%+ occupancy for memory-bound kernels
- 70%+ occupancy for compute-bound kernels

### Power Profiling

**Monitor power consumption:**
```bash
nvidia-smi --query-gpu=power.draw --format=csv -l 1

# During processing, expect:
# - Idle: 50-100W
# - Light load: 150-250W
# - Full load: 400-575W (TDP)
```

**Power efficiency:**
- Current: ~64 CPIs/sec at ~300W = 0.21 CPIs/sec/W
- Target: Maintain performance at lower power (nvpmodel on Jetson)

### Build Optimization

**Current flags:**
```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
```

**Additional options:**
```cmake
# For production (remove debug symbols)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")

# For profiling (keep debug info)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 --use_fast_math -lineinfo")

# For Blackwell-specific optimization
set(CMAKE_CUDA_ARCHITECTURES "89")  # Remove older architectures
```

---

## Profiling Checklist

**Before Optimization:**
- [ ] Establish baseline metrics (run integration test)
- [ ] Profile with Nsight Systems (identify hotspots)
- [ ] Profile critical kernels with Nsight Compute
- [ ] Document current performance

**During Optimization:**
- [ ] Change ONE thing at a time
- [ ] Re-profile after each change
- [ ] Verify correctness (run pytest suite)
- [ ] Document speedup/regression

**After Optimization:**
- [ ] Validate on multiple platforms (Jetson, Cloud)
- [ ] Stress test (long-duration, thermal throttling)
- [ ] Update documentation with new benchmarks

---

## Support

**Tools Documentation:**
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling](https://docs.nvidia.com/cuda/profiler-users-guide/)

**Related Docs:**
- [GPU Performance Benchmarks](GPU_PERFORMANCE.md)
- [GPU User Guide](GPU_USER_GUIDE.md)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-08
**Author:** Dr. Robert W McGwier, PhD, N4HY
**GPU Implementation:** Claude (Anthropic)
