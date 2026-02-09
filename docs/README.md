# PassiveRadar_Kraken Documentation Index

**Version:** 0.2.0
**Last Updated:** 2026-02-08

---

## Quick Links

| Document | Description | Audience |
|----------|-------------|----------|
| [Main README](../README.md) | Project overview, installation, quick start | All users |
| [GPU User Guide](GPU_USER_GUIDE.md) | Complete GPU setup and configuration | GPU users |
| [GPU Performance](GPU_PERFORMANCE.md) | Benchmarks and optimization tips | Performance tuning |
| [GPU API Reference](GPU_API_REFERENCE.md) | Complete API documentation | Developers |
| [GPU Deployment](GPU_DEPLOYMENT.md) | Platform-specific deployment guides | DevOps, deployment |
| [Changelog](../CHANGELOG.md) | Version history and release notes | All users |

---

## Documentation Structure

### User Documentation

**Getting Started:**
1. Read [Main README](../README.md) for project overview
2. Follow installation instructions for your platform
3. Run first application with CPU or GPU backend

**GPU Acceleration:**
1. Check [GPU User Guide](GPU_USER_GUIDE.md) for prerequisites
2. Build with GPU support following platform guide
3. Configure backend selection (auto/gpu/cpu)
4. Review [GPU Performance](GPU_PERFORMANCE.md) for optimization tips

### Developer Documentation

**API Reference:**
- [GPU API Reference](GPU_API_REFERENCE.md) - Complete Python and C/CUDA API
- Python blocks: See `gr-kraken_passive_radar/python/kraken_passive_radar/`
- C++ blocks: See `gr-kraken_passive_radar/include/gnuradio/kraken_passive_radar/`

**Architecture:**
- Signal processing chain: See [Main README § Signal Processing Chain](../README.md#signal-processing-chain)
- GPU architecture: See [GPU Performance § Technical Details](GPU_PERFORMANCE.md#technical-details)
- System architecture: See [Main README § System Architecture](../README.md#system-architecture)

### Deployment Documentation

**Platform Guides:**
- Raspberry Pi 5 (CPU-only): [GPU Deployment § Scenario 1](GPU_DEPLOYMENT.md#scenario-1-raspberry-pi-5-cpu-only)
- Desktop + RTX GPU: [GPU Deployment § Scenario 2](GPU_DEPLOYMENT.md#scenario-2-desktop-with-rtx-gpu)
- NVIDIA Jetson Orin: [GPU Deployment § Scenario 3](GPU_DEPLOYMENT.md#scenario-3-nvidia-jetson-orin)
- Cloud (AWS/Azure): [GPU Deployment § Scenario 4](GPU_DEPLOYMENT.md#scenario-4-cloud-gpu-awsazure)
- Docker: [GPU Deployment § Docker Deployment](GPU_DEPLOYMENT.md#docker-deployment)

---

## Feature Comparison

### CPU vs GPU Capabilities

| Feature | CPU (RPi5) | GPU (RTX 5090) | Notes |
|---------|------------|----------------|-------|
| **Doppler Processing** | ~15 ms | **1.27 ms** | ✅ Validated (1.0 correlation) |
| **CFAR Detection** | 592 ms | **1.94 ms** | ✅ Validated (99.99% agreement) |
| **CAF Processing** | 46.7 ms | **2.03 ms** | ⚠️ Performance only (debugging WIP) |
| **Update Rate** | ~10 Hz | **100-200 Hz** | When CAF fixed |
| **Power Consumption** | 10-15W | 50-300W | GPU varies by workload |
| **Cost** | $80 | $1500-3500 | Hardware only |

### Platform Support Matrix

| Platform | Build Mode | GPU Support | Status |
|----------|-----------|-------------|--------|
| Raspberry Pi 5 | CPU-only | ❌ None | ✅ Validated |
| Desktop (RTX 2000-5000) | GPU-enabled | ✅ Full | ✅ Validated (RTX 5090) |
| NVIDIA Jetson Orin | GPU-enabled | ✅ Full | ⚠️ Expected (not tested) |
| AWS/Azure GPU | GPU-enabled | ✅ Full | ⚠️ Expected (not tested) |

---

## Testing Documentation

### Test Coverage

**CPU Tests** (196 total):
- 191 passed
- 5 skipped (display imports, headless environment)

**GPU Tests:**
- Doppler GPU: ✅ Comprehensive validation (random, impulse, tone)
- CFAR GPU: ✅ Synthetic target detection
- CAF GPU: Performance benchmarking
- Infrastructure: ✅ Device detection, backend selection

**Test Locations:**
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- GPU tests: `tests/gpu/`
- Benchmarks: `tests/benchmarks/`

### Running Tests

```bash
# All CPU tests
python3 -m pytest tests/ -v

# GPU-specific tests
python3 tests/gpu/test_gpu_doppler.py
python3 tests/gpu/test_gpu_cfar.py

# Benchmarks
python3 -m pytest tests/benchmarks/ -v
```

---

## Troubleshooting Index

### Common Issues

| Issue | Document | Section |
|-------|----------|---------|
| GPU not detected | [GPU User Guide](GPU_USER_GUIDE.md) | Troubleshooting § GPU Not Detected |
| Build errors (CUDA) | [GPU User Guide](GPU_USER_GUIDE.md) | Building § Build Troubleshooting |
| Poor GPU performance | [GPU User Guide](GPU_USER_GUIDE.md) | Troubleshooting § Poor GPU Performance |
| Out of memory | [GPU User Guide](GPU_USER_GUIDE.md) | Troubleshooting § Out of Memory |
| CPU fallback issues | [GPU Deployment](GPU_DEPLOYMENT.md) | Troubleshooting § Fallback to CPU |
| Production deployment | [GPU Deployment](GPU_DEPLOYMENT.md) | Production Checklist |

---

## Performance Guides

### Optimization Tips

**For Maximum Throughput:**
1. Use larger data sizes (4K+ samples, 512+ Doppler bins)
2. Enable GPU backend: `export KRAKEN_GPU_BACKEND=gpu`
3. Use MAXN power mode on Jetson
4. Monitor GPU utilization (should be >80%)

**For Power Efficiency:**
1. Use power-efficient mode on Jetson: `nvpmodel -m 2`
2. Reduce data sizes for lower GPU load
3. Consider CPU-only for low-power scenarios (RPi5)

**For Debugging:**
1. Force CPU backend: `export KRAKEN_GPU_BACKEND=cpu`
2. Enable CUDA debugging: `export CUDA_LAUNCH_BLOCKING=1`
3. Use memory checker: `cuda-memcheck python3 run_passive_radar.py`

See [GPU Performance](GPU_PERFORMANCE.md) for detailed optimization guides.

---

## API Quick Reference

### Python High-Level API

```python
from kraken_passive_radar import (
    is_gpu_available,          # GPU detection
    get_gpu_info,              # Device info
    set_processing_backend,    # Backend selection
    get_active_backend,        # Query backend
)

# Check GPU
if is_gpu_available():
    info = get_gpu_info()
    print(f"GPU: {info['name']}")

# Set backend
set_processing_backend('auto')  # auto/gpu/cpu

# Query
backend = get_active_backend()  # 'gpu' or 'cpu'
```

See [GPU API Reference](GPU_API_REFERENCE.md) for complete documentation.

---

## Release Notes

### Version 0.2.0 (2026-02-08)

**Major Features:**
- ✅ GPU acceleration for Doppler, CFAR, and CAF kernels
- ✅ 10-305x performance improvements on RTX 5090
- ✅ 100% backward compatibility with RPi5 CPU-only builds
- ✅ Runtime backend selection (auto/gpu/cpu)

**Validated Kernels:**
- ✅ Doppler GPU: Perfect correctness (1.0 correlation)
- ✅ CFAR GPU: 99.99% agreement, all targets detected

**In Progress:**
- ⚠️ CAF GPU: Performance excellent (23x), correctness debugging WIP

See [Changelog](../CHANGELOG.md) for full release notes.

---

## Support and Contributing

**Report Issues:**
- GitHub Issues: https://github.com/n4hy/PassiveRadar_Kraken/issues
- Tag GPU-related issues with `gpu` label

**Contact:**
- Author: Dr. Robert W McGwier, PhD (N4HY)
- GPU Implementation: Claude (Anthropic)

**License:**
- MIT License (see [LICENSE](../LICENSE))

---

## Document Versions

| Document | Version | Last Updated |
|----------|---------|--------------|
| README.md | 1.0 | 2026-02-08 |
| GPU_USER_GUIDE.md | 1.0 | 2026-02-08 |
| GPU_PERFORMANCE.md | 1.0 | 2026-02-08 |
| GPU_API_REFERENCE.md | 1.0 | 2026-02-08 |
| GPU_DEPLOYMENT.md | 1.0 | 2026-02-08 |
| CHANGELOG.md | 1.0 | 2026-02-08 |

---

**Documentation maintained by:** Claude (Anthropic)
**Project Author:** Dr. Robert W McGwier, PhD, N4HY
**Last Updated:** 2026-02-08
