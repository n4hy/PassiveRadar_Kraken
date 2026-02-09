# GPU Deployment Guide
## PassiveRadar_Kraken CUDA Deployment

**Version:** 0.2.0
**Date:** 2026-02-08

---

## Table of Contents

- [Overview](#overview)
- [Deployment Scenarios](#deployment-scenarios)
- [Platform-Specific Guides](#platform-specific-guides)
- [Docker Deployment](#docker-deployment)
- [Production Checklist](#production-checklist)
- [Monitoring and Diagnostics](#monitoring-and-diagnostics)
- [Troubleshooting](#troubleshooting)

---

## Overview

PassiveRadar_Kraken GPU acceleration supports multiple deployment scenarios from entry-level hobbyist (RPi5) to professional/commercial (RTX GPUs) to cloud/enterprise (multi-GPU servers).

### Deployment Tiers

| Tier | Platform | Performance | Cost | Target Users |
|------|----------|-------------|------|--------------|
| **Entry** | Raspberry Pi 5 | ~10 Hz | $80 | Hobbyist, research, education |
| **Professional** | Desktop + RTX GPU | 100-200 Hz | $1500-3500 | Professional, commercial R&D |
| **Embedded** | NVIDIA Jetson Orin | 80-150 Hz | $500-900 | Field deployment, mobile |
| **Cloud** | AWS/Azure GPU instances | 200+ Hz | $3-8/hour | Enterprise, elastic scaling |

---

## Deployment Scenarios

### Scenario 1: Raspberry Pi 5 (CPU-Only)

**Hardware:**
- Raspberry Pi 5 (4GB or 8GB)
- KrakenSDR (~$400)
- microSD card (64GB+)
- Power supply (5V/5A)

**Software:**
- Raspberry Pi OS 64-bit (Debian Bookworm)
- No CUDA required

**Performance:**
- Update rate: ~10 Hz
- Range: 15-20 km
- Resolution: 300-600 m

**Deployment Steps:**

1. **Install OS:**
   ```bash
   # Download Raspberry Pi OS 64-bit
   # Flash to microSD using Raspberry Pi Imager
   ```

2. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install -y \
       build-essential cmake pkg-config \
       gnuradio gnuradio-dev \
       libfftw3-dev libvolk2-dev pybind11-dev \
       python3-dev python3-numpy python3-pytest

   pip3 install matplotlib
   ```

3. **Build (CPU-Only):**
   ```bash
   cd /home/n4hy/PassiveRadar_Kraken/src
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j4
   ```

4. **Run:**
   ```bash
   export KRAKEN_GPU_BACKEND=cpu  # Explicit CPU-only
   python3 run_passive_radar.py --freq 103.7e6 --gain 30 --visualize
   ```

**Validation:**
- All 191 tests should pass
- No GPU libraries should be built
- Performance identical to v0.1.0

---

### Scenario 2: Desktop with RTX GPU

**Hardware:**
- Desktop PC (Intel/AMD CPU)
- NVIDIA RTX 2060 or better
- 16GB+ RAM
- Ubuntu 24.04 LTS

**Software:**
- CUDA Toolkit 12.0+
- NVIDIA drivers 545+

**Performance:**
- Update rate: 100-200 Hz
- Range: 15-20 km
- Resolution: 75-150 m (4x better due to higher update rate)

**Deployment Steps:**

1. **Install NVIDIA Drivers:**
   ```bash
   sudo ubuntu-drivers autoinstall
   sudo reboot

   # Verify
   nvidia-smi
   ```

2. **Install CUDA Toolkit:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install cuda-toolkit-12-6

   # Add to PATH
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc

   # Verify
   nvcc --version
   ```

3. **Install Dependencies:**
   ```bash
   sudo apt install -y \
       build-essential cmake pkg-config \
       gnuradio gnuradio-dev \
       libfftw3-dev libvolk2-dev pybind11-dev \
       python3-dev python3-numpy python3-pytest
   ```

4. **Build (GPU-Enabled):**
   ```bash
   cd /home/n4hy/PassiveRadar_Kraken/src
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON
   make -j$(nproc)

   # Verify GPU libraries
   ls -lh ../libkraken_*gpu*.so
   ```

5. **Test GPU:**
   ```bash
   python3 -c "
   from kraken_passive_radar import is_gpu_available, get_gpu_info
   if is_gpu_available():
       info = get_gpu_info()
       print(f'GPU detected: {info[\"name\"]}')
   else:
       print('ERROR: GPU not detected')
   "
   ```

6. **Run:**
   ```bash
   export KRAKEN_GPU_BACKEND=gpu  # Require GPU
   python3 run_passive_radar.py --freq 103.7e6 --gain 30 --visualize
   ```

**Production Settings:**

```bash
# Persistent GPU configuration
cat >> ~/.bashrc << 'EOF'
# PassiveRadar_Kraken GPU
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export KRAKEN_GPU_BACKEND=auto
EOF
```

---

### Scenario 3: NVIDIA Jetson Orin

**Hardware:**
- Jetson Orin NX (16GB) or Orin Nano (8GB)
- Carrier board
- KrakenSDR
- Power supply (5V/4A or 9-20V/3A depending on carrier)

**Software:**
- JetPack SDK 5.x or 6.x
- CUDA pre-installed

**Performance:**
- Update rate: 80-150 Hz (estimated)
- Power: 10-30W
- Compact, fanless deployment

**Deployment Steps:**

1. **Flash JetPack:**
   - Use NVIDIA SDK Manager to flash JetPack 6.0
   - CUDA and drivers included automatically

2. **Verify CUDA:**
   ```bash
   nvcc --version
   # Should show: release 11.4+ (JetPack 5.x) or 12.x (JetPack 6.x)

   nvidia-smi  # Not available on Jetson, use:
   sudo jetson_clocks --show
   ```

3. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install -y \
       build-essential cmake pkg-config \
       libfftw3-dev libvolk2-dev pybind11-dev \
       python3-dev python3-numpy python3-pytest

   # Install GNU Radio (may need to build from source)
   sudo apt install gnuradio gnuradio-dev
   ```

4. **Build (GPU-Enabled):**
   ```bash
   cd /home/n4hy/PassiveRadar_Kraken/src
   mkdir build && cd build

   # Jetson Orin is sm_87
   cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON
   make -j$(nproc)
   ```

5. **Power Mode:**
   ```bash
   # Max performance mode
   sudo nvpmodel -m 0
   sudo jetson_clocks

   # Or power-efficient mode
   sudo nvpmodel -m 2
   ```

6. **Run:**
   ```bash
   export KRAKEN_GPU_BACKEND=gpu
   python3 run_passive_radar.py --freq 103.7e6 --gain 30
   # Note: May run headless without --visualize
   ```

**Embedded Optimizations:**
- Disable GUI for lower power
- Use MAXN power mode for performance
- Monitor temperature: `tegrastats`

---

### Scenario 4: Cloud GPU (AWS/Azure)

**AWS p3.2xlarge Specifications:**
- GPU: 1× NVIDIA Tesla V100 (16GB)
- vCPU: 8
- RAM: 61 GB
- Cost: ~$3/hour (on-demand)

**Deployment Steps:**

1. **Launch Instance:**
   ```bash
   # AWS EC2 Console
   # AMI: Deep Learning AMI (Ubuntu 22.04)
   # Instance type: p3.2xlarge
   # Security: Open SSH (22)
   ```

2. **Connect:**
   ```bash
   ssh -i your-key.pem ubuntu@<instance-ip>
   ```

3. **Verify CUDA:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

4. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install -y \
       build-essential cmake pkg-config \
       gnuradio gnuradio-dev \
       libfftw3-dev libvolk2-dev pybind11-dev \
       python3-dev python3-numpy python3-pytest
   ```

5. **Clone and Build:**
   ```bash
   git clone https://github.com/n4hy/PassiveRadar_Kraken
   cd PassiveRadar_Kraken/src
   mkdir build && cd build

   # V100 is sm_70 - need to add to CMakeLists.txt
   # Edit src/gpu/CMakeLists.txt: add "70" to CMAKE_CUDA_ARCHITECTURES

   cmake .. -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

6. **Run Headless:**
   ```bash
   export KRAKEN_GPU_BACKEND=gpu
   # No --visualize flag (headless)
   python3 run_passive_radar.py --freq 103.7e6 --gain 30
   ```

**Cloud Optimizations:**
- Use Spot Instances for 70% cost savings (acceptable for non-critical workloads)
- Auto-scaling for variable load
- S3 for data storage/retrieval
- CloudWatch for monitoring

---

## Docker Deployment

### Dockerfile (GPU-Enabled)

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config git \
    gnuradio gnuradio-dev \
    libfftw3-dev libvolk2-dev pybind11-dev \
    python3-dev python3-numpy python3-pytest \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
WORKDIR /opt
RUN git clone https://github.com/n4hy/PassiveRadar_Kraken
WORKDIR /opt/PassiveRadar_Kraken

# Build with GPU support
WORKDIR /opt/PassiveRadar_Kraken/src
RUN mkdir build && cd build && \
    cmake .. -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Set environment
ENV KRAKEN_GPU_BACKEND=gpu
WORKDIR /opt/PassiveRadar_Kraken

# Entry point
CMD ["python3", "run_passive_radar.py", "--freq", "103.7e6", "--gain", "30"]
```

### Build and Run

```bash
# Build image
docker build -t passiveradar-kraken-gpu .

# Run with GPU support
docker run --gpus all \
    -e KRAKEN_GPU_BACKEND=gpu \
    -v /dev:/dev \
    passiveradar-kraken-gpu

# Check GPU inside container
docker run --gpus all passiveradar-kraken-gpu nvidia-smi
```

---

## Production Checklist

### Pre-Deployment

- [ ] GPU detected: `is_gpu_available()` returns `True`
- [ ] GPU libraries built: `ls src/libkraken_*gpu*.so` shows 4 files
- [ ] Backend set: `KRAKEN_GPU_BACKEND=gpu` or `auto`
- [ ] Tests pass: `python3 tests/gpu/test_gpu_doppler.py`
- [ ] Doppler correctness: correlation = 1.0
- [ ] CFAR correctness: agreement > 99%

### Production Settings

**Environment:**
```bash
export KRAKEN_GPU_BACKEND=auto     # Auto-detect with fallback
export CUDA_VISIBLE_DEVICES=0       # Use first GPU (if multiple)
export CUDA_CACHE_PATH=/tmp/cuda_cache  # Kernel cache
```

**Monitoring:**
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Expected during processing:
# - GPU utilization: 80-100%
# - Memory used: 200-500 MB (depends on config)
# - Temperature: < 85°C
# - Power: 50-300W (depends on workload)
```

### Post-Deployment Validation

```bash
# Run performance tests
python3 tests/gpu/test_gpu_doppler.py
python3 tests/gpu/test_gpu_cfar.py

# Check for memory leaks (run for 10 minutes)
python3 run_passive_radar.py --freq 103.7e6 --gain 30
# Monitor nvidia-smi - memory should stabilize, not grow

# Verify graceful CPU fallback
export CUDA_VISIBLE_DEVICES=-1  # Hide GPU
python3 run_passive_radar.py --freq 103.7e6 --gain 30
# Should run on CPU without errors
```

---

## Monitoring and Diagnostics

### Real-Time GPU Monitoring

```bash
# Basic monitoring
nvidia-smi

# Continuous monitoring (1 sec updates)
watch -n 1 nvidia-smi

# Detailed stats
nvidia-smi dmon -s puc -c 100  # 100 samples, power/util/clock

# Log to file
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used \
           --format=csv -l 1 > gpu_monitor.csv
```

### Performance Profiling

```bash
# Profile Doppler kernel
nsys profile -o doppler_profile python3 tests/gpu/test_gpu_doppler.py

# View in Nsight Systems
nsys-ui doppler_profile.nsys-rep

# Kernel-level profiling
ncu --set full -o doppler_kernel python3 tests/gpu/test_gpu_doppler.py

# View in Nsight Compute
ncu-ui doppler_kernel.ncu-rep
```

### Logging

**Enable CUDA debugging:**
```bash
export CUDA_LAUNCH_BLOCKING=1  # Synchronous kernel launches
export CUDA_LOG_LEVEL=warn     # CUDA logging
```

**Check for memory errors:**
```bash
cuda-memcheck python3 run_passive_radar.py --freq 103.7e6 --gain 30
```

---

## Troubleshooting

### GPU Not Detected in Production

**Check:**
```bash
# 1. Verify GPU visible
nvidia-smi

# 2. Check CUDA libraries
ldd /home/n4hy/PassiveRadar_Kraken/src/libkraken_gpu_runtime.so

# 3. Test GPU detection
python3 -c "
from kraken_passive_radar import is_gpu_available
print('GPU available:', is_gpu_available())
"

# 4. Check environment
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

**Solution:**
```bash
# Ensure CUDA in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Rebuild
cd src/build
cmake .. -DENABLE_GPU=ON
make clean && make -j$(nproc)
```

---

### Out of Memory

**Symptoms:**
- Process crashes with CUDA out-of-memory
- nvidia-smi shows 100% memory usage

**Solutions:**

1. **Reduce data size:**
   ```python
   # Reduce range/Doppler bins
   num_range_bins = 2048  # Instead of 8192
   num_doppler_bins = 256  # Instead of 1024
   ```

2. **Free GPU memory:**
   ```bash
   # Kill other GPU processes
   nvidia-smi
   # Find PID, then:
   kill <pid>
   ```

3. **Use smaller GPU:**
   ```bash
   # If multiple GPUs, use smaller one for PassiveRadar
   export CUDA_VISIBLE_DEVICES=1
   ```

---

### Performance Degradation

**Symptoms:**
- GPU slower than expected
- GPU utilization < 50%

**Diagnose:**

```bash
# Check thermal throttling
nvidia-smi -q | grep -i temp
nvidia-smi -q | grep -i throttle

# Check power limit
nvidia-smi -q | grep -i power

# Check PCIe link
nvidia-smi -q | grep -i "link width"
# Should be: x16 (not x1, x4, x8)
```

**Solutions:**

1. **Thermal:** Improve cooling, clean fans
2. **Power:** Increase power limit (if safe)
   ```bash
   sudo nvidia-smi -pl 350  # Set 350W limit (check GPU spec)
   ```
3. **PCIe:** Move GPU to x16 slot

---

### Fallback to CPU

**Verify CPU fallback works:**

```bash
# Hide GPU temporarily
export CUDA_VISIBLE_DEVICES=-1

# Should run on CPU without errors
python3 run_passive_radar.py --freq 103.7e6 --gain 30

# Check backend
python3 -c "
from kraken_passive_radar import get_active_backend
print('Backend:', get_active_backend())
"
# Should print: Backend: cpu
```

---

## Support

**Documentation:**
- [GPU User Guide](GPU_USER_GUIDE.md)
- [GPU Performance Benchmarks](GPU_PERFORMANCE.md)
- [GPU API Reference](GPU_API_REFERENCE.md)

**Issues:**
- GitHub: https://github.com/n4hy/PassiveRadar_Kraken/issues
- Tag with `gpu` label

**Commercial Support:**
- Contact: N4HY (Dr. Robert W McGwier, PhD)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-08
**Author:** Dr. Robert W McGwier, PhD, N4HY
**GPU Implementation:** Claude (Anthropic)
