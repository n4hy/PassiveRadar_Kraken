# Production Deployment Guide
## PassiveRadar_Kraken v0.2.0 GPU

**Date:** 2026-02-08
**Status:** Production-Ready
**Target:** RTX GPUs, Jetson, Cloud

---

## Quick Start

### Docker Deployment (Recommended)

```bash
# Build GPU-enabled image
docker build -f Dockerfile.gpu -t passiveradar-kraken:gpu-latest .

# Run with GPU support
docker run --gpus all \
    -e KRAKEN_GPU_BACKEND=auto \
    passiveradar-kraken:gpu-latest

# Expected output:
# GPU Available: True
# GPU: NVIDIA GeForce RTX 5090
```

### Docker Compose

```bash
# Start GPU service
docker-compose -f docker-compose.gpu.yml up -d

# Check logs
docker-compose -f docker-compose.gpu.yml logs -f

# Stop service
docker-compose -f docker-compose.gpu.yml down
```

---

## Pre-Deployment Checklist

### Hardware Validation

- [ ] **GPU Detection**
  ```bash
  nvidia-smi
  # Should show your GPU (RTX 2000+, Jetson Orin, etc.)
  ```

- [ ] **CUDA Installation**
  ```bash
  nvcc --version
  # Should show CUDA 12.0 or later
  ```

- [ ] **Driver Version**
  ```bash
  nvidia-smi | grep "Driver Version"
  # Should be 545+ for RTX 5090, 535+ for older GPUs
  ```

### Software Validation

- [ ] **Build Test**
  ```bash
  cd src/build
  cmake .. -DENABLE_GPU=ON
  make -j$(nproc)
  # Should build without errors
  ```

- [ ] **GPU Test**
  ```bash
  python3 -c "from kraken_passive_radar import is_gpu_available; assert is_gpu_available()"
  # Should pass without assertion error
  ```

- [ ] **Pytest Suite**
  ```bash
  python3 -m pytest tests/gpu/ -v
  # All 6 CAF tests should pass
  ```

### Performance Validation

- [ ] **Integration Test**
  ```bash
  python3 test_gpu_integration.py
  # Should show:
  #   CAF: 2-10 ms (depends on n_samples)
  #   End-to-end: 60-150 Hz
  ```

- [ ] **Benchmark**
  ```bash
  python3 tests/gpu/test_gpu_caf.py::TestGPUCAFPerformance::test_gpu_throughput -v
  # Should complete in <100 ms
  ```

---

## Deployment Scenarios

### Scenario 1: Development Workstation

**Hardware:** Desktop + RTX GPU
**OS:** Ubuntu 24.04 LTS
**Use Case:** Development, testing, profiling

**Deployment:**
```bash
# Native installation (no Docker)
cd /home/n4hy/PassiveRadar_Kraken

# Build
cd src/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON
make -j$(nproc)
sudo make install

# Verify
python3 -c "from kraken_passive_radar import is_gpu_available; print('GPU:', is_gpu_available())"

# Run
export KRAKEN_GPU_BACKEND=gpu
python3 run_passive_radar.py --freq 103.7e6 --gain 30
```

**Monitoring:**
```bash
# Terminal 1: Run application
python3 run_passive_radar.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

### Scenario 2: Docker Container (Desktop)

**Hardware:** Desktop + RTX GPU
**OS:** Ubuntu with Docker + NVIDIA Container Toolkit
**Use Case:** Isolated deployment, easy updates

**Prerequisites:**
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Deployment:**
```bash
# Build image
docker build -f Dockerfile.gpu -t passiveradar:gpu .

# Run container
docker run --gpus all \
    --name passiveradar \
    -v $(pwd)/data:/data \
    -e KRAKEN_GPU_BACKEND=auto \
    passiveradar:gpu

# Check logs
docker logs -f passiveradar

# Stop
docker stop passiveradar
docker rm passiveradar
```

### Scenario 3: NVIDIA Jetson Orin

**Hardware:** Jetson Orin NX/Nano
**OS:** JetPack 6.0 (Ubuntu 22.04)
**Use Case:** Embedded deployment, field operation

**Deployment:**
```bash
# CUDA is pre-installed on JetPack
nvcc --version  # Verify CUDA 12.x

# Clone repo
git clone https://github.com/n4hy/PassiveRadar_Kraken
cd PassiveRadar_Kraken

# Build (Jetson is sm_87)
cd src/build
cmake .. -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# Set power mode (for performance)
sudo nvpmodel -m 0  # MAXN mode
sudo jetson_clocks

# Run
export KRAKEN_GPU_BACKEND=gpu
python3 run_passive_radar.py --freq 103.7e6 --gain 30
```

**Power Modes:**
```bash
# Show available modes
sudo nvpmodel -q

# MAXN (maximum performance, ~30W)
sudo nvpmodel -m 0

# Balanced (10-15W)
sudo nvpmodel -m 2

# Power efficient (5-7W)
sudo nvpmodel -m 4
```

**Monitoring:**
```bash
# GPU stats on Jetson
tegrastats

# Alternative
jtop  # Install: sudo -H pip install jetson-stats
```

### Scenario 4: Cloud GPU (AWS/Azure)

**Instance:** AWS p3.2xlarge (Tesla V100) or p4d (A100)
**OS:** Deep Learning AMI (Ubuntu)
**Use Case:** Elastic scaling, multi-channel processing

**AWS Deployment:**
```bash
# Launch instance (AWS Console or CLI)
aws ec2 run-instances \
    --image-id ami-0d73f667f8b8b8e5c \  # Deep Learning AMI
    --instance-type p3.2xlarge \
    --key-name your-key \
    --security-groups ssh-access

# Connect
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone and build
git clone https://github.com/n4hy/PassiveRadar_Kraken
cd PassiveRadar_Kraken/src/build

# Note: V100 is sm_70, need to add to CMakeLists.txt
# Edit src/gpu/CMakeLists.txt: set(CMAKE_CUDA_ARCHITECTURES "70;75;86;87;89")
cmake .. -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run headless (no display)
export KRAKEN_GPU_BACKEND=gpu
python3 run_passive_radar.py --freq 103.7e6 --gain 30 --no-gui
```

**Cost Optimization:**
```bash
# Use Spot Instances (70% cost savings)
aws ec2 request-spot-instances \
    --spot-price "1.00" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification file://spot-spec.json

# Auto-shutdown when idle (save costs)
echo "sudo shutdown -h +60" | at now  # Shutdown in 60 min if no user activity
```

---

## Production Configuration

### Environment Variables

```bash
# GPU backend selection
export KRAKEN_GPU_BACKEND=auto  # Auto-detect (default)
# export KRAKEN_GPU_BACKEND=gpu   # Require GPU
# export KRAKEN_GPU_BACKEND=cpu   # Force CPU

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0    # Use first GPU
export CUDA_CACHE_PATH=/tmp/cuda_cache  # Kernel cache location

# Performance
export CUDA_LAUNCH_BLOCKING=0    # Async kernel launches (production)
# export CUDA_LAUNCH_BLOCKING=1  # Sync launches (debugging)
```

### Systemd Service (Auto-start on boot)

Create `/etc/systemd/system/passiveradar.service`:

```ini
[Unit]
Description=PassiveRadar_Kraken GPU Service
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=radar
WorkingDirectory=/opt/PassiveRadar_Kraken
Environment="KRAKEN_GPU_BACKEND=auto"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/usr/bin/python3 run_passive_radar.py --freq 103.7e6 --gain 30
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable passiveradar
sudo systemctl start passiveradar
sudo systemctl status passiveradar
```

### Logging

**Application logs:**
```bash
# Redirect stdout/stderr
python3 run_passive_radar.py 2>&1 | tee /var/log/passiveradar/app.log

# Rotate logs
sudo apt install logrotate
```

**GPU monitoring logs:**
```bash
# Log GPU stats every 10 seconds
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used \
           --format=csv -l 10 > /var/log/passiveradar/gpu.csv &
```

---

## Monitoring and Alerting

### Health Checks

```python
# /opt/passiveradar/healthcheck.py
from kraken_passive_radar import is_gpu_available, get_gpu_info
import sys

if not is_gpu_available():
    print("ERROR: GPU not available")
    sys.exit(1)

info = get_gpu_info()
print(f"OK: GPU {info['name']} available")
sys.exit(0)
```

**Run periodically:**
```bash
# Cron job (every 5 minutes)
*/5 * * * * /usr/bin/python3 /opt/passiveradar/healthcheck.py || echo "GPU ERROR" | mail -s "PassiveRadar Alert" admin@example.com
```

### Performance Monitoring

```bash
# Monitor end-to-end latency
python3 test_gpu_integration.py > /tmp/perf_check.txt

# Parse results
grep "Total:" /tmp/perf_check.txt
# Should show: Total: 15-20 ms (50-65 Hz) for 16K samples
```

### Alerting

**GPU temperature alert:**
```bash
#!/bin/bash
# /opt/passiveradar/gpu_temp_alert.sh
TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
if [ $TEMP -gt 85 ]; then
    echo "GPU temperature high: ${TEMP}°C" | mail -s "GPU ALERT" admin@example.com
fi
```

**GPU error alert:**
```bash
# Check for CUDA errors in logs
grep -i "cuda error\|gpu error" /var/log/passiveradar/app.log && \
    echo "CUDA error detected" | mail -s "PassiveRadar ERROR" admin@example.com
```

---

## Backup and Recovery

### Configuration Backup

```bash
# Backup critical config
tar -czf passiveradar-config-$(date +%Y%m%d).tar.gz \
    /opt/PassiveRadar_Kraken/config/ \
    /etc/systemd/system/passiveradar.service \
    /opt/passiveradar/healthcheck.py

# Upload to S3 (optional)
aws s3 cp passiveradar-config-*.tar.gz s3://my-backup-bucket/
```

### Disaster Recovery

**Scenario: GPU failure**
```bash
# Automatic fallback to CPU
export KRAKEN_GPU_BACKEND=cpu
systemctl restart passiveradar

# Performance degradation expected (10 Hz vs 100 Hz)
# Replace GPU hardware, then:
export KRAKEN_GPU_BACKEND=auto
systemctl restart passiveradar
```

**Scenario: Library corruption**
```bash
# Rebuild from source
cd /opt/PassiveRadar_Kraken/src/build
make clean
cmake .. -DENABLE_GPU=ON
make -j$(nproc)
sudo make install
systemctl restart passiveradar
```

---

## Security Considerations

### Container Security

```bash
# Run as non-root user
docker run --gpus all \
    --user 1000:1000 \
    passiveradar:gpu

# Limit resources
docker run --gpus all \
    --memory="4g" \
    --cpus="4" \
    passiveradar:gpu

# Read-only filesystem (except /tmp, /data)
docker run --gpus all \
    --read-only \
    --tmpfs /tmp \
    -v /data:/data:rw \
    passiveradar:gpu
```

### Network Isolation

```bash
# No network access (for standalone operation)
docker run --gpus all \
    --network=none \
    passiveradar:gpu

# Custom network
docker network create --driver bridge radar-net
docker run --gpus all \
    --network=radar-net \
    passiveradar:gpu
```

---

## Troubleshooting

See [GPU_USER_GUIDE.md](GPU_USER_GUIDE.md#troubleshooting) for detailed troubleshooting steps.

**Common production issues:**

| Issue | Cause | Solution |
|-------|-------|----------|
| Container can't access GPU | Missing nvidia-container-toolkit | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |
| Out of memory | Too large n_samples | Reduce CPI size or upgrade GPU memory |
| Thermal throttling | Poor cooling | Check fans, improve airflow, reduce power limit |
| Low throughput | CPU backend used | Check `KRAKEN_GPU_BACKEND` env var |

---

## Support

**Documentation:**
- [GPU User Guide](GPU_USER_GUIDE.md)
- [GPU Performance](GPU_PERFORMANCE.md)
- [GPU Profiling](GPU_PROFILING.md)

**Contact:**
- Author: Dr. Robert W McGwier, PhD (N4HY)
- Issues: https://github.com/n4hy/PassiveRadar_Kraken/issues

---

**Document Version:** 1.0
**Last Updated:** 2026-02-08
**Production Status:** ✅ Ready for deployment
