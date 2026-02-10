# README.md Updates for Block B3

**Date:** 2026-02-10

## Summary of Changes

The README.md has been comprehensively updated to document Block B3 (Multi-Signal Reference Reconstructor).

---

## Sections Updated

### 1. Table of Contents
- ✅ Added new section: "Block B3: Reference Signal Reconstruction"

### 2. Test Results (Header)
- ✅ Updated date to 2026-02-10
- ✅ Added note about separate Block B3 test suite (5 tests passing)

### 3. NEW SECTION: Block B3: Reference Signal Reconstruction
**Added complete section covering:**
- Performance gains table (FM, ATSC 3.0, DVB-T)
- How it works (FM demodulation, OFDM processing)
- Quick start examples
- GNU Radio Companion flowgraph
- Documentation links

**Key Information:**
- FM Radio: 10-15 dB improvement, 8% CPU, production ready
- ATSC 3.0: 15-20 dB improvement, 49% CPU, OFDM complete
- DVB-T: Skeleton implementation

### 4. System Architecture
**Updated ASCII diagram to include:**
```
  Conditioning / AGC
        |
        v (Ch0 reference only)
+-------------------+
| Block B3          |  *** NEW: Reference Signal Reconstructor ***
| (C++/FFTW/Eigen3) |  FM/ATSC3/DVB-T demod-remod
| Multi-signal      |  10-20 dB SNR improvement
| FM: Audio filter  |  Enables weak signal detection
| OFDM: SVD pilots  |
+-------------------+
        |
        v
  ECA Canceller...
```

### 5. Project Structure
**Updated to show:**
- 8 C++ blocks (was 7)
- `dvbt_reconstructor_impl.cc/h` (750+ lines)
- `dvbt_reconstructor_python.cc` (Python bindings)
- `dvbt_reconstructor.block.yml` (GRC definition)
- New files: `passive_radar_block_b3.grc`, `test_block_b3.py`, `measure_b3_improvement.py`
- Block B3 documentation files (8 .md files)

### 6. Signal Processing Chain
**Updated processing chain:**
```
Source -> PhaseCorr -> AGC -> Block B3 (NEW!) -> ECA(C++) -> CAF -> ...
```

**Added Stage 3b to table:**
| Stage | Block | Language | Description |
|-------|-------|----------|-------------|
| **3b** | **`dvbt_reconstructor`** | **C++ (FFTW/Eigen3)** | **Reference signal reconstruction (10-20 dB improvement)** |

### 7. GNU Radio Blocks
**Updated count:**
- Changed from "14 blocks: 7 C++ and 7 Python"
- To "15 blocks: 8 C++ and 7 Python"

**Added to C++ blocks table:**
| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **`dvbt_reconstructor`** | **Multi-signal reference reconstructor (Block B3)** | **`signal_type`, `fm_deviation`, `fft_size`, `enable_svd`** |

### 8. Prerequisites
**Added Block B3 dependencies:**
```bash
# Block B3 dependencies (for reference signal reconstruction)
# gr-dtv: OFDM processing for ATSC 3.0/DVB-T
# gr-filter: FIR filter design for FM Radio
sudo apt install -y gnuradio-dtv gnuradio-filter
```

### 9. Running
**Added Block B3 examples:**
```bash
# With Block B3 reference reconstruction (FM Radio - 10-15 dB improvement)
python3 run_passive_radar.py --freq 100e6 --gain 30 --b3-signal fm --visualize

# With Block B3 (ATSC 3.0 - US urban areas, 15-20 dB improvement)
python3 run_passive_radar.py --freq 500e6 --gain 30 --b3-signal atsc3 --b3-fft-size 8192 --visualize
```

**Added Block B3 command-line options:**
- `--b3-signal`: Signal type (passthrough/fm/atsc3/dvbt)
- `--b3-fft-size`: OFDM FFT size
- `--b3-guard-interval`: Guard interval in samples

### 10. GNU Radio Companion
**Added new flowgraph:**
```bash
# Complete flowgraph with Block B3 reference reconstruction
gnuradio-companion passive_radar_block_b3.grc
```

### 11. API Reference
**Updated imports:**
```python
from gnuradio.kraken_passive_radar import (
    dvbt_reconstructor,  # Block B3 - reference reconstruction
    eca_canceller, doppler_processor, cfar_detector,
    ...
)
```

**Added complete Block B3 examples:**
- FM Radio mode configuration
- ATSC 3.0 mode configuration
- Passthrough mode
- Runtime controls (SNR estimation, signal type switching)

### 12. Acknowledgments
**Updated to include:**
- Block B3 reference reconstruction system
- Multi-signal demodulation-remodulation (FM/ATSC3/DVB-T)
- Performance metrics (10-20 dB, 8-49% CPU)
- GRC flowgraph and documentation

### 13. Last Updated Date
- Changed from 2026-02-09 to 2026-02-10

---

## Statistics

**Lines Added:** ~150 lines
**Sections Modified:** 13 sections
**New Section:** 1 major section (Block B3)
**Code Examples Added:** 6 examples
**Tables Added/Updated:** 2 tables

---

## Documentation Cross-References

README now links to these Block B3 documents:
- `BLOCK_B3_READY_TO_USE.md` - Quick start guide
- `BLOCK_B3_GRC_GUIDE.md` - GRC usage guide
- `BLOCK_B3_COMPLETE_PACKAGE.md` - Full package summary
- `ATSC3_OFDM_COMPLETE.md` - Technical details

---

## Key Messages Communicated

1. **Performance Improvement**: 10-20 dB SNR gain clearly stated
2. **Easy to Use**: Command-line examples provided
3. **Multiple Modes**: FM (production), ATSC 3.0 (advanced), DVB-T (future)
4. **GRC Integration**: Complete flowgraph available
5. **Well Documented**: 8 documentation files referenced
6. **Production Ready**: FM mode fully validated, ATSC 3.0 OFDM complete

---

## User Impact

Users reading the README will now:
- ✅ Learn about Block B3 immediately (new section in TOC)
- ✅ See it in the system architecture diagram
- ✅ Find command-line examples for FM and ATSC 3.0
- ✅ Know which signal type to use (FM recommended)
- ✅ Have links to comprehensive documentation
- ✅ See Block B3 in the API examples
- ✅ Understand the performance benefits (10-20 dB)

---

## Completeness Check

- [x] Table of Contents updated
- [x] New section added
- [x] Architecture diagram updated
- [x] Signal chain updated
- [x] Block list updated
- [x] Prerequisites updated
- [x] Running examples updated
- [x] API reference updated
- [x] GRC section updated
- [x] Acknowledgments updated
- [x] Date updated
- [x] Project structure updated

**README.md is now fully up-to-date with Block B3! ✅**
