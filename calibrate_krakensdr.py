#!/usr/bin/env python3
import sys
import time
import numpy as np
import argparse
import ctypes
import os
import json
from pathlib import Path

# Add python path
sys.path.append(os.path.join(os.path.dirname(__file__), "gr-kraken_passive_radar/python"))

from kraken_passive_radar.krakensdr_source import krakensdr_source
from kraken_passive_radar.custom_blocks import TimeAlignmentBlock

def calibrate(args):
    print("Starting KrakenSDR Calibration...")
    print("Ensure antennas are disconnected or noise source is dominant.")

    # 1. Initialize Source
    # We need high gain or just enough to see noise source.
    # Noise source is usually strong.
    source = krakensdr_source(
        center_freq=args.freq,
        sample_rate=2.4e6,
        gain=args.gain,
        num_channels=5
    )

    # Turn on Noise Source
    print("Enabling Noise Source...")
    source.set_noise_source(True)
    time.sleep(1.0) # Settle

    # 2. Capture Data
    # We can't easily run a blocking flowgraph and return data without a Sink.
    # krakensdr_source is a HierBlock.
    # We will instantiate a top block just for capture.

    from gnuradio import gr, blocks

    tb = gr.top_block()

    # Capture 1 second
    nsamps = int(2.4e6)

    # Vector Sinks for 5 channels
    sinks = []
    for i in range(5):
        snk = blocks.vector_sink_c()
        sinks.append(snk)
        tb.connect((source, i), snk)

    # Run
    print("Capturing calibration frames...")
    tb.start()
    time.sleep(2.0)
    tb.stop()
    tb.wait()

    # Disable Noise Source
    source.set_noise_source(False)

    # 3. Process
    print("Processing Phase Offsets...")

    # Load Alignment Lib manually to use static compute
    # Or use TimeAlignmentBlock wrapper logic if we refactor it?
    # TimeAlignmentBlock wrapper is a GR block.
    # We can use the C library directly.

    # Find Lib
    repo_root = Path(__file__).resolve().parent
    lib_path = repo_root / "src" / "libkraken_time_alignment.so"
    if not lib_path.exists():
        # Try python install path
        import sysconfig
        lib_path = Path(sysconfig.get_paths()["purelib"]) / "kraken_passive_radar" / "libkraken_time_alignment.so"

    if not lib_path.exists():
        print("Error: libkraken_time_alignment.so not found. Build OOT first.")
        sys.exit(1)

    lib = ctypes.cdll.LoadLibrary(str(lib_path))
    lib.align_create.restype = ctypes.c_void_p
    lib.align_create.argtypes = [ctypes.c_int]
    lib.align_destroy.restype = None
    lib.align_compute.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]

    align_len = 4096
    obj = lib.align_create(align_len)

    # Get Data
    # Ch0 is Reference
    data = [np.array(snk.data(), dtype=np.complex64) for snk in sinks]

    # Calibrate Ch1-4 relative to Ch0
    # Also check Ch0 relative to Ch0 (should be 0)

    offsets = {}

    # Average over multiple chunks
    n_chunks = len(data[0]) // align_len
    if n_chunks > 100: n_chunks = 100

    print(f"Averaging over {n_chunks} blocks...")

    for ch in range(5):
        total_phase = 0.0
        # We need vector averaging for phase!
        # sum(exp(j*phase))
        phasor_sum = 0.0 + 0j

        for i in range(n_chunks):
            start = i * align_len
            end = start + align_len

            ref_ptr = data[0][start:end].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            surv_ptr = data[ch][start:end].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            delay = ctypes.c_int(0)
            phase = ctypes.c_float(0.0)

            lib.align_compute(obj, ref_ptr, surv_ptr, ctypes.byref(delay), ctypes.byref(phase))

            # We care about Phase. Delay should be constant (hardware buffer align).
            # But osmosdr sometimes has integer sample slips.
            # Ideally delay is 0 if synchronized.

            phasor_sum += np.exp(1j * phase.value)

        avg_phase = np.angle(phasor_sum)
        offsets[f"ch{ch}"] = float(avg_phase)
        print(f"Channel {ch}: Phase Offset = {np.degrees(avg_phase):.2f} deg")

    lib.align_destroy(obj)

    # Save
    out_file = "calibration.json"
    with open(out_file, "w") as f:
        json.dump(offsets, f, indent=4)

    print(f"Calibration saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KrakenSDR Calibration Tool")
    parser.add_argument("--freq", type=float, default=100e6, help="Frequency (Hz)")
    parser.add_argument("--gain", type=float, default=30.0, help="Gain (dB)")
    args = parser.parse_args()

    calibrate(args)
