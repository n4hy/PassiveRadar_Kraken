#!/usr/bin/env python3
"""
Block B3 CAF Improvement Measurement Script

Measures the Cross-Ambiguity Function (CAF) peak improvement when using Block B3
reference signal reconstruction compared to no reconstruction.

Usage:
    # Run with Block B3 enabled (FM mode)
    python3 run_passive_radar.py --freq 100e6 --b3-signal fm &
    sleep 30
    python3 measure_b3_improvement.py --duration 60 --output results_fm.json

    # Run without Block B3 (baseline)
    python3 run_passive_radar.py --freq 100e6 --b3-signal passthrough &
    sleep 30
    python3 measure_b3_improvement.py --duration 60 --output results_baseline.json

    # Compare results
    python3 measure_b3_improvement.py --compare results_baseline.json results_fm.json
"""

import sys
import time
import argparse
import json
import numpy as np
from datetime import datetime

def measure_caf_performance(duration_sec, sample_interval=1.0):
    """
    Measure CAF performance metrics over time.

    Returns:
        dict: Performance metrics including peak SNR, average noise floor, etc.
    """
    print(f"Measuring CAF performance for {duration_sec} seconds...")
    print("Note: This is a placeholder - integrate with actual CAF output")

    measurements = {
        'timestamp': datetime.now().isoformat(),
        'duration_sec': duration_sec,
        'sample_interval_sec': sample_interval,
        'samples': [],
        'statistics': {}
    }

    num_samples = int(duration_sec / sample_interval)

    for i in range(num_samples):
        # TODO: Replace with actual CAF data reading
        # For now, simulate measurements

        # In real implementation, read from:
        # - CAF output file/pipe
        # - Shared memory
        # - GNU Radio message port

        sample = {
            'time': i * sample_interval,
            'peak_snr_db': 0.0,        # Replace with actual CAF peak SNR
            'noise_floor_db': 0.0,     # Replace with actual noise floor
            'peak_location': (0, 0),   # (range_bin, doppler_bin)
            'num_detections': 0        # Number of targets detected
        }

        measurements['samples'].append(sample)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_samples} samples collected")

        time.sleep(sample_interval)

    # Calculate statistics
    peak_snrs = [s['peak_snr_db'] for s in measurements['samples']]
    noise_floors = [s['noise_floor_db'] for s in measurements['samples']]

    measurements['statistics'] = {
        'peak_snr_mean': np.mean(peak_snrs),
        'peak_snr_std': np.std(peak_snrs),
        'peak_snr_max': np.max(peak_snrs),
        'peak_snr_min': np.min(peak_snrs),
        'noise_floor_mean': np.mean(noise_floors),
        'noise_floor_std': np.std(noise_floors),
    }

    return measurements

def compare_results(baseline_file, test_file):
    """
    Compare baseline vs Block B3 enabled results.

    Args:
        baseline_file: JSON file with baseline measurements (passthrough)
        test_file: JSON file with Block B3 enabled measurements (fm/atsc3)
    """
    print("\n" + "="*60)
    print("BLOCK B3 CAF IMPROVEMENT ANALYSIS")
    print("="*60 + "\n")

    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    with open(test_file, 'r') as f:
        test = json.load(f)

    base_stats = baseline['statistics']
    test_stats = test['statistics']

    # Calculate improvement
    snr_improvement = test_stats['peak_snr_mean'] - base_stats['peak_snr_mean']
    noise_reduction = base_stats['noise_floor_mean'] - test_stats['noise_floor_mean']

    print("Baseline (No Block B3):")
    print(f"  Peak SNR:     {base_stats['peak_snr_mean']:.2f} ± {base_stats['peak_snr_std']:.2f} dB")
    print(f"  Noise Floor:  {base_stats['noise_floor_mean']:.2f} ± {base_stats['noise_floor_std']:.2f} dB")
    print(f"  Duration:     {baseline['duration_sec']} seconds")
    print(f"  Samples:      {len(baseline['samples'])}")

    print("\nWith Block B3 Enabled:")
    print(f"  Peak SNR:     {test_stats['peak_snr_mean']:.2f} ± {test_stats['peak_snr_std']:.2f} dB")
    print(f"  Noise Floor:  {test_stats['noise_floor_mean']:.2f} ± {test_stats['noise_floor_std']:.2f} dB")
    print(f"  Duration:     {test['duration_sec']} seconds")
    print(f"  Samples:      {len(test['samples'])}")

    print("\n" + "="*60)
    print("IMPROVEMENT METRICS")
    print("="*60)
    print(f"  SNR Improvement:    {snr_improvement:+.2f} dB")
    print(f"  Noise Reduction:    {noise_reduction:+.2f} dB")

    if snr_improvement >= 10:
        print("\n✓ Excellent improvement (10+ dB)")
    elif snr_improvement >= 5:
        print("\n✓ Good improvement (5-10 dB)")
    elif snr_improvement >= 2:
        print("\n✓ Moderate improvement (2-5 dB)")
    elif snr_improvement > 0:
        print("\n⚠ Marginal improvement (<2 dB)")
    else:
        print("\n✗ No improvement (check configuration)")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Measure Block B3 CAF improvement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Measure with Block B3 enabled (FM)
    python3 measure_b3_improvement.py --duration 60 --output results_fm.json

    # Measure baseline (no reconstruction)
    python3 measure_b3_improvement.py --duration 60 --output results_baseline.json

    # Compare results
    python3 measure_b3_improvement.py --compare results_baseline.json results_fm.json

Expected Improvements:
    FM Radio:   10-15 dB CAF peak SNR improvement
    ATSC 3.0:   15-20 dB CAF peak SNR improvement (strong signals)
        """
    )

    parser.add_argument("--duration", type=float, default=60.0,
                        help="Measurement duration in seconds (default: 60)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sample interval in seconds (default: 1.0)")
    parser.add_argument("--output", type=str,
                        help="Output JSON file for measurements")
    parser.add_argument("--compare", nargs=2, metavar=('BASELINE', 'TEST'),
                        help="Compare two measurement files (baseline vs test)")

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        if not args.output:
            print("Error: --output required for measurement mode")
            sys.exit(1)

        measurements = measure_caf_performance(args.duration, args.interval)

        with open(args.output, 'w') as f:
            json.dump(measurements, f, indent=2)

        print(f"\nMeasurements saved to {args.output}")
        print(f"Peak SNR: {measurements['statistics']['peak_snr_mean']:.2f} dB")
        print(f"Noise Floor: {measurements['statistics']['noise_floor_mean']:.2f} dB")

if __name__ == '__main__':
    main()
