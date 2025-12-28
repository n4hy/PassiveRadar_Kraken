#!/usr/bin/env python3
import sys
import osmosdr
import time

def test_single_serial(serial):
    print(f"\n--- Testing Single Device Serial {serial} ---")
    try:
        # Try to open just ONE device by serial
        src = osmosdr.source(args=f"rtl={serial}")
        src.set_sample_rate(2.048e6)
        src.set_center_freq(100e6)
        src.set_gain(30)

        # We need to actually start it to see if it streams
        # But osmosdr python object doesn't have a simple 'read' method easily accessible without a flowgraph usually.
        # However, successfully initializing without crash is step 1.
        print(f"Device {serial} initialized successfully.")

        # Just hold it open for a second
        time.sleep(1)
        del src
        print(f"Device {serial} closed cleanly.")
        return True
    except Exception as e:
        print(f"FAILED to initialize Serial {serial}: {e}")
        return False

def main():
    serials = [1000, 1001, 1002, 1003, 1004]
    print(f"Testing individual initialization of serials: {serials}")

    results = {}
    for s in serials:
        results[s] = test_single_serial(s)
        time.sleep(0.5) # Give libusb a moment to breath

    print("\n--- SUMMARY ---")
    for s in serials:
        status = "OK" if results[s] else "FAIL"
        print(f"Serial {s} (Physical CH{s-1000}): {status}")

    print("\nIf one specific serial fails or hangs, that specific tuner hardware has an issue.")

if __name__ == "__main__":
    main()
