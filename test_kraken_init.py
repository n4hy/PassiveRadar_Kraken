#!/usr/bin/env python3
import sys
import osmosdr
import time

def test_init(args_str):
    print(f"\n--- Testing osmosdr.source(args='{args_str}') ---")
    try:
        src = osmosdr.source(args=args_str)
        num_chans = src.get_num_channels()
        print(f"SUCCESS: Initialized {num_chans} channels.")
        return num_chans
    except Exception as e:
        print(f"FAILED: {e}")
        return 0

def main():
    print("Detecting KrakenSDR configuration...")

    # Test 1: Standard numchan=5
    n = test_init("numchan=5")
    if n == 5:
        print("Standard configuration works.")
        sys.exit(0)

    # Test 2: Explicit indices
    # Sometimes providing specific rtl indices helps librtlsdr identify separate devices
    n = test_init("numchan=5 rtl=0 rtl=1 rtl=2 rtl=3 rtl=4")
    if n == 5:
        print("Explicit indices configuration works.")
        sys.exit(0)

    # Test 3: String with comma separation (some versions prefer this)
    n = test_init("numchan=5,rtl=0,rtl=1,rtl=2,rtl=3,rtl=4")
    if n == 5:
        print("Comma separated configuration works.")
        sys.exit(0)

    print("\nSUMMARY: Could not initialize 5 channels with standard arguments.")
    print("Please check: ")
    print("1. Power supply (2.5A+ required)")
    print("2. USB connection")
    print("3. Udev rules (run setup_krakensdr_permissions.sh)")

if __name__ == "__main__":
    main()
