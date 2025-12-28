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

        # Try to print some info if possible (though osmosdr python api is limited)
        # We rely on stdout logs from the C++ driver
        return num_chans
    except Exception as e:
        print(f"FAILED: {e}")
        return 0

def main():
    print("Detecting KrakenSDR configuration...")
    print("Goal: Ensure Channel 0 maps to Serial 1000, etc.")

    # Test 1: Explicit Serials (Standard Kraken Order)
    # Note: syntax might be "rtl=1000 rtl=1001..." or "rtl0=1000..."
    # We will try the most common multi-device syntax

    # Attempt A: Just listing them. gr-osmosdr usually takes the first match for ch0, second for ch1...
    # If we pass specific serials, it should match them.
    # Note: We must ensure unique identifiers are passed.

    cmd_serials = "numchan=5 rtl=1000 rtl=1001 rtl=1002 rtl=1003 rtl=1004"
    n = test_init(cmd_serials)
    if n == 5:
        print("Serial number mapping works! This ensures Ch0 is SN 1000.")
        print(f"Recommended String: '{cmd_serials}'")
        sys.exit(0)

    # Attempt B: comma separated
    cmd_serials_comma = "numchan=5,rtl=1000,rtl=1001,rtl=1002,rtl=1003,rtl=1004"
    n = test_init(cmd_serials_comma)
    if n == 5:
        print("Comma serial mapping works.")
        sys.exit(0)

    print("\nSUMMARY: Serial number addressing failed. Stick to index based, but verify mapping manually.")
    print("Check the logs above. If 'Device #0' is 'SN: 1004', your channels are reversed.")

if __name__ == "__main__":
    main()
