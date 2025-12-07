
import gnuradio.filter
import gnuradio.gr as gr
import sys

try:
    # Check for adaptive blocks
    print("Checking for LMS filter...")
    # usually lms_dd_equalizer_cc, but that's decision directed.
    # We want generic adaptive filter.

    # In some versions, there might be 'adaptive_fir_filter_ccf' or similar?
    # Let's inspect gnuradio.filter
    print(dir(gnuradio.filter))

except Exception as e:
    print(e)
