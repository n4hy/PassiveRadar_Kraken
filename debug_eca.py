import os
import ctypes
import numpy as np

# Load library
lib = ctypes.cdll.LoadLibrary("./src/libkraken_eca_b_clutter_canceller.so")

# Define signatures
lib.eca_b_create.restype = ctypes.c_void_p
lib.eca_b_create.argtypes = [ctypes.c_int]

lib.eca_b_process.restype = None
lib.eca_b_process.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

# Create ECA
num_taps = 4
state = lib.eca_b_create(num_taps)

# Create input
n = 4096
ref = (np.random.randn(n) + 1j*np.random.randn(n)).astype(np.complex64)
# Make surv exactly correlated
surv = ref.copy() # coeff = 1.0 at delay 0

out = np.zeros(n, dtype=np.complex64)

# Call
lib.eca_b_process(
    state,
    ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    surv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    n
)

# Check
power_in = np.mean(np.abs(surv)**2)
power_out = np.mean(np.abs(out)**2)
print(f"Power In: {power_in}")
print(f"Power Out: {power_out}")
if power_out < 1e-5:
    print("SUCCESS: Cancellation works")
else:
    print("FAIL: No cancellation")
