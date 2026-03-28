#!/usr/bin/env python3
"""
Test: can raw libusb toggle GPIO while rtlsdr holds the device?
This is the EXACT scenario in our flowgraph (osmosdr holds device,
we try raw libusb for noise source GPIO).

Sequence:
  1. Open device with rtlsdr (like osmosdr does)
  2. Capture baseline power via rtlsdr
  3. Use rtlsdr_set_bias_tee to enable noise (KNOWN WORKING) -> measure power
  4. Use rtlsdr_set_bias_tee to disable noise -> confirm power drops
  5. Use RAW LIBUSB (second handle) to enable noise -> measure power
  6. If power rises, raw libusb works concurrently. If not, that's the bug.
"""

import ctypes
import ctypes.util
import time
import numpy as np


def load_rtlsdr():
    lib = ctypes.CDLL(ctypes.util.find_library('rtlsdr') or 'librtlsdr.so.0')
    VP = ctypes.c_void_p
    lib.rtlsdr_get_index_by_serial.argtypes = [ctypes.c_char_p]
    lib.rtlsdr_get_index_by_serial.restype = ctypes.c_int
    lib.rtlsdr_open.argtypes = [ctypes.POINTER(VP), ctypes.c_uint32]
    lib.rtlsdr_open.restype = ctypes.c_int
    lib.rtlsdr_close.argtypes = [VP]
    lib.rtlsdr_close.restype = ctypes.c_int
    lib.rtlsdr_set_sample_rate.argtypes = [VP, ctypes.c_uint32]
    lib.rtlsdr_set_center_freq.argtypes = [VP, ctypes.c_uint32]
    lib.rtlsdr_set_tuner_gain_mode.argtypes = [VP, ctypes.c_int]
    lib.rtlsdr_set_tuner_gain.argtypes = [VP, ctypes.c_int]
    lib.rtlsdr_reset_buffer.argtypes = [VP]
    lib.rtlsdr_read_sync.argtypes = [
        VP, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.rtlsdr_read_sync.restype = ctypes.c_int
    lib.rtlsdr_set_bias_tee.argtypes = [VP, ctypes.c_int]
    lib.rtlsdr_set_bias_tee.restype = ctypes.c_int
    return lib


def capture_power(lib, dev, label=""):
    n_bytes = 262144
    buf = (ctypes.c_uint8 * n_bytes)()
    n_read = ctypes.c_int(0)
    lib.rtlsdr_reset_buffer(dev)
    lib.rtlsdr_read_sync(dev, buf, n_bytes, ctypes.byref(n_read))
    raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    iq = (raw[0::2] - 127.5) / 127.5 + 1j * (raw[1::2] - 127.5) / 127.5
    pwr = float(10 * np.log10(np.mean(np.abs(iq)**2) + 1e-20))
    print(f"  {label:40s}  power = {pwr:+.1f} dB")
    return pwr


def libusb_set_gpio0(serial, value):
    """Set GPIO 0 via raw libusb (second handle). Returns read-back GPO."""
    libusb = ctypes.CDLL(ctypes.util.find_library('usb-1.0') or 'libusb-1.0.so.0')

    class DevDesc(ctypes.Structure):
        _fields_ = [
            ('bLength', ctypes.c_uint8), ('bDescriptorType', ctypes.c_uint8),
            ('bcdUSB', ctypes.c_uint16), ('bDeviceClass', ctypes.c_uint8),
            ('bDeviceSubClass', ctypes.c_uint8), ('bDeviceProtocol', ctypes.c_uint8),
            ('bMaxPacketSize0', ctypes.c_uint8), ('idVendor', ctypes.c_uint16),
            ('idProduct', ctypes.c_uint16), ('bcdDevice', ctypes.c_uint16),
            ('iManufacturer', ctypes.c_uint8), ('iProduct', ctypes.c_uint8),
            ('iSerialNumber', ctypes.c_uint8), ('bNumConfigurations', ctypes.c_uint8),
        ]

    VP = ctypes.c_void_p
    libusb.libusb_init.argtypes = [ctypes.POINTER(VP)]
    libusb.libusb_init.restype = ctypes.c_int
    libusb.libusb_exit.argtypes = [VP]
    libusb.libusb_exit.restype = None
    libusb.libusb_get_device_list.argtypes = [VP, ctypes.POINTER(ctypes.POINTER(VP))]
    libusb.libusb_get_device_list.restype = ctypes.c_ssize_t
    libusb.libusb_free_device_list.argtypes = [ctypes.POINTER(VP), ctypes.c_int]
    libusb.libusb_free_device_list.restype = None
    libusb.libusb_get_device_descriptor.argtypes = [VP, ctypes.POINTER(DevDesc)]
    libusb.libusb_get_device_descriptor.restype = ctypes.c_int
    libusb.libusb_open.argtypes = [VP, ctypes.POINTER(VP)]
    libusb.libusb_open.restype = ctypes.c_int
    libusb.libusb_close.argtypes = [VP]
    libusb.libusb_close.restype = None
    libusb.libusb_get_string_descriptor_ascii.argtypes = [
        VP, ctypes.c_uint8, ctypes.c_char_p, ctypes.c_int]
    libusb.libusb_get_string_descriptor_ascii.restype = ctypes.c_int
    libusb.libusb_control_transfer.argtypes = [
        VP, ctypes.c_uint8, ctypes.c_uint8,
        ctypes.c_uint16, ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint16, ctypes.c_uint]
    libusb.libusb_control_transfer.restype = ctypes.c_int

    CTRL_OUT, CTRL_IN = 0x40, 0xC0
    SYSB, GPO, GPOE, GPD = 2, 0x3001, 0x3003, 0x3004
    TIMEOUT = 1000

    ctx = VP()
    libusb.libusb_init(ctypes.byref(ctx))
    try:
        dev_list = ctypes.POINTER(VP)()
        cnt = libusb.libusb_get_device_list(ctx, ctypes.byref(dev_list))
        handle = None
        try:
            for i in range(cnt):
                dev = dev_list[i]
                if not dev:
                    break
                desc = DevDesc()
                if libusb.libusb_get_device_descriptor(VP(dev), ctypes.byref(desc)) != 0:
                    continue
                if desc.idVendor != 0x0bda or desc.iSerialNumber == 0:
                    continue
                h = VP()
                if libusb.libusb_open(VP(dev), ctypes.byref(h)) != 0:
                    continue
                buf = ctypes.create_string_buffer(256)
                slen = libusb.libusb_get_string_descriptor_ascii(
                    h, desc.iSerialNumber, buf, 256)
                if slen > 0 and buf.value == serial.encode():
                    handle = h
                    break
                libusb.libusb_close(h)
        finally:
            libusb.libusb_free_device_list(dev_list, 1)

        if handle is None:
            print(f"  ERROR: device {serial} not found via libusb")
            return None

        try:
            def read_reg(addr, block):
                data = (ctypes.c_uint8 * 1)()
                r = libusb.libusb_control_transfer(
                    handle, CTRL_IN, 0, addr, block << 8, data, 1, TIMEOUT)
                return r, data[0]

            def write_reg(addr, block, val):
                data = (ctypes.c_uint8 * 1)(val & 0xFF)
                r = libusb.libusb_control_transfer(
                    handle, CTRL_OUT, 0, addr, (block << 8) | 0x10, data, 1, TIMEOUT)
                return r

            # Direction + output enable
            r, gpd = read_reg(GPD, SYSB)
            write_reg(GPD, SYSB, gpd & ~1)
            r, gpoe = read_reg(GPOE, SYSB)
            write_reg(GPOE, SYSB, gpoe | 1)

            # Set value
            r, gpo = read_reg(GPO, SYSB)
            new_gpo = (gpo | 1) if value else (gpo & ~1)
            write_reg(GPO, SYSB, new_gpo)

            # Read-back
            r, gpo_rb = read_reg(GPO, SYSB)
            print(f"  libusb: GPO 0x{gpo:02x} -> wrote 0x{new_gpo:02x} -> readback 0x{gpo_rb:02x} "
                  f"{'OK' if gpo_rb == (new_gpo & 0xFF) else 'MISMATCH'}")
            return gpo_rb
        finally:
            libusb.libusb_close(handle)
    finally:
        libusb.libusb_exit(ctx)


def main():
    print("=" * 60)
    print("  Concurrent libusb + rtlsdr GPIO test")
    print("=" * 60)

    lib = load_rtlsdr()
    idx = lib.rtlsdr_get_index_by_serial(b"1000")
    if idx < 0:
        print(f"Device serial 1000 not found (err {idx})")
        return

    dev = ctypes.c_void_p()
    r = lib.rtlsdr_open(ctypes.byref(dev), idx)
    if r != 0:
        print(f"rtlsdr_open failed: {r}")
        return

    lib.rtlsdr_set_sample_rate(dev, 2_400_000)
    lib.rtlsdr_set_center_freq(dev, int(103.7e6))
    lib.rtlsdr_set_tuner_gain_mode(dev, 1)
    lib.rtlsdr_set_tuner_gain(dev, 0)
    print(f"\nDevice opened with rtlsdr (like osmosdr does)\n")

    # Step 1: Baseline
    p_off = capture_power(lib, dev, "1. Baseline (noise OFF)")

    # Step 2: rtlsdr enables noise (known working)
    lib.rtlsdr_set_bias_tee(dev, 1)
    time.sleep(0.3)
    p_rtl_on = capture_power(lib, dev, "2. rtlsdr_set_bias_tee(1) -> ON")

    # Step 3: rtlsdr disables noise
    lib.rtlsdr_set_bias_tee(dev, 0)
    time.sleep(0.3)
    p_rtl_off = capture_power(lib, dev, "3. rtlsdr_set_bias_tee(0) -> OFF")

    # Step 4: RAW LIBUSB enables noise (while rtlsdr holds device!)
    print(f"\n  --- Now testing raw libusb while rtlsdr holds device ---")
    libusb_set_gpio0("1000", value=1)
    time.sleep(0.3)
    p_libusb_on = capture_power(lib, dev, "4. raw libusb GPIO 0 HIGH -> ???")

    # Step 5: RAW LIBUSB disables noise
    libusb_set_gpio0("1000", value=0)
    time.sleep(0.3)
    p_libusb_off = capture_power(lib, dev, "5. raw libusb GPIO 0 LOW  -> ???")

    lib.rtlsdr_close(dev)

    # Verdict
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  rtlsdr ON delta:  {p_rtl_on - p_off:+.1f} dB (expect +30 dB)")
    print(f"  libusb ON delta:  {p_libusb_on - p_rtl_off:+.1f} dB (expect +30 dB if working)")

    if p_libusb_on - p_rtl_off > 10:
        print(f"\n  PASS: raw libusb GPIO works while rtlsdr holds the device")
    else:
        print(f"\n  FAIL: raw libusb GPIO does NOT work while rtlsdr holds device")
        print(f"  This explains why noise source fails in the flowgraph.")
        print(f"  Fix: use rtlsdr_set_bias_tee via osmosdr's own handle.")


if __name__ == '__main__':
    main()
