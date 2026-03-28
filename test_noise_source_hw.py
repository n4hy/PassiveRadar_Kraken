#!/usr/bin/env python3
"""
Direct hardware test for KrakenSDR noise source.
No GNU Radio - uses rtlsdr library directly, then raw libusb.

IMPORTANT: Stop any running flowgraph first (osmosdr must not hold the device).

Test 1: rtlsdr_set_bias_tee -- the gold-standard GPIO control path
Test 2: Raw libusb control transfers -- same approach as krakensdr_source.py
Test 3: Two-device cross-correlation -- definitive noise-source verification

Usage:
    python3 test_noise_source_hw.py [--serial 1000] [--freq 103.7e6]
"""

import ctypes
import ctypes.util
import time
import sys
import argparse
import numpy as np


# ── rtlsdr helpers ──────────────────────────────────────────────────

def load_rtlsdr():
    """Load librtlsdr with proper argtypes."""
    lib = ctypes.CDLL(ctypes.util.find_library('rtlsdr') or 'librtlsdr.so.0')

    VP = ctypes.c_void_p
    lib.rtlsdr_get_device_count.restype = ctypes.c_uint32
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
    lib.rtlsdr_read_sync.argtypes = [VP, ctypes.c_void_p,
                                      ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.rtlsdr_read_sync.restype = ctypes.c_int
    lib.rtlsdr_set_bias_tee.argtypes = [VP, ctypes.c_int]
    lib.rtlsdr_set_bias_tee.restype = ctypes.c_int
    lib.rtlsdr_set_bias_tee_gpio.argtypes = [VP, ctypes.c_int, ctypes.c_int]
    lib.rtlsdr_set_bias_tee_gpio.restype = ctypes.c_int
    return lib


def rtlsdr_open(lib, serial):
    """Open RTL-SDR by serial string.  Returns (dev_handle, index) or raises."""
    idx = lib.rtlsdr_get_index_by_serial(serial.encode())
    if idx < 0:
        raise RuntimeError(f"Serial '{serial}' not found (err {idx})")
    dev = ctypes.c_void_p()
    r = lib.rtlsdr_open(ctypes.byref(dev), idx)
    if r != 0:
        raise RuntimeError(f"rtlsdr_open({idx}) failed: {r}")
    return dev, idx


def rtlsdr_configure(lib, dev, freq, gain_tenth_db=0):
    lib.rtlsdr_set_sample_rate(dev, 2_400_000)
    lib.rtlsdr_set_center_freq(dev, int(freq))
    lib.rtlsdr_set_tuner_gain_mode(dev, 1)          # manual gain
    lib.rtlsdr_set_tuner_gain(dev, gain_tenth_db)    # 0 dB for noise cal
    lib.rtlsdr_reset_buffer(dev)


def rtlsdr_capture(lib, dev, n_samples=131072):
    """Capture n_samples complex IQ.  Returns np.complex64 array."""
    n_bytes = n_samples * 2
    buf = (ctypes.c_uint8 * n_bytes)()
    n_read = ctypes.c_int(0)
    r = lib.rtlsdr_read_sync(dev, buf, n_bytes, ctypes.byref(n_read))
    if r != 0:
        raise RuntimeError(f"rtlsdr_read_sync failed: {r}")
    raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return (raw[0::2] - 127.5) / 127.5 + 1j * (raw[1::2] - 127.5) / 127.5


def iq_power_db(iq):
    return float(10 * np.log10(np.mean(np.abs(iq)**2) + 1e-20))


# ── raw libusb GPIO (mirrors krakensdr_source._gpio_control_libusb) ─

def libusb_set_gpio(serial, gpio, value):
    """
    Set RTL2832U GPIO via raw libusb control transfers.
    This is the SAME code path as krakensdr_source._gpio_control_libusb.
    Returns dict of debug info.
    """
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
    libusb.libusb_get_string_descriptor_ascii.argtypes = [VP, ctypes.c_uint8, ctypes.c_char_p, ctypes.c_int]
    libusb.libusb_get_string_descriptor_ascii.restype = ctypes.c_int
    libusb.libusb_control_transfer.argtypes = [
        VP, ctypes.c_uint8, ctypes.c_uint8,
        ctypes.c_uint16, ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint16, ctypes.c_uint]
    libusb.libusb_control_transfer.restype = ctypes.c_int

    CTRL_OUT = 0x40
    CTRL_IN = 0xC0
    SYSB = 2
    GPO = 0x3001
    GPOE = 0x3003
    GPD = 0x3004
    TIMEOUT = 1000
    gpio_bit = 1 << gpio
    info = {}

    ctx = VP()
    ret = libusb.libusb_init(ctypes.byref(ctx))
    assert ret == 0, f"libusb_init failed: {ret}"

    try:
        dev_list = ctypes.POINTER(VP)()
        cnt = libusb.libusb_get_device_list(ctx, ctypes.byref(dev_list))
        assert cnt > 0, f"No USB devices found ({cnt})"

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
                sn = buf.value.decode() if slen > 0 else ""
                if sn == serial:
                    handle = h
                    info['idProduct'] = f"0x{desc.idProduct:04x}"
                    info['serial'] = sn
                    break
                libusb.libusb_close(h)
        finally:
            libusb.libusb_free_device_list(dev_list, 1)

        if handle is None:
            raise RuntimeError(f"Device serial '{serial}' not found via libusb")

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

            # Read-back before any writes
            r, gpd_before = read_reg(GPD, SYSB)
            info['GPD_read_ret'] = r
            info['GPD_before'] = f"0x{gpd_before:02x}"

            r, gpoe_before = read_reg(GPOE, SYSB)
            info['GPOE_read_ret'] = r
            info['GPOE_before'] = f"0x{gpoe_before:02x}"

            r, gpo_before = read_reg(GPO, SYSB)
            info['GPO_read_ret'] = r
            info['GPO_before'] = f"0x{gpo_before:02x}"

            # Set direction: output (clear bit in GPD)
            new_gpd = gpd_before & ~gpio_bit
            r = write_reg(GPD, SYSB, new_gpd)
            info['GPD_write_ret'] = r
            info['GPD_written'] = f"0x{new_gpd:02x}"

            # Enable output (set bit in GPOE)
            new_gpoe = gpoe_before | gpio_bit
            r = write_reg(GPOE, SYSB, new_gpoe)
            info['GPOE_write_ret'] = r
            info['GPOE_written'] = f"0x{new_gpoe:02x}"

            # Set value (set or clear bit in GPO)
            if value:
                new_gpo = gpo_before | gpio_bit
            else:
                new_gpo = gpo_before & ~gpio_bit
            r = write_reg(GPO, SYSB, new_gpo)
            info['GPO_write_ret'] = r
            info['GPO_written'] = f"0x{new_gpo:02x}"

            # Read-back after writes to verify
            r, gpd_after = read_reg(GPD, SYSB)
            info['GPD_after'] = f"0x{gpd_after:02x}"
            r, gpoe_after = read_reg(GPOE, SYSB)
            info['GPOE_after'] = f"0x{gpoe_after:02x}"
            r, gpo_after = read_reg(GPO, SYSB)
            info['GPO_after'] = f"0x{gpo_after:02x}"

            # Check if writes took effect
            info['GPD_ok'] = (gpd_after == new_gpd)
            info['GPOE_ok'] = (gpoe_after == new_gpoe)
            info['GPO_ok'] = (gpo_after == new_gpo)

        finally:
            libusb.libusb_close(handle)
    finally:
        libusb.libusb_exit(ctx)

    return info


# ── Main tests ──────────────────────────────────────────────────────

def test1_rtlsdr_bias_tee(lib, serial, freq):
    """Test 1: Use rtlsdr's own bias_tee function on a single device."""
    print("\n" + "=" * 60)
    print("  TEST 1: rtlsdr_set_bias_tee (gold-standard path)")
    print("=" * 60)

    dev, idx = rtlsdr_open(lib, serial)
    print(f"Opened device {idx} (serial {serial})")

    rtlsdr_configure(lib, dev, freq, gain_tenth_db=0)

    # Baseline (OFF)
    iq_off = rtlsdr_capture(lib, dev)
    pwr_off = iq_power_db(iq_off)
    print(f"\nNoise OFF:  {pwr_off:+.1f} dB")

    # Enable bias tee (= GPIO 0 = noise source)
    r = lib.rtlsdr_set_bias_tee(dev, 1)
    print(f"rtlsdr_set_bias_tee(1) returned {r}")
    time.sleep(0.5)
    lib.rtlsdr_reset_buffer(dev)

    iq_on = rtlsdr_capture(lib, dev)
    pwr_on = iq_power_db(iq_on)
    print(f"Noise ON:   {pwr_on:+.1f} dB")

    delta = pwr_on - pwr_off
    print(f"Delta:      {delta:+.1f} dB")

    # Also try each GPIO 0-7
    print(f"\nTrying all GPIOs 0-7:")
    lib.rtlsdr_set_bias_tee(dev, 0)
    time.sleep(0.2)
    lib.rtlsdr_reset_buffer(dev)
    iq_baseline = rtlsdr_capture(lib, dev)
    pwr_base = iq_power_db(iq_baseline)

    for g in range(8):
        r = lib.rtlsdr_set_bias_tee_gpio(dev, g, 1)
        time.sleep(0.3)
        lib.rtlsdr_reset_buffer(dev)
        iq = rtlsdr_capture(lib, dev)
        pwr = iq_power_db(iq)
        lib.rtlsdr_set_bias_tee_gpio(dev, g, 0)
        time.sleep(0.1)
        print(f"  GPIO {g}: {pwr:+.1f} dB  (delta {pwr - pwr_base:+.1f} dB)  ret={r}")

    lib.rtlsdr_close(dev)
    return delta > 3.0


def test2_raw_libusb(serial):
    """Test 2: Raw libusb GPIO control with full debug output."""
    print("\n" + "=" * 60)
    print("  TEST 2: Raw libusb GPIO control (krakensdr_source path)")
    print("=" * 60)

    print(f"\nSetting GPIO 0 HIGH (noise source ON) on serial {serial}:")
    info_on = libusb_set_gpio(serial, gpio=0, value=1)
    for k, v in info_on.items():
        print(f"  {k}: {v}")

    time.sleep(0.5)

    print(f"\nSetting GPIO 0 LOW (noise source OFF) on serial {serial}:")
    info_off = libusb_set_gpio(serial, gpio=0, value=0)
    for k, v in info_off.items():
        print(f"  {k}: {v}")

    all_ok = (info_on.get('GPD_ok') and info_on.get('GPOE_ok') and
              info_on.get('GPO_ok') and info_off.get('GPO_ok'))
    print(f"\nRegister read-back verification: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test3_cross_correlation(lib, serial_ref, serial_surv, freq):
    """Test 3: Two-device cross-correlation with noise source."""
    print("\n" + "=" * 60)
    print("  TEST 3: Two-device cross-correlation")
    print("=" * 60)

    dev0, idx0 = rtlsdr_open(lib, serial_ref)
    dev1, idx1 = rtlsdr_open(lib, serial_surv)
    print(f"Opened {serial_ref} (idx {idx0}) and {serial_surv} (idx {idx1})")

    rtlsdr_configure(lib, dev0, freq, gain_tenth_db=0)
    rtlsdr_configure(lib, dev1, freq, gain_tenth_db=0)

    # Baseline (OFF)
    iq0_off = rtlsdr_capture(lib, dev0)
    iq1_off = rtlsdr_capture(lib, dev1)
    xc = np.vdot(iq0_off[:4096], iq1_off[:4096])
    p0 = np.sqrt(np.vdot(iq0_off[:4096], iq0_off[:4096]).real)
    p1 = np.sqrt(np.vdot(iq1_off[:4096], iq1_off[:4096]).real)
    corr_off = float(np.abs(xc) / (p0 * p1)) if p0 > 0 and p1 > 0 else 0
    print(f"\nNoise OFF correlation: {corr_off:.4f}")

    # Enable on device 0 (serial_ref controls noise source)
    r = lib.rtlsdr_set_bias_tee(dev0, 1)
    print(f"Bias tee ON (ret={r})")
    time.sleep(0.5)
    lib.rtlsdr_reset_buffer(dev0)
    lib.rtlsdr_reset_buffer(dev1)

    iq0_on = rtlsdr_capture(lib, dev0)
    iq1_on = rtlsdr_capture(lib, dev1)

    # FFT cross-correlation to find delay
    N = min(4096, len(iq0_on), len(iq1_on))
    Xr = np.fft.fft(iq0_on[:N])
    Xs = np.fft.fft(iq1_on[:N])
    xc_full = np.fft.ifft(Xr * np.conj(Xs))
    peak = int(np.argmax(np.abs(xc_full)))
    delay = peak if peak <= N // 2 else peak - N
    peak_val = float(np.abs(xc_full[peak]))
    mean_val = float(np.mean(np.abs(xc_full)))
    snr = peak_val / mean_val if mean_val > 0 else 0

    # Align and compute correlation
    overlap = N - abs(delay)
    if delay >= 0:
        a0, a1 = iq0_on[:overlap], iq1_on[delay:delay + overlap]
    else:
        a0, a1 = iq0_on[-delay:-delay + overlap], iq1_on[:overlap]
    xc = np.vdot(a0, a1)
    p0 = np.sqrt(np.vdot(a0, a0).real)
    p1 = np.sqrt(np.vdot(a1, a1).real)
    corr_on = float(np.abs(xc) / (p0 * p1)) if p0 > 0 and p1 > 0 else 0
    phase = float(np.degrees(np.angle(xc)))

    print(f"Noise ON  correlation: {corr_on:.4f}  (delay={delay}, SNR={snr:.1f})")
    print(f"Phase offset: {phase:+.1f} deg")

    lib.rtlsdr_set_bias_tee(dev0, 0)
    lib.rtlsdr_close(dev1)
    lib.rtlsdr_close(dev0)

    return corr_on > 0.5


def main():
    parser = argparse.ArgumentParser(
        description="KrakenSDR noise source hardware test (no GNU Radio)")
    parser.add_argument('--serial', default='1000',
                        help='Serial of noise-source controller (default: 1000)')
    parser.add_argument('--serial2', default='1001',
                        help='Serial of second device for cross-corr test')
    parser.add_argument('--freq', type=float, default=103.7e6,
                        help='Center frequency in Hz')
    args = parser.parse_args()

    print("=" * 60)
    print("  KrakenSDR Noise Source Hardware Diagnostic")
    print("=" * 60)
    print(f"  Serial (noise ctrl): {args.serial}")
    print(f"  Serial (cross-corr): {args.serial2}")
    print(f"  Frequency: {args.freq / 1e6:.3f} MHz")
    print()
    print("  NOTE: Stop any running flowgraph first!")
    print()

    lib = load_rtlsdr()
    n_devs = lib.rtlsdr_get_device_count()
    print(f"Found {n_devs} RTL-SDR device(s)")

    results = {}

    # Test 1: rtlsdr bias tee
    try:
        results['test1'] = test1_rtlsdr_bias_tee(lib, args.serial, args.freq)
    except Exception as e:
        print(f"  TEST 1 ERROR: {e}")
        results['test1'] = False

    # Test 2: raw libusb
    try:
        results['test2'] = test2_raw_libusb(args.serial)
    except Exception as e:
        print(f"  TEST 2 ERROR: {e}")
        results['test2'] = False

    # Test 3: cross-correlation
    try:
        results['test3'] = test3_cross_correlation(
            lib, args.serial, args.serial2, args.freq)
    except Exception as e:
        print(f"  TEST 3 ERROR: {e}")
        results['test3'] = False

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        label = {'test1': 'rtlsdr bias_tee (power)',
                 'test2': 'Raw libusb GPIO (register read-back)',
                 'test3': 'Cross-correlation (2 devices)'}[name]
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")


if __name__ == '__main__':
    main()
