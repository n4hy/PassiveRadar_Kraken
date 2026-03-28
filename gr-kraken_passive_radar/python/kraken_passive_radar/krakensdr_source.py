import numpy as np
from gnuradio import gr
import osmosdr
import sys
import ctypes
import ctypes.util

class krakensdr_source(gr.hier_block2):
    """
    KrakenSDR Source Block
    Wraps osmosdr.source configured for 5 coherent channels.
    """
    def __init__(self, frequency=100e6, sample_rate=2.4e6, gain=30.0):
        gr.hier_block2.__init__(
            self,
            "KrakenSDR Source",
            gr.io_signature(0, 0, 0),  # No inputs
            gr.io_signature(5, 5, gr.sizeof_gr_complex)  # 5 Complex outputs
        )

        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain

        # Create the underlying OSMOSDR source
        # "numchan=5" is the key argument for KrakenSDR.
        # We use explicit SERIAL NUMBERS (1000..1004) to ensure Channel 0 maps to Port 1, etc.
        # Attach buffer settings to EACH device string to avoid parsing errors.
        # buffers=128 (increased from 32) and buflen=65536 (64kB) to absorb USB latency.

        channel_args = []
        for i in range(5):
            channel_args.append(f"rtl={1000+i},buffers=128,buflen=65536")

        source_args = "numchan=5 " + " ".join(channel_args)

        self.osmosdr = osmosdr.source(args=source_args)

        self.osmosdr.set_sample_rate(self.sample_rate)

        # Configure all 5 channels
        for i in range(5):
            self.osmosdr.set_center_freq(self.frequency, i)
            # Explicitly enable Manual Gain mode so the set_gain() call is respected.
            # For RTL-SDR in gr-osmosdr: set_gain_mode(False) = Manual Gain.
            self.osmosdr.set_gain_mode(False, i)
            self.osmosdr.set_gain(self.gain, i)
            self.osmosdr.set_bandwidth(0, i) # 0 = Auto
            self.osmosdr.set_freq_corr(0, i)
            self.osmosdr.set_dc_offset_mode(0, i) # 0 = Off (Manual)
            self.osmosdr.set_iq_balance_mode(0, i) # 0 = Off (Manual)

            # Ensure proper clock source (internal) and antenna selection
            try:
                self.osmosdr.set_clock_source("internal", i)
            except (AttributeError, RuntimeError, NotImplementedError):
                # Some drivers (e.g., RTL-SDR) don't implement clock source control
                pass

            # Connect the internal osmosdr ports to the hier block outputs
            self.connect((self.osmosdr, i), (self, i))

    def set_frequency(self, freq):
        self.frequency = freq
        for i in range(5):
            self.osmosdr.set_center_freq(freq, i)

    def set_sample_rate(self, rate):
        self.sample_rate = rate
        self.osmosdr.set_sample_rate(rate)

    def set_gain(self, gain):
        self.gain = gain
        for i in range(5):
            self.osmosdr.set_gain(gain, i)

    def set_noise_source(self, enable):
        """
        Enables or disables the KrakenSDR internal noise source.

        Uses direct libusb vendor control transfers to the RTL2832U GPIO
        registers, bypassing rtlsdr_open() which would fail with
        usb_claim_interface error -6 because osmosdr holds the bulk interface.

        Control transfers to endpoint 0 don't require interface claiming,
        so they work concurrently with osmosdr's data streaming.

        IMPORTANT: When enabled, the hardware silicon switch DISCONNECTS all
        antennas and routes ONLY the internal noise source to all receivers.

        Args:
            enable (bool): True to enable, False to disable.
        """
        self._gpio_control_libusb(gpio=0, value=1 if enable else 0)
        state = "ENABLED" if enable else "DISABLED"
        print(f"Noise source {state} via direct USB control transfer")

    def _gpio_control_libusb(self, gpio, value):
        """
        Set RTL2832U GPIO via direct libusb vendor control transfer.

        RTL2832U register access uses vendor control transfers (endpoint 0):
          - CTRL_OUT (0x40): write register
          - CTRL_IN  (0xC0): read register
          - Block SYSB (2): system registers (GPO, GPOE, GPD)
          - GPO (0x3001): GPIO output value
          - GPOE (0x3003): GPIO output enable
          - GPD (0x3004): GPIO direction
        """
        libusb = ctypes.CDLL(
            ctypes.util.find_library('usb-1.0') or 'libusb-1.0.so.0')

        class DevDesc(ctypes.Structure):
            _fields_ = [
                ('bLength', ctypes.c_uint8),
                ('bDescriptorType', ctypes.c_uint8),
                ('bcdUSB', ctypes.c_uint16),
                ('bDeviceClass', ctypes.c_uint8),
                ('bDeviceSubClass', ctypes.c_uint8),
                ('bDeviceProtocol', ctypes.c_uint8),
                ('bMaxPacketSize0', ctypes.c_uint8),
                ('idVendor', ctypes.c_uint16),
                ('idProduct', ctypes.c_uint16),
                ('bcdDevice', ctypes.c_uint16),
                ('iManufacturer', ctypes.c_uint8),
                ('iProduct', ctypes.c_uint8),
                ('iSerialNumber', ctypes.c_uint8),
                ('bNumConfigurations', ctypes.c_uint8),
            ]

        # argtypes/restype required on 64-bit: without them ctypes passes
        # pointer-valued integers as c_int (32-bit), truncating addresses
        # above 4 GB and causing segfaults under memory pressure.
        libusb.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        libusb.libusb_init.restype = ctypes.c_int
        libusb.libusb_exit.argtypes = [ctypes.c_void_p]
        libusb.libusb_exit.restype = None
        libusb.libusb_get_device_list.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))]
        libusb.libusb_get_device_list.restype = ctypes.c_ssize_t
        libusb.libusb_free_device_list.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        libusb.libusb_free_device_list.restype = None
        libusb.libusb_get_device_descriptor.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(DevDesc)]
        libusb.libusb_get_device_descriptor.restype = ctypes.c_int
        libusb.libusb_open.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        libusb.libusb_open.restype = ctypes.c_int
        libusb.libusb_close.argtypes = [ctypes.c_void_p]
        libusb.libusb_close.restype = None
        libusb.libusb_get_string_descriptor_ascii.argtypes = [
            ctypes.c_void_p, ctypes.c_uint8, ctypes.c_char_p, ctypes.c_int]
        libusb.libusb_get_string_descriptor_ascii.restype = ctypes.c_int
        libusb.libusb_control_transfer.argtypes = [
            ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.c_uint16, ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint16, ctypes.c_uint]
        libusb.libusb_control_transfer.restype = ctypes.c_int

        CTRL_OUT = 0x40
        CTRL_IN = 0xC0
        SYSB = 2          # System block (block enum: DEMODB=0, USBB=1, SYSB=2)
        GPO = 0x3001       # GPIO output value
        GPOE = 0x3003      # GPIO output enable
        GPD = 0x3004       # GPIO direction
        TIMEOUT = 1000
        gpio_bit = 1 << gpio

        ctx = ctypes.c_void_p()
        ret = libusb.libusb_init(ctypes.byref(ctx))
        if ret != 0:
            raise RuntimeError(f"libusb_init failed: {ret}")

        try:
            dev_list = ctypes.POINTER(ctypes.c_void_p)()
            cnt = libusb.libusb_get_device_list(ctx, ctypes.byref(dev_list))
            if cnt < 0:
                raise RuntimeError(f"libusb_get_device_list failed: {cnt}")

            handle = None
            try:
                for i in range(cnt):
                    dev = dev_list[i]
                    if not dev:
                        break
                    desc = DevDesc()
                    if libusb.libusb_get_device_descriptor(
                            ctypes.c_void_p(dev), ctypes.byref(desc)) != 0:
                        continue
                    if desc.idVendor != 0x0bda or desc.iSerialNumber == 0:
                        continue

                    h = ctypes.c_void_p()
                    if libusb.libusb_open(
                            ctypes.c_void_p(dev), ctypes.byref(h)) != 0:
                        continue

                    buf = ctypes.create_string_buffer(256)
                    slen = libusb.libusb_get_string_descriptor_ascii(
                        h, desc.iSerialNumber, buf, 256)
                    if slen > 0 and buf.value == b"1000":
                        handle = h
                        break
                    libusb.libusb_close(h)
            finally:
                libusb.libusb_free_device_list(dev_list, 1)

            if handle is None:
                raise RuntimeError("KrakenSDR SN 1000 not found via libusb")

            try:
                def read_reg(addr, block):
                    data = (ctypes.c_uint8 * 1)()
                    r = libusb.libusb_control_transfer(
                        handle, CTRL_IN, 0,
                        addr, block << 8,
                        data, 1, TIMEOUT)
                    if r < 0:
                        raise RuntimeError(f"USB read reg 0x{addr:04x} failed: {r}")
                    return data[0]

                def write_reg(addr, block, val):
                    data = (ctypes.c_uint8 * 1)(val & 0xFF)
                    r = libusb.libusb_control_transfer(
                        handle, CTRL_OUT, 0,
                        addr, block << 8,
                        data, 1, TIMEOUT)
                    if r < 0:
                        raise RuntimeError(f"USB write reg 0x{addr:04x} failed: {r}")
                    # Dummy read to flush (RTL2832U demod quirk)
                    dummy = (ctypes.c_uint8 * 1)()
                    libusb.libusb_control_transfer(
                        handle, CTRL_IN, 0,
                        0x01, 0x0A << 8,
                        dummy, 1, TIMEOUT)

                # Set GPIO as output (matches rtlsdr_set_gpio_output)
                gpd = read_reg(GPD, SYSB)
                write_reg(GPD, SYSB, gpd & ~gpio_bit)
                gpoe = read_reg(GPOE, SYSB)
                write_reg(GPOE, SYSB, gpoe | gpio_bit)

                # Set GPIO value (matches rtlsdr_set_gpio_bit)
                gpo = read_reg(GPO, SYSB)
                if value:
                    write_reg(GPO, SYSB, gpo | gpio_bit)
                else:
                    write_reg(GPO, SYSB, gpo & ~gpio_bit)
            finally:
                libusb.libusb_close(handle)
        finally:
            libusb.libusb_exit(ctx)
