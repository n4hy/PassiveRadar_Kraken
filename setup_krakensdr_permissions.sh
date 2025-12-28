#!/bin/bash
# setup_krakensdr_permissions.sh
# Configures udev rules and kernel blacklisting for KrakenSDR (RTL-SDR based)

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./setup_krakensdr_permissions.sh)"
  exit 1
fi

echo "Configuring KrakenSDR permissions..."

# 1. Blacklist the default DVB-T kernel driver
# This prevents the kernel from claiming the device as a TV tuner, allowing librtlsdr to access it.
echo "Creating /etc/modprobe.d/blacklist-krakensdr.conf..."
echo "blacklist dvb_usb_rtl28xxu" > /etc/modprobe.d/blacklist-krakensdr.conf

# 2. Create udev rules
# Grants access to the device for users in the 'plugdev' group (or everyone via 0666)
# KrakenSDR uses standard RTL-SDR VID/PID: 0bda:2838
echo "Creating /etc/udev/rules.d/52-krakensdr.rules..."
cat <<EOF > /etc/udev/rules.d/52-krakensdr.rules
# KrakenSDR (RTL-SDR)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666", SYMLINK+="krakensdr%n"
EOF

# 3. Reload and Trigger udev
echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

# 4. Increase USBFS memory buffer (Critical for 5-channel operation)
# Default is 16MB, which is insufficient for 5 RTL-SDRs. 0 means unlimited/auto.
echo "Increasing usbfs_memory_mb to unlimited (0)..."
if [ -d "/sys/module/usbcore/parameters" ]; then
    echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb
    # Make it persistent across reboots via GRUB or modprobe.d?
    # For now, let's verify it works for the session.
    echo "USB buffer limit removed."
else
    echo "Warning: /sys/module/usbcore/parameters/usbfs_memory_mb not found."
fi

echo "Done. You may need to unplug and replug your KrakenSDR."
