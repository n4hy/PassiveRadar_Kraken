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

echo "Done. You may need to unplug and replug your KrakenSDR."
