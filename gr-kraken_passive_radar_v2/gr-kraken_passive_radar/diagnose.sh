#!/bin/bash
# Diagnostic script for GNU Radio OOT module installation
echo "============================================="
echo "GNU Radio OOT Module Diagnostic"
echo "============================================="

echo ""
echo "[1] GNU Radio Installation"
echo "-------------------------------------------"
if command -v gnuradio-config-info &> /dev/null; then
    echo "  Version: $(gnuradio-config-info --version)"
    echo "  Prefix:  $(gnuradio-config-info --prefix)"
    GR_PREFIX=$(gnuradio-config-info --prefix)
else
    echo "  ERROR: gnuradio-config-info not found"
    exit 1
fi

echo ""
echo "[2] GRC Block Search Paths"
echo "-------------------------------------------"
echo "  Standard paths checked by GRC:"
echo "    - ${GR_PREFIX}/share/gnuradio/grc/blocks/"
echo "    - /usr/local/share/gnuradio/grc/blocks/"
echo "    - ~/.grc_gnuradio/"

echo ""
echo "[3] Kraken Module YAML Files"
echo "-------------------------------------------"
FOUND=0
for path in "${GR_PREFIX}/share/gnuradio/grc/blocks" "/usr/local/share/gnuradio/grc/blocks" "/usr/share/gnuradio/grc/blocks"; do
    if ls "${path}"/*kraken* 2>/dev/null; then
        echo "  Found in: ${path}"
        FOUND=1
    fi
done
if [ $FOUND -eq 0 ]; then
    echo "  WARNING: No kraken YAML files found!"
    echo "  This is why blocks don't appear in GRC."
fi

echo ""
echo "[4] Shared Library"
echo "-------------------------------------------"
if ldconfig -p | grep -q kraken; then
    ldconfig -p | grep kraken
else
    echo "  WARNING: gnuradio-kraken library not found in ldconfig"
    echo "  Checking manual paths..."
    ls -la /usr/local/lib/*kraken* 2>/dev/null || echo "  Not in /usr/local/lib"
    ls -la /usr/lib/*kraken* 2>/dev/null || echo "  Not in /usr/lib"
fi

echo ""
echo "[5] Python Module"
echo "-------------------------------------------"
python3 << 'EOF'
try:
    from gnuradio import kraken_passive_radar
    print("  SUCCESS: Module imported")
    print("  Location:", kraken_passive_radar.__file__ if hasattr(kraken_passive_radar, '__file__') else "builtin")
    attrs = [a for a in dir(kraken_passive_radar) if not a.startswith('_')]
    print("  Available blocks:", attrs)
except ImportError as e:
    print(f"  FAILED: {e}")
    print("  Checking Python path...")
    import sys
    for p in sys.path:
        print(f"    {p}")
EOF

echo ""
echo "[6] YAML Syntax Validation"
echo "-------------------------------------------"
for yml in grc/*.block.yml; do
    if [ -f "$yml" ]; then
        echo -n "  Checking $yml: "
        python3 -c "import yaml; yaml.safe_load(open('$yml'))" 2>&1 && echo "OK" || echo "SYNTAX ERROR"
    fi
done

echo ""
echo "[7] pybind11 Availability"
echo "-------------------------------------------"
python3 -c "import pybind11; print('  pybind11 version:', pybind11.__version__)" 2>&1 || echo "  WARNING: pybind11 not found"

echo ""
echo "============================================="
echo "Diagnosis Complete"
echo "============================================="
