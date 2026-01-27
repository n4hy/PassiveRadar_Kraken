#!/bin/bash
#===============================================================================
# PassiveRadar_Kraken Comprehensive Test Runner
#===============================================================================
#
# Runs all unit tests, integration tests, and benchmarks for the passive radar
# system. Tests are organized by category and can be run selectively.
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh unit         # Run only unit tests
#   ./run_tests.sh integration  # Run only integration tests
#   ./run_tests.sh benchmark    # Run only benchmarks
#   ./run_tests.sh quick        # Run quick subset of tests
#   ./run_tests.sh cpp          # Run C++ library tests only
#   ./run_tests.sh display      # Run display module tests only
#   ./run_tests.sh -v           # Verbose output
#   ./run_tests.sh -h           # Show help
#
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
VERBOSE=""
TEST_CATEGORY="all"
PYTEST_ARGS=""
FAILED=0
PASSED=0
SKIPPED=0

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++)) || true
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++)) || true
}

print_skip() {
    echo -e "${YELLOW}○ $1 (skipped)${NC}"
    ((SKIPPED++)) || true
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

show_help() {
    echo "PassiveRadar_Kraken Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] [CATEGORY]"
    echo ""
    echo "Categories:"
    echo "  all          Run all tests (default)"
    echo "  unit         Run unit tests only"
    echo "  integration  Run integration tests only"
    echo "  benchmark    Run performance benchmarks"
    echo "  quick        Run quick subset of tests"
    echo "  cpp          Run C++ library tests"
    echo "  display      Run display module tests"
    echo "  fixtures     Run fixture tests only"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Verbose output"
    echo "  -h, --help       Show this help"
    echo "  -k PATTERN       Only run tests matching PATTERN"
    echo "  --failfast       Stop on first failure"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all tests"
    echo "  $0 unit -v            # Run unit tests verbosely"
    echo "  $0 -k caf             # Run tests matching 'caf'"
    echo "  $0 benchmark          # Run performance benchmarks"
}

check_prerequisites() {
    print_section "Checking Prerequisites"

    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        print_success "Python: $PYTHON_VERSION"
    else
        print_failure "Python3 not found"
        exit 1
    fi

    # Check pytest
    if python3 -c "import pytest" 2>/dev/null; then
        PYTEST_VERSION=$(python3 -c "import pytest; print(pytest.__version__)")
        print_success "pytest: $PYTEST_VERSION"
    else
        print_info "pytest not found, using unittest"
    fi

    # Check numpy
    if python3 -c "import numpy" 2>/dev/null; then
        NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
        print_success "numpy: $NUMPY_VERSION"
    else
        print_failure "numpy not found"
        exit 1
    fi

    # Check C++ libraries
    local libs_found=0
    for lib in libkraken_caf_processing.so libkraken_eca_b_clutter_canceller.so libkraken_doppler_processing.so; do
        if [ -f "src/$lib" ]; then
            print_success "Found: src/$lib"
            ((libs_found++)) || true
        else
            print_skip "Not found: src/$lib"
        fi
    done

    if [ $libs_found -eq 0 ]; then
        print_info "No C++ libraries found. Run 'cd src && mkdir -p build && cd build && cmake .. && make' to build them."
    fi

    # Check display modules
    for module in range_doppler_display radar_display calibration_panel metrics_dashboard; do
        if [ -f "kraken_passive_radar/${module}.py" ]; then
            print_success "Found: kraken_passive_radar/${module}.py"
        fi
    done

    echo ""
}

#-------------------------------------------------------------------------------
# Test runners
#-------------------------------------------------------------------------------

run_unittest_discovery() {
    local test_dir="$1"
    local description="$2"

    print_section "$description"

    if [ -d "$test_dir" ]; then
        export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/tests:$PYTHONPATH"

        if python3 -c "import pytest" 2>/dev/null; then
            # Use pytest if available
            python3 -m pytest "$test_dir" $VERBOSE $PYTEST_ARGS --tb=short || {
                print_failure "$description"
                return 1
            }
        else
            # Fall back to unittest
            python3 -m unittest discover -s "$test_dir" $VERBOSE || {
                print_failure "$description"
                return 1
            }
        fi
        print_success "$description completed"
    else
        print_skip "$test_dir not found"
    fi
}

run_unit_tests() {
    print_header "UNIT TESTS"

    run_unittest_discovery "tests/unit" "Unit Tests"

    # Also run legacy tests in tests/ directory
    if [ -f "tests/test_caf_cpp.py" ]; then
        print_section "Legacy C++ Wrapper Tests"
        export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
        python3 -m unittest tests.test_caf_cpp $VERBOSE || print_failure "test_caf_cpp"
        python3 -m unittest tests.test_eca_b_cpp $VERBOSE || print_failure "test_eca_b_cpp"
        python3 -m unittest tests.test_doppler_cpp $VERBOSE || print_failure "test_doppler_cpp"
        python3 -m unittest tests.test_aoa_cpp $VERBOSE || print_failure "test_aoa_cpp"
    fi
}

run_integration_tests() {
    print_header "INTEGRATION TESTS"

    run_unittest_discovery "tests/integration" "Integration Tests"
}

run_benchmark_tests() {
    print_header "PERFORMANCE BENCHMARKS"

    run_unittest_discovery "tests/benchmarks" "Performance Benchmarks"
}

run_cpp_tests() {
    print_header "C++ LIBRARY TESTS"

    print_section "C++ Library Tests"

    # Test CAF library
    if [ -f "src/libkraken_caf_processing.so" ]; then
        python3 -m unittest tests.test_caf_cpp -v || print_failure "CAF tests"
        print_success "CAF library tests"
    else
        print_skip "CAF library not built"
    fi

    # Test ECA library
    if [ -f "src/libkraken_eca_b_clutter_canceller.so" ]; then
        python3 -m unittest tests.test_eca_b_cpp -v || print_failure "ECA tests"
        print_success "ECA library tests"
    else
        print_skip "ECA library not built"
    fi

    # Test Doppler library
    if [ -f "src/libkraken_doppler_processing.so" ]; then
        python3 -m unittest tests.test_doppler_cpp -v || print_failure "Doppler tests"
        print_success "Doppler library tests"
    else
        print_skip "Doppler library not built"
    fi
}

run_display_tests() {
    print_header "DISPLAY MODULE TESTS"

    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

    print_section "Display Module Tests"
    python3 -m unittest tests.unit.test_display_modules $VERBOSE || print_failure "Display tests"
}

run_fixture_tests() {
    print_header "FIXTURE TESTS"

    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

    print_section "Synthetic Target Generator Tests"
    python3 -c "
from tests.fixtures.synthetic_targets import BistaticTargetGenerator, TargetSpec, single_target_snr15
import numpy as np

# Test basic generation
gen = BistaticTargetGenerator()
ref = gen.generate_fm_waveform(4096)
print(f'  FM waveform: {len(ref)} samples, power={np.mean(np.abs(ref)**2):.3f}')

# Test scenario generation
scenario = single_target_snr15()
print(f'  Scenario: {len(scenario.targets)} targets, ref={len(scenario.ref_signal)} samples')

print('✓ Fixture tests passed')
" || print_failure "Fixture tests"
}

run_quick_tests() {
    print_header "QUICK TESTS"

    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

    print_section "Quick Smoke Tests"

    # Basic imports
    python3 -c "
import numpy as np
print('  numpy import: OK')

# Test fixtures
from tests.fixtures.synthetic_targets import BistaticTargetGenerator
print('  synthetic_targets import: OK')

from tests.fixtures.clutter_models import ClutterGenerator
print('  clutter_models import: OK')

from tests.fixtures.noise_models import NoiseGenerator
print('  noise_models import: OK')

# Quick functional test
gen = BistaticTargetGenerator()
ref = gen.generate_fm_waveform(1024)
assert len(ref) == 1024
print('  FM waveform generation: OK')

print('✓ Quick tests passed')
" && print_success "Quick smoke tests"

    # Quick C++ library test
    if [ -f "src/libkraken_caf_processing.so" ]; then
        python3 -c "
import ctypes
import numpy as np
lib = ctypes.cdll.LoadLibrary('src/libkraken_caf_processing.so')
print('  C++ CAF library load: OK')
print('✓ C++ library load test passed')
" && print_success "C++ library quick test"
    fi
}

run_all_tests() {
    run_unit_tests
    run_integration_tests
    run_fixture_tests
    run_cpp_tests
    run_display_tests
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -k)
            PYTEST_ARGS="$PYTEST_ARGS -k $2"
            shift 2
            ;;
        --failfast)
            PYTEST_ARGS="$PYTEST_ARGS --exitfirst"
            shift
            ;;
        unit|integration|benchmark|quick|cpp|display|fixtures|all)
            TEST_CATEGORY="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print banner
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     PassiveRadar_Kraken Comprehensive Test Suite             ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Test Category: $TEST_CATEGORY"
echo "  Verbose: ${VERBOSE:-no}"
echo ""

# Check prerequisites
check_prerequisites

# Run tests based on category
case $TEST_CATEGORY in
    all)
        run_all_tests
        ;;
    unit)
        run_unit_tests
        ;;
    integration)
        run_integration_tests
        ;;
    benchmark)
        run_benchmark_tests
        ;;
    quick)
        run_quick_tests
        ;;
    cpp)
        run_cpp_tests
        ;;
    display)
        run_display_tests
        ;;
    fixtures)
        run_fixture_tests
        ;;
    *)
        echo "Unknown category: $TEST_CATEGORY"
        exit 1
        ;;
esac

# Print summary
print_header "TEST SUMMARY"

echo -e "${GREEN}Passed:  $PASSED${NC}"
echo -e "${RED}Failed:  $FAILED${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
