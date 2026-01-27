cmake_minimum_required(VERSION 3.8)
project(gr-kraken_passive_radar CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GNU Radio (adjust version as needed)
find_package(Gnuradio "3.10" REQUIRED COMPONENTS runtime blocks fft filter)

# Find pybind11
find_package(pybind11 REQUIRED)

# Set installation directories
include(GNUInstallDirs)
set(GR_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(GR_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR})
set(GR_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(GR_DATA_DIR ${CMAKE_INSTALL_DATAROOTDIR})
set(GR_DOC_DIR ${GR_DATA_DIR}/doc)

# Find FFTW3 (required for Doppler processor and CAF)
find_package(PkgConfig)
pkg_check_modules(FFTW3 fftw3f)
if(NOT FFTW3_FOUND)
    find_library(FFTW3F_LIB fftw3f)
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    if(FFTW3F_LIB AND FFTW3_INCLUDE_DIR)
        set(FFTW3_FOUND TRUE)
        set(FFTW3_LIBRARIES ${FFTW3F_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "FFTW3f not found. Install: libfftw3-dev (Debian/Ubuntu), fftw3-devel (CentOS/RHEL)")
    endif()
endif()
message(STATUS "FFTW3 found: ${FFTW3_LIBRARIES}")

# Find Volk (required for GNU Radio)
find_package(Volk REQUIRED)

# Find where gnuradio's Python package is installed (critical for namespace integration)
execute_process(
    COMMAND python3 -c "import os, gnuradio; print(os.path.dirname(gnuradio.__file__))"
    OUTPUT_VARIABLE GNURADIO_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE GNURADIO_PYTHON_RESULT
)
# Validate the result
if(NOT GNURADIO_PYTHON_DIR OR GNURADIO_PYTHON_RESULT)
    message(FATAL_ERROR "Could not find gnuradio Python installation. Make sure GNU Radio is installed and python3 can import gnuradio.")
endif()
# Install our module INTO the gnuradio namespace
set(GR_PYTHON_DIR ${GNURADIO_PYTHON_DIR})
message(STATUS "GnuRadio Python location: ${GNURADIO_PYTHON_DIR}")
message(STATUS "Module will install to: ${GR_PYTHON_DIR}/kraken_passive_radar/")

# GRC blocks path
set(GRC_BLOCKS_DIR ${GR_DATA_DIR}/gnuradio/grc/blocks)
message(STATUS "GRC blocks dir: ${GRC_BLOCKS_DIR}")

# Subdirectories
add_subdirectory(include/gnuradio/kraken_passive_radar)
add_subdirectory(lib)
add_subdirectory(python/kraken_passive_radar)
add_subdirectory(grc)
