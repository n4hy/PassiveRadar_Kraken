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

# Python site-packages path
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(GR_PYTHON_DIR ${PYTHON_SITE_PACKAGES})
message(STATUS "Python install dir: ${GR_PYTHON_DIR}")

# GRC blocks path
set(GRC_BLOCKS_DIR ${GR_DATA_DIR}/gnuradio/grc/blocks)
message(STATUS "GRC blocks dir: ${GRC_BLOCKS_DIR}")

# Subdirectories
add_subdirectory(include/gnuradio/kraken_passive_radar)
add_subdirectory(lib)
add_subdirectory(python/kraken_passive_radar)
add_subdirectory(grc)
