# Top-level CMakeLists.txt for PassiveRadar_Kraken
# Builds both the signal processing kernels and GNU Radio OOT module
#
# Standard usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#   sudo make install && sudo ldconfig
#
# Options:
#   -DBUILD_KERNELS=ON       Build C++ signal processing kernels (default: ON)
#   -DBUILD_OOT_MODULE=ON    Build GNU Radio OOT module (default: ON)
#   -DENABLE_GPU=ON          Build GPU-accelerated kernels if CUDA available (default: ON)
#   -DNATIVE_OPTIMIZATION=ON Build with -march=native (default: OFF, for portability)

cmake_minimum_required(VERSION 3.18)
project(PassiveRadar_Kraken VERSION 1.0.0 LANGUAGES C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Installation directories
include(GNUInstallDirs)

# Options
option(BUILD_KERNELS "Build C++ signal processing kernels" ON)
option(BUILD_OOT_MODULE "Build GNU Radio OOT module" ON)

# Detect Python site-packages for GR_PYTHON_DIR if not specified
if(NOT DEFINED GR_PYTHON_DIR)
    execute_process(
        COMMAND python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
        OUTPUT_VARIABLE GR_PYTHON_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT GR_PYTHON_DIR)
        set(GR_PYTHON_DIR "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages")
        message(WARNING "Could not detect Python path, using default: ${GR_PYTHON_DIR}")
    else()
        message(STATUS "Detected Python install path: ${GR_PYTHON_DIR}")
    endif()
endif()

# Export GR_PYTHON_DIR to subdirectories
set(GR_PYTHON_DIR "${GR_PYTHON_DIR}" CACHE PATH "Python site-packages directory")

# Build signal processing kernels
if(BUILD_KERNELS)
    message(STATUS "Building signal processing kernels...")
    add_subdirectory(src)
endif()

# Build GNU Radio OOT module (requires GNU Radio to be installed)
if(BUILD_OOT_MODULE)
    find_package(Gnuradio "3.10" QUIET COMPONENTS runtime)
    if(Gnuradio_FOUND)
        message(STATUS "GNU Radio found - building OOT module...")
        add_subdirectory(gr-kraken_passive_radar)
    else()
        message(WARNING "GNU Radio not found - skipping OOT module build")
        message(STATUS "Install GNU Radio 3.10+ to build the OOT module")
    endif()
endif()

# Summary
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "GR_PYTHON_DIR: ${GR_PYTHON_DIR}")
message(STATUS "BUILD_KERNELS: ${BUILD_KERNELS}")
message(STATUS "BUILD_OOT_MODULE: ${BUILD_OOT_MODULE}")
message(STATUS "===========================")
