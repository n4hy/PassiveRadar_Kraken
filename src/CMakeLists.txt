# Build the ECA-B clutter canceller shared library for use via ctypes
cmake_minimum_required(VERSION 3.8)
project(kraken_passive_radar_kernels C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enforce Release build for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output .so files to source directory for backward compatibility with tests
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Define GR_PYTHON_DIR if not set by parent project
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

# Add optimization flags (with option for portable builds)
# Default OFF for cross-platform safety (RPi5 aarch64 + x86_64)
option(NATIVE_OPTIMIZATION "Build with -march=native (non-portable)" OFF)
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -ffast-math -funroll-loops)
    if(NATIVE_OPTIMIZATION)
        add_compile_options(-march=native)
    else()
        # Portable tuning per architecture
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
            add_compile_options(-mcpu=cortex-a76)  # RPi5
        endif()
    endif()
    # Position-independent code for shared libs linking static OptMathKernels
    add_compile_options(-fPIC)
endif()

# Common Link Libraries
find_package(PkgConfig)
find_package(Threads REQUIRED)
pkg_check_modules(FFTW3 fftw3f)
pkg_check_modules(FFTW3_THREADS fftw3f_threads)

# Fallback for fftw3f_threads if pkg-config fails
if(NOT FFTW3_THREADS_FOUND)
    find_library(FFTW3F_THREADS_LIB fftw3f_threads)
    if(FFTW3F_THREADS_LIB)
        set(FFTW3_THREADS_FOUND TRUE)
        set(FFTW3_THREADS_LIBRARIES ${FFTW3F_THREADS_LIB})
        message(STATUS "Found fftw3f_threads via find_library: ${FFTW3F_THREADS_LIB}")
    endif()
endif()

# OptMathKernels for NEON acceleration (optional, requires >= 0.2.1)
find_package(OptMathKernels 0.2.1 QUIET)
if(OptMathKernels_FOUND)
    message(STATUS "OptMathKernels ${OptMathKernels_VERSION} found - enabling NEON optimization")
else()
    message(STATUS "OptMathKernels >= 0.2.1 not found - using scalar implementation")
endif()

# Fallback if pkg-config fails (e.g. if environment isn't set up)
if(NOT FFTW3_FOUND)
    find_library(FFTW3F_LIB fftw3f)
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    if(FFTW3F_LIB AND FFTW3_INCLUDE_DIR)
        set(FFTW3_FOUND TRUE)
        set(FFTW3_LIBRARIES ${FFTW3F_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    endif()
endif()

# 1. ECA-B clutter canceller (pure C++, no external dependencies)
add_library(kraken_eca_b_clutter_canceller SHARED
    eca_b_clutter_canceller.cpp
)
target_link_libraries(kraken_eca_b_clutter_canceller m)
install(TARGETS kraken_eca_b_clutter_canceller LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

# 2. Conditioning
add_library(kraken_conditioning SHARED conditioning.cpp)
target_link_libraries(kraken_conditioning m)
install(TARGETS kraken_conditioning LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

# Shared FFTW init library (ensures fftwf_init_threads called once per process)
if(FFTW3_FOUND)
    add_library(kraken_fftw_init SHARED fftw_init.cpp)
    target_link_libraries(kraken_fftw_init ${FFTW3_LIBRARIES} ${FFTW3_THREADS_LIBRARIES} Threads::Threads)
    target_include_directories(kraken_fftw_init PRIVATE ${FFTW3_INCLUDE_DIRS})
    install(TARGETS kraken_fftw_init LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

    # 3. Time Alignment
    add_library(kraken_time_alignment SHARED time_alignment.cpp)
    target_link_libraries(kraken_time_alignment kraken_fftw_init ${FFTW3_LIBRARIES} ${FFTW3_THREADS_LIBRARIES} Threads::Threads m)
    target_include_directories(kraken_time_alignment PRIVATE ${FFTW3_INCLUDE_DIRS})
    install(TARGETS kraken_time_alignment LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

    # 4. CAF Processing (with optional OptMathKernels NEON optimization)
    add_library(kraken_caf_processing SHARED caf_processing.cpp)
    target_link_libraries(kraken_caf_processing kraken_fftw_init ${FFTW3_LIBRARIES} ${FFTW3_THREADS_LIBRARIES} Threads::Threads m)
    target_include_directories(kraken_caf_processing PRIVATE ${FFTW3_INCLUDE_DIRS})
    if(OptMathKernels_FOUND)
        target_compile_definitions(kraken_caf_processing PRIVATE HAVE_OPTMATHKERNELS=1)
        target_link_libraries(kraken_caf_processing OptMathKernels::OptMathKernels)
    endif()
    install(TARGETS kraken_caf_processing LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

    # 5. Doppler Processing
    add_library(kraken_doppler_processing SHARED doppler_processing.cpp)
    target_link_libraries(kraken_doppler_processing kraken_fftw_init ${FFTW3_LIBRARIES} ${FFTW3_THREADS_LIBRARIES} Threads::Threads m)
    target_include_directories(kraken_doppler_processing PRIVATE ${FFTW3_INCLUDE_DIRS})
    install(TARGETS kraken_doppler_processing LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)
else()
    message(FATAL_ERROR "FFTW3f library not found. Install: libfftw3-dev (Debian/Ubuntu), fftw3-devel (CentOS/RHEL), fftw (Fedora/Arch)")
endif()

# 6. Backend (CFAR/Fusion)
add_library(kraken_backend SHARED backend.cpp)
target_link_libraries(kraken_backend m)
install(TARGETS kraken_backend LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

# 7. AoA Processing
add_library(kraken_aoa_processing SHARED aoa_processing.cpp)
target_link_libraries(kraken_aoa_processing m)
install(TARGETS kraken_aoa_processing LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

# 8. Resampler
add_library(kraken_resampler SHARED resampler.cpp)
target_link_libraries(kraken_resampler m)
install(TARGETS kraken_resampler LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

# 9. NLMS Clutter Canceller
add_library(kraken_nlms_clutter_canceller SHARED nlms_clutter_canceller.cpp)
target_compile_definitions(kraken_nlms_clutter_canceller PRIVATE LIBRARY_BUILD)
target_link_libraries(kraken_nlms_clutter_canceller m)
install(TARGETS kraken_nlms_clutter_canceller LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)
