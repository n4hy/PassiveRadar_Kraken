# Build the ECA-B clutter canceller shared library for use via ctypes

# Enforce Release build for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -ffast-math -funroll-loops)
endif()

# Common Link Libraries
find_package(PkgConfig)
pkg_check_modules(FFTW3 fftw3f)

# 1. ECA-B
add_library(kraken_eca_b_clutter_canceller SHARED
    eca_b_clutter_canceller.cpp
)
target_link_libraries(kraken_eca_b_clutter_canceller m)
install(TARGETS kraken_eca_b_clutter_canceller LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

# 2. Conditioning
add_library(kraken_conditioning SHARED conditioning.cpp)
target_link_libraries(kraken_conditioning m)
install(TARGETS kraken_conditioning LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

# 3. Time Alignment (Needs FFTW)
if(FFTW3_FOUND)
    add_library(kraken_time_alignment SHARED time_alignment.cpp)
    target_link_libraries(kraken_time_alignment ${FFTW3_LIBRARIES} m)
    target_include_directories(kraken_time_alignment PRIVATE ${FFTW3_INCLUDE_DIRS})
    install(TARGETS kraken_time_alignment LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

    # 4. CAF Processing
    add_library(kraken_caf_processing SHARED caf_processing.cpp)
    target_link_libraries(kraken_caf_processing ${FFTW3_LIBRARIES} m)
    target_include_directories(kraken_caf_processing PRIVATE ${FFTW3_INCLUDE_DIRS})
    install(TARGETS kraken_caf_processing LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)

    # 5. Doppler Processing
    add_library(kraken_doppler_processing SHARED doppler_processing.cpp)
    target_link_libraries(kraken_doppler_processing ${FFTW3_LIBRARIES} m)
    target_include_directories(kraken_doppler_processing PRIVATE ${FFTW3_INCLUDE_DIRS})
    install(TARGETS kraken_doppler_processing LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)
else()
    message(WARNING "FFTW3f not found. Skipping FFT-dependent libraries.")
endif()

# 6. Backend (CFAR/Fusion)
add_library(kraken_backend SHARED backend.cpp)
target_link_libraries(kraken_backend m)
install(TARGETS kraken_backend LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar)
