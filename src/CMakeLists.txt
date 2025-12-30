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

add_library(kraken_eca_b_clutter_canceller SHARED
    eca_b_clutter_canceller.cpp
)

# Explicitly link libm for math functions
target_link_libraries(kraken_eca_b_clutter_canceller m)

# Install the shared library next to the Python kraken_passive_radar package
install(TARGETS kraken_eca_b_clutter_canceller
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# Doppler Processing Library (FFTW3 dependency)
find_package(PkgConfig)
pkg_check_modules(FFTW3 fftw3f)

if(FFTW3_FOUND)
    add_library(kraken_doppler_processing SHARED
        doppler_processing.cpp
    )
    target_link_libraries(kraken_doppler_processing ${FFTW3_LIBRARIES} m)
    target_include_directories(kraken_doppler_processing PRIVATE ${FFTW3_INCLUDE_DIRS})

    install(TARGETS kraken_doppler_processing
        LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
    )
else()
    message(WARNING "FFTW3f not found. Doppler processing library will not be built.")
endif()
