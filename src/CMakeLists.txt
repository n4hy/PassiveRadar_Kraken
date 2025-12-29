# Build the ECA-B clutter canceller shared library for use via ctypes

add_library(kraken_eca_b_clutter_canceller SHARED
    eca_b_clutter_canceller.cpp
    optmath/neon_kernels.cpp
)

# Install the shared library next to the Python kraken_passive_radar package
install(TARGETS kraken_eca_b_clutter_canceller
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)
