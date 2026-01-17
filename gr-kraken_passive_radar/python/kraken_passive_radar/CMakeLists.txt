include(GrPython)

GR_PYTHON_INSTALL(
    FILES
        __init__.py
        doppler_processing.py
        eca_b_clutter_canceller.py
        krakensdr_source.py
        custom_blocks.py
    DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# Shared libraries are installed by src/CMakeLists.txt
# Do not install them here to avoid looking for non-existent source files
