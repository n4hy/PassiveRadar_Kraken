# Install Python package files
install(FILES
    __init__.py
    krakensdr_source.py
    calibration_controller.py
    custom_blocks.py
    doppler_processing.py
    eca_b_clutter_canceller.py
    DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# Add bindings subdirectory
add_subdirectory(bindings)
