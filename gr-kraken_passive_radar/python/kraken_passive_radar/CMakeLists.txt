include(GrPython)

GR_PYTHON_INSTALL(
    FILES
        __init__.py
        clutter_cancellation.py
        doppler_processing.py
        range_doppler_widget.py
        eca_b_clutter_canceller.py
        krakensdr_source.py
    DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)
