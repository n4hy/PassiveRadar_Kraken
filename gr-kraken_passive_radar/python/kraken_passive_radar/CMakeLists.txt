include(GrPython)

GR_PYTHON_INSTALL(
    FILES
        __init__.py
        doppler_processing.py
        eca_b_clutter_canceller.py
        krakensdr_source.py
    DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

install(
    FILES
        libkraken_eca_b_clutter_canceller.so
        libkraken_doppler_processing.so
        libkraken_aoa_processing.so
        libkraken_resampler.so
    DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)
