# pybind11 module
pybind11_add_module(kraken_passive_radar_python
    python_bindings.cc
    eca_canceller_python.cc
    doppler_processor_python.cc
    cfar_detector_python.cc
    coherence_monitor_python.cc
)

target_link_libraries(kraken_passive_radar_python PRIVATE
    gnuradio-kraken_passive_radar
    gnuradio::gnuradio-runtime
    gnuradio::gnuradio-blocks
)

target_include_directories(kraken_passive_radar_python PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install to Python site-packages
install(TARGETS kraken_passive_radar_python
    LIBRARY DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)
