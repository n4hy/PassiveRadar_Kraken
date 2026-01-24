# Source files
list(APPEND kraken_passive_radar_sources
    eca_canceller_impl.cc
)

# Build shared library
add_library(gnuradio-kraken_passive_radar SHARED ${kraken_passive_radar_sources})

target_link_libraries(gnuradio-kraken_passive_radar PUBLIC
    gnuradio::gnuradio-runtime
    gnuradio::gnuradio-blocks
)

target_include_directories(gnuradio-kraken_passive_radar PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set library version (optional but recommended)
set_target_properties(gnuradio-kraken_passive_radar PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# Install the library
install(TARGETS gnuradio-kraken_passive_radar
    EXPORT gnuradio-kraken_passive_radar-export
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
