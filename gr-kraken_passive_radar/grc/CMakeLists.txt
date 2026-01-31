# Install GRC block definition YAML files
# THIS IS THE CRITICAL STEP THAT MAKES BLOCKS APPEAR IN GRC PALETTE

install(FILES
    kraken_passive_radar_krakensdr_source.block.yml
    kraken_passive_radar_eca_canceller.block.yml
    kraken_passive_radar_eca_b_clutter_canceller.block.yml
    kraken_passive_radar_clutter_canceller.block.yml
    kraken_passive_radar_doppler_processing.block.yml
    DESTINATION ${GRC_BLOCKS_DIR}
)

message(STATUS "GRC block YAMLs will be installed to: ${GRC_BLOCKS_DIR}")
