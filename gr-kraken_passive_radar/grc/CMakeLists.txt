# Install GRC block definitions for Kraken Passive Radar

install(
    FILES
        kraken_passive_radar_clutter_canceller.block.yml
        kraken_passive_radar_eca_b_clutter_canceller.block.yml
        kraken_passive_radar_krakensdr_source.block.yml
    DESTINATION share/gnuradio/grc/blocks
)
