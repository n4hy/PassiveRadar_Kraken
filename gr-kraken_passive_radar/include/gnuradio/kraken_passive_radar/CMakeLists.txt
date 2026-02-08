install(FILES
    api.h
    eca_canceller.h
    doppler_processor.h
    cfar_detector.h
    coherence_monitor.h
    detection_cluster.h
    aoa_estimator.h
    tracker.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gnuradio/kraken_passive_radar
)
