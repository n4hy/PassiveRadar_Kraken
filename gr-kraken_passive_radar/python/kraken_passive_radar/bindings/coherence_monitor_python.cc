/*
 * pybind11 bindings for Coherence Monitor
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>
#include <gnuradio/kraken_passive_radar/coherence_monitor.h>

namespace py = pybind11;

void bind_coherence_monitor(py::module& m)
{
    using coherence_monitor = gr::kraken_passive_radar::coherence_monitor;

    py::class_<coherence_monitor, gr::sync_block, std::shared_ptr<coherence_monitor>>(
        m, "coherence_monitor")
        
        .def_static("make",
            &coherence_monitor::make,
            py::arg("num_channels") = 5,
            py::arg("sample_rate") = 2.4e6f,
            py::arg("measure_interval_ms") = 1000.0f,
            py::arg("measure_duration_ms") = 10.0f,
            py::arg("corr_threshold") = 0.95f,
            py::arg("phase_threshold_deg") = 5.0f,
            R"doc(
Create coherence monitor block.

Parameters:
    num_channels: Number of input channels (typically 5)
    sample_rate: Sample rate in Hz
    measure_interval_ms: Interval between measurements (ms)
    measure_duration_ms: Duration of each measurement (ms)
    corr_threshold: Minimum acceptable correlation coefficient
    phase_threshold_deg: Maximum acceptable phase std dev (degrees)
)doc")
        
        .def("is_calibration_needed",
            &coherence_monitor::is_calibration_needed,
            "Check if calibration is needed")
            
        .def("get_correlation",
            &coherence_monitor::get_correlation,
            py::arg("channel"),
            "Get correlation coefficient for channel")
            
        .def("get_phase_offset",
            &coherence_monitor::get_phase_offset,
            py::arg("channel"),
            "Get phase offset for channel (radians)")
            
        .def("get_phase_variance",
            &coherence_monitor::get_phase_variance,
            py::arg("channel"),
            "Get phase variance for channel (radians)")
            
        .def("set_measure_interval",
            &coherence_monitor::set_measure_interval,
            py::arg("interval_ms"),
            "Set measurement interval (ms)")
            
        .def("set_corr_threshold",
            &coherence_monitor::set_corr_threshold,
            py::arg("threshold"),
            "Set correlation threshold")
            
        .def("set_phase_threshold",
            &coherence_monitor::set_phase_threshold,
            py::arg("threshold_deg"),
            "Set phase threshold (degrees)")
            
        .def("request_calibration",
            &coherence_monitor::request_calibration,
            "Manually request calibration")
            
        .def("acknowledge_calibration",
            &coherence_monitor::acknowledge_calibration,
            "Acknowledge calibration complete");
}
