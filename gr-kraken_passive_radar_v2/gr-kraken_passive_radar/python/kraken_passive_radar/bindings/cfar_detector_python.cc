/*
 * pybind11 bindings for CFAR Detector
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>
#include <gnuradio/kraken_passive_radar/cfar_detector.h>

namespace py = pybind11;

void bind_cfar_detector(py::module& m)
{
    using cfar_detector = gr::kraken_passive_radar::cfar_detector;

    py::class_<cfar_detector, gr::sync_block, std::shared_ptr<cfar_detector>>(
        m, "cfar_detector")
        
        .def_static("make",
            &cfar_detector::make,
            py::arg("num_range_bins"),
            py::arg("num_doppler_bins"),
            py::arg("guard_cells_range") = 2,
            py::arg("guard_cells_doppler") = 2,
            py::arg("ref_cells_range") = 8,
            py::arg("ref_cells_doppler") = 8,
            py::arg("pfa") = 1e-6f,
            py::arg("cfar_type") = 0,
            py::arg("os_k") = 0,
            R"doc(
Create CFAR detector block.

Parameters:
    num_range_bins: Number of range bins
    num_doppler_bins: Number of Doppler bins
    guard_cells_range: Guard cells in range dimension (each side)
    guard_cells_doppler: Guard cells in Doppler dimension (each side)
    ref_cells_range: Reference cells in range dimension (each side)
    ref_cells_doppler: Reference cells in Doppler dimension (each side)
    pfa: Probability of false alarm
    cfar_type: 0=CA, 1=GO, 2=SO, 3=OS
    os_k: For OS-CFAR, k-th ordered statistic
)doc")
        
        .def("set_pfa",
            &cfar_detector::set_pfa,
            py::arg("pfa"),
            "Set probability of false alarm")
            
        .def("set_cfar_type",
            &cfar_detector::set_cfar_type,
            py::arg("cfar_type"),
            "Set CFAR type (0=CA, 1=GO, 2=SO, 3=OS)")
            
        .def("set_guard_cells",
            &cfar_detector::set_guard_cells,
            py::arg("range"),
            py::arg("doppler"),
            "Set guard cells")
            
        .def("set_ref_cells",
            &cfar_detector::set_ref_cells,
            py::arg("range"),
            py::arg("doppler"),
            "Set reference cells")
            
        .def("get_num_detections",
            &cfar_detector::get_num_detections,
            "Get number of detections from last frame");
}
