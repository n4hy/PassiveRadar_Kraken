/*
 * pybind11 bindings for AoA Estimator
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gnuradio/kraken_passive_radar/aoa_estimator.h>

namespace py = pybind11;

void bind_aoa_estimator(py::module& m)
{
    using aoa_estimator = gr::kraken_passive_radar::aoa_estimator;
    using aoa_result_t = gr::kraken_passive_radar::aoa_result_t;
    using array_type_t = gr::kraken_passive_radar::array_type_t;

    // Bind array type enum
    py::enum_<array_type_t>(m, "array_type_t")
        .value("ULA", array_type_t::ULA)
        .value("UCA", array_type_t::UCA);

    // Bind AoA result struct
    py::class_<aoa_result_t>(m, "aoa_result_t")
        .def(py::init<>())
        .def_readwrite("detection_id", &aoa_result_t::detection_id)
        .def_readwrite("aoa_deg", &aoa_result_t::aoa_deg)
        .def_readwrite("aoa_confidence", &aoa_result_t::aoa_confidence)
        .def_readwrite("spectrum_peak", &aoa_result_t::spectrum_peak)
        .def_readwrite("spectrum_width_deg", &aoa_result_t::spectrum_width_deg);

    py::class_<aoa_estimator, gr::sync_block, std::shared_ptr<aoa_estimator>>(
        m, "aoa_estimator")

        .def_static("make",
            &aoa_estimator::make,
            py::arg("num_elements") = 4,
            py::arg("d_lambda") = 0.5f,
            py::arg("n_angles") = 181,
            py::arg("min_angle_deg") = -90.0f,
            py::arg("max_angle_deg") = 90.0f,
            py::arg("array_type") = 0,
            py::arg("num_range_bins") = 256,
            py::arg("num_doppler_bins") = 64,
            py::arg("max_detections") = 100,
            R"doc(
Create AoA estimator block.

Parameters:
    num_elements: Number of array elements (typically 4)
    d_lambda: Element spacing in wavelengths (default 0.5)
    n_angles: Number of angles in scan
    min_angle_deg: Minimum scan angle (degrees)
    max_angle_deg: Maximum scan angle (degrees)
    array_type: ULA (0) or UCA (1)
    num_range_bins: Range bins in CAF
    num_doppler_bins: Doppler bins in CAF
    max_detections: Maximum detections per frame
)doc")

        .def("set_d_lambda",
            &aoa_estimator::set_d_lambda,
            py::arg("d_lambda"),
            "Set element spacing in wavelengths")

        .def("set_scan_range",
            &aoa_estimator::set_scan_range,
            py::arg("min_deg"),
            py::arg("max_deg"),
            "Set angular scan range")

        .def("set_array_type",
            &aoa_estimator::set_array_type,
            py::arg("type"),
            "Set array type (0=ULA, 1=UCA)")

        .def("get_aoa_results",
            &aoa_estimator::get_aoa_results,
            "Get AoA results from last frame")

        .def("get_spectrum",
            &aoa_estimator::get_spectrum,
            "Get angular spectrum for debugging");
}
