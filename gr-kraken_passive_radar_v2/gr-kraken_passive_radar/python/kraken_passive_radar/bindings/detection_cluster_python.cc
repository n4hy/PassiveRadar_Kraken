/*
 * pybind11 bindings for Detection Clustering
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gnuradio/kraken_passive_radar/detection_cluster.h>

namespace py = pybind11;

void bind_detection_cluster(py::module& m)
{
    using detection_cluster = gr::kraken_passive_radar::detection_cluster;
    using detection_t = gr::kraken_passive_radar::detection_t;

    // Bind the detection struct
    py::class_<detection_t>(m, "detection_t")
        .def(py::init<>())
        .def_readwrite("id", &detection_t::id)
        .def_readwrite("range_bin", &detection_t::range_bin)
        .def_readwrite("doppler_bin", &detection_t::doppler_bin)
        .def_readwrite("range_m", &detection_t::range_m)
        .def_readwrite("doppler_hz", &detection_t::doppler_hz)
        .def_readwrite("snr_db", &detection_t::snr_db)
        .def_readwrite("power_sum", &detection_t::power_sum)
        .def_readwrite("cluster_size", &detection_t::cluster_size)
        .def_readwrite("peak_range", &detection_t::peak_range)
        .def_readwrite("peak_doppler", &detection_t::peak_doppler);

    py::class_<detection_cluster, gr::sync_block, std::shared_ptr<detection_cluster>>(
        m, "detection_cluster")

        .def_static("make",
            &detection_cluster::make,
            py::arg("num_range_bins"),
            py::arg("num_doppler_bins"),
            py::arg("min_cluster_size") = 1,
            py::arg("max_cluster_extent") = 50,
            py::arg("range_resolution_m") = 600.0f,
            py::arg("doppler_resolution_hz") = 3.9f,
            py::arg("max_detections") = 100,
            R"doc(
Create detection clustering block.

Parameters:
    num_range_bins: Number of range bins
    num_doppler_bins: Number of Doppler bins
    min_cluster_size: Minimum cells to form valid detection
    max_cluster_extent: Maximum cluster size before split
    range_resolution_m: Range resolution in meters
    doppler_resolution_hz: Doppler resolution in Hz
    max_detections: Maximum detections per frame
)doc")

        .def("set_min_cluster_size",
            &detection_cluster::set_min_cluster_size,
            py::arg("size"),
            "Set minimum cluster size")

        .def("set_max_cluster_extent",
            &detection_cluster::set_max_cluster_extent,
            py::arg("extent"),
            "Set maximum cluster extent")

        .def("set_range_resolution",
            &detection_cluster::set_range_resolution,
            py::arg("res_m"),
            "Set range resolution in meters")

        .def("set_doppler_resolution",
            &detection_cluster::set_doppler_resolution,
            py::arg("res_hz"),
            "Set Doppler resolution in Hz")

        .def("get_detections",
            &detection_cluster::get_detections,
            "Get list of detections from last frame")

        .def("get_num_detections",
            &detection_cluster::get_num_detections,
            "Get number of detections from last frame");
}
