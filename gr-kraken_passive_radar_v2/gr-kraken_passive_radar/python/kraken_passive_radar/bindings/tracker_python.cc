/*
 * pybind11 bindings for Multi-Target Tracker
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gnuradio/kraken_passive_radar/tracker.h>

namespace py = pybind11;

void bind_tracker(py::module& m)
{
    using tracker = gr::kraken_passive_radar::tracker;
    using track_t = gr::kraken_passive_radar::track_t;
    using track_status_t = gr::kraken_passive_radar::track_status_t;

    // Bind track status enum
    py::enum_<track_status_t>(m, "track_status_t")
        .value("TENTATIVE", track_status_t::TENTATIVE)
        .value("CONFIRMED", track_status_t::CONFIRMED)
        .value("COASTING", track_status_t::COASTING);

    // Bind track struct
    py::class_<track_t>(m, "track_t")
        .def(py::init<>())
        .def_readwrite("id", &track_t::id)
        .def_readwrite("status", &track_t::status)
        .def_readwrite("state", &track_t::state)
        .def_readwrite("covariance", &track_t::covariance)
        .def_readwrite("hits", &track_t::hits)
        .def_readwrite("misses", &track_t::misses)
        .def_readwrite("age", &track_t::age)
        .def_readwrite("score", &track_t::score)
        .def_readwrite("history", &track_t::history)
        .def_property_readonly("range_m", [](const track_t& t) { return t.state[0]; })
        .def_property_readonly("doppler_hz", [](const track_t& t) { return t.state[1]; })
        .def_property_readonly("range_rate", [](const track_t& t) { return t.state[2]; })
        .def_property_readonly("doppler_rate", [](const track_t& t) { return t.state[3]; });

    py::class_<tracker, gr::sync_block, std::shared_ptr<tracker>>(
        m, "tracker")

        .def_static("make",
            &tracker::make,
            py::arg("dt") = 0.1f,
            py::arg("process_noise_range") = 50.0f,
            py::arg("process_noise_doppler") = 5.0f,
            py::arg("meas_noise_range") = 100.0f,
            py::arg("meas_noise_doppler") = 2.0f,
            py::arg("gate_threshold") = 9.21f,
            py::arg("confirm_hits") = 3,
            py::arg("delete_misses") = 5,
            py::arg("max_tracks") = 50,
            py::arg("max_detections") = 100,
            R"doc(
Create multi-target tracker block.

Parameters:
    dt: Frame period in seconds
    process_noise_range: Process noise std dev for range (m)
    process_noise_doppler: Process noise std dev for Doppler (Hz)
    meas_noise_range: Measurement noise std dev for range (m)
    meas_noise_doppler: Measurement noise std dev for Doppler (Hz)
    gate_threshold: Chi-squared gate threshold for association
    confirm_hits: Hits required to confirm tentative track
    delete_misses: Misses before track deletion
    max_tracks: Maximum simultaneous tracks
    max_detections: Maximum detections per frame
)doc")

        .def("set_process_noise",
            &tracker::set_process_noise,
            py::arg("range"),
            py::arg("doppler"),
            "Set process noise std dev")

        .def("set_measurement_noise",
            &tracker::set_measurement_noise,
            py::arg("range"),
            py::arg("doppler"),
            "Set measurement noise std dev")

        .def("set_gate_threshold",
            &tracker::set_gate_threshold,
            py::arg("threshold"),
            "Set association gate threshold")

        .def("set_confirm_hits",
            &tracker::set_confirm_hits,
            py::arg("hits"),
            "Set hits to confirm track")

        .def("set_delete_misses",
            &tracker::set_delete_misses,
            py::arg("misses"),
            "Set misses to delete track")

        .def("get_tracks",
            &tracker::get_tracks,
            "Get all tracks")

        .def("get_confirmed_tracks",
            &tracker::get_confirmed_tracks,
            "Get confirmed tracks only")

        .def("get_num_tracks",
            &tracker::get_num_tracks,
            "Get total track count")

        .def("get_num_confirmed_tracks",
            &tracker::get_num_confirmed_tracks,
            "Get confirmed track count")

        .def("reset",
            &tracker::reset,
            "Reset tracker state");
}
