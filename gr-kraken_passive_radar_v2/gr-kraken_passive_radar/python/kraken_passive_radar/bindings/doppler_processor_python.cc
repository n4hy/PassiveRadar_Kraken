/*
 * pybind11 bindings for Doppler Processor
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include <pybind11/pybind11.h>
#include <gnuradio/kraken_passive_radar/doppler_processor.h>

namespace py = pybind11;

void bind_doppler_processor(py::module& m)
{
    using doppler_processor = gr::kraken_passive_radar::doppler_processor;

    py::class_<doppler_processor, gr::sync_block, std::shared_ptr<doppler_processor>>(
        m, "doppler_processor")
        
        .def_static("make",
            &doppler_processor::make,
            py::arg("num_range_bins"),
            py::arg("num_doppler_bins"),
            py::arg("window_type") = 1,
            py::arg("output_power") = true,
            R"doc(
Create Doppler processor block.

Parameters:
    num_range_bins: Number of range bins per CPI (input vector length)
    num_doppler_bins: Number of CPIs to accumulate (Doppler FFT size)
    window_type: Window function (0=rect, 1=hamming, 2=hann, 3=blackman)
    output_power: If True, output |X|²; if False, output complex X
)doc")
        
        .def("set_num_doppler_bins",
            &doppler_processor::set_num_doppler_bins,
            py::arg("num_doppler_bins"),
            "Set number of Doppler bins (CPIs)")
            
        .def("set_window_type",
            &doppler_processor::set_window_type,
            py::arg("window_type"),
            "Set window type (0=rect, 1=hamming, 2=hann, 3=blackman)");
}
