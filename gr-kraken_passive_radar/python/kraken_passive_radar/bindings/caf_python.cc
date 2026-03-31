#include <pybind11/pybind11.h>
#include <gnuradio/kraken_passive_radar/caf.h>
#include <gnuradio/sync_decimator.h>

namespace py = pybind11;

void bind_caf(py::module& m)
{
    using caf = gr::kraken_passive_radar::caf;

    py::class_<caf, gr::sync_decimator, std::shared_ptr<caf>>(m, "caf",
                   R"pbdoc(
                   Cross-Ambiguity Function (range profile) for passive radar.
                   FFT cross-correlation of reference and surveillance signals.
                   )pbdoc")

        .def(py::init(&caf::make),
             py::arg("n_samples") = 2048,
             R"pbdoc(Create CAF block)pbdoc");
}
