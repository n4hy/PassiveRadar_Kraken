#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations of binding functions
void bind_eca_canceller(py::module& m);
void bind_doppler_processor(py::module& m);
void bind_cfar_detector(py::module& m);
void bind_coherence_monitor(py::module& m);
void bind_detection_cluster(py::module& m);
void bind_tracker(py::module& m);
void bind_aoa_estimator(py::module& m);

PYBIND11_MODULE(kraken_passive_radar_python, m)
{
    m.doc() = "Kraken Passive Radar GNU Radio Module - Python Bindings";
    
    // Bind all blocks
    bind_eca_canceller(m);
    bind_doppler_processor(m);
    bind_cfar_detector(m);
    bind_coherence_monitor(m);
    bind_detection_cluster(m);
    bind_tracker(m);
    bind_aoa_estimator(m);
}
