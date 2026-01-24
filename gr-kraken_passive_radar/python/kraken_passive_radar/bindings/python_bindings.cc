#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations of binding functions
void bind_eca_canceller(py::module& m);

PYBIND11_MODULE(kraken_passive_radar_python, m)
{
    m.doc() = "Kraken Passive Radar GNU Radio Module - Python Bindings";
    
    // Bind all blocks
    bind_eca_canceller(m);
}
