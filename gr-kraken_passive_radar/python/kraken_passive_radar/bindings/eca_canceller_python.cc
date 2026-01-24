#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <gnuradio/kraken_passive_radar/eca_canceller.h>

namespace py = pybind11;

void bind_eca_canceller(py::module& m)
{
    using eca_canceller = gr::kraken_passive_radar::eca_canceller;
    
    py::class_<eca_canceller,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<eca_canceller>>(m, "eca_canceller", 
                   R"pbdoc(
                   ECA-B Clutter Canceller for Passive Radar
                   
                   Removes direct-path and multipath clutter from surveillance 
                   channels using the reference channel.
                   
                   Args:
                       num_taps: Number of filter taps (clutter delay spread)
                       reg_factor: Diagonal loading regularization factor
                       num_surv: Number of surveillance channels
                   )pbdoc")
        
        .def(py::init(&eca_canceller::make),
             py::arg("num_taps") = 128,
             py::arg("reg_factor") = 0.001f,
             py::arg("num_surv") = 4,
             R"pbdoc(Create ECA canceller block)pbdoc")
        
        .def("set_num_taps", 
             &eca_canceller::set_num_taps,
             py::arg("num_taps"),
             R"pbdoc(Set number of filter taps)pbdoc")
        
        .def("set_reg_factor",
             &eca_canceller::set_reg_factor,
             py::arg("reg_factor"),
             R"pbdoc(Set regularization factor)pbdoc");
}
