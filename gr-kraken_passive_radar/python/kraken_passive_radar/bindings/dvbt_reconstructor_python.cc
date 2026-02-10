#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <gnuradio/kraken_passive_radar/dvbt_reconstructor.h>

namespace py = pybind11;

void bind_dvbt_reconstructor(py::module& m)
{
    using dvbt_reconstructor = gr::kraken_passive_radar::dvbt_reconstructor;

    py::class_<dvbt_reconstructor, gr::sync_block, std::shared_ptr<dvbt_reconstructor>>(
        m, "dvbt_reconstructor",
        R"pbdoc(
        Multi-Signal Reference Signal Reconstructor (Block B3)

        Reconstructs a clean reference signal using demodulation-remodulation
        with signal-specific error correction. Provides a "perfect" reference
        signal for passive radar processing, improving sensitivity by 10-20 dB
        in weak signal environments.

        Supported signal types:
          - "fm": FM Radio (88-108 MHz, US/worldwide)
          - "atsc3": ATSC 3.0 NextGen TV (US digital TV)
          - "dvbt": DVB-T (Europe/Australia digital TV)
          - "passthrough": No processing

        Args:
            signal_type: Signal type string ("fm", "atsc3", "dvbt", "passthrough")
            fm_deviation: FM deviation in Hz (FM only, default 75 kHz for US)
            enable_stereo: Process stereo (FM only)
            enable_pilot_regen: Regenerate 19 kHz pilot (FM only)
            audio_bw: Audio bandwidth in Hz (FM only)
            fft_size: OFDM FFT size (ATSC3/DVB-T only)
            guard_interval: Guard interval (ATSC3/DVB-T only)
            constellation: Modulation (DVB-T only)
            code_rate: FEC code rate (DVB-T only)
            pilot_pattern: Pilot pattern (ATSC3 only)
            enable_svd: Enable SVD pilot enhancement (OFDM only)
        )pbdoc")

        .def_static("make",
             &dvbt_reconstructor::make,
             py::arg("signal_type") = "passthrough",
             py::arg("fm_deviation") = 75e3,
             py::arg("enable_stereo") = true,
             py::arg("enable_pilot_regen") = true,
             py::arg("audio_bw") = 15e3,
             py::arg("fft_size") = 2048,
             py::arg("guard_interval") = 4,
             py::arg("constellation") = 2,
             py::arg("code_rate") = 2,
             py::arg("pilot_pattern") = 0,
             py::arg("enable_svd") = true,
             R"pbdoc(
             Create multi-signal reconstructor block

             Examples:
                 # FM Radio mode (US)
                 fm_recon = dvbt_reconstructor.make("fm")

                 # ATSC 3.0 mode (US NextGen TV)
                 atsc_recon = dvbt_reconstructor.make("atsc3", fft_size=8192)

                 # DVB-T mode (Europe)
                 dvbt_recon = dvbt_reconstructor.make("dvbt", fft_size=2048)

                 # Passthrough (no processing)
                 passthrough = dvbt_reconstructor.make("passthrough")
             )pbdoc")

        .def("set_enable_svd",
             &dvbt_reconstructor::set_enable_svd,
             py::arg("enable"),
             R"pbdoc(Enable/disable SVD pilot enhancement (OFDM signals only))pbdoc")

        .def("set_enable_pilot_regen",
             &dvbt_reconstructor::set_enable_pilot_regen,
             py::arg("enable"),
             R"pbdoc(Enable/disable pilot regeneration (FM only))pbdoc")

        .def("get_snr_estimate",
             &dvbt_reconstructor::get_snr_estimate,
             R"pbdoc(Get current SNR estimate in dB)pbdoc")

        .def("get_signal_type",
             &dvbt_reconstructor::get_signal_type,
             R"pbdoc(Get current signal type string)pbdoc")

        .def("set_signal_type",
             &dvbt_reconstructor::set_signal_type,
             py::arg("signal_type"),
             R"pbdoc(Set signal type (runtime switching))pbdoc");
}
