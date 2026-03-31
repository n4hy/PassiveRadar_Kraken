#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_CAF_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_CAF_IMPL_H

#include <gnuradio/kraken_passive_radar/caf.h>
#include <fftw3.h>

namespace gr {
namespace kraken_passive_radar {

class caf_impl : public caf
{
private:
    int d_n_samples;
    int d_fft_len;

    // FFTW plans (in-place)
    fftwf_plan d_fwd_ref;
    fftwf_plan d_fwd_surv;
    fftwf_plan d_inv;

    // FFTW-aligned buffers
    fftwf_complex* d_buf_ref;
    fftwf_complex* d_buf_surv;
    fftwf_complex* d_buf_prod;

public:
    caf_impl(int n_samples);
    ~caf_impl();

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif
