#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H

#include <gnuradio/kraken_passive_radar/eca_canceller.h>
#include <vector>
#include <complex>

namespace gr {
namespace kraken_passive_radar {

class eca_canceller_impl : public eca_canceller
{
private:
    int d_num_taps;
    float d_reg_factor;
    int d_num_surv;
    
    // Adaptive filter weights: num_surv x num_taps
    std::vector<std::vector<std::complex<float>>> d_weights;
    
    // History buffer for reference signal
    std::vector<std::complex<float>> d_ref_history;

public:
    eca_canceller_impl(int num_taps, float reg_factor, int num_surv);
    ~eca_canceller_impl();

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    void set_num_taps(int num_taps) override;
    void set_reg_factor(float reg_factor) override;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_IMPL_H */
