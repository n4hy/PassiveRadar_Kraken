#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_H

#include <gnuradio/kraken_passive_radar/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief ECA-B Clutter Canceller for Passive Radar
 * \ingroup kraken_passive_radar
 *
 * Removes direct-path and multipath clutter from surveillance channels
 * using the reference channel via Extensive Cancellation Algorithm.
 *
 * Input ports:
 *   - Port 0: Reference channel (complex)
 *   - Ports 1 to num_surv: Surveillance channels (complex)
 *
 * Output ports:
 *   - Ports 0 to num_surv-1: Clutter-cancelled surveillance channels (complex)
 */
class KRAKEN_PASSIVE_RADAR_API eca_canceller : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<eca_canceller> sptr;

    /*!
     * \brief Create a new ECA canceller instance
     * \param num_taps Number of filter taps (clutter delay spread in samples)
     * \param reg_factor Diagonal loading regularization factor (e.g., 0.001)
     * \param num_surv Number of surveillance channels (typically 4 for KrakenSDR)
     */
    static sptr make(int num_taps, float reg_factor, int num_surv);

    // Runtime parameter setters (callbacks from GRC)
    virtual void set_num_taps(int num_taps) = 0;
    virtual void set_reg_factor(float reg_factor) = 0;
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_ECA_CANCELLER_H */
