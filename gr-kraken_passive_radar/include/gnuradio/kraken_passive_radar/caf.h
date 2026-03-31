#ifndef INCLUDED_KRAKEN_PASSIVE_RADAR_CAF_H
#define INCLUDED_KRAKEN_PASSIVE_RADAR_CAF_H

#include <gnuradio/kraken_passive_radar/api.h>
#include <gnuradio/sync_decimator.h>

namespace gr {
namespace kraken_passive_radar {

/*!
 * \brief Cross-Ambiguity Function (range profile) for passive radar
 * \ingroup kraken_passive_radar
 *
 * FFT-based cross-correlation of reference and surveillance signals.
 * Produces one complex range profile vector per CPI.
 *
 * Input ports:
 *   - Port 0: Reference channel (complex scalar stream)
 *   - Port 1: Surveillance channel (complex scalar stream)
 *
 * Output port:
 *   - Port 0: Range profile (complex vector of n_samples)
 */
class KRAKEN_PASSIVE_RADAR_API caf : virtual public gr::sync_decimator
{
public:
    typedef std::shared_ptr<caf> sptr;

    /*!
     * \brief Create a new CAF block
     * \param n_samples Number of samples per CPI (range profile length)
     */
    static sptr make(int n_samples);
};

} // namespace kraken_passive_radar
} // namespace gr

#endif /* INCLUDED_KRAKEN_PASSIVE_RADAR_CAF_H */
