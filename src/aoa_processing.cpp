#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>

using Complex = std::complex<float>;
const float PI = 3.14159265358979323846f;

enum ArrayType {
    ULA = 0,    // Uniform Linear Array (1D)
    URA = 1     // Uniform Rectangular Array (2x2 Square)
};

class AoAProcessor {
private:
    int num_antennas;
    float d_spacing; // Spacing in meters. If 0, assumes lambda/2.
    ArrayType type;

public:
    AoAProcessor(int n_ant, float spacing, int array_type)
        : num_antennas(n_ant), d_spacing(spacing), type(static_cast<ArrayType>(array_type)) {
    }

    // 1D Bartlett (Azimuth Scan)
    // Supports ULA (native) and URA (projected to Azimuth?)
    // For URA 2x2: Elements at (0,0), (d,0), (0,d), (d,d).
    // If scanning Azimuth only, we assume Elevation=0? Or broadside.
    // Let's implement generic 2D scanning but "aoa_process" usually implies 1D spectrum for simple plots.
    // If URA is selected, we scan Azimuth at Elevation=0.

    // Steering Vector Calculation
    // Azimuth (theta), Elevation (phi)
    // ULA (x-axis): pos = [m*d, 0, 0]
    // URA (x-y plane):
    //   0: (0, 0)
    //   1: (d, 0)
    //   2: (0, d)
    //   3: (d, d)
    // Phase delay: k * (x*cos(theta)*cos(phi) + y*sin(theta)*cos(phi) + z*sin(phi))?
    // Coordinate system:
    //   Azimuth theta defined from X-axis? Or Broadside (Y)?
    //   Standard Physics: Theta from Z-axis?
    //   Radar Standard: Azimuth from North (Y).
    // Let's use:
    //   ULA along X-axis. Broadside is Y-axis (90 deg) or Z-axis.
    //   Let's stick to standard "Angle from Broadside".
    //   Broadside = 0 deg. Endfire = +/- 90 deg.
    //   This means array axis is normal to Look vector at 0 deg.
    //   So array along X. Look vector in Y direction is 0 deg.
    //   Vector u = [sin(theta), cos(theta), 0].
    //   Pos p = [x, y, 0].
    //   Phase = k * dot(p, u).

    // ULA positions: (0,0), (d,0), (2d,0), (3d,0).
    // Phase m: k * (m*d) * sin(theta). (Matches previous impl).

    // URA positions:
    // 0: (0,0) -> 0
    // 1: (d,0) -> k*d*sin(theta)
    // 2: (0,d) -> k*d*cos(theta) (because y-component projects onto cos(theta) direction?)
    // Wait.
    // Look vector v = [sin(theta), cos(theta)].
    // Ant 2 at (0,d). Dot = d*cos(theta).
    // Ant 3 at (d,d). Dot = d*sin(theta) + d*cos(theta).

    // This assumes 2D plane wave in XY plane (Elevation=0).

    void compute_bartlett(const Complex* antenna_data, float lambda, float* output_spectrum, int n_angles) {
        float d = d_spacing;
        if (d <= 1e-9f) d = 0.5f * lambda;

        float k = 2.0f * PI / lambda;

        for (int i = 0; i < n_angles; ++i) {
            // Map index to angle -90..90 (from broadside)
            float theta_deg = -90.0f + (180.0f * i / (n_angles - 1));
            float theta_rad = theta_deg * PI / 180.0f;

            // Unit vector direction
            // Broadside (0 deg) -> Y axis.
            // Theta -> Rotation around Z.
            // u_x = sin(theta)
            // u_y = cos(theta)
            float ux = std::sin(theta_rad);
            float uy = std::cos(theta_rad);

            Complex sum_val(0.0f, 0.0f);

            for (int m = 0; m < num_antennas; ++m) {
                float px = 0.0f;
                float py = 0.0f;

                if (type == ULA) {
                    // Linear along X
                    px = m * d;
                    py = 0.0f;
                } else if (type == URA) {
                    // 2x2 Square
                    // 0: (0,0), 1: (d,0), 2: (0,d), 3: (d,d)
                    // (Assuming row-major mapping of channels)
                    int row = m / 2; // 0, 0, 1, 1
                    int col = m % 2; // 0, 1, 0, 1
                    // Actually:
                    // m=0: (0,0)
                    // m=1: (d,0) ? Or (0,d)?
                    // "Adjacent corners lambda/2".
                    // Let's map:
                    // Ch0: (0,0)
                    // Ch1: (d,0)
                    // Ch2: (0,d)
                    // Ch3: (d,d)
                    // User must connect cables accordingly.
                    px = (m % 2) * d;
                    py = (m / 2) * d;
                }

                // Phase delay = k * dot(p, u)
                // If wavefront comes from u, element at p sees phase LEAD or LAG?
                // Standard: exp(j * k * dot(r, u)).
                // If source is far field at u.
                // Signal arriving at p relative to origin (0):
                // Travel distance diff = dot(p, u).
                // If dot > 0, p is closer to source? No.
                // u points TO source? Or FROM source?
                // Standard DOA u points TO source.
                // p is closer to source by projection.
                // So p receives EARLIER. Phase LEAD.
                // x(t) = exp(j(wt + phi)). Earlier means t -> t + delta.
                // So exp(j*k*dist).

                float phase_shift = k * (px * ux + py * uy);

                // Weight: w = exp(-j * phase_shift) to cancel it.
                // We sum: x[m] * exp(-j * phase)
                Complex w = std::polar(1.0f, -phase_shift);

                sum_val += antenna_data[m] * w;
            }

            float p = std::norm(sum_val) / (num_antennas * num_antennas);
            output_spectrum[i] = 10.0f * std::log10(p + 1e-12f);
        }
    }
};

extern "C" {
    void* aoa_create(int n_ant, float spacing, int type) {
        return new AoAProcessor(n_ant, spacing, type);
    }

    void aoa_destroy(void* ptr) {
        if (ptr) delete static_cast<AoAProcessor*>(ptr);
    }

    void aoa_process(void* ptr, const float* inputs, float lambda, float* output, int n_angles) {
        if (!ptr) return;
        AoAProcessor* obj = static_cast<AoAProcessor*>(ptr);
        const Complex* c_inputs = reinterpret_cast<const Complex*>(inputs);
        obj->compute_bartlett(c_inputs, lambda, output, n_angles);
    }
}
