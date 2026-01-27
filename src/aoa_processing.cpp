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

    // 2D Bartlett (Azimuth + Elevation Scan)
    // Azimuth: -180..180
    // Elevation: 0..90
    //
    // Coordinate System:
    // Broadside (North) = Y axis.
    // East = X axis.
    // Up = Z axis.
    //
    // Azimuth (theta): Angle from North (Y) towards East (X).
    //   Range: -180 to 180. 0=North, 90=East, -90=West.
    // Elevation (phi): Angle from XY plane towards Z.
    //   Range: 0 to 90.
    //
    // Unit Vector u:
    //   x = cos(phi) * sin(theta)
    //   y = cos(phi) * cos(theta)
    //   z = sin(phi)
    //
    // Positions p[m] = (x,y,z).
    // Phase delay = k * dot(p, u).
    //
    // Support for 5 antennas:
    // If num_antennas == 5, we assume Ch0 is Reference at (0,0,0) and Ch1..4 are array.
    // Wait, usually Ref is directional.
    // If user selects "Include Ref", we use all 5.
    // Geometry definition:
    //   ULA: Ch0..N-1 along X axis centered? Or starting at 0?
    //   URA: Ch0..N-1 on grid.
    // If 5 elements and URA:
    //   Ref (Ch0) at (0,0,0).
    //   Ch1..4 form the square. But where?
    //   Let's put the square centered at (0,0,0) or offset?
    //   If Ref is "directional gain antenna pointed at emitting source", it is likely physically separated.
    //   Beamforming with it requires precise relative position.
    //   Since that is unknown, we will assume a default layout:
    //   Ref at (0,0,0).
    //   Square Array center at (0,0,0)? No, occluded.
    //   Let's assume Square Array is Ch1-4 as defined before: (0,0), (d,0), (0,d), (d,d).
    //   And Ref is Ch0 at (-d, -d) or similar?
    //   Without user input on geometry, "Include Ref" is best effort.
    //   I will place Ref at (0,0,0) and the URA starting at (0,0,0) too? Overlap?
    //   Let's assume Ref is Ch0 and URA is Ch1-4.
    //   URA points:
    //     1: (0,0,0)
    //     2: (d,0,0)
    //     3: (0,d,0)
    //     4: (d,d,0)
    //   If Ref is included, where is it?
    //   Maybe Ref is Ch0.
    //   I will just treat Ch0 as part of the list.
    //   ULA: 0, d, 2d, 3d, 4d.
    //   URA (5 elements?):
    //     0: (0,0)
    //     1: (d,0)
    //     2: (2d,0)
    //     3: (0,d)
    //     4: (d,d)
    //     Uneven.
    //
    // Let's stick to strict 4-element processing for URA 2x2 unless user forces 5.
    // If 5, I'll use a cross shape or just warn geometry is undefined.
    // Actually, "Reference antenna IS NOT part of the array... FOR AOA is can be selected".
    // This implies we take Ch0 + Ch1-4.
    // I will use Ch0 at (0,0,0) and shift the URA to start at (d, 0, 0)?
    // I'll assume Ref is at Center of the square?
    // Square: (-d/2, -d/2), (d/2, -d/2)... Ref at (0,0).
    // This makes sense for a "Four Square" with a central element.

    void compute_bartlett_3d(const Complex* antenna_data, float lambda, float* output_spectrum,
                             int n_az, int n_el, bool use_ref) {
        float d = d_spacing;
        if (d <= 1e-9f) d = 0.5f * lambda;

        float k = 2.0f * PI / lambda;

        // Define Positions
        // Max 5 antennas
        struct Pos { float x, y, z; };
        std::vector<Pos> positions;

        int start_idx = use_ref ? 0 : 1; // If not use_ref, skip Ch0
        // But antenna_data input might include Ch0?
        // Yes, input is always 5 channels from Doppler block.
        // We select which ones to use.

        // If ULA
        if (type == ULA) {
            // Elements along X
            // If use_ref (5 ants): 0, d, 2d, 3d, 4d
            // If !use_ref (4 ants): Ch1 at 0? Or Ch1 at d?
            // Usually we center the array or start at 0.
            // Let's start at 0 for the *active* subarray.
            int count = 0;
            for (int m = start_idx; m < 5; ++m) { // Ch0..4 or Ch1..4
                if (m >= num_antennas) break; // Safety
                positions.push_back({(float)count * d, 0.0f, 0.0f});
                count++;
            }
        }
        else if (type == URA) {
            // 2x2 Square
            // Ch1..4 are the square.
            // 1:(0,0), 2:(d,0), 3:(0,d), 4:(d,d)
            // If Ref (Ch0) is used, place it at center (0.5d, 0.5d)?
            // Or if Ref is Ch0, maybe Ch0 is corner?
            // "Two adjacent corners are lambda/2 apart".
            // That defines the square size d=lambda/2.

            // Standard layout for 2x2:
            // (0,0), (d,0)
            // (0,d), (d,d)

            // If Ref is included, I'll put it at center: (d/2, d/2, 0).

            // Channel Mapping:
            // Ch0: Ref
            // Ch1: BL
            // Ch2: BR
            // Ch3: TL
            // Ch4: TR

            // If use_ref: Include Ch0.
            if (use_ref) {
                // Ch0 (Ref) -> Center
                positions.push_back({0.5f * d, 0.5f * d, 0.0f});
            }

            // Ch1..4 (Square)
            positions.push_back({0.0f, 0.0f, 0.0f}); // Ch1
            positions.push_back({d,    0.0f, 0.0f}); // Ch2
            positions.push_back({0.0f, d,    0.0f}); // Ch3
            positions.push_back({d,    d,    0.0f}); // Ch4
        }

        // Output size: n_az * n_el
        // Scan loop
        for (int el_idx = 0; el_idx < n_el; ++el_idx) {
            float phi_deg = 90.0f * el_idx / (n_el - 1); // 0 to 90
            float phi_rad = phi_deg * PI / 180.0f;
            float cos_phi = std::cos(phi_rad);
            float sin_phi = std::sin(phi_rad);

            for (int az_idx = 0; az_idx < n_az; ++az_idx) {
                float theta_deg = -180.0f + 360.0f * az_idx / n_az; // -180 to 180
                float theta_rad = theta_deg * PI / 180.0f;

                // Unit vector
                // North (Y) reference
                float ux = cos_phi * std::sin(theta_rad);
                float uy = cos_phi * std::cos(theta_rad);
                float uz = sin_phi;

                Complex sum_val(0.0f, 0.0f);
                int active_ants = 0;

                // Iterate over *active* positions
                // We need to map positions back to input indices.
                // Positions list corresponds to:
                // ULA: [Ch_start, Ch_start+1 ...]
                // URA: [Ch0(opt), Ch1, Ch2, Ch3, Ch4]

                int p_idx = 0;

                // Map logical position index back to channel index
                // This is slightly messy. Let's iterate channels.

                if (type == ULA) {
                    for (int m = start_idx; m < 5; ++m) {
                        if (m >= num_antennas) break;
                        Pos p = positions[p_idx++];
                        float phase = k * (p.x * ux + p.y * uy + p.z * uz);
                        Complex w = std::polar(1.0f, -phase);
                        sum_val += antenna_data[m] * w;
                        active_ants++;
                    }
                } else {
                    // URA
                    if (use_ref) {
                        // Ch0
                        Pos p = positions[0];
                        float phase = k * (p.x * ux + p.y * uy + p.z * uz);
                        sum_val += antenna_data[0] * std::polar(1.0f, -phase);
                        active_ants++;
                        p_idx = 1;
                    }
                    // Ch1..4
                    for(int m=1; m<5; ++m) {
                        Pos p = positions[p_idx++];
                        float phase = k * (p.x * ux + p.y * uy + p.z * uz);
                        sum_val += antenna_data[m] * std::polar(1.0f, -phase);
                        active_ants++;
                    }
                }

                float p = std::norm(sum_val) / (active_ants * active_ants);

                // Output grid is [El][Az] or [Az][El]?
                // Let's do Row-Major: El * n_az + Az
                output_spectrum[el_idx * n_az + az_idx] = 10.0f * std::log10(p + 1e-12f);
            }
        }
    }
};

extern "C" {
    void* aoa_create(int n_ant, float spacing, int type) {
        if (n_ant <= 0) return nullptr;
        return new AoAProcessor(n_ant, spacing, type);
    }

    void aoa_destroy(void* ptr) {
        if (ptr) delete static_cast<AoAProcessor*>(ptr);
    }

    // 1D (Legacy/Simple) - Scans Azimuth -90 to +90 at Elevation 0
    void aoa_process(void* ptr, const float* inputs, float lambda, float* output, int n_angles) {
        if (!ptr || !inputs || !output || n_angles <= 0 || lambda <= 0.0f) return;
        AoAProcessor* obj = static_cast<AoAProcessor*>(ptr);
        // Temporary buffer for 3D output (full az/el scan is inefficient here, so just do 1D logic)
        // Re-implementing simplified 1D for compatibility:
        const Complex* antenna_data = reinterpret_cast<const Complex*>(inputs);
        float d = 0.5f * lambda; // default
        float k = 2.0f * PI / lambda;

        // Assume ULA Ch1..4 (No Ref) for legacy 1D
        int start_idx = 1;

        for (int i = 0; i < n_angles; ++i) {
            float theta_deg = -90.0f + (180.0f * i / (n_angles - 1));
            float theta_rad = theta_deg * PI / 180.0f;
            float ux = std::sin(theta_rad);
            float uy = std::cos(theta_rad);
            // El=0 -> uz=0

            Complex sum_val(0.0f, 0.0f);
            int count = 0;
            // Scan Ch1..4 as ULA
            for(int m=start_idx; m<5; ++m) {
                float px = (float)count * d;
                float phase = k * px * ux; // Dot product (y=0)
                sum_val += antenna_data[m] * std::polar(1.0f, -phase);
                count++;
            }
            float p = std::norm(sum_val) / (count * count);
            output[i] = 10.0f * std::log10(p + 1e-12f);
        }
    }

    // 3D Process
    // output size: n_az * n_el
    void aoa_process_3d(void* ptr, const float* inputs, float lambda, float* output,
                        int n_az, int n_el, bool use_ref) {
        if (!ptr || !inputs || !output || n_az <= 0 || n_el <= 0 || lambda <= 0.0f) return;
        AoAProcessor* obj = static_cast<AoAProcessor*>(ptr);
        const Complex* c_inputs = reinterpret_cast<const Complex*>(inputs);
        obj->compute_bartlett_3d(c_inputs, lambda, output, n_az, n_el, use_ref);
    }
}
