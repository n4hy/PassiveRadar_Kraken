#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>

using Complex = std::complex<float>;
const float PI = 3.14159265358979323846f;
const float C_SPEED = 299792458.0f; // Speed of light

class AoAProcessor {
private:
    int num_antennas;
    float d_spacing; // Spacing between elements (usually in wavelengths or meters)

    // We assume Uniform Linear Array (ULA) for simplicity.
    // Ideally, we'd accept arbitrary geometry.
    // d_spacing is meters. If 0, assume normalized 0.5 lambda.

public:
    AoAProcessor(int n_ant, float spacing) : num_antennas(n_ant), d_spacing(spacing) {
    }

    // Bartlett Beamformer (Steering Vector Match)
    // Inputs:
    // - antenna_data: Array of [num_antennas] complex samples (single snapshot).
    // - lambda: Wavelength of the signal (c / freq).
    // - output_spectrum: Array of [n_angles] floats (power).
    // - n_angles: Resolution.
    //
    // Note: If d_spacing is 0, we assume d = 0.5 * lambda, so kd = pi.
    // Steering vector a(theta) = [1, exp(-j*k*d*sin(theta)), exp(-j*2*k*d*sin(theta))...]
    void compute_bartlett(const Complex* antenna_data, float lambda, float* output_spectrum, int n_angles) {
        float d = d_spacing;
        if (d <= 1e-9f) d = 0.5f * lambda; // default to half-wavelength

        float k = 2.0f * PI / lambda;

        // Scan from -90 to +90 degrees
        for (int i = 0; i < n_angles; ++i) {
            // Map index to angle -90..90
            float theta_deg = -90.0f + (180.0f * i / (n_angles - 1));
            float theta_rad = theta_deg * PI / 180.0f;

            // Calculate steering vector response: y = w^H * x
            // Bartlett weights w = a(theta) / N
            // P(theta) = |w^H * x|^2 = |sum(conj(a_m)*x_m)|^2 / N^2

            Complex sum_val(0.0f, 0.0f);

            for (int m = 0; m < num_antennas; ++m) {
                // Phase shift for element m: -k * m * d * sin(theta)
                float phase = -k * m * d * std::sin(theta_rad);
                Complex a_m = std::polar(1.0f, phase);

                // conj(a_m) * x_m = exp(-j*phase) * x_m
                // Actually conj(exp(j*phase)) = exp(-j*phase).
                // Wait, standard definition:
                // a(theta) = [1, e^{j psi}, e^{j 2 psi}...]
                // where psi = -k d sin(theta) (depends on coordinate system def)
                // Let's stick to: delay relative to first element.
                // If source is at angle theta from broadside:
                // path difference = m * d * sin(theta)
                // phase lag = k * path_diff
                // signal at m: x_m = s * exp(-j * k * m * d * sin(theta))
                // Steering vector a_m should match this.
                // So we project x onto a(theta).
                // response = sum( x_m * conj(a_m) )

                // conj(a_m) cancels the phase lag if theta is correct.
                // conj(exp(-j*...)) = exp(+j*...)
                Complex w_conj = std::polar(1.0f, -phase); // this is effectively conj(a_m)

                sum_val += antenna_data[m] * w_conj;
            }

            float p = std::norm(sum_val) / (num_antennas * num_antennas); // Normalize
            output_spectrum[i] = 10.0f * std::log10(p + 1e-12f);
        }
    }
};

extern "C" {
    void* aoa_create(int n_ant, float spacing) {
        return new AoAProcessor(n_ant, spacing);
    }

    void aoa_destroy(void* ptr) {
        if (ptr) delete static_cast<AoAProcessor*>(ptr);
    }

    // Process a single snapshot (spatial vector)
    // inputs: interleaved complex float array of size n_ant
    // lambda: signal wavelength
    // output: float array of size n_angles
    void aoa_process(void* ptr, const float* inputs, float lambda, float* output, int n_angles) {
        if (!ptr) return;
        AoAProcessor* obj = static_cast<AoAProcessor*>(ptr);
        const Complex* c_inputs = reinterpret_cast<const Complex*>(inputs);
        obj->compute_bartlett(c_inputs, lambda, output, n_angles);
    }
}
