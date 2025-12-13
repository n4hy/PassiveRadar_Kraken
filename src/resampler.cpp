#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "optmath/neon_kernels.hpp"

using ComplexFloat = std::complex<float>;

class PolyphaseResampler {
private:
    int interpolation;
    int decimation;
    int num_taps;
    int taps_per_phase;

    // Polyphase filter bank: poly_taps[phase][k]
    // Each phase vector is REVERSED to allow direct dot product with [idx, idx-1, ...]
    std::vector<std::vector<float>> poly_taps;

    // History: Split into Real and Imaginary for SoA processing
    std::vector<float> history_re;
    std::vector<float> history_im;

    // State
    int current_phase;
    int excess_input_advance;

public:
    PolyphaseResampler(int interp, int decim, const float* taps_in, int n_taps)
        : interpolation(interp), decimation(decim), num_taps(n_taps), current_phase(0), excess_input_advance(0) {

        taps_per_phase = (num_taps + interp - 1) / interp;

        // Pre-process taps into polyphase bank
        poly_taps.resize(interp);
        for (int p = 0; p < interp; ++p) {
            poly_taps[p].reserve(taps_per_phase);
            for (int k = 0; k < taps_per_phase; ++k) {
                int tap_idx = p + k * interp;
                if (tap_idx < num_taps) {
                    poly_taps[p].push_back(taps_in[tap_idx]);
                } else {
                    poly_taps[p].push_back(0.0f);
                }
            }
            // Reverse allow: dot(x[n, n-1...], h[0, 1...])
            std::reverse(poly_taps[p].begin(), poly_taps[p].end());
        }

        // History size
        int history_len = taps_per_phase + 2;
        history_re.resize(history_len, 0.0f);
        history_im.resize(history_len, 0.0f);
    }

    int process(const ComplexFloat* input, int n_input, ComplexFloat* output, int max_output) {
        // 1. Construct Work Buffers (SoA)
        int history_sz = history_re.size();
        int total_len = history_sz + n_input;

        std::vector<float> work_re;
        std::vector<float> work_im;
        work_re.reserve(total_len);
        work_im.reserve(total_len);

        work_re.insert(work_re.end(), history_re.begin(), history_re.end());
        work_im.insert(work_im.end(), history_im.begin(), history_im.end());

        // Unzip input to SoA
        for(int i=0; i<n_input; ++i) {
            work_re.push_back(input[i].real());
            work_im.push_back(input[i].imag());
        }

        int output_count = 0;
        int idx = history_sz + excess_input_advance;

        // Pointer access
        const float* re_ptr = work_re.data();
        const float* im_ptr = work_im.data();

        // 2. Process Loop
        while (idx < total_len) {
            if (output_count >= max_output) break;

            // We need samples ending at idx.
            // Start of segment: idx - taps_per_phase + 1
            int start_idx = idx - taps_per_phase + 1;

            if (start_idx < 0) {
                 idx++;
                 continue;
            }

            const std::vector<float>& phase_taps = poly_taps[current_phase];

            float val_re = optmath::neon::neon_dot_f32(re_ptr + start_idx, phase_taps.data(), taps_per_phase);
            float val_im = optmath::neon::neon_dot_f32(im_ptr + start_idx, phase_taps.data(), taps_per_phase);

            output[output_count++] = ComplexFloat(val_re, val_im);

            // Advance
            current_phase += decimation;
            int advance = current_phase / interpolation;
            current_phase %= interpolation;
            idx += advance;
        }

        // 3. Save State (excess advance)
        excess_input_advance = idx - total_len;

        // 4. Update History
        int start_copy = total_len - history_sz;
        if (start_copy < 0) start_copy = 0;

        std::copy(work_re.begin() + start_copy, work_re.end(), history_re.begin());
        std::copy(work_im.begin() + start_copy, work_im.end(), history_im.begin());

        return output_count;
    }
};

extern "C" {
    void* resampler_create(int interp, int decim, const float* taps, int num_taps) {
        return new PolyphaseResampler(interp, decim, taps, num_taps);
    }

    void resampler_destroy(void* ptr) {
        if (ptr) delete static_cast<PolyphaseResampler*>(ptr);
    }

    int resampler_process(void* ptr, const float* input, int n_input, float* output, int max_output) {
        if (!ptr) return 0;
        PolyphaseResampler* obj = static_cast<PolyphaseResampler*>(ptr);
        const ComplexFloat* c_in = reinterpret_cast<const ComplexFloat*>(input);
        ComplexFloat* c_out = reinterpret_cast<ComplexFloat*>(output);
        return obj->process(c_in, n_input, c_out, max_output);
    }
}
