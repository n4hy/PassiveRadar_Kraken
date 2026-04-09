#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <algorithm>

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#else
#define HAVE_OPTMATHKERNELS 0
#endif

#if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
#endif

using ComplexFloat = std::complex<float>;

/**
 * PolyphaseResampler - Efficient rational-rate sample rate converter
 *
 * Technique: Implements polyphase decomposition of an FIR lowpass filter
 * for L/M rational resampling. The prototype filter taps are partitioned
 * into L polyphase subfilters (one per interpolation phase), each reversed
 * for direct dot-product convolution. Uses Structure-of-Arrays (SoA) layout
 * for real/imaginary components and NEON-optimized dot products when available.
 * Maintains inter-block state via history buffers and phase tracking.
 */
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

    // Cached work buffers to avoid allocation in hot path
    std::vector<float> work_re;
    std::vector<float> work_im;

    // State
    int current_phase;
    int excess_input_advance;

    /**
     * dot_prod - SIMD-accelerated real-valued dot product
     *
     * Technique: Uses NEON dot product via OptMathKernels when available,
     * otherwise 8-way unrolled scalar accumulation.
     */
    static FORCE_INLINE float dot_prod(const float* a, const float* b, int n) {
#if HAVE_OPTMATHKERNELS
        return optmath::neon::neon_dot_f32(a, b, static_cast<std::size_t>(n));
#else
        float sum = 0.0f;
        int i = 0;
        for (; i <= n - 8; i += 8) {
            sum += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3] +
                   a[i+4] * b[i+4] + a[i+5] * b[i+5] + a[i+6] * b[i+6] + a[i+7] * b[i+7];
        }
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
#endif
    }

public:
    /**
     * PolyphaseResampler - Construct resampler with L/M ratio and prototype filter
     *
     * Technique: Decomposes prototype lowpass filter into L polyphase subfilters,
     * reverses each for dot-product convolution, and initializes SoA history buffers.
     */
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

    /**
     * process - Resample a block of complex IQ samples at L/M rate
     *
     * Technique: Prepends SoA history to input, iterates through polyphase
     * phases computing dot products between reversed subfilter taps and
     * input segments. Phase and input index advance by decimation/interpolation
     * ratio per output sample. Saves tail of work buffer as history for
     * next block continuity. Returns number of output samples produced.
     */
    int process(const ComplexFloat* input, int n_input, ComplexFloat* output, int max_output) {
        // 1. Construct Work Buffers (SoA) - reuse cached buffers
        int history_sz = static_cast<int>(history_re.size());
        int total_len = history_sz + n_input;

        // Resize cached buffers only if needed (avoids allocation in hot path)
        work_re.clear();
        work_im.clear();
        if (work_re.capacity() < static_cast<size_t>(total_len)) {
            work_re.reserve(total_len);
            work_im.reserve(total_len);
        }

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

            float val_re = dot_prod(re_ptr + start_idx, phase_taps.data(), taps_per_phase);
            float val_im = dot_prod(im_ptr + start_idx, phase_taps.data(), taps_per_phase);

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
    /** resampler_create - C API: Allocate polyphase resampler with given L/M ratio and filter taps */
    void* resampler_create(int interp, int decim, const float* taps, int num_taps) {
        if (interp <= 0 || decim <= 0 || !taps || num_taps <= 0) return nullptr;
        return new PolyphaseResampler(interp, decim, taps, num_taps);
    }

    /** resampler_destroy - C API: Free polyphase resampler instance */
    void resampler_destroy(void* ptr) {
        if (ptr) delete static_cast<PolyphaseResampler*>(ptr);
    }

    /**
     * resampler_process - C API: Resample interleaved complex float buffer at L/M rate
     *
     * Technique: Reinterprets float buffers as complex, delegates to
     * PolyphaseResampler::process. Returns number of output samples produced.
     */
    int resampler_process(void* ptr, const float* input, int n_input, float* output, int max_output) {
        if (!ptr || !input || !output || n_input <= 0 || max_output <= 0) return 0;
        PolyphaseResampler* obj = static_cast<PolyphaseResampler*>(ptr);
        const ComplexFloat* c_in = reinterpret_cast<const ComplexFloat*>(input);
        ComplexFloat* c_out = reinterpret_cast<ComplexFloat*>(output);
        return obj->process(c_in, n_input, c_out, max_output);
    }
}
