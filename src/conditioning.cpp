#include <complex>
#include <cmath>
#include <vector>

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#endif

using Complex = std::complex<float>;

class Conditioning {
    float gain;
    float rate;
    float target;
    std::vector<float> mag_buf;  // Pre-allocated magnitude buffer
    std::vector<float> re_buf, im_buf;  // Deinterleave buffers
public:
    Conditioning(float alpha=1e-5f) : gain(1.0f), rate(alpha), target(1.0f) {}

    void process(Complex* in, int n) {
#ifdef HAVE_OPTMATHKERNELS
        // Batch-compute all magnitudes using NEON
        if ((int)mag_buf.size() < n) {
            mag_buf.resize(n);
            re_buf.resize(n);
            im_buf.resize(n);
        }
        // Deinterleave complex data
        for (int i = 0; i < n; ++i) {
            re_buf[i] = in[i].real();
            im_buf[i] = in[i].imag();
        }
        optmath::neon::neon_complex_magnitude_f32(mag_buf.data(), re_buf.data(), im_buf.data(), n);

        for (int i = 0; i < n; ++i) {
            float mag = mag_buf[i];
#else
        for(int i=0; i<n; ++i) {
            float mag = std::abs(in[i]);
#endif
            // Update gain slowly
            // target = |x| * gain
            // err = target_level - |x| * gain
            // gain += rate * err ?
            // Usually AGC updates based on output power or input power.
            // Let's use simple error feedback.
            // err = target - mag * gain;
            // gain += rate * err;

            // Or simpler: update gain based on mag.
            // err = target - mag; (if mag > target, reduce gain?)
            // No, gain scales input.
            // Desired: mag * gain -> target.
            // gain_new = gain + rate * (target - mag * gain).

            float current_level = mag * gain;
            float err = target - current_level;
            // Limit gain increase rate on very low signal to prevent spike on signal return
            if (mag < 1e-4f) {
                // Decay-only mode: slowly reduce gain toward 1.0 during silence
                gain += rate * (1.0f - gain) * 0.01f;
            } else {
                gain += rate * err;
            }

            if (gain < 1e-6f) gain = 1e-6f;
            if (gain > 1e3f) gain = 1e3f;  // 60 dB max gain (was 1e6 = 120 dB)

            in[i] *= gain;
        }
    }
};

extern "C" {
    void* cond_create(float rate) { return new Conditioning(rate); }
    void cond_destroy(void* p) { if (p) delete static_cast<Conditioning*>(p); }
    void cond_process(void* p, float* buf, int n) {
        if (!p || !buf || n <= 0) return;
        static_cast<Conditioning*>(p)->process(reinterpret_cast<Complex*>(buf), n);
    }
}
