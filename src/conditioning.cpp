#include <complex>
#include <cmath>
#include <vector>

using Complex = std::complex<float>;

class Conditioning {
    float gain;
    float rate;
    float target;
public:
    Conditioning(float alpha=1e-5f) : gain(1.0f), rate(alpha), target(1.0f) {}

    void process(Complex* in, int n) {
        for(int i=0; i<n; ++i) {
            float mag = std::abs(in[i]);
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
            gain += rate * err;

            if (gain < 1e-6f) gain = 1e-6f;
            if (gain > 1e6f) gain = 1e6f;

            in[i] *= gain;
        }
    }
};

extern "C" {
    void* cond_create(float rate) { return new Conditioning(rate); }
    void cond_destroy(void* p) { delete static_cast<Conditioning*>(p); }
    void cond_process(void* p, float* buf, int n) {
        static_cast<Conditioning*>(p)->process(reinterpret_cast<Complex*>(buf), n);
    }
}
