#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#endif

// Use float for compatibility with np.complex64 (GNU Radio standard)
using Complex = std::complex<float>;

/**
 * AdaptiveSignalConditioner - NLMS adaptive filter for clutter cancellation
 *
 * Technique: Implements Normalized Least Mean Squares (NLMS) adaptive filtering
 * to cancel direct-path interference from surveillance channels. Uses a circular
 * buffer for reference signal history and per-sample weight updates with
 * energy-normalized step size: w[k] += (mu * conj(e)) / (||r||^2 + eps) * r[k].
 * Supports multi-channel processing and optionally uses NEON-optimized
 * complex operations via OptMathKernels.
 */
class AdaptiveSignalConditioner {
private:
    size_t filter_length;
    float mu;      // Step size (learning rate)
    float epsilon; // Regularization term

    // Weights for each surveillance channel.
    std::vector<std::vector<Complex>> weights;

    // Circular buffer for the Reference signal R(t)
    std::vector<Complex> r_history;
    size_t r_history_pos;  // Points to most recent sample (index 0 in logical view)

public:
    /**
     * AdaptiveSignalConditioner - Construct NLMS filter with given length and step size
     *
     * Technique: Initializes circular buffer for reference history and sets
     * NLMS learning rate (mu) and regularization epsilon.
     */
    AdaptiveSignalConditioner(size_t len, float step_size = 0.1)
        : filter_length(len), mu(step_size), epsilon(1e-8f), r_history_pos(0) {
        // Initialize history buffer
        r_history.resize(filter_length, Complex(0.0f, 0.0f));
    }

    /**
     * process - Multi-channel NLMS processing returning correlated component estimates
     *
     * Technique: For each sample, updates reference history, computes filter
     * output y = w^H * r for each surveillance channel, calculates error
     * e = d - y, and updates weights via NLMS rule. Returns the estimated
     * correlated component (y) for each channel.
     */
    std::vector<std::vector<Complex>> process(const std::vector<std::vector<Complex>*>& signal_ptrs) {
        if (signal_ptrs.empty()) throw std::runtime_error("No signals provided.");

        const std::vector<Complex>& R = *signal_ptrs[0];
        size_t num_samples = R.size();
        size_t num_surveillance = signal_ptrs.size() - 1;

        if (weights.size() != num_surveillance) {
            weights.assign(num_surveillance, std::vector<Complex>(filter_length, Complex(0.0f, 0.0f)));
        }

        std::vector<std::vector<Complex>> output_signals(num_surveillance);
        for(auto& vec : output_signals) vec.reserve(num_samples);

        for (size_t n = 0; n < num_samples; ++n) {
            update_history(R[n]);
            
            float r_energy = get_history_energy();

            for (size_t ch = 0; ch < num_surveillance; ++ch) {
                const std::vector<Complex>& S = *signal_ptrs[ch + 1];
                Complex d = S[n]; 

                Complex y = filter(ch);
                Complex e = d - y;

                output_signals[ch].push_back(y);

                update_weights(ch, e, r_energy);
            }
        }
        return output_signals;
    }

    /**
     * process_cancellation_single - Single-channel NLMS cancellation (C API backend)
     *
     * Technique: Per-sample NLMS processing for one surveillance channel.
     * Outputs the error signal e = d - y (clutter-cancelled surveillance).
     * Used by the C API nlms_process function.
     */
    void process_cancellation_single(const Complex* ref_in, const Complex* surv_in, Complex* out_err, size_t n_samples) {
         if (n_samples == 0) return;
         if (weights.empty()) {
             weights.assign(1, std::vector<Complex>(filter_length, Complex(0.0f, 0.0f)));
         }

         for (size_t n = 0; n < n_samples; ++n) {
             update_history(ref_in[n]);

             float r_energy = get_history_energy();

             Complex d = surv_in[n];
             Complex y = filter(0);
             Complex e = d - y;

             out_err[n] = e;

             update_weights(0, e, r_energy);
         }
    }

private:
    /** update_history - Insert new sample into circular reference buffer (O(1)) */
    void update_history(Complex val) {
        // Circular buffer: O(1) instead of O(n) shift
        r_history_pos = (r_history_pos == 0) ? filter_length - 1 : r_history_pos - 1;
        r_history[r_history_pos] = val;
    }

    /** history_at - Access history element k (0 = most recent) from circular buffer */
    inline Complex history_at(size_t k) const {
        return r_history[(r_history_pos + k) % filter_length];
    }

    /**
     * get_history_energy - Compute total energy of reference history buffer
     *
     * Technique: Sum of |z|^2 over all history elements. Uses NEON dot product
     * on interleaved re/im data when OptMathKernels is available.
     */
    float get_history_energy() {
#ifdef HAVE_OPTMATHKERNELS
        // r_history is contiguous; treat as interleaved re/im pairs
        // |z|^2 = re^2 + im^2, sum over all elements
        const float* data = reinterpret_cast<const float*>(r_history.data());
        return optmath::neon::neon_dot_f32(data, data, filter_length * 2);
#else
        float r_energy = 0.0f;
        for (const auto& val : r_history) {
            r_energy += std::norm(val);
        }
        return r_energy;
#endif
    }

    /**
     * linearize_history - Copy circular buffer to contiguous scratch buffer
     *
     * Technique: Linearizes the circular reference history into a contiguous
     * array suitable for NEON vectorized operations.
     */
    void linearize_history(std::vector<Complex>& scratch) const {
        scratch.resize(filter_length);
        for (size_t k = 0; k < filter_length; ++k) {
            scratch[k] = r_history[(r_history_pos + k) % filter_length];
        }
    }

    /**
     * filter - Compute FIR filter output y = sum(conj(w[k]) * h[k])
     *
     * Technique: Complex inner product between weight vector and reference
     * history. Uses NEON complex dot product when OptMathKernels is available,
     * otherwise scalar loop with std::conj.
     */
    Complex filter(size_t ch) {
#ifdef HAVE_OPTMATHKERNELS
        // Linearize history for vectorized dot product
        linearize_history(scratch_buf);
        // sum(conj(w) * h) = conj(sum(w * conj(h))) = conj of neon_complex_dot(w, h)
        // neon_complex_dot computes sum(a * conj(b)), so dot(h, w) = sum(h * conj(w))
        float dot_re, dot_im;
        const float* h_re = reinterpret_cast<const float*>(scratch_buf.data());
        const float* w_re = reinterpret_cast<const float*>(weights[ch].data());
        // Deinterleave for neon_complex_dot_f32
        std::vector<float> hr(filter_length), hi(filter_length), wr(filter_length), wi(filter_length);
        for (size_t k = 0; k < filter_length; ++k) {
            hr[k] = scratch_buf[k].real(); hi[k] = scratch_buf[k].imag();
            wr[k] = weights[ch][k].real(); wi[k] = weights[ch][k].imag();
        }
        optmath::neon::neon_complex_dot_f32(&dot_re, &dot_im, hr.data(), hi.data(), wr.data(), wi.data(), filter_length);
        // neon_complex_dot computes sum(a * conj(b)) = sum(h * conj(w))
        return Complex(dot_re, dot_im);
#else
        Complex y = 0.0f;
        for (size_t k = 0; k < filter_length; ++k) {
            y += std::conj(weights[ch][k]) * history_at(k);
        }
        return y;
#endif
    }

    /**
     * update_weights - NLMS weight update rule
     *
     * Technique: w[k] += (mu * conj(e) / (||r||^2 + eps)) * r[k].
     * Normalizes step by reference energy to ensure convergence regardless
     * of input power level. Skips update on near-zero energy to prevent
     * numerical instability. Uses NEON-optimized complex scale when available.
     */
    void update_weights(size_t ch, Complex e, float r_energy) {
        // Skip update if energy is too low to avoid numerical instability
        if (r_energy < epsilon) return;
        Complex step_scale = (mu * std::conj(e)) / (r_energy + epsilon);
        // Check for NaN/Inf before applying
        if (!std::isfinite(step_scale.real()) || !std::isfinite(step_scale.imag())) return;
#ifdef HAVE_OPTMATHKERNELS
        // Linearize history, scale, and add to weights
        linearize_history(scratch_buf);
        std::vector<float> hr(filter_length), hi(filter_length), sr(filter_length), si(filter_length);
        for (size_t k = 0; k < filter_length; ++k) {
            hr[k] = scratch_buf[k].real(); hi[k] = scratch_buf[k].imag();
        }
        optmath::neon::neon_complex_scale_f32(sr.data(), si.data(), hr.data(), hi.data(),
                                               step_scale.real(), step_scale.imag(), filter_length);
        for (size_t k = 0; k < filter_length; ++k) {
            weights[ch][k] += Complex(sr[k], si[k]);
        }
#else
        for (size_t k = 0; k < filter_length; ++k) {
            weights[ch][k] += step_scale * history_at(k);
        }
#endif
    }

    mutable std::vector<Complex> scratch_buf; // Scratch for linearized history
};

extern "C" {
    /** nlms_create - C API: Allocate NLMS adaptive filter with given taps and step size */
    void* nlms_create(int taps, float mu) {
        return new AdaptiveSignalConditioner(static_cast<size_t>(taps), mu);
    }

    /** nlms_destroy - C API: Free NLMS filter instance */
    void nlms_destroy(void* ptr) {
        if (ptr) delete static_cast<AdaptiveSignalConditioner*>(ptr);
    }

    /**
     * nlms_process - C API: Run single-channel NLMS cancellation on interleaved complex buffers
     *
     * Technique: Reinterprets float buffers as complex, delegates to
     * AdaptiveSignalConditioner::process_cancellation_single.
     */
    void nlms_process(void* ptr, const float* ref_in, const float* surv_in, float* out_err, int n_samples) {
        if (!ptr || !ref_in || !surv_in || !out_err || n_samples <= 0) return;
        AdaptiveSignalConditioner* obj = static_cast<AdaptiveSignalConditioner*>(ptr);

        const Complex* c_ref = reinterpret_cast<const Complex*>(ref_in);
        const Complex* c_surv = reinterpret_cast<const Complex*>(surv_in);
        Complex* c_out = reinterpret_cast<Complex*>(out_err);

        obj->process_cancellation_single(c_ref, c_surv, c_out, static_cast<size_t>(n_samples));
    }
}

// ==========================================
//              UNIT TEST (MAIN)
// ==========================================
#ifndef LIBRARY_BUILD

/** get_power - Compute average power of a complex signal vector (unit test helper) */
float get_power(const std::vector<Complex>& v) {
    float p = 0;
    for(auto c : v) p += std::norm(c);
    return p / v.size();
}

/**
 * generate_test_data - Create synthetic reference + surveillance test signals
 *
 * Technique: Generates reference signal R as Gaussian noise, creates
 * surveillance S as delayed/phase-rotated R plus strong clutter tone
 * (simulating direct-path interference) plus additive noise. IdealS
 * holds the clutter-free signal for EVM computation.
 */
void generate_test_data(size_t N, std::vector<Complex>& R, std::vector<Complex>& S, std::vector<Complex>& IdealS) {
    R.resize(N);
    S.resize(N);
    IdealS.resize(N);

    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0, 1.0f);

    Complex phase_rotation = std::polar(1.0f, (float)(M_PI / 3.0));
    float gain = 0.8f;

    for(size_t i=0; i<N; ++i) {
        R[i] = Complex(dist(gen), dist(gen));

        size_t delay = 2;
        Complex delayed_R = (i >= delay) ? R[i-delay] : Complex(0,0);
        
        IdealS[i] = delayed_R * gain * phase_rotation;

        Complex clutter = 5.0f * std::exp(Complex(0, 0.5f * i));
        Complex noise = 0.1f * Complex(dist(gen), dist(gen));

        S[i] = IdealS[i] + clutter + noise;
    }
}

/** main - Unit test: verify NLMS clutter cancellation achieves EVM < -15 dB */
int main() {
    std::cout << "--- NLMS Signal Conditioner Unit Test ---" << std::endl;
    
    // Increased sample count to ensure convergence
    size_t num_samples = 20000;
    std::vector<Complex> R, S1, IdealS1;
    generate_test_data(num_samples, R, S1, IdealS1);

    std::vector<std::vector<Complex>*> inputs = { &R, &S1 };

    AdaptiveSignalConditioner conditioner(16, 0.05f);

    auto outputs = conditioner.process(inputs);
    std::vector<Complex>& recovered_S1 = outputs[0];

    // Check last 2000 samples
    size_t start = num_samples - 2000;
    
    float total_error_sq = 0.0f;
    float ideal_power = 0.0f;
    
    for(size_t i = start; i < num_samples; ++i) {
        total_error_sq += std::norm(recovered_S1[i] - IdealS1[i]);
        ideal_power += std::norm(IdealS1[i]);
    }

    float evm_db = 10.0f * std::log10(total_error_sq / (ideal_power + 1e-12f));
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Recovered vs Ideal EVM: " << evm_db << " dB" << std::endl;
    
    if (evm_db < -15.0) {
        std::cout << "\nSUCCESS: Clutter removed, R-phase preserved." << std::endl;
    } else {
        std::cout << "\nFAILURE: Poor reconstruction." << std::endl;
    }

    return 0;
}
#endif
