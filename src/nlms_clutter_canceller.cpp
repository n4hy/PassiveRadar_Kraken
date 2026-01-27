#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

// Use float for compatibility with np.complex64 (GNU Radio standard)
using Complex = std::complex<float>;

class AdaptiveSignalConditioner {
private:
    size_t filter_length;
    float mu;      // Step size (learning rate)
    float epsilon; // Regularization term

    // Weights for each surveillance channel.
    std::vector<std::vector<Complex>> weights;

    // History buffer for the Reference signal R(t)
    std::vector<Complex> r_history;

public:
    AdaptiveSignalConditioner(size_t len, float step_size = 0.1)
        : filter_length(len), mu(step_size), epsilon(1e-8f) {
        // Initialize history buffer
        r_history.resize(filter_length, Complex(0.0f, 0.0f));
    }

    // Original process method (returns y, the estimate/correlated part)
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

    // New method for C interface: Single channel, returns Error (e)
    // out_err must be pre-allocated with size n_samples
    void process_cancellation_single(const Complex* ref_in, const Complex* surv_in, Complex* out_err, size_t n_samples) {
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
    void update_history(Complex val) {
        for (size_t i = filter_length - 1; i > 0; --i) {
            r_history[i] = r_history[i-1];
        }
        r_history[0] = val;
    }

    float get_history_energy() {
        float r_energy = 0.0f;
        for (const auto& val : r_history) {
            r_energy += std::norm(val);
        }
        return r_energy;
    }

    Complex filter(size_t ch) {
        Complex y = 0.0f;
        for (size_t k = 0; k < filter_length; ++k) {
            y += std::conj(weights[ch][k]) * r_history[k];
        }
        return y;
    }

    void update_weights(size_t ch, Complex e, float r_energy) {
        // Skip update if energy is too low to avoid numerical instability
        if (r_energy < epsilon) return;
        Complex step_scale = (mu * std::conj(e)) / (r_energy + epsilon);
        // Check for NaN/Inf before applying
        if (!std::isfinite(step_scale.real()) || !std::isfinite(step_scale.imag())) return;
        for (size_t k = 0; k < filter_length; ++k) {
            weights[ch][k] += step_scale * r_history[k];
        }
    }
};

extern "C" {
    void* nlms_create(int taps, float mu) {
        return new AdaptiveSignalConditioner(static_cast<size_t>(taps), mu);
    }

    void nlms_destroy(void* ptr) {
        if (ptr) delete static_cast<AdaptiveSignalConditioner*>(ptr);
    }

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

float get_power(const std::vector<Complex>& v) {
    float p = 0;
    for(auto c : v) p += std::norm(c);
    return p / v.size();
}

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
