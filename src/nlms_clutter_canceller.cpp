#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

// Standard complex double precision
using Complex = std::complex<double>;

class AdaptiveSignalConditioner {
private:
    size_t filter_length;
    double mu;      // Step size (learning rate)
    double epsilon; // Regularization term

    // Weights for each surveillance channel.
    // weights[ch] is the FIR filter that maps R -> S[ch]
    std::vector<std::vector<Complex>> weights;

    // History buffer for the Reference signal R(t)
    std::vector<Complex> r_history;

public:
    /**
     * @param len: Filter length (model order). 
     * @param step_size: NLMS step size. 
     * - Low (0.01-0.1): High stability, ignores transient noise, slow convergence.
     * - High (>0.5): Fast convergence, higher misadjustment (noise jitter).
     */
    AdaptiveSignalConditioner(size_t len, double step_size = 0.1) 
        : filter_length(len), mu(step_size), epsilon(1e-8) {
        // Initialize history buffer
        r_history.resize(filter_length, Complex(0.0, 0.0));
    }

    /**
     * Process block.
     * @param signal_ptrs: [0] -> Reference (R), [1..N] -> Surveillance (S1, S2...)
     * @return cleaned_signals: The FILTER OUTPUTS (y). 
     * These are the components of S that correlate with R (Signal of Interest).
     * The Clutter (Error) is discarded.
     */
    std::vector<std::vector<Complex>> process(const std::vector<std::vector<Complex>*>& signal_ptrs) {
        if (signal_ptrs.empty()) throw std::runtime_error("No signals provided.");

        const std::vector<Complex>& R = *signal_ptrs[0];
        size_t num_samples = R.size();
        size_t num_surveillance = signal_ptrs.size() - 1;

        // Lazy initialization of weights
        if (weights.size() != num_surveillance) {
            weights.assign(num_surveillance, std::vector<Complex>(filter_length, Complex(0.0, 0.0)));
        }

        // Prepare output containers
        std::vector<std::vector<Complex>> output_signals(num_surveillance);
        for(auto& vec : output_signals) vec.reserve(num_samples);

        // --- Sample-by-Sample Processing ---
        for (size_t n = 0; n < num_samples; ++n) {
            
            // 1. Update Reference History (R is the input x)
            // Shift history: oldest falls off, new sample R[n] enters at front (index 0)
            // Note: Efficient circular buffers preferred for production, using rotate for clarity.
            std::rotate(r_history.rbegin(), r_history.rbegin() + 1, r_history.rend());
            r_history[0] = R[n];

            // Calculate Energy of Reference History |x|^2
            double r_energy = 0.0;
            for (const auto& val : r_history) {
                r_energy += std::norm(val);
            }

            // 2. Process each Surveillance Channel independently
            for (size_t ch = 0; ch < num_surveillance; ++ch) {
                const std::vector<Complex>& S = *signal_ptrs[ch + 1];
                
                // Desired signal d(n) is the noisy Surveillance sample
                Complex d = S[n]; 

                // A. Filter Step: y(n) = w^H(n) * x(n)
                // This 'y' is our ESTIMATE of the signal component in S
                Complex y = 0.0;
                for (size_t k = 0; k < filter_length; ++k) {
                    y += std::conj(weights[ch][k]) * r_history[k];
                }

                // B. Error Calculation: e(n) = d(n) - y(n)
                // This 'e' represents the Clutter/Noise (uncorrelated part)
                Complex e = d - y;

                // C. Save the CLEAN signal (y)
                // We return 'y' because we want the part of S that looks like R.
                output_signals[ch].push_back(y);

                // D. NLMS Weight Update
                // w(n+1) = w(n) + mu * (e(n) / |x(n)|^2) * x(n)*
                // Note on Conjugates: To minimize E[|d - w^H x|^2], update is:
                // w += mu * conj(e) * x / |x|^2
                Complex step_scale = (mu * std::conj(e)) / (r_energy + epsilon);
                
                for (size_t k = 0; k < filter_length; ++k) {
                    weights[ch][k] += step_scale * r_history[k];
                }
            }
        }

        return output_signals;
    }
};

// ==========================================
//              UNIT TEST
// ==========================================

// Helper for power calc
double get_power(const std::vector<Complex>& v) {
    double p = 0;
    for(auto c : v) p += std::norm(c);
    return p / v.size();
}

void generate_test_data(size_t N, std::vector<Complex>& R, std::vector<Complex>& S, std::vector<Complex>& IdealS) {
    R.resize(N);
    S.resize(N);
    IdealS.resize(N);

    std::mt19937 gen(1337);
    std::normal_distribution<double> dist(0, 1.0);

    // 1. Signal Parameters
    // We want S to be a phase-rotated version of R + Interference
    Complex phase_rotation = std::polar(1.0, M_PI / 3.0); // 60 degrees rotation
    double gain = 0.8; 

    for(size_t i=0; i<N; ++i) {
        // Reference R: Random Gaussian data (e.g., OFDM-like)
        R[i] = Complex(dist(gen), dist(dist(gen)));

        // The "Ideal" Signal component embedded in S (Phase rotated R)
        // We add a tiny delay (e.g., 2 samples) to prove the filter aligns it
        size_t delay = 2;
        Complex delayed_R = (i >= delay) ? R[i-delay] : Complex(0,0);
        
        IdealS[i] = delayed_R * gain * phase_rotation;

        // Interference (Clutter): Strong non-gaussian signal uncorrelated with R
        // e.g., A continuous wave (CW) jammer
        Complex clutter = 5.0 * std::exp(Complex(0, 0.5 * i)); 

        // Noise: Low level Gaussian
        Complex noise = 0.1 * Complex(dist(gen), dist(gen));

        // Total Surveillance Signal
        S[i] = IdealS[i] + clutter + noise;
    }
}

int main() {
    std::cout << "--- NLMS Signal Conditioner Unit Test ---" << std::endl;
    
    // Setup
    size_t num_samples = 4000;
    std::vector<Complex> R, S1, IdealS1;
    generate_test_data(num_samples, R, S1, IdealS1);

    // Wrap in pointer array
    std::vector<std::vector<Complex>*> inputs = { &R, &S1 };

    // Instantiate Filter
    // Length 16 enough to capture small delays. Mu 0.05 for stability.
    AdaptiveSignalConditioner conditioner(16, 0.05);

    // Process
    auto outputs = conditioner.process(inputs);
    std::vector<Complex>& recovered_S1 = outputs[0];

    // --- Analysis ---
    // Ignore convergence period (first 500 samples)
    size_t start = 500;
    
    // Calculate Error Vector Magnitude (EVM) between Recovered and Ideal
    double total_error_sq = 0.0;
    double ideal_power = 0.0;
    
    for(size_t i = start; i < num_samples; ++i) {
        total_error_sq += std::norm(recovered_S1[i] - IdealS1[i]);
        ideal_power += std::norm(IdealS1[i]);
    }

    double evm_db = 10 * std::log10(total_error_sq / ideal_power);
    double input_sinr = 10 * std::log10(get_power(IdealS1) / (get_power(S1) - get_power(IdealS1)));

    std::cout << "Input Details:" << std::endl;
    std::cout << "  Input SINR (Signal to Clutter ratio): " << input_sinr << " dB (Dominated by Clutter)" << std::endl;
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Recovered vs Ideal EVM: " << evm_db << " dB" << std::endl;
    
    // Check Phase alignment
    // We take the dot product of recovered vs ideal to see if phase matches
    Complex dot_prod = 0;
    for(size_t i = start; i < num_samples; ++i) {
        dot_prod += recovered_S1[i] * std::conj(IdealS1[i]);
    }
    double avg_phase_diff = std::arg(dot_prod);

    std::cout << "  Phase Alignment Error: " << avg_phase_diff << " radians" << std::endl;

    if (evm_db < -15.0) {
        std::cout << "\nSUCCESS: Clutter removed, R-phase preserved." << std::endl;
    } else {
        std::cout << "\nFAILURE: Poor reconstruction." << std::endl;
    }

    return 0;
}
