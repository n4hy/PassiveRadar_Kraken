#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <algorithm>

using ComplexFloat = std::complex<float>;
using ComplexDouble = std::complex<double>;

class PolyphaseResampler {
private:
    int interpolation;
    int decimation;
    std::vector<float> taps;
    int num_taps;

    // Polyphase filters: taps rearranged
    // We can just index the linear taps array with stride logic to avoid copying
    // But typically polyphase filters are stored as a bank.
    // Let's store as linear and index: h[phase + k*interp]

    // State
    // We need a history buffer to store previous inputs required for the filter convolution
    std::vector<ComplexFloat> history;
    size_t history_idx; // Circular buffer or slide?

    // Usually for resamplers it's easier to slide/copy for simplicity unless performance is critical
    // Given the Python overhead, memmove is negligible.

    // Phase accumulator for the polyphase commutator
    // Tracks position in the upsampled grid
    // We track `current_phase` (0 to interp-1) and `input_consumed`.
    // Actually, simpler model:
    // We compute output y[m].
    // Input index required is n. Phase is p.
    // We increment m. n advances by floor((p + decim)/interp). p updates to (p + decim)%interp.
    int current_phase;

public:
    PolyphaseResampler(int interp, int decim, const float* taps_in, int n_taps)
        : interpolation(interp), decimation(decim), num_taps(n_taps), current_phase(0) {

        taps.assign(taps_in, taps_in + n_taps);

        // History size needs to cover the filter length (in input samples)
        // Filter length in upsampled domain is num_taps.
        // In input domain, it spans approx ceil(num_taps / interp).
        // Let's keep a safe margin.
        int history_len = (num_taps + interp - 1) / interp + 2;
        history.resize(history_len, ComplexFloat(0, 0));
    }

    // Process a block of samples
    // input: array of complex float
    // output: array of complex float (capacity must be sufficient)
    // Returns number of output samples produced
    int process(const ComplexFloat* input, int n_input, ComplexFloat* output, int max_output) {

        // Append new input to history
        // To handle the streaming nature, we need to keep the tail of the previous history
        // and append the new input.
        // Strategy:
        // 1. Existing history contains the "overlap" from previous call.
        // 2. We logically construct a buffer: [Old History] + [New Input].
        // 3. We process.
        // 4. We save the last N samples as the new history.

        // However, we don't want to re-allocate huge vectors every time.
        // Let's use a working buffer.

        // Filter span in input samples
        int taps_per_phase = (num_taps + interpolation - 1) / interpolation;

        // We need at least 'taps_per_phase' samples.
        std::vector<ComplexFloat> work_buffer;
        work_buffer.reserve(history.size() + n_input);
        work_buffer.insert(work_buffer.end(), history.begin(), history.end());
        work_buffer.insert(work_buffer.end(), input, input + n_input);

        const ComplexFloat* buf_ptr = work_buffer.data();
        int total_len = work_buffer.size();

        int output_count = 0;
        int input_idx = 0; // Index in work_buffer where the "current" input sample corresponds to phase 0?

        // Wait, precise tracking:
        // We are at `input_idx` in the stream (relative to work_buffer start).
        // `current_phase` tracks the sub-sample position.
        // We produce an output.
        // Then we advance.

        // input_idx starts at index where history ended?
        // No, `history` was the *tail* of the previous block, which is needed for looking *back*.
        // So the "current" input starts at `history.size()`.

        // Let's redefine input_idx to be the index of the "newest" sample required?
        // Standard Polyphase: y[m] = sum( h[k*L + p] * x[n - k] )
        // Here x[n] is the sample corresponding to the current time.
        // If we are at phase p, we are "between" x[n] and x[n+1].

        // Let's align such that:
        // We start processing from the first new sample.
        // The algorithm determines when we need the next input.

        // Let 'n' be the index in work_buffer.
        // Initially n is `history.size()`. (The first new sample).
        // But we might be mid-way through a sample from previous block if decim < interp?
        // Or if decim > interp, we might skip samples.

        // Correct state maintenance:
        // We maintained `current_phase` from last time.
        // We effectively have a continuous stream of samples.
        // work_buffer[0] corresponds to sample index K (from past).
        // work_buffer[history.size()] corresponds to sample index K + history.size().

        // Let's just iterate generating outputs until we run out of input.
        // To generate an output, we need `taps_per_phase` samples looking back from the current input pointer.

        // Start 'ptr' at the beginning of the "valid" new data region?
        // No, we continue from where we left off.
        // We need a state variable `samples_consumed_since_history_start`?

        // Actually, we can just say:
        // current_phase is the phase of the *next* output sample.
        // But relative to which input sample?
        // We need to know which input sample corresponds to the current output time.
        // Let's track `sample_offset` which is the fractional position in input samples.
        // `sample_offset` = integer part + phase/L.
        // We want to generate outputs as long as `integer part` is within range.

        // Let's reset the coordinate system:
        // work_buffer[0] is index 0.
        // The "next" input sample we process is at index `history.size()`.
        // Wait, standard GR resampler maintains an index into the input buffer.

        // Let's assume we simply process outputs.
        // Each step advances phase by `decimation`.
        // If phase >= interpolation, we wrap and advance input index by `phase / interpolation`.

        // We need an initial input index.
        // Since we saved history, `work_buffer` is continuous.
        // The "current" input index (n) corresponds to the sample we are currently aligned with.
        // Last time, we stopped at some input index.
        // We need to know where that was relative to the history.
        // Since we copy `history` (which is the last samples of previous block),
        // the index 0 in work_buffer is (TotalInputIndex - history.size()).

        // Let's simplify:
        // We just need to know the index `n` in `work_buffer` that corresponds to the current sampling instant.
        // We effectively start `n` at `history.size()` - but offset by however many samples we "didn't consume" fully last time?
        // No, polyphase usually consumes inputs as it goes.
        // Let's say we have `current_phase` state.
        // And we assume we are aligned with the *first sample of the new input*?
        // No, we might be partly through the previous sample interval.

        // Standard implementation:
        // State: `current_phase`.
        // Loop:
        //   Calculate dot product using `current_phase` and samples ending at `current_input_ptr`.
        //   Increment `current_phase` by `decimation`.
        //   While `current_phase` >= `interpolation`:
        //     `current_phase` -= `interpolation`
        //     `current_input_ptr`++

        // We need to know if `current_input_ptr` runs off the end of the buffer.

        // Where does `current_input_ptr` start?
        // It starts at `history.size() - 1`? (The last sample of history).
        // If we wrapped exactly at the boundary last time, we are at `history.size()`.

        // Actually, it's safer to just track "how many inputs we are waiting for".
        // But simpler:
        // Reset `current_input_ptr` to `history.size() - 1`?
        // No, that implies we are aligned exactly there.
        // We need to persist the "sub-sample" position, which is `current_phase`.
        // But we also need to persist "how many input samples we need to advance before valid".
        // If `decimation` is large, we might skip input samples.

        // Correct approach with history buffer:
        // We always keep the last `taps_per_phase` samples as history.
        // When `process` is called, we prepend history.
        // We start our input pointer `ptr` at `history.size()`.
        // But wait, if we had left-over phase increments from last time?
        // i.e., last time we output a sample, advanced phase, and realized we need to skip 3 inputs.
        // But we ran out of inputs.
        // So we need to skip 3 inputs *in the new block*.

        // So state: `skip_count` (integers) and `current_phase` (fraction).
        // Actually, we can just say `current_phase` can be > `interpolation`.
        // No, `phase` implies coefficient selection.

        // Let's store `accumulated_phase` or similar?
        // No.

        // Let's just iterate.
        // `input_index` relative to `work_buffer`.
        // Initial `input_index`:
        // We want it to be continuous from last call.
        // Last call ended. We kept the last `history_len` samples.
        // So `work_buffer` [0 ... history_len-1] are the previous tail.
        // `work_buffer` [history_len ... ] are new.

        // We need a state variable: `samples_to_drop`.
        // i.e. how many input samples to advance before producing the next output?
        // But `decimation` steps are usually fractional input steps (D/I).

        // Let's look at `rational_resampler_base`.
        // It tracks `d_phase`.
        // When `d_phase >= d_interp`, it decrements `d_phase` and consumes input.

        // Let's assume we start aligned with the sample at `history.size() - 1` (last sample of history)?
        // Or `history.size()` (first new sample)?
        // Let's assume we are aligned with `work_buffer[current_input_offset]`.
        // Initially `current_input_offset` = `history.size()`?
        // But we might need to skip samples *before* we produce the first output of this block,
        // if the previous block ended pending a large skip.

        // Let's maintain `current_input_ptr_offset` relative to the *beginning of the new data*.
        // Can be negative (meaning we are still in history).
        // Can be positive (in new data).

        // Ideally: `current_sample_index` relative to stream start.
        // But that grows indefinitely.

        // Let's just simply use a loop and check bounds.
        // Start index `n = 0`. This points to `work_buffer[start_index]`.
        // Where is `start_index`?
        // It is determined by the previous state.
        // `start_index` should be `history.size()`.
        // But if we had a large decimation step pending, we might need to start at `history.size() + skip`.

        // We need to save the state of "how far along the input stream we are".
        // Let's just save `current_phase`.
        // And we implicitly assume that we processed as much as possible last time.
        // "As much as possible" means we stopped because `input_idx` went out of bounds.
        // So `input_idx` was somewhere past the end of valid data.
        // Let's say `valid_data_end` was `N`.
        // We stopped because `input_idx >= N`.
        // The "excess" is `input_idx - N`.
        // This excess is how far into the *new* buffer we must start.

        // So state: `excess_input_needed`.
        // And `current_phase`.

        // In the new call:
        // `input_idx` starts at `history.size() + excess_input_needed`.
        // Wait, `history` is constructed from the *end* of the previous block.
        // So `work_buffer[history.size()]` is the first new sample.
        // So yes, `input_idx` starts at `history.size() + excess_input_needed`.
        // `excess_input_needed` is 0 usually, unless D >> I.

        int idx = history.size() + excess_input_advance;

        // Limit is `work_buffer.size()`. We need `taps_per_phase` history *behind* `idx`?
        // Polyphase convolution:
        // $y = \sum h[p + k*L] \cdot x[n - k]$
        // Here $n$ is the current input index `idx`.
        // We need $x[idx]$ down to $x[idx - taps_per_phase + 1]$.
        // So we need `idx >= taps_per_phase - 1`.
        // Our history size should ensure this.

        // Processing loop:
        while (idx < total_len) {
            if (output_count >= max_output) break;

            // Compute output
            ComplexDouble accum(0.0, 0.0);

            for (int k = 0; k < taps_per_phase; ++k) {
                // Taps index: current_phase + k * interpolation
                // Input index: idx - k
                if (idx - k < 0) continue; // Should not happen with sufficient history

                int tap_idx = current_phase + k * interpolation;
                if (tap_idx >= num_taps) break;

                // Accumulate with double precision
                accum += static_cast<ComplexDouble>(work_buffer[idx - k]) * static_cast<double>(taps[tap_idx]);
            }

            output[output_count++] = static_cast<ComplexFloat>(accum);

            // Advance
            current_phase += decimation;
            int advance = current_phase / interpolation;
            current_phase %= interpolation;
            idx += advance;
        }

        // Save state for next block
        // `idx` is now >= total_len (or we stopped due to output full).
        // `excess_input_advance` for next time is `idx - total_len`.
        // Wait, `total_len` is where the new data ends.
        // `work_buffer` indices: 0 ... total_len-1.
        // Next sample is at `total_len`.
        // So if `idx` points to `total_len + 5`, we need to skip 5 samples in next block.
        excess_input_advance = idx - total_len; // Can be negative? No, loop condition `idx < total_len` ensures we stop before running off?
        // No, the increment `idx += advance` happens at end of loop.
        // So `idx` can overshoot `total_len`.
        // `excess_input_advance` = `idx - total_len`.
        // Since `idx` starts relative to `history.size()`, and `total_len` = `history.size() + n_input`.
        // The logic holds.

        // Update history
        // Keep last `history_len` samples of `work_buffer`.
        int start_copy = work_buffer.size() - history.size();
        if (start_copy < 0) start_copy = 0; // Should not happen
        for(size_t i=0; i<history.size(); ++i) {
            if (start_copy + i < work_buffer.size())
                history[i] = work_buffer[start_copy + i];
            else
                history[i] = ComplexFloat(0,0);
        }

        return output_count;
    }

private:
    int excess_input_advance = 0;
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
