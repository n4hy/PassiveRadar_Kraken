/*
 * ECA Canceller — Batch Toeplitz + FFT FIR/Cross-Correlation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Autocorrelation: NEON dot products + Toeplitz diagonal recursion
 * Cross-correlation: FFT-based (conj(FFT(ref)) * FFT(surv))
 * FIR application: FFT-based (FFT(w) * FFT(ref))
 * Cholesky solve: O(L³), computed once, reused across channels
 *
 * FFT(ref) is computed once per work() call and cached for all channels.
 */

#include "eca_canceller_impl.h"
#include <gnuradio/io_signature.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef HAVE_OPTMATHKERNELS
#include <optmath/neon_kernels.hpp>
#endif

namespace gr {
namespace kraken_passive_radar {

// ---------- helpers ----------

inline float eca_canceller_impl::dot_f32(const float* a, const float* b, int n)
{
#ifdef HAVE_OPTMATHKERNELS
    return optmath::neon::neon_dot_f32(a, b, static_cast<std::size_t>(n));
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

bool eca_canceller_impl::cholesky_decompose()
{
    const int L = d_num_taps;
    std::fill(d_L.begin(), d_L.end(), std::complex<float>(0));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j <= i; j++) {
            std::complex<float> sum(0);
            for (int k = 0; k < j; k++)
                sum += d_L[i * L + k] * std::conj(d_L[j * L + k]);
            if (i == j) {
                float val = (d_R[i * L + i] - sum).real();
                if (val <= 0.0f) return false;
                d_L[i * L + j] = std::sqrt(val);
            } else {
                d_L[i * L + j] = (d_R[i * L + j] - sum) / d_L[j * L + j];
            }
        }
    }
    return true;
}

void eca_canceller_impl::cholesky_solve()
{
    const int L = d_num_taps;
    for (int i = 0; i < L; i++) {
        std::complex<float> sum(0);
        for (int j = 0; j < i; j++) sum += d_L[i * L + j] * d_y[j];
        d_y[i] = (d_p[i] - sum) / d_L[i * L + i];
    }
    for (int i = L - 1; i >= 0; i--) {
        std::complex<float> sum(0);
        for (int j = i + 1; j < L; j++) sum += std::conj(d_L[j * L + i]) * d_w[j];
        d_w[i] = (d_y[i] - sum) / std::conj(d_L[i * L + i]);
    }
}

void eca_canceller_impl::cleanup_fir_fft()
{
    if (d_fir_fft_len > 0) {
        fftwf_destroy_plan(d_fir_fwd);
        fftwf_destroy_plan(d_fir_inv);
        fftwf_free(d_fir_ref_fft);
        fftwf_free(d_fir_a);
        fftwf_free(d_fir_b);
        d_fir_fft_len = 0;
    }
}

void eca_canceller_impl::setup_fir_fft(int n_output)
{
    int total = n_output + d_num_taps - 1;
    int F = 1;
    while (F < total) F <<= 1;

    if (F == d_fir_fft_len) return;  // already set up

    cleanup_fir_fft();

    d_fir_ref_fft = fftwf_alloc_complex(F);
    d_fir_a       = fftwf_alloc_complex(F);
    d_fir_b       = fftwf_alloc_complex(F);

    d_fir_fwd = fftwf_plan_dft_1d(F, d_fir_a, d_fir_a, FFTW_FORWARD,  FFTW_ESTIMATE);
    d_fir_inv = fftwf_plan_dft_1d(F, d_fir_b, d_fir_b, FFTW_BACKWARD, FFTW_ESTIMATE);

    d_fir_fft_len = F;
}

// ---------- block lifecycle ----------

eca_canceller::sptr eca_canceller::make(int num_taps, float reg_factor, int num_surv)
{
    return gnuradio::make_block_sptr<eca_canceller_impl>(num_taps, reg_factor, num_surv);
}

eca_canceller_impl::eca_canceller_impl(int num_taps, float reg_factor, int num_surv)
    : gr::sync_block("eca_canceller",
                     gr::io_signature::make(1 + num_surv, 1 + num_surv, sizeof(gr_complex)),
                     gr::io_signature::make(num_surv, num_surv, sizeof(gr_complex))),
      d_num_taps(num_taps),
      d_diag_loading(reg_factor),
      d_num_surv(num_surv),
      d_fir_fft_len(0),
      d_fir_fwd(nullptr),
      d_fir_inv(nullptr),
      d_fir_ref_fft(nullptr),
      d_fir_a(nullptr),
      d_fir_b(nullptr)
{
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ULL << 24);
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#endif

    const int L = num_taps;
    d_R.resize(L * L);
    d_L.resize(L * L);
    d_p.resize(L);
    d_w.resize(L);
    d_y.resize(L);

    set_history(num_taps);
    set_output_multiple(std::max(num_taps * 8, 2048));
}

eca_canceller_impl::~eca_canceller_impl()
{
    cleanup_fir_fft();
}

// ---------- work ----------

int eca_canceller_impl::work(int noutput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{
    gr::thread::scoped_lock guard(d_setlock);

    const gr_complex* ref = static_cast<const gr_complex*>(input_items[0]);
    const int N = noutput_items;
    const int L = d_num_taps;
    const int total = N + L - 1;
    const int base = L - 1;

    // Resize SoA buffers
    if (static_cast<int>(d_ref_re.size()) < total) {
        d_ref_re.resize(total);
        d_ref_im.resize(total);
    }
    if (static_cast<int>(d_surv_re.size()) < N) {
        d_surv_re.resize(N);
        d_surv_im.resize(N);
    }

    // Deinterleave reference
    for (int i = 0; i < total; i++) {
        d_ref_re[i] = ref[i].real();
        d_ref_im[i] = ref[i].imag();
    }

    // Setup FFT plans (lazy, only recreate if N changed)
    setup_fir_fft(N);
    const int F = d_fir_fft_len;
    const float fft_scale = 1.0f / F;

    // ===== FFT(ref) — computed once, cached for all channels =====
    std::memset(d_fir_ref_fft, 0, F * sizeof(fftwf_complex));
    std::memcpy(d_fir_ref_fft, ref, total * sizeof(fftwf_complex));
    fftwf_execute_dft(d_fir_fwd, d_fir_ref_fft, d_fir_ref_fft);

    // ===== Autocorrelation R (dot products + Toeplitz recursion) =====
    {
        const float* r0_re = d_ref_re.data() + base;
        const float* r0_im = d_ref_im.data() + base;
        for (int k = 0; k < L; k++) {
            const float* rk_re = d_ref_re.data() + (base - k);
            const float* rk_im = d_ref_im.data() + (base - k);
            float rr = dot_f32(r0_re, rk_re, N);
            float ii = dot_f32(r0_im, rk_im, N);
            float ri = dot_f32(r0_re, rk_im, N);
            float ir = dot_f32(r0_im, rk_re, N);
            d_R[k] = std::complex<float>(rr + ii, ir - ri);
        }
    }

    // Toeplitz diagonal recursion
    for (int k_start = 0; k_start < L; k_start++) {
        const int max_steps = L - 1 - k_start;
        for (int i = 0; i < max_steps; i++) {
            const int j = i, k = k_start + i;
            std::complex<float> curr = d_R[j * L + k];
            int pj = base - 1 - j, pk = base - 1 - k;
            int lj = base + N - 1 - j, lk = base + N - 1 - k;
            if (pj < 0 || pk < 0 || lj >= total || lk >= total) {
                d_R[(j+1)*L + (k+1)] = curr;
                continue;
            }
            std::complex<float> prev_j(d_ref_re[pj], d_ref_im[pj]);
            std::complex<float> prev_k(d_ref_re[pk], d_ref_im[pk]);
            std::complex<float> last_j(d_ref_re[lj], d_ref_im[lj]);
            std::complex<float> last_k(d_ref_re[lk], d_ref_im[lk]);
            d_R[(j+1)*L + (k+1)] = curr
                + std::conj(prev_j) * prev_k
                - std::conj(last_j) * last_k;
        }
    }

    // Hermitian fill + diagonal loading
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < j; k++)
            d_R[j * L + k] = std::conj(d_R[k * L + j]);
        d_R[j * L + j] += d_diag_loading;
    }

    // Cholesky decompose R → L (reused for all channels)
    bool chol_ok = cholesky_decompose();

    // ===== Process each surveillance channel =====
    for (int ch = 0; ch < d_num_surv; ch++) {
        const gr_complex* surv = static_cast<const gr_complex*>(input_items[1 + ch]);
        gr_complex* out = static_cast<gr_complex*>(output_items[ch]);

        // Deinterleave surveillance (output-aligned region)
        for (int i = 0; i < N; i++) {
            d_surv_re[i] = surv[i + base].real();
            d_surv_im[i] = surv[i + base].imag();
        }

        if (!chol_ok) {
            for (int i = 0; i < N; i++) out[i] = surv[i + base];
            continue;
        }

        // --- Cross-correlation p via FFT ---
        // p[j] = IFFT(conj(FFT(ref)) * FFT(surv))[L-1-j]
        std::memset(d_fir_a, 0, F * sizeof(fftwf_complex));
        for (int i = 0; i < N; i++) {
            d_fir_a[i][0] = d_surv_re[i];
            d_fir_a[i][1] = d_surv_im[i];
        }
        fftwf_execute_dft(d_fir_fwd, d_fir_a, d_fir_a);

        // conj(ref_fft) * surv_fft → d_fir_b, with 1/F scaling
        for (int i = 0; i < F; i++) {
            float rr = d_fir_ref_fft[i][0], ri = d_fir_ref_fft[i][1];
            float sr = d_fir_a[i][0],       si = d_fir_a[i][1];
            // conj(ref) * surv = (rr - j*ri)(sr + j*si)
            d_fir_b[i][0] = (rr * sr + ri * si) * fft_scale;
            d_fir_b[i][1] = (rr * si - ri * sr) * fft_scale;
        }
        fftwf_execute_dft(d_fir_inv, d_fir_b, d_fir_b);

        // Extract p[j] = result[L-1-j]
        for (int j = 0; j < L; j++) {
            int idx = L - 1 - j;
            d_p[j] = std::complex<float>(d_fir_b[idx][0], d_fir_b[idx][1]);
        }

        // Solve R*w = p
        cholesky_solve();

        // --- FIR application via FFT ---
        // y = conv(w, ref)[L-1 : L-1+N] = IFFT(FFT(w) * FFT(ref))[L-1+n]
        std::memset(d_fir_a, 0, F * sizeof(fftwf_complex));
        for (int k = 0; k < L; k++) {
            d_fir_a[k][0] = d_w[k].real();
            d_fir_a[k][1] = d_w[k].imag();
        }
        fftwf_execute_dft(d_fir_fwd, d_fir_a, d_fir_a);

        // w_fft * ref_fft → d_fir_b, with 1/F scaling
        for (int i = 0; i < F; i++) {
            float wr = d_fir_a[i][0], wi = d_fir_a[i][1];
            float rr = d_fir_ref_fft[i][0], ri = d_fir_ref_fft[i][1];
            d_fir_b[i][0] = (wr * rr - wi * ri) * fft_scale;
            d_fir_b[i][1] = (wr * ri + wi * rr) * fft_scale;
        }
        fftwf_execute_dft(d_fir_inv, d_fir_b, d_fir_b);

        // Output: out[n] = surv[n] - conv_result[L-1+n]
        for (int n = 0; n < N; n++) {
            int ci = base + n;  // = L-1+n
            out[n] = gr_complex(d_surv_re[n] - d_fir_b[ci][0],
                                d_surv_im[n] - d_fir_b[ci][1]);
        }
    }

    return noutput_items;
}

// ---------- setters ----------

void eca_canceller_impl::set_num_taps(int num_taps)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_num_taps = num_taps;
    const int L = num_taps;
    d_R.resize(L * L);
    d_L.resize(L * L);
    d_p.resize(L);
    d_w.resize(L);
    d_y.resize(L);
    set_history(num_taps);
    set_output_multiple(std::max(num_taps * 8, 2048));
    cleanup_fir_fft();  // force replan on next work()
}

void eca_canceller_impl::set_reg_factor(float reg_factor)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_diag_loading = reg_factor;
}

void eca_canceller_impl::set_mu(float /*mu*/) {}

} // namespace kraken_passive_radar
} // namespace gr
