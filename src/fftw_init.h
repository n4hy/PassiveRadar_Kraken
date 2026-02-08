/**
 * Centralized FFTW thread initialization.
 *
 * fftwf_init_threads() must be called exactly once per process.
 * Since multiple shared libraries (caf, doppler, time_alignment) each need FFTW,
 * we centralize the init here in a single shared library that all link against.
 *
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */
#ifndef KRAKEN_FFTW_INIT_H
#define KRAKEN_FFTW_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize FFTW threads. Safe to call multiple times; only the first
 * call actually invokes fftwf_init_threads(). Thread-safe via pthread_once.
 */
void kraken_fftw_init(void);

#ifdef __cplusplus
}
#endif

#endif /* KRAKEN_FFTW_INIT_H */
