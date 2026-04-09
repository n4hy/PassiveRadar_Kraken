/**
 * Centralized FFTW thread initialization.
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */
#include "fftw_init.h"
#include <fftw3.h>
#include <pthread.h>

static pthread_once_t fftw_once = PTHREAD_ONCE_INIT;

/**
 * do_fftw_init - One-time FFTW library initialization callback
 *
 * Technique: Initializes FFTW thread support and sets single-threaded
 * planning mode. Called exactly once via pthread_once to ensure
 * thread-safe initialization across all library users.
 */
static void do_fftw_init(void) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(1);
}

/**
 * kraken_fftw_init - Thread-safe FFTW initialization entry point
 *
 * Technique: Uses pthread_once to guarantee that FFTW thread setup
 * (fftwf_init_threads) runs exactly once regardless of how many
 * modules or threads call this function concurrently.
 */
void kraken_fftw_init(void) {
    pthread_once(&fftw_once, do_fftw_init);
}
