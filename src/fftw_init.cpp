/**
 * Centralized FFTW thread initialization.
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */
#include "fftw_init.h"
#include <fftw3.h>
#include <pthread.h>

static pthread_once_t fftw_once = PTHREAD_ONCE_INIT;

static void do_fftw_init(void) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(1);
}

void kraken_fftw_init(void) {
    pthread_once(&fftw_once, do_fftw_init);
}
