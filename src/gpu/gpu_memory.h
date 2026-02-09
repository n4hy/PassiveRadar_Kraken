/**
 * GPU Memory Management for PassiveRadar_Kraken
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Memory pool, pinned allocations, and transfer utilities.
 */

#ifndef KRAKEN_GPU_MEMORY_H
#define KRAKEN_GPU_MEMORY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Device Memory Allocation
 */

// Allocate device memory
void* kraken_gpu_alloc(size_t size);

// Free device memory
void kraken_gpu_free(void* ptr);

// Allocate pinned host memory (faster transfers)
void* kraken_gpu_alloc_host(size_t size);

// Free pinned host memory
void kraken_gpu_free_host(void* ptr);

/**
 * Memory Transfers
 */

// Host to Device (synchronous)
int kraken_gpu_memcpy_h2d(void* dst, const void* src, size_t size);

// Device to Host (synchronous)
int kraken_gpu_memcpy_d2h(void* dst, const void* src, size_t size);

// Device to Device
int kraken_gpu_memcpy_d2d(void* dst, const void* src, size_t size);

// Async transfers (require cudaStream_t, for advanced use)
int kraken_gpu_memcpy_h2d_async(void* dst, const void* src, size_t size, void* stream);
int kraken_gpu_memcpy_d2h_async(void* dst, const void* src, size_t size, void* stream);

/**
 * Memory Pool (for persistent allocations)
 */

typedef struct KrakenGPUMemoryPool KrakenGPUMemoryPool;

// Create memory pool
KrakenGPUMemoryPool* kraken_gpu_memory_pool_create(void);

// Destroy memory pool (frees all allocations)
void kraken_gpu_memory_pool_destroy(KrakenGPUMemoryPool* pool);

// Allocate from pool (reuses freed memory if available)
void* kraken_gpu_memory_pool_alloc(KrakenGPUMemoryPool* pool, size_t size);

// Return memory to pool (doesn't actually free, marks as available)
void kraken_gpu_memory_pool_free(KrakenGPUMemoryPool* pool, void* ptr);

// Get memory usage statistics
void kraken_gpu_memory_pool_stats(KrakenGPUMemoryPool* pool,
                                   size_t* total_allocated,
                                   size_t* total_in_use,
                                   int* num_allocations);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_GPU_MEMORY_H
