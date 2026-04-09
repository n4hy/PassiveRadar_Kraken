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

/**
 * kraken_gpu_alloc - Allocate device (GPU) memory
 */
void* kraken_gpu_alloc(size_t size);

/**
 * kraken_gpu_free - Free device (GPU) memory
 */
void kraken_gpu_free(void* ptr);

/**
 * kraken_gpu_alloc_host - Allocate pinned (page-locked) host memory for faster DMA transfers
 */
void* kraken_gpu_alloc_host(size_t size);

/**
 * kraken_gpu_free_host - Free pinned host memory
 */
void kraken_gpu_free_host(void* ptr);

/**
 * Memory Transfers
 */

/**
 * kraken_gpu_memcpy_h2d - Synchronous host-to-device memory copy
 */
int kraken_gpu_memcpy_h2d(void* dst, const void* src, size_t size);

/**
 * kraken_gpu_memcpy_d2h - Synchronous device-to-host memory copy
 */
int kraken_gpu_memcpy_d2h(void* dst, const void* src, size_t size);

/**
 * kraken_gpu_memcpy_d2d - Device-to-device memory copy
 */
int kraken_gpu_memcpy_d2d(void* dst, const void* src, size_t size);

/**
 * kraken_gpu_memcpy_h2d_async - Asynchronous host-to-device memory copy on a CUDA stream
 */
int kraken_gpu_memcpy_h2d_async(void* dst, const void* src, size_t size, void* stream);

/**
 * kraken_gpu_memcpy_d2h_async - Asynchronous device-to-host memory copy on a CUDA stream
 */
int kraken_gpu_memcpy_d2h_async(void* dst, const void* src, size_t size, void* stream);

/**
 * Memory Pool (for persistent allocations)
 */

typedef struct KrakenGPUMemoryPool KrakenGPUMemoryPool;

/**
 * kraken_gpu_memory_pool_create - Create a GPU memory pool for persistent allocation reuse
 */
KrakenGPUMemoryPool* kraken_gpu_memory_pool_create(void);

/**
 * kraken_gpu_memory_pool_destroy - Destroy memory pool and free all device allocations
 */
void kraken_gpu_memory_pool_destroy(KrakenGPUMemoryPool* pool);

/**
 * kraken_gpu_memory_pool_alloc - Allocate from pool, reusing freed blocks when possible
 */
void* kraken_gpu_memory_pool_alloc(KrakenGPUMemoryPool* pool, size_t size);

/**
 * kraken_gpu_memory_pool_free - Return memory to pool without freeing (marks as available for reuse)
 */
void kraken_gpu_memory_pool_free(KrakenGPUMemoryPool* pool, void* ptr);

/**
 * kraken_gpu_memory_pool_stats - Query total allocated bytes, in-use bytes, and allocation count
 */
void kraken_gpu_memory_pool_stats(KrakenGPUMemoryPool* pool,
                                   size_t* total_allocated,
                                   size_t* total_in_use,
                                   int* num_allocations);

#ifdef __cplusplus
}
#endif

#endif // KRAKEN_GPU_MEMORY_H
