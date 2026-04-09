/**
 * GPU Memory Management Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 */

#include "gpu_memory.h"
#include "gpu_common.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <mutex>

/**
 * Memory Pool Implementation
 */

/** MemoryBlock - Tracks a single GPU allocation within the memory pool */
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
};

/**
 * KrakenGPUMemoryPool - Thread-safe GPU memory pool with block reuse
 *
 * Technique: Maintains a list of allocated GPU memory blocks with usage tracking.
 * On allocation, first tries to reuse a freed block of sufficient size before
 * calling cudaMalloc. Thread-safe via std::mutex.
 */
struct KrakenGPUMemoryPool {
    std::vector<MemoryBlock> blocks;
    std::unordered_map<void*, size_t> ptr_to_index;
    std::mutex mutex;
    size_t total_allocated;
    size_t total_in_use;
};

/**
 * kraken_gpu_alloc - Allocate GPU device memory via cudaMalloc
 *
 * Technique: Direct cudaMalloc wrapper with error reporting to stderr.
 */
extern "C"
void* kraken_gpu_alloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

/** kraken_gpu_free - Free GPU device memory via cudaFree */
extern "C"
void kraken_gpu_free(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

/**
 * kraken_gpu_alloc_host - Allocate pinned (page-locked) host memory
 *
 * Technique: Uses cudaMallocHost for DMA-capable pinned memory,
 * enabling faster async host-device transfers.
 */
extern "C"
void* kraken_gpu_alloc_host(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

/** kraken_gpu_free_host - Free pinned host memory via cudaFreeHost */
extern "C"
void kraken_gpu_free_host(void* ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

/** kraken_gpu_memcpy_h2d - Synchronous host-to-device memory copy */
extern "C"
int kraken_gpu_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/** kraken_gpu_memcpy_d2h - Synchronous device-to-host memory copy */
extern "C"
int kraken_gpu_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/** kraken_gpu_memcpy_d2d - Synchronous device-to-device memory copy */
extern "C"
int kraken_gpu_memcpy_d2d(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/** kraken_gpu_memcpy_h2d_async - Asynchronous host-to-device copy on given CUDA stream */
extern "C"
int kraken_gpu_memcpy_h2d_async(void* dst, const void* src, size_t size, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                      (cudaStream_t)stream);
    return (err == cudaSuccess) ? 0 : -1;
}

/** kraken_gpu_memcpy_d2h_async - Asynchronous device-to-host copy on given CUDA stream */
extern "C"
int kraken_gpu_memcpy_d2h_async(void* dst, const void* src, size_t size, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                      (cudaStream_t)stream);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * kraken_gpu_memory_pool_create - Allocate and initialize a GPU memory pool
 *
 * Technique: Creates empty pool with zero-initialized tracking counters.
 */
extern "C"
KrakenGPUMemoryPool* kraken_gpu_memory_pool_create(void) {
    return new KrakenGPUMemoryPool{
        .blocks = {},
        .ptr_to_index = {},
        .mutex = {},
        .total_allocated = 0,
        .total_in_use = 0
    };
}

/**
 * kraken_gpu_memory_pool_destroy - Free all pool allocations and destroy pool
 *
 * Technique: Iterates all tracked blocks, calls cudaFree on each, then deletes pool.
 */
extern "C"
void kraken_gpu_memory_pool_destroy(KrakenGPUMemoryPool* pool) {
    if (pool == nullptr) {
        return;
    }

    // Free all allocated blocks
    for (auto& block : pool->blocks) {
        if (block.ptr != nullptr) {
            cudaFree(block.ptr);
        }
    }

    delete pool;
}

/**
 * kraken_gpu_memory_pool_alloc - Allocate from pool with block reuse
 *
 * Technique: Thread-safe (mutex-locked). First scans existing freed blocks
 * for one of sufficient size; if found, marks it in-use and returns its
 * pointer (O(n) scan, first-fit). Otherwise calls cudaMalloc and adds
 * the new block to the pool's tracking list.
 */
extern "C"
void* kraken_gpu_memory_pool_alloc(KrakenGPUMemoryPool* pool, size_t size) {
    if (pool == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(pool->mutex);

    // Try to reuse existing block
    for (size_t i = 0; i < pool->blocks.size(); i++) {
        auto& block = pool->blocks[i];
        if (!block.in_use && block.size >= size) {
            // Reuse this block
            block.in_use = true;
            pool->total_in_use += block.size;
            return block.ptr;
        }
    }

    // No suitable block found, allocate new one
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Memory pool allocation failed: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }

    // Add to pool
    size_t index = pool->blocks.size();
    pool->blocks.push_back({ptr, size, true});
    pool->ptr_to_index[ptr] = index;
    pool->total_allocated += size;
    pool->total_in_use += size;

    return ptr;
}

/**
 * kraken_gpu_memory_pool_free - Return allocation to pool for future reuse
 *
 * Technique: Thread-safe. Looks up pointer in hash map, marks block as not
 * in-use (does NOT call cudaFree - block is retained for reuse).
 */
extern "C"
void kraken_gpu_memory_pool_free(KrakenGPUMemoryPool* pool, void* ptr) {
    if (pool == nullptr || ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(pool->mutex);

    auto it = pool->ptr_to_index.find(ptr);
    if (it == pool->ptr_to_index.end()) {
        fprintf(stderr, "Warning: Attempting to free unknown pointer\n");
        return;
    }

    size_t index = it->second;
    auto& block = pool->blocks[index];

    if (!block.in_use) {
        fprintf(stderr, "Warning: Attempting to free already-freed memory\n");
        return;
    }

    block.in_use = false;
    pool->total_in_use -= block.size;
}

/**
 * kraken_gpu_memory_pool_stats - Query pool usage statistics
 *
 * Technique: Thread-safe read of total allocated bytes, in-use bytes,
 * and number of allocation blocks.
 */
extern "C"
void kraken_gpu_memory_pool_stats(KrakenGPUMemoryPool* pool,
                                   size_t* total_allocated,
                                   size_t* total_in_use,
                                   int* num_allocations) {
    if (pool == nullptr) {
        if (total_allocated) *total_allocated = 0;
        if (total_in_use) *total_in_use = 0;
        if (num_allocations) *num_allocations = 0;
        return;
    }

    std::lock_guard<std::mutex> lock(pool->mutex);

    if (total_allocated) {
        *total_allocated = pool->total_allocated;
    }

    if (total_in_use) {
        *total_in_use = pool->total_in_use;
    }

    if (num_allocations) {
        *num_allocations = static_cast<int>(pool->blocks.size());
    }
}
