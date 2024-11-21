#include <cstdint> // 定义 uint8_t, uint32_t 等

#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

void call_compute_pq_distances(const uint8_t* d_vectors, const float* d_pq_dists, const uint32_t* d_ids,
                               uint64_t n_ids, uint64_t n_chunks, float* d_dists_out);

#endif // GPU_KERNELS_H
