#include <cuda_runtime.h>
#include "gpu_kernels.h"

// 核函数实现
__global__ void compute_pq_distances(const uint8_t* d_vectors, const float* d_pq_dists, const uint32_t* d_ids,
                                     uint64_t n_ids, uint64_t n_chunks, float* d_dists_out) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx >= n_ids) return;

    // uint32_t id = d_ids[idx];
    // float dist = 0.0f;

    // // 通过ID索引对应的PQ向量
    // const uint8_t* pq_vector = d_vectors + id * n_chunks;
    // for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
    //     uint8_t pq_code = pq_vector[chunk];
    //     dist += d_pq_dists[chunk * 256 + pq_code];
    // }
    // d_dists_out[idx] = dist;

    if (idx < n_ids * n_chunks)
    {
        int id_offset = idx / n_chunks;
        int ch_offset = idx % n_chunks;
        float *ptr = (float *)d_pq_dists;
        ptr += 256 * ch_offset;
        uint32_t v_id = d_ids[id_offset];
        uint32_t offset = sizeof(uint8_t) * v_id * n_chunks + ch_offset;
        // uint32_t offset = v_id * n_chunks + ch_offset;
        atomicAdd(&d_dists_out[id_offset], ptr[d_vectors[offset]]);
    }
}

void call_compute_pq_distances(const uint8_t* d_vectors, const float* d_pq_dists, const uint32_t* d_ids,
                               uint64_t n_ids, uint64_t n_chunks, float* d_dists_out) {
    // dim3 threadsPerBlock(256);
    // dim3 numBlocks((n_ids * n_chunks + threadsPerBlock.x - 1) / threadsPerBlock.x);

    int block = 256;
    int grid = (n_ids * n_chunks + block - 1) / block;
    compute_pq_distances<<<grid, block>>>(
        d_vectors, d_pq_dists, d_ids, n_ids, n_chunks, d_dists_out);
    cudaDeviceSynchronize();
}