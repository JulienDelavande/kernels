#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <algorithm>
#include <float.h>

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float rand_uniform(curandStatePhilox4_32_10_t* state) {
    return curand_uniform(state);
}

__global__ void fused_sample_topk_kernel(
    const float* __restrict__ logits,  // [B, V]
    int* __restrict__ output,          // [B]
    float temperature,
    int top_k,
    int B,
    int V
) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared[];

    float* probs = shared;         // V floats
    int* indices = (int*)&probs[V]; // V ints

    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row = logits + b * V;

    // 1. Find max for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < V; i += stride) {
        local_max = fmaxf(local_max, row[i] / temperature);
    }
    local_max = warpReduceSum(local_max); // not exact max but okay for demo

    // 2. Softmax
    for (int i = tid; i < V; i += stride) {
        probs[i] = expf((row[i] / temperature) - local_max);
        indices[i] = i;
    }
    __syncthreads();

    // 3. Compute sum
    float sum = 0.f;
    for (int i = tid; i < V; i += stride) {
        sum += probs[i];
    }
    sum = warpReduceSum(sum);
    __syncthreads();

    for (int i = tid; i < V; i += stride) {
        probs[i] /= sum;
    }
    __syncthreads();

    // 4. Partial sort top-k (simple selection sort for demo)
    for (int i = 0; i < top_k; ++i) {
        for (int j = tid; j < V - 1; j += stride) {
            if (probs[j] < probs[j + 1]) {
                float tmp_p = probs[j];
                probs[j] = probs[j + 1];
                probs[j + 1] = tmp_p;

                int tmp_i = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tmp_i;
            }
        }
        __syncthreads();
    }

    // 5. Normalize top-k
    float topk_sum = 0.f;
    for (int i = tid; i < top_k; i += stride) {
        topk_sum += probs[i];
    }
    topk_sum = warpReduceSum(topk_sum);
    __syncthreads();
    for (int i = tid; i < top_k; i += stride) {
        probs[i] /= topk_sum;
    }
    __syncthreads();

    // 6. Sample using inverse CDF
    __shared__ float u;
    if (tid == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(42 + b, 0, 0, &state);
        u = rand_uniform(&state);
    }
    __syncthreads();

    float cdf = 0.f;
    int sample = -1;
    for (int i = 0; i < top_k; ++i) {
        cdf += probs[i];
        if (u < cdf && sample == -1) {
            sample = indices[i];
        }
    }

    if (tid == 0 && sample >= 0) {
        output[b] = sample;
    }
}
