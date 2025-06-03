#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <tuple>
#include <torch/torch.h>
#include <curand_kernel.h>
#include<chrono>

#include <mma.h>

#include <bits/stdc++.h>
#define WARP 2
#define WARP4MMA 8
#define ReduceWARP 2
#define EMB_DIM 100

#define BLK_H 16 
#define BLK_W 8

using namespace std;
using namespace nvcuda;


__global__ void forward_ComplEx_kernel_mma(
    const float* __restrict__ src_pos_embeddings,
    const float* __restrict__ dst_pos_embeddings,
    const float* __restrict__ src_relation_embeddings,
    const float* __restrict__ dst_relation_embeddings, 
    const float* __restrict__ src_neg_embeddings, 
    const float* __restrict__ dst_neg_embeddings,
    int pos_num,
    int neg_num,
    int emb_dim,
    int chunk_num,
    float* lhs_pos_scores,
    float* rhs_pos_scores,
    float* lhs_neg_scores,
    float* rhs_neg_scores, 
    float* adjusted_src_pos,
    float* adjusted_dst_pos
){
    // int row = blockIdx.x * BLK_H + threadIdx.y / 2;
    // int tid = threadIdx.x + (threadIdx.y % 2) * 32 + row * emb_dim;
    float pos_score_lhs = 0.0, neg_score_lhs = 0.0, pos_score_rhs = 0.0, neg_score_rhs = 0.0;
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    // __shared__ float adj_src[(EMB_DIM + 4) * BLK_H];
    __shared__ float tmp_store[WARP4MMA];
    float adj_src1, adj_src2;
    // lhs_loss
    for(int row = blockIdx.x * BLK_H + threadIdx.y / 2; row < (blockIdx.x + 1) * BLK_H; row += (WARP4MMA / 2)){
        int tid = threadIdx.x + (threadIdx.y % 2) * 32 + row * emb_dim;
        if(tid % emb_dim < emb_dim / 2){
            adj_src1 = src_pos_embeddings[tid] * src_relation_embeddings[tid] - src_pos_embeddings[tid + emb_dim / 2] * src_relation_embeddings[tid + emb_dim / 2];
            adj_src2 = src_pos_embeddings[tid] * src_relation_embeddings[tid + emb_dim / 2] + src_pos_embeddings[tid + emb_dim / 2] * src_relation_embeddings[tid];
            pos_score_lhs = (adj_src1 * dst_pos_embeddings[tid] + adj_src2 * dst_pos_embeddings[tid + emb_dim / 2]);
            adjusted_src_pos[tid] = adj_src1;
            adjusted_src_pos[tid + emb_dim / 2] = adj_src2;
        }
        for(int it = 16; it >= 1; it /= 2)
            pos_score_lhs += __shfl_down_sync(threadIdx.y % 2 == 1 ? 0x0003ffff : 0xffffffff, pos_score_lhs, it);
        if(threadIdx.x == 0) tmp_store[threadIdx.y] = pos_score_lhs;
        __syncthreads();
        if (threadIdx.y % 2 == 0 && threadIdx.x == 0) 
        {
            lhs_pos_scores[row] = tmp_store[threadIdx.y] + tmp_store[threadIdx.y + 1];
        }
        __syncthreads();
        // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) printf("AA %f, %d\n", adj_src[104], row);
    }

    // __threadfence();
    // __syncthreads();
    
    for(unsigned warp_id = threadIdx.y; warp_id < (neg_num + BLK_H - 1) / BLK_H; warp_id += WARP4MMA){
        wmma::fill_fragment(acc_frag, 0.0f);
        // __syncthreads();
        // __threadfence();
        for (int i = 0; i < (emb_dim + BLK_W - 1) / BLK_W; i++) {
            // wmma::load_matrix_sync(a_frag, adj_src + i * BLK_W, emb_dim + 4);
            wmma::load_matrix_sync(a_frag, adjusted_src_pos + blockIdx.x*BLK_H*emb_dim + i * BLK_W, emb_dim);
            wmma::load_matrix_sync(b_frag, dst_neg_embeddings + (((blockIdx.x * BLK_H + threadIdx.y / 2) / (pos_num / chunk_num)) * neg_num + warp_id * BLK_H) * (emb_dim + 4) + i * BLK_W, emb_dim + 4);
    
            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        #pragma unroll
        for (unsigned t = 0; t < acc_frag.num_elements; t++) {
            acc_frag.x[t] = expf(acc_frag.x[t]);
        }
        wmma::store_matrix_sync(lhs_neg_scores + blockIdx.x * BLK_H * (neg_num+8) + warp_id * BLK_H, acc_frag, neg_num+8, wmma::mem_row_major);
    }
    // rhs_loss
    for(unsigned row = blockIdx.x * BLK_H + threadIdx.y / 2; row < (blockIdx.x + 1) * BLK_H; row += (WARP4MMA / 2)){
        int tid = threadIdx.x + (threadIdx.y % 2) * 32 + row * emb_dim;
        if(tid % emb_dim < emb_dim / 2){
            adj_src1 = dst_pos_embeddings[tid] * dst_relation_embeddings[tid] - dst_pos_embeddings[tid + emb_dim / 2] * dst_relation_embeddings[tid + emb_dim / 2];
            adj_src2 = dst_pos_embeddings[tid] * dst_relation_embeddings[tid + emb_dim / 2] + dst_pos_embeddings[tid + emb_dim / 2] * dst_relation_embeddings[tid];
            pos_score_rhs = (adj_src1 * src_pos_embeddings[tid] + adj_src2 * src_pos_embeddings[tid + emb_dim / 2]);
            // adj_src[tid % emb_dim + (row % BLK_H) * emb_dim] = adj_src1;
            // adj_src[tid % emb_dim + (row % BLK_H) * emb_dim + emb_dim / 2] = adj_src2;
            adjusted_dst_pos[tid] = adj_src1;
            adjusted_dst_pos[tid + emb_dim / 2] = adj_src2;
        }
        for(int it = 16; it >= 1; it /= 2)
            pos_score_rhs += __shfl_down_sync(threadIdx.y % 2 == 1 ? 0x0003ffff : 0xffffffff, pos_score_rhs, it);
        if(threadIdx.x == 0) tmp_store[threadIdx.y] = pos_score_rhs;
        __syncthreads();
        if (threadIdx.y % 2 == 0 && threadIdx.x == 0) 
        {
            rhs_pos_scores[row] = tmp_store[threadIdx.y] + tmp_store[threadIdx.y + 1];
            // if(row == 0 && blockIdx.x == 0 && threadIdx.y == 0) printf("pos_score: %f\n", pos_score_lhs_lst[0]);
        }
        __syncthreads();
    }
    
    // __threadfence();
    // __syncthreads();
    
    for(unsigned warp_id = threadIdx.y; warp_id < (neg_num + BLK_H - 1) / BLK_H; warp_id += WARP4MMA){
        wmma::fill_fragment(acc_frag, 0.0f);
        // __syncthreads();
        // __threadfence();
        for (unsigned i = 0; i < (emb_dim + BLK_W - 1) / BLK_W; i++) {
            // wmma::load_matrix_sync(a_frag, adj_src + i * BLK_W, BLK_W);
            wmma::load_matrix_sync(a_frag, adjusted_dst_pos + (blockIdx.x*BLK_H)*emb_dim + i * BLK_W, emb_dim);
            wmma::load_matrix_sync(b_frag, src_neg_embeddings + (((blockIdx.x * BLK_H + threadIdx.y / 2) / (pos_num / chunk_num)) * neg_num + warp_id * BLK_H) * (emb_dim + 4) + i * BLK_W, emb_dim + 4);

            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                // if(blockIdx.x == 624 && warp_id == 5 && i == 0) printf("A: %f %d\n", a_frag.x[t], threadIdx.x);
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                // if(blockIdx.x == 624 && (warp_id == 0 || warp_id == 2) && i == 0) printf("A: %f %d %d\n", a_frag.x[t], threadIdx.x, warp_id);
            }
    
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                // if(blockIdx.x == 624 && warp_id == 5 && i == 0) printf("%f %d\n", b_frag.x[t], threadIdx.x);
            }

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        #pragma unroll
        for (unsigned t = 0; t < acc_frag.num_elements; t++) {
            acc_frag.x[t] = expf(acc_frag.x[t]);
        }
        wmma::store_matrix_sync(rhs_neg_scores + blockIdx.x * BLK_H * (neg_num+8) + warp_id * BLK_H, acc_frag, neg_num+8, wmma::mem_row_major);
    }
    
}

__global__ void grad_cal_fused(
    const float* __restrict__ src_pos_embeddings,
    const float* __restrict__ dst_pos_embeddings,
    const float* __restrict__ src_relation_embeddings,
    const float* __restrict__ dst_relation_embeddings, 
    const float* __restrict__ tmp_grad_src,
    const float* __restrict__ tmp_grad_dst,
    const float* __restrict__ tmp1,
    const float* __restrict__ tmp2,
    int emb_dim,
    float* grad_src,
    float* grad_dst,
    float* grad_src_rel,
    float* grad_dst_rel
){
    int row = blockIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x + row * emb_dim;
    if(tid % emb_dim < emb_dim / 2){
        // grad_src[tid] = (src_relation_embeddings[tid] * dst_neg_embeddings[tid] + src_relation_embeddings[tid + emb_dim / 2] * dst_neg_embeddings[tid + emb_dim / 2]);
        // grad_src[tid + emb_dim / 2] = (src_relation_embeddings[tid] * dst_neg_embeddings[tid + emb_dim / 2] - src_relation_embeddings[tid + emb_dim / 2] * dst_neg_embeddings[tid]);
        // grad_dst[tid] = (dst_relation_embeddings[tid] * src_neg_embeddings[tid] + dst_relation_embeddings[tid + emb_dim / 2] * src_neg_embeddings[tid + emb_dim / 2]);
        // grad_dst[tid + emb_dim / 2] = (dst_relation_embeddings[tid] * src_neg_embeddings[tid + emb_dim / 2] - dst_relation_embeddings[tid + emb_dim / 2] * src_neg_embeddings[tid]);
        // grad_src_rel[tid] = (src_pos_embeddings[tid] * dst_neg_embeddings[tid] + src_pos_embeddings[tid + emb_dim / 2] * dst_neg_embeddings[tid + emb_dim / 2]);
        // grad_src_rel[tid + emb_dim / 2] = (src_pos_embeddings[tid] * dst_neg_embeddings[tid + emb_dim / 2] - src_pos_embeddings[tid + emb_dim / 2] * dst_neg_embeddings[tid]);
        // grad_dst_rel[tid] = (dst_pos_embeddings[tid] * src_neg_embeddings[tid] + dst_pos_embeddings[tid + emb_dim / 2] * src_neg_embeddings[tid + emb_dim / 2]); 
        // grad_dst_rel[tid + emb_dim / 2] = (dst_pos_embeddings[tid] * src_neg_embeddings[tid + emb_dim / 2] - dst_pos_embeddings[tid + emb_dim / 2] * src_neg_embeddings[tid]);

        grad_src[tid] = (tmp1[row] * ((src_relation_embeddings[tid] * tmp_grad_src[tid] + src_relation_embeddings[tid + emb_dim / 2] * tmp_grad_src[tid + emb_dim / 2]) - src_relation_embeddings[tid] * dst_pos_embeddings[tid] - src_relation_embeddings[tid + emb_dim / 2] * dst_pos_embeddings[tid + emb_dim / 2])) - (tmp2[row] * (dst_relation_embeddings[tid] * dst_pos_embeddings[tid] - dst_relation_embeddings[tid + emb_dim / 2] * dst_pos_embeddings[tid + emb_dim / 2]));
        grad_src[tid + emb_dim / 2] = tmp1[row] * ((src_relation_embeddings[tid] * tmp_grad_src[tid + emb_dim / 2] - src_relation_embeddings[tid + emb_dim / 2] * tmp_grad_src[tid]) - src_relation_embeddings[tid] * dst_pos_embeddings[tid + emb_dim / 2] + src_relation_embeddings[tid + emb_dim / 2] * dst_pos_embeddings[tid]) - tmp2[row] * (dst_relation_embeddings[tid] * dst_pos_embeddings[tid + emb_dim / 2] + dst_relation_embeddings[tid + emb_dim / 2] * dst_pos_embeddings[tid]);
        grad_dst[tid] = tmp1[row] * (src_relation_embeddings[tid + emb_dim / 2] * src_pos_embeddings[tid + emb_dim / 2] - src_relation_embeddings[tid] * src_pos_embeddings[tid]) + tmp2[row] * ((dst_relation_embeddings[tid] * tmp_grad_dst[tid] + dst_relation_embeddings[tid + emb_dim / 2] * tmp_grad_dst[tid + emb_dim / 2]) - dst_relation_embeddings[tid] * src_pos_embeddings[tid] - dst_relation_embeddings[tid + emb_dim / 2] * src_pos_embeddings[tid + emb_dim / 2]);
        grad_dst[tid + emb_dim / 2] = -tmp1[row] * (src_relation_embeddings[tid] * src_pos_embeddings[tid + emb_dim / 2] + src_relation_embeddings[tid + emb_dim / 2] * src_pos_embeddings[tid]) + tmp2[row] * ((dst_relation_embeddings[tid] * tmp_grad_dst[tid + emb_dim / 2] - dst_relation_embeddings[tid + emb_dim / 2] * tmp_grad_dst[tid]) - dst_relation_embeddings[tid] * src_pos_embeddings[tid + emb_dim / 2] + dst_relation_embeddings[tid + emb_dim / 2] * src_pos_embeddings[tid]);

        grad_src_rel[tid] = tmp1[row] * ((src_pos_embeddings[tid] * tmp_grad_src[tid] + src_pos_embeddings[tid + emb_dim / 2] * tmp_grad_src[tid + emb_dim / 2]) - src_pos_embeddings[tid] * dst_pos_embeddings[tid] - src_pos_embeddings[tid + emb_dim / 2] * dst_pos_embeddings[tid + emb_dim / 2]);
        grad_src_rel[tid + emb_dim / 2] = tmp1[row] * ((src_pos_embeddings[tid] * tmp_grad_src[tid + emb_dim / 2] - src_pos_embeddings[tid + emb_dim / 2] * tmp_grad_src[tid]) - src_pos_embeddings[tid] * dst_pos_embeddings[tid + emb_dim / 2] + src_pos_embeddings[tid + emb_dim / 2] * dst_pos_embeddings[tid]);
        grad_dst_rel[tid] = tmp2[row] * ((dst_pos_embeddings[tid] * tmp_grad_dst[tid] + dst_pos_embeddings[tid + emb_dim / 2] * tmp_grad_dst[tid + emb_dim / 2]) - dst_pos_embeddings[tid] * src_pos_embeddings[tid] - dst_pos_embeddings[tid + emb_dim / 2] * src_pos_embeddings[tid + emb_dim / 2]); 
        grad_dst_rel[tid + emb_dim / 2] = tmp2[row] * ((dst_pos_embeddings[tid] * tmp_grad_dst[tid + emb_dim / 2] - dst_pos_embeddings[tid + emb_dim / 2] * tmp_grad_dst[tid]) - dst_pos_embeddings[tid] * src_pos_embeddings[tid + emb_dim / 2] + dst_pos_embeddings[tid + emb_dim / 2] * src_pos_embeddings[tid]);
    }
}

torch::Tensor ComplexHadamardOperator(const torch::Tensor &embs, const torch::Tensor &rels) {
    if (!rels.defined()) {
        // cout <<"BB" << endl;
        return embs;
    }
    int dim = embs.size(1);

    int real_len = dim / 2;
    int imag_len = dim - dim / 2;

    torch::Tensor real_emb = embs.narrow(1, 0, real_len);
    torch::Tensor imag_emb = embs.narrow(1, real_len, imag_len);

    torch::Tensor real_rel = rels.narrow(1, 0, real_len);
    torch::Tensor imag_rel = rels.narrow(1, real_len, imag_len);

    torch::Tensor out = torch::zeros_like(embs);

    out.narrow(1, 0, real_len) = (real_emb * real_rel) - (imag_emb * imag_rel);
    out.narrow(1, real_len, imag_len) = (real_emb * imag_rel) + (imag_emb * real_rel);

    return out;
}

tuple<torch::Tensor, torch::Tensor> DotCompare(const torch::Tensor &src, const torch::Tensor &dst, const torch::Tensor &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int) ceil((float) num_pos / num_chunks);

    // apply relation operator
    torch::Tensor adjusted_src = src;
    torch::Tensor adjusted_dst = dst;

    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        adjusted_src = torch::nn::functional::pad(adjusted_src, options);
        adjusted_dst = torch::nn::functional::pad(adjusted_dst, options);
    }

    torch::Tensor pos_scores = (adjusted_src * adjusted_dst).sum(-1);
    adjusted_src = adjusted_src.view({num_chunks, num_per_chunk, src.size(1)});
    // auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor neg_scores = adjusted_src.bmm(negs.transpose(-1, -2)).flatten(0, 1);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed = end - start;
    return make_tuple(move(pos_scores), move(neg_scores));
}

torch::Tensor SoftMax(const torch::Tensor &pos_scores, const torch::Tensor &neg_scores) {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    auto scores = torch::cat({pos_scores.unsqueeze(1), neg_scores.logsumexp(1, true)}, 1);
    // cout << scores[0] << endl;
    torch::nn::functional::CrossEntropyFuncOptions options;
    options.reduction(torch::kSum);
    auto loss = torch::nn::functional::cross_entropy(scores, pos_scores.new_zeros({}, device_options).expand(pos_scores.size(0)), options);
    return loss;
}

__global__ void grad_cal_fused_no_rel(
    const float* __restrict__ src_pos_embeddings,
    const float* __restrict__ dst_pos_embeddings,
    const float* __restrict__ tmp_grad_src,
    const float* __restrict__ tmp_grad_dst,
    const float* __restrict__ tmp1,
    const float* __restrict__ tmp2,
    int emb_dim,
    float* grad_src,
    float* grad_dst,
    float* grad_src_rel,
    float* grad_dst_rel
){
    int row = blockIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x + row * emb_dim;
    if(tid % emb_dim < emb_dim / 2){
        grad_src[tid] = (tmp1[row] * (tmp_grad_src[tid] -  dst_pos_embeddings[tid])) - (tmp2[row] * dst_pos_embeddings[tid]);
        grad_src[tid + emb_dim / 2] = tmp1[row] * (tmp_grad_src[tid + emb_dim / 2] -  dst_pos_embeddings[tid + emb_dim / 2]) - tmp2[row] * (dst_pos_embeddings[tid + emb_dim / 2]);
        grad_dst[tid] = tmp1[row] * (- src_pos_embeddings[tid]) + tmp2[row] * (tmp_grad_dst[tid] - src_pos_embeddings[tid]);
        grad_dst[tid + emb_dim / 2] = -tmp1[row] * (src_pos_embeddings[tid + emb_dim / 2]) + tmp2[row] * (tmp_grad_dst[tid + emb_dim / 2] - src_pos_embeddings[tid + emb_dim / 2]);
    }
}

__global__ void forward_ComplEx_kernel_mma_no_rel(
    const float* __restrict__ src_pos_embeddings,
    const float* __restrict__ dst_pos_embeddings,
    const float* __restrict__ src_neg_embeddings, 
    const float* __restrict__ dst_neg_embeddings,
    int pos_num,
    int neg_num,
    int emb_dim,
    int chunk_num,
    float* lhs_pos_scores,
    float* rhs_pos_scores,
    float* lhs_neg_scores,
    float* rhs_neg_scores, 
    float* adjusted_src_pos,
    float* adjusted_dst_pos
){
    // int row = blockIdx.x * BLK_H + threadIdx.y / 2;
    // int tid = threadIdx.x + (threadIdx.y % 2) * 32 + row * emb_dim;

    float pos_score_lhs = 0.0, neg_score_lhs = 0.0, pos_score_rhs = 0.0, neg_score_rhs = 0.0;
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    // __shared__ float adj_src[(EMB_DIM + 4) * BLK_H];
    __shared__ float tmp_store[WARP4MMA];
    float adj_src1, adj_src2;
    // lhs_loss
    for(int row = blockIdx.x * BLK_H + threadIdx.y / 2; row < (blockIdx.x + 1) * BLK_H; row += (WARP4MMA / 2)){
        int tid = threadIdx.x + (threadIdx.y % 2) * 32 + row * emb_dim;
        if(tid % emb_dim < emb_dim / 2){
            adj_src1 = src_pos_embeddings[tid];
            adj_src2 = src_pos_embeddings[tid + emb_dim / 2];
            pos_score_lhs = (adj_src1 * dst_pos_embeddings[tid] + adj_src2 * dst_pos_embeddings[tid + emb_dim / 2]);
            adjusted_src_pos[tid] = adj_src1;
            adjusted_src_pos[tid + emb_dim / 2] = adj_src2;
        }
        for(int it = 16; it >= 1; it /= 2)
            pos_score_lhs += __shfl_down_sync(threadIdx.y % 2 == 1 ? 0x0003ffff : 0xffffffff, pos_score_lhs, it);
        if(threadIdx.x == 0) tmp_store[threadIdx.y] = pos_score_lhs;
        __syncthreads();
        if (threadIdx.y % 2 == 0 && threadIdx.x == 0) 
        {
            lhs_pos_scores[row] = tmp_store[threadIdx.y] + tmp_store[threadIdx.y + 1];
        }
        __syncthreads();
    }

    // __threadfence();
    // __syncthreads();
    
    for(unsigned warp_id = threadIdx.y; warp_id < (neg_num + BLK_H - 1) / BLK_H; warp_id += WARP4MMA){
        wmma::fill_fragment(acc_frag, 0.0f);
        // __syncthreads();
        // __threadfence();
        for (int i = 0; i < (emb_dim + BLK_W - 1) / BLK_W; i++) {
            // wmma::load_matrix_sync(a_frag, adj_src + i * BLK_W, emb_dim + 4);
            wmma::load_matrix_sync(a_frag, adjusted_src_pos + blockIdx.x*BLK_H*emb_dim + i * BLK_W, emb_dim);
            wmma::load_matrix_sync(b_frag, dst_neg_embeddings + (((blockIdx.x * BLK_H + threadIdx.y / 2) / (pos_num / chunk_num)) * neg_num + warp_id * BLK_H) * (emb_dim + 4) + i * BLK_W, emb_dim + 4);
    
            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        #pragma unroll
        for (unsigned t = 0; t < acc_frag.num_elements; t++) {
            acc_frag.x[t] = expf(acc_frag.x[t]);
        }
        wmma::store_matrix_sync(lhs_neg_scores + blockIdx.x * BLK_H * (neg_num+8) + warp_id * BLK_H, acc_frag, neg_num+8, wmma::mem_row_major);
    }
    // rhs_loss
    for(unsigned row = blockIdx.x * BLK_H + threadIdx.y / 2; row < (blockIdx.x + 1) * BLK_H; row += (WARP4MMA / 2)){
        int tid = threadIdx.x + (threadIdx.y % 2) * 32 + row * emb_dim;
        if(tid % emb_dim < emb_dim / 2){
            adj_src1 = dst_pos_embeddings[tid];
            adj_src2 = dst_pos_embeddings[tid + emb_dim / 2];
            pos_score_rhs = (adj_src1 * src_pos_embeddings[tid] + adj_src2 * src_pos_embeddings[tid + emb_dim / 2]);
            // adj_src[tid % emb_dim + (row % BLK_H) * emb_dim] = adj_src1;
            // adj_src[tid % emb_dim + (row % BLK_H) * emb_dim + emb_dim / 2] = adj_src2;
            adjusted_dst_pos[tid] = adj_src1;
            adjusted_dst_pos[tid + emb_dim / 2] = adj_src2;
        }
        for(int it = 16; it >= 1; it /= 2)
            pos_score_rhs += __shfl_down_sync(threadIdx.y % 2 == 1 ? 0x0003ffff : 0xffffffff, pos_score_rhs, it);
        if(threadIdx.x == 0) tmp_store[threadIdx.y] = pos_score_rhs;
        __syncthreads();
        if (threadIdx.y % 2 == 0 && threadIdx.x == 0) 
        {
            rhs_pos_scores[row] = tmp_store[threadIdx.y] + tmp_store[threadIdx.y + 1];
        }
        __syncthreads();
    }
    
    // __threadfence();
    // __syncthreads();
    
    for(unsigned warp_id = threadIdx.y; warp_id < (neg_num + BLK_H - 1) / BLK_H; warp_id += WARP4MMA){
        wmma::fill_fragment(acc_frag, 0.0f);
        // __syncthreads();
        // __threadfence();
        for (unsigned i = 0; i < (emb_dim + BLK_W - 1) / BLK_W; i++) {
            // wmma::load_matrix_sync(a_frag, adj_src + i * BLK_W, BLK_W);
            wmma::load_matrix_sync(a_frag, adjusted_dst_pos + (blockIdx.x*BLK_H)*emb_dim + i * BLK_W, emb_dim);
            wmma::load_matrix_sync(b_frag, src_neg_embeddings + (((blockIdx.x * BLK_H + threadIdx.y / 2) / (pos_num / chunk_num)) * neg_num + warp_id * BLK_H) * (emb_dim + 4) + i * BLK_W, emb_dim + 4);

            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
    
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        #pragma unroll
        for (unsigned t = 0; t < acc_frag.num_elements; t++) {
            acc_frag.x[t] = expf(acc_frag.x[t]);
        }
        wmma::store_matrix_sync(rhs_neg_scores + blockIdx.x * BLK_H * (neg_num+8) + warp_id * BLK_H, acc_frag, neg_num+8, wmma::mem_row_major);
    }
}

void forward_our_front_mma(
    torch::Tensor &src_pos_embeddings_, 
    torch::Tensor &src_relation_emebeddings_, 
    torch::Tensor &dst_pos_embeddings_, 
    torch::Tensor &dst_relation_emebeddings_, 
    torch::Tensor &src_all_neg_embeddings_, 
    torch::Tensor &dst_all_neg_embeddings_, 
    bool train, int pos_num, int neg_num, int emb_dim, int chunk_num, 
    torch::Tensor &grad_src, torch::Tensor &grad_dst, torch::Tensor &grad_src_rel, 
    torch::Tensor &grad_dst_rel, torch::Tensor &grad_src_neg, torch::Tensor &grad_dst_neg, 
    torch::Tensor &lhs_score_exp, torch::Tensor &rhs_score_exp, 
    torch::Tensor &lhs_pos_scores, torch::Tensor &rhs_pos_scores, 
    torch::Tensor &adjusted_dst_pos, torch::Tensor &adjusted_src_pos, 
    torch::Tensor &new_src_neg_c, torch::Tensor &new_dst_neg_c, int batch_size, int rel_num) {
    // torch::Tensor lhs_pos_scores;
    torch::Tensor lhs_neg_scores;
    // torch::Tensor rhs_pos_scores;
    torch::Tensor rhs_neg_scores;

    torch::Tensor loss;
    torch::Tensor lhs_loss;
    torch::Tensor rhs_loss;

    int regularization_coef = 1;
    int regularization_norm = 3;

    dim3 block(32, WARP4MMA, 1);
    dim3 grid(625*chunk_num, 1, 1);
    
    if (train) {
        if(rel_num > 1)
            forward_ComplEx_kernel_mma<<<grid, block>>>(src_pos_embeddings_.data<float>(),
                                                    dst_pos_embeddings_.data<float>(),
                                                    src_relation_emebeddings_.data<float>(),
                                                    dst_relation_emebeddings_.data<float>(),
                                                    new_src_neg_c.data<float>(),
                                                    new_dst_neg_c.data<float>(),
                                                    pos_num,
                                                    neg_num,
                                                    emb_dim,
                                                    chunk_num,
                                                    rhs_pos_scores.data<float>(),
                                                    lhs_pos_scores.data<float>(),
                                                    rhs_score_exp.data<float>(),
                                                    lhs_score_exp.data<float>(),
                                                    adjusted_src_pos.data<float>(),
                                                    adjusted_dst_pos.data<float>());
        else{
            forward_ComplEx_kernel_mma_no_rel<<<grid, block>>>(src_pos_embeddings_.data<float>(),
                                                    dst_pos_embeddings_.data<float>(),
                                                    new_src_neg_c.data<float>(),
                                                    new_dst_neg_c.data<float>(),
                                                    pos_num,
                                                    neg_num,
                                                    emb_dim,
                                                    chunk_num,
                                                    rhs_pos_scores.data<float>(),
                                                    lhs_pos_scores.data<float>(),
                                                    rhs_score_exp.data<float>(),
                                                    lhs_score_exp.data<float>(),
                                                    adjusted_src_pos.data<float>(),
                                                    adjusted_dst_pos.data<float>());
        }
        cudaDeviceSynchronize();

        auto rhs_score_exp_nar = rhs_score_exp.view({pos_num, neg_num+8}).narrow(1, 0, neg_num);
        auto lhs_score_exp_nar = lhs_score_exp.view({pos_num, neg_num+8}).narrow(1, 0, neg_num);
        // auto lhs_max = get<0>(lhs_score_exp_nar.max(1, true)), rhs_max = get<0>(rhs_score_exp_nar.max(1, true));
        // lhs_score_exp_nar = torch::exp(lhs_score_exp_nar - lhs_max);
        // rhs_score_exp_nar = torch::exp(rhs_score_exp_nar - rhs_max);

        adjusted_src_pos = adjusted_src_pos.view({pos_num, emb_dim});
        adjusted_dst_pos = adjusted_dst_pos.view({pos_num, emb_dim});

        auto lhs_score_exp_sum = (lhs_score_exp_nar.sum(-1)).unsqueeze(1);
        auto rhs_score_exp_sum = (rhs_score_exp_nar.sum(-1)).unsqueeze(1);

        auto tmp1 = 1 / (torch::exp((lhs_pos_scores.unsqueeze(1) - torch::log(lhs_score_exp_sum))) + 1);
        auto tmp2 = 1 / (torch::exp((rhs_pos_scores.unsqueeze(1) - torch::log(rhs_score_exp_sum))) + 1);

        lhs_score_exp_sum = lhs_score_exp_sum.repeat({1,emb_dim});
        rhs_score_exp_sum = rhs_score_exp_sum.repeat({1,emb_dim});
        
        torch::Tensor op1 = (adjusted_dst_pos * tmp1.repeat({1,emb_dim}) / (lhs_score_exp_sum)).view({chunk_num, pos_num / chunk_num, emb_dim});
        torch::Tensor op2 = (adjusted_src_pos * tmp2.repeat({1,emb_dim}) / rhs_score_exp_sum).view({chunk_num, pos_num / chunk_num, emb_dim});    
        
        grad_src_neg = (lhs_score_exp_nar).view({chunk_num, pos_num / chunk_num, neg_num}).transpose(-1, -2).bmm(op1);
        grad_dst_neg = (rhs_score_exp_nar).view({chunk_num, pos_num / chunk_num, neg_num}).transpose(-1, -2).bmm(op2);

        torch::Tensor tmp_grad_src = rhs_score_exp_nar.view({chunk_num, pos_num / chunk_num, neg_num}).bmm(dst_all_neg_embeddings_);
        torch::Tensor tmp_grad_dst = lhs_score_exp_nar.view({chunk_num, pos_num / chunk_num, neg_num}).bmm(src_all_neg_embeddings_);

        if(rel_num > 1)
            grad_cal_fused<<<pos_num, WARP*32>>>(src_pos_embeddings_.flatten().data<float>(),dst_pos_embeddings_.flatten().data<float>(),src_relation_emebeddings_.flatten().data<float>(),dst_relation_emebeddings_.flatten().data<float>(), (tmp_grad_src.view({pos_num, emb_dim})/rhs_score_exp_sum).flatten().data<float>(), (tmp_grad_dst.view({pos_num, emb_dim})/lhs_score_exp_sum).flatten().data<float>(), tmp2.flatten().data<float>(), tmp1.flatten().data<float>(),emb_dim,grad_src.flatten().data<float>(),grad_dst.flatten().data<float>(),grad_src_rel.flatten().data<float>(),grad_dst_rel.flatten().data<float>());
        else
            grad_cal_fused_no_rel<<<pos_num, WARP*32>>>(src_pos_embeddings_.flatten().data<float>(),dst_pos_embeddings_.flatten().data<float>(), (tmp_grad_src.view({pos_num, emb_dim})/rhs_score_exp_sum).flatten().data<float>(), (tmp_grad_dst.view({pos_num, emb_dim})/lhs_score_exp_sum).flatten().data<float>(), tmp2.flatten().data<float>(), tmp1.flatten().data<float>(),emb_dim,grad_src.flatten().data<float>(),grad_dst.flatten().data<float>(),grad_src_rel.flatten().data<float>(),grad_dst_rel.flatten().data<float>());
        cudaDeviceSynchronize();
    }
    
}

void forward_our_front_mma_eva(
    torch::Tensor &src_pos_embeddings_, 
    torch::Tensor &src_relation_emebeddings_, 
    torch::Tensor &dst_pos_embeddings_, 
    torch::Tensor &dst_relation_emebeddings_, 
    torch::Tensor &src_all_neg_embeddings_, 
    torch::Tensor &dst_all_neg_embeddings_, 
    bool train, int pos_num, int neg_num, int emb_dim, int chunk_num, int batch_size, float &total_auc, torch::Tensor &all_ranks) {
    
    torch::Tensor lhs_pos_scores;
    torch::Tensor lhs_neg_scores;
    torch::Tensor rhs_pos_scores;
    torch::Tensor rhs_neg_scores;

    torch::Tensor loss;
    torch::Tensor lhs_loss;
    torch::Tensor rhs_loss;

    int regularization_coef = 1;
    int regularization_norm = 3;

    // corrupt destination

    // auto options = torch::TensorOptions.dtype(torch::kFloat32);
    
    torch::Tensor lhs_ranks;
    torch::Tensor rhs_ranks;
    torch::Tensor auc;
    torch::Tensor lhs_auc;
    torch::Tensor rhs_auc;

    torch::Tensor adjusted_src_pos = ComplexHadamardOperator(src_pos_embeddings_, src_relation_emebeddings_);
    tie(rhs_pos_scores, rhs_neg_scores) = DotCompare(adjusted_src_pos, dst_pos_embeddings_, dst_all_neg_embeddings_);

    torch::Tensor adjusted_dst_pos = ComplexHadamardOperator(dst_pos_embeddings_, dst_relation_emebeddings_);
    tie(lhs_pos_scores, lhs_neg_scores) = DotCompare(adjusted_dst_pos, src_pos_embeddings_, src_all_neg_embeddings_);
   
    lhs_ranks = (lhs_neg_scores >= lhs_pos_scores.unsqueeze(1)).sum(1) + 1;
    rhs_ranks = (rhs_neg_scores >= rhs_pos_scores.unsqueeze(1)).sum(1) + 1;

    auto auc_opts = torch::TensorOptions().dtype(torch::kInt64).device(src_pos_embeddings_.device());
    lhs_auc = (lhs_pos_scores.index_select(0, torch::randint(lhs_pos_scores.size(0), {batch_size}, auc_opts))
        > lhs_neg_scores.flatten(0, 1).index_select(0, torch::randint(lhs_neg_scores.flatten(0, 1).size(0), {batch_size}, auc_opts))).to(torch::kFloat32).mean();

    rhs_auc = (rhs_pos_scores.index_select(0, torch::randint(rhs_pos_scores.size(0), {batch_size}, auc_opts))
        > rhs_neg_scores.flatten(0, 1).index_select(0, torch::randint(rhs_neg_scores.flatten(0, 1).size(0), {batch_size}, auc_opts))).to(torch::kFloat32).mean();

    auc = (lhs_auc + rhs_auc) / 2;

    total_auc += auc.to(torch::kCPU).item<float>();
    if (all_ranks.numel() == 0) {
        all_ranks = torch::cat({lhs_ranks, rhs_ranks});
    } else {
        all_ranks = torch::cat({all_ranks, lhs_ranks, rhs_ranks});
    }
}

void globalSample(torch::Tensor &src_neg_indices_, torch::Tensor &dst_neg_indices_ ,int chunk_num, int neg_num, int node_num) 
{
    float fraction=0.5;
    auto ind_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
    torch::Tensor rand_idx;
    vector<torch::Tensor> ret_indices(chunk_num);
    for (int ch = 0; ch < chunk_num; ch++) {
        rand_idx = torch::randint(0, node_num, {neg_num}, ind_opts);
        ret_indices[ch] = rand_idx;
    }
    src_neg_indices_ = torch::stack(ret_indices).flatten(0, 1);

    for (int ch = 0; ch < chunk_num; ch++) {
        rand_idx = torch::randint(0, node_num, {neg_num}, ind_opts);
        ret_indices[ch] = rand_idx;
    }
    dst_neg_indices_ = torch::stack(ret_indices).flatten(0, 1);
}
__global__ void generateNegativeSamples(int *d_negative_samples,int batch_size,int chunks,int num_negative_samples,int node_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 2*chunks*num_negative_samples) return;
    if (idx < chunks*num_negative_samples) {
        unsigned long long seed=123456789;
        curandState state;
        curand_init(seed, idx, 0, &state);
        int randInt = curand(&state) % (batch_size + 1);
        d_negative_samples[idx]=randInt;
    }
    if(idx<2*chunks*num_negative_samples){
        unsigned long long seed=123456789;
        curandState state;
        curand_init(seed, idx, 0, &state);
        int randInt = curand(&state) % (node_num + 1);
        d_negative_samples[idx]=randInt;
    }

}
void evaluate(bool test, bool all, torch::Tensor &node_embedding, torch::Tensor &src_rel_embedding, torch::Tensor &dst_rel_embedding){

    // ifstream node_file("node_mapping.txt");
    // int total_nodes = 0;
    // string line;
    // while (getline(node_file, line)) {
    //     total_nodes++;
    // }
    // node_file.close();

    vector<int> all_batch;
    unsigned file_size;
    unsigned char* buff;
    // ifstream input_file;
    if(test){
        // ifstream infile("test_edges.pt", std::ios::binary);
        ifstream infile("livejournal/test_edges.pt", std::ios::binary);
        infile.seekg(0, std::ios::end);
        file_size = (unsigned)(infile.tellg());
        buff = new unsigned char[file_size];
        infile.seekg(0, std::ios::beg);
        infile.read((char*)buff, file_size);
        infile.close();
        cout << "End read data" << endl;
        // input_file.open("test_edges.txt");
    }
    else{
        ifstream infile("livejournal/valid_edges.pt", std::ios::binary);
        infile.seekg(0, std::ios::end);
        file_size = (unsigned)(infile.tellg());
        buff = new unsigned char[file_size];
        infile.seekg(0, std::ios::beg);
        infile.read((char*)buff, file_size);
        infile.close();
        cout << "End read data" << endl;
        // input_file.open("valid_edges.txt");
    }

    unsigned src, rel, tgt;
    for(unsigned i = 0; i < file_size / 12; i++) {
        src = ((unsigned*)buff)[i * 3];
        rel = ((unsigned*)buff)[i * 3 + 1];
        tgt = ((unsigned*)buff)[i * 3 + 2]; 
        all_batch.push_back(src);
        all_batch.push_back(tgt);
        all_batch.push_back(rel);
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int elements_per_group = 3;
    int pos_num = 1000, neg_num = 10000, embedding_dim = 100, chunk_num = 1;
    float learning_rate=0.1;
    int batch_size=pos_num;
    // int node_num = 14951, rel_num = 1345;
    int node_num = 4847571, rel_num = 1, edge_num = 62094395;
    int num_negative_samples = neg_num;
    torch::Tensor src_pos_c, dst_pos_c, src_rel_c, dst_rel_c, src_neg_c, dst_neg_c;

    float total_auc = 0.0;

    int num_batchs = (all_batch.size()+(3 * pos_num)-1) / (3 * pos_num);
    torch::Tensor d_flattened_partitions = torch::from_blob(all_batch.data(), {static_cast<int64_t>(all_batch.size())}, torch::kInt32).to(torch::kCUDA);
    torch::Tensor all_ranks = torch::empty({0});

    for (int j = 0; j < num_batchs; ++j) {
        int batch_start = j * batch_size * 3;

        torch::Tensor d_indexed_batch = d_flattened_partitions.slice(0, batch_start, batch_start + batch_size*3);
        if(batch_size * 3 > d_indexed_batch.sizes()[0]){
            batch_size = d_indexed_batch.sizes()[0] / 3;
        }
        torch::Tensor neg_src_node_id;
        torch::Tensor neg_tgt_node_id;
        
        if(all){
            neg_num=node_num;
            neg_src_node_id = torch::arange(0, node_num, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
            neg_tgt_node_id = torch::arange(0, node_num, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
        }
        else{
            globalSample(neg_src_node_id,neg_tgt_node_id,chunk_num,neg_num,node_num);
        }

        torch::Tensor src_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 0);
        torch::Tensor tgt_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 1);
        torch::Tensor rel_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 2);

        torch::Tensor emb_idx = torch::cat({src_node_id, tgt_node_id, neg_src_node_id, neg_tgt_node_id});
        
        auto unique_tup = torch::_unique2(emb_idx, true, true, false);
        torch::Tensor unique_node_indices_ = get<0>(unique_tup);
        torch::Tensor emb_mapping = get<1>(unique_tup);
        int64_t curr = 0;
        int64_t size = batch_size;
        torch::Tensor src_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
        curr += size;
        torch::Tensor dst_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
        curr += size;
        size = neg_src_node_id.size(0);
        torch::Tensor src_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);
        curr += size;
        torch::Tensor dst_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);


        torch::Tensor unique_node_embedding = node_embedding.index_select(0, unique_node_indices_.toType(at::kLong));
        
        src_pos_c = unique_node_embedding.index_select(0, src_pos_indices_mapping_.toType(at::kLong));
        dst_pos_c = unique_node_embedding.index_select(0, dst_pos_indices_mapping_.toType(at::kLong));
        src_neg_c = unique_node_embedding.index_select(0, src_neg_indices_mapping_.toType(at::kLong));
        dst_neg_c = unique_node_embedding.index_select(0, dst_neg_indices_mapping_.toType(at::kLong));
        if(rel_num > 1){
            unique_tup = torch::_unique2(rel_id, true, true, false);
            torch::Tensor unique_rel_indices_ = get<0>(unique_tup);
            torch::Tensor rel_indices_mapping_ = get<1>(unique_tup);
            torch::Tensor unique_src_rel_embedding = src_rel_embedding.index_select(0, unique_rel_indices_.toType(at::kLong));
            torch::Tensor unique_dst_rel_embedding = dst_rel_embedding.index_select(0, unique_rel_indices_.toType(at::kLong));
            src_rel_c = unique_src_rel_embedding.index_select(0, rel_indices_mapping_.toType(at::kLong));
            dst_rel_c = unique_dst_rel_embedding.index_select(0, rel_indices_mapping_.toType(at::kLong));
        }
        
        src_neg_c = src_neg_c.view({chunk_num, neg_num, embedding_dim});
        dst_neg_c = dst_neg_c.view({chunk_num, neg_num, embedding_dim});

        torch::Tensor new_src_neg_c;
        torch::Tensor new_dst_neg_c;

        forward_our_front_mma_eva(
            src_pos_c, src_rel_c, dst_pos_c, dst_rel_c, src_neg_c, dst_neg_c, 
            true, pos_num, neg_num, embedding_dim, chunk_num, batch_size, total_auc, all_ranks);
        if(j >= 1000) break;
    }
    cout << all_ranks.sizes() << endl;

    all_ranks = all_ranks.to(torch::kDouble).to(torch::kCPU);
    double avg_ranks = all_ranks.mean().item<double>();
    double mrr = all_ranks.reciprocal().mean().item<double>();
    
    double ranks1 = (double) all_ranks.le(1).nonzero().size(0) / all_ranks.size(0);
    double ranks5 = (double) all_ranks.le(5).nonzero().size(0) / all_ranks.size(0);
    double ranks10 = (double) all_ranks.le(10).nonzero().size(0) / all_ranks.size(0);
    double ranks20 = (double) all_ranks.le(20).nonzero().size(0) / all_ranks.size(0);
    double ranks50 = (double) all_ranks.le(50).nonzero().size(0) / all_ranks.size(0);
    double ranks100 = (double) all_ranks.le(100).nonzero().size(0) / all_ranks.size(0);
    printf("Auc: %.3f, Avg Ranks: %.3f, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f, Hits@20: %.3f, Hits@50: %.3f, Hits@100: %.3f\n",
       total_auc / num_batchs, avg_ranks, mrr, ranks1, ranks5, ranks10, ranks20, ranks50, ranks100);
}

int main(){
    int num_parts = 4;
    // int node_num = 14951, rel_num = 1345;
    int node_num = 4847571, rel_num = 1, edge_num = 62094395;
    int part_size = (node_num+num_parts-1) / num_parts;
    // ifstream infile("train_edges.pt", std::ios::binary);
    ifstream infile("livejournal/train_edges.pt", std::ios::binary);

    infile.seekg(0, std::ios::end);

    unsigned file_size = (unsigned)(infile.tellg());
    unsigned char* buff = new unsigned char[file_size];
    infile.seekg(0, std::ios::beg);
    infile.read((char*)buff, file_size);
    infile.close();
    cout << "End read data" << endl;
    // vector<vector<int>> partitions(num_parts * num_parts);
    vector<vector<int>> partitions(1);
    unsigned src, rel, tgt;
    for(unsigned i = 0; i < file_size / 12; i++) {

        src = ((unsigned*)buff)[i * 3];
        rel = ((unsigned*)buff)[i * 3 + 1];
        tgt = ((unsigned*)buff)[i * 3 + 2];
        unsigned src_part = src / part_size;
        unsigned tgt_part = tgt / part_size;
        if (src_part >= num_parts) src_part = num_parts - 1;
        if (tgt_part >= num_parts) tgt_part = num_parts - 1;
        
        unsigned part_index = src_part * num_parts + tgt_part;

        partitions[0].push_back(src);
        partitions[0].push_back(tgt);
        partitions[0].push_back(rel);
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int elements_per_group = 3;
    int pos_num = 100000, neg_num = 1000, embedding_dim = 100, chunk_num = 10;
    float learning_rate=0.1;
    int batch_size=pos_num;
    
    int num_negative_samples = neg_num;
    torch::Tensor src_pos_c, dst_pos_c, src_rel_c, dst_rel_c, src_neg_c, dst_neg_c;
    // torch::Tensor unique_node_embedding, unique_rel_embedding;
    // torch::Tensor node_embedding = torch::rand({node_num, embedding_dim},dtype(torch::kFloat32)).cuda(), 
    torch::Tensor node_embedding = torch::randn({node_num * embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).mul_(0.001);
    node_embedding = node_embedding.view({node_num, embedding_dim});
    // src_rel_embedding = torch::rand({rel_num, embedding_dim},dtype(torch::kFloat32)).cuda(), 
    // dst_rel_embedding = torch::rand({rel_num, embedding_dim},dtype(torch::kFloat32)).cuda();
    torch::Tensor src_rel_embedding = torch::zeros({rel_num, embedding_dim}, opts);
    src_rel_embedding.narrow(1, 0, (embedding_dim / 2) - 1).fill_(1);
    torch::Tensor dst_rel_embedding = torch::zeros({rel_num, embedding_dim}, opts);
    dst_rel_embedding.narrow(1, 0, (embedding_dim / 2) - 1).fill_(1);
    torch::Tensor node_grad_state = torch::zeros({node_num, embedding_dim}).cuda(), src_rel_grad_state = torch::zeros({rel_num, embedding_dim}).cuda(), dst_rel_grad_state = torch::zeros({rel_num, embedding_dim}).cuda();
    cout << node_embedding.max() << endl;
    // torch::Tensor src_pos = torch::rand({pos_num, embedding_dim}), dst_pos = torch::rand({pos_num, embedding_dim}), 
    // src_rel = torch::rand({pos_num, embedding_dim}), dst_rel = torch::rand({pos_num, embedding_dim}), 
    // src_neg = torch::rand({chunk_num, neg_num, embedding_dim}), dst_neg = torch::rand({chunk_num, neg_num, embedding_dim});

    // src_pos_c = src_pos.cuda(); dst_pos_c = dst_pos.cuda(); src_rel_c = src_rel.cuda(); dst_rel_c = dst_rel.cuda(); src_neg_c = src_neg.cuda(); dst_neg_c = dst_neg.cuda();
    torch::Tensor grad_src = torch::zeros({pos_num, embedding_dim}, opts), 
    grad_dst = torch::zeros({pos_num, embedding_dim}, opts), 
    grad_src_rel = torch::zeros({pos_num, embedding_dim}, opts), 
    grad_dst_rel = torch::zeros({pos_num, embedding_dim}, opts),
    grad_src_neg,
    grad_dst_neg;

    torch::Tensor unique_node_gradients_ = torch::zeros({pos_num, embedding_dim}).cuda();
    torch::Tensor unique_node_gradients2_ = torch::zeros({pos_num, embedding_dim}).cuda();
    
    torch::Tensor unique_src_rel_gradients_ = torch::zeros({rel_num, embedding_dim}).cuda();
    torch::Tensor unique_src_rel_gradients2_ = torch::zeros({rel_num, embedding_dim}).cuda();
    torch::Tensor unique_dst_rel_gradients_ = torch::zeros({rel_num, embedding_dim}).cuda();
    torch::Tensor unique_dst_rel_gradients2_ = torch::zeros({rel_num, embedding_dim}).cuda();

    auto lhs_score_exp = torch::rand({chunk_num, pos_num/chunk_num, neg_num+8}).flatten().cuda();
    auto rhs_score_exp = torch::rand({chunk_num, pos_num/chunk_num, neg_num+8}).flatten().cuda();
    auto lhs_pos_scores = torch::rand({pos_num, 1}).flatten().cuda();
    auto rhs_pos_scores = torch::rand({pos_num, 1}).flatten().cuda();

    auto adjusted_dst_pos = torch::zeros({pos_num, embedding_dim}, opts).flatten();
    auto adjusted_src_pos = torch::zeros({pos_num, embedding_dim}, opts).flatten();

    auto torch_zeros = torch::zeros({pos_num, embedding_dim}).cuda();
    auto torch_neg_pad = torch::zeros({1, neg_num, embedding_dim+4}).cuda();
    auto torch_rand = torch::zeros({chunk_num, neg_num, 4}).cuda();
    int epoch=30;
    for(int epoch_=0; epoch_<epoch; epoch_++){
        auto start = std::chrono::high_resolution_clock::now();
        int all_num_batch=0;
        vector<int>flattened_partitions;
        for(auto part : partitions){
            flattened_partitions.insert(flattened_partitions.end(), part.begin(), part.end());
        }
        torch::Tensor d_flattened_partitions = torch::from_blob(flattened_partitions.data(), {static_cast<int64_t>(flattened_partitions.size())}, torch::kInt32).cuda();
        int num_batchs = (flattened_partitions.size()+(3 * pos_num)-1) / (3 * pos_num);
        cout << num_batchs << endl;

        for (int j = 0; j < num_batchs; ++j) {
            batch_size = pos_num;
            int batch_start = j * batch_size * 3;

            torch::Tensor d_indexed_batch = d_flattened_partitions.slice(0, batch_start, batch_start + batch_size*3);
            
            torch::Tensor neg_src_node_id;
            torch::Tensor neg_tgt_node_id;
            globalSample(neg_src_node_id,neg_tgt_node_id,chunk_num,neg_num,node_num);

            if(batch_size * 3 > d_indexed_batch.sizes()[0]){
                batch_size = d_indexed_batch.sizes()[0] / 3;
            }
            torch::Tensor src_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 0).clone();
            torch::Tensor tgt_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 1).clone();

            torch::Tensor rel_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 2).clone();

            torch::Tensor emb_idx = torch::cat({src_node_id, tgt_node_id, neg_src_node_id, neg_tgt_node_id });

            auto unique_tup = torch::_unique2(emb_idx, true, true, false);
            torch::Tensor unique_node_indices_ = get<0>(unique_tup);
            torch::Tensor emb_mapping = get<1>(unique_tup);
            int64_t curr = 0;
            int64_t size = batch_size;
            torch::Tensor src_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
            curr += size;
            torch::Tensor dst_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
            curr += size;
            size = neg_src_node_id.size(0);
            torch::Tensor src_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);
            curr += size;
            torch::Tensor dst_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);

            torch::Tensor unique_node_embedding = node_embedding.index_select(0, unique_node_indices_.toType(at::kLong));
            src_pos_c = unique_node_embedding.index_select(0, src_pos_indices_mapping_.toType(at::kLong));

            dst_pos_c = unique_node_embedding.index_select(0, dst_pos_indices_mapping_.toType(at::kLong));

            src_neg_c = unique_node_embedding.index_select(0, src_neg_indices_mapping_.toType(at::kLong));

            dst_neg_c = unique_node_embedding.index_select(0, dst_neg_indices_mapping_.toType(at::kLong));
            torch::Tensor unique_node_states = node_grad_state.index_select(0, unique_node_indices_.toType(at::kLong));


            torch::Tensor unique_rel_indices_, rel_indices_mapping_, unique_src_rel_embedding, unique_dst_rel_embedding, unique_src_rel_states, unique_dst_rel_states;
            if(rel_num > 1){
                unique_tup = torch::_unique2(rel_id, true, true, false);
                unique_rel_indices_ = get<0>(unique_tup);
                rel_indices_mapping_ = get<1>(unique_tup);
                unique_src_rel_embedding = src_rel_embedding.index_select(0, unique_rel_indices_.toType(at::kLong));
                unique_dst_rel_embedding = dst_rel_embedding.index_select(0, unique_rel_indices_.toType(at::kLong));
                unique_src_rel_states = src_rel_grad_state.index_select(0, unique_rel_indices_.toType(at::kLong));
                unique_dst_rel_states = dst_rel_grad_state.index_select(0, unique_rel_indices_.toType(at::kLong));
                src_rel_c = unique_src_rel_embedding.index_select(0, rel_indices_mapping_.toType(at::kLong));
                dst_rel_c = unique_dst_rel_embedding.index_select(0, rel_indices_mapping_.toType(at::kLong));
            }

            if(batch_size < pos_num){
                auto to_cat = torch_zeros.narrow(0, 0, pos_num - batch_size);
                src_pos_c = torch::cat({src_pos_c, to_cat});
                dst_pos_c = torch::cat({dst_pos_c, to_cat});
                if(rel_num > 1){
                    src_rel_c = torch::cat({src_rel_c, to_cat});
                    dst_rel_c = torch::cat({dst_rel_c, to_cat});
                }
            }

            src_neg_c = src_neg_c.view({chunk_num, neg_num, embedding_dim});
            dst_neg_c = dst_neg_c.view({chunk_num, neg_num, embedding_dim});
            auto new_dst_neg_c = torch::cat({dst_neg_c, torch_rand}, 2);
            auto new_src_neg_c = torch::cat({src_neg_c, torch_rand}, 2);
            
            new_src_neg_c = torch::cat({new_src_neg_c, torch_neg_pad});
            new_dst_neg_c = torch::cat({new_dst_neg_c, torch_neg_pad});
            
            forward_our_front_mma(
                src_pos_c, src_rel_c, dst_pos_c, dst_rel_c, src_neg_c, dst_neg_c, 
                true, pos_num, neg_num, embedding_dim, chunk_num, 
                grad_src, grad_dst, grad_src_rel, grad_dst_rel, grad_src_neg, grad_dst_neg, 
                lhs_score_exp, rhs_score_exp, lhs_pos_scores, rhs_pos_scores, adjusted_dst_pos, adjusted_src_pos, 
                new_src_neg_c, new_dst_neg_c, batch_size, rel_num);

            grad_src = grad_src.view({pos_num, embedding_dim});
            grad_dst = grad_dst.view({pos_num, embedding_dim});

            grad_src_neg = grad_src_neg.view({chunk_num*neg_num, embedding_dim});
            grad_dst_neg = grad_dst_neg.view({chunk_num*neg_num, embedding_dim});
            
            unique_node_gradients_ = torch::zeros_like(unique_node_embedding);
            unique_node_gradients_.index_add_(0, src_pos_indices_mapping_.toType(at::kLong), grad_src.narrow(0, 0, batch_size));
            unique_node_gradients_.index_add_(0, src_neg_indices_mapping_.toType(at::kLong), grad_src_neg);
            unique_node_gradients_.index_add_(0, dst_pos_indices_mapping_.toType(at::kLong), grad_dst.narrow(0, 0, batch_size));
            unique_node_gradients_.index_add_(0, dst_neg_indices_mapping_.toType(at::kLong), grad_dst_neg);
            
            unique_node_gradients2_ = unique_node_gradients_.pow(2);
            unique_node_states.add_(unique_node_gradients2_);
            
            unique_node_gradients_ = -learning_rate * (unique_node_gradients_ / ((unique_node_states).sqrt().add_(1e-9)));

            node_embedding.index_add_(0, unique_node_indices_.toType(at::kLong), unique_node_gradients_);
            node_grad_state.index_add_(0, unique_node_indices_.toType(at::kLong), unique_node_gradients2_);
            
            if(rel_num > 1){
                unique_src_rel_gradients_ = torch::zeros_like(unique_src_rel_embedding);
                unique_dst_rel_gradients_ = torch::zeros_like(unique_dst_rel_embedding);
                unique_src_rel_gradients_.index_add_(0, rel_indices_mapping_.toType(at::kLong), grad_src_rel.narrow(0, 0, batch_size));
                unique_dst_rel_gradients_.index_add_(0, rel_indices_mapping_.toType(at::kLong), grad_dst_rel.narrow(0, 0, batch_size));
                
                // unique_rel_gradients2_.zero_();
                unique_src_rel_gradients2_ = unique_src_rel_gradients_.pow(2);
                unique_src_rel_states.add_(unique_src_rel_gradients2_);
                unique_src_rel_gradients_ = -learning_rate * (unique_src_rel_gradients_ / ((unique_src_rel_states).sqrt().add_(1e-9)));
                unique_dst_rel_gradients2_ = unique_dst_rel_gradients_.pow(2);
                unique_dst_rel_states.add_(unique_dst_rel_gradients2_);
                unique_dst_rel_gradients_ = -learning_rate * (unique_dst_rel_gradients_ / ((unique_dst_rel_states).sqrt().add_(1e-9)));

                src_rel_embedding.index_add_(0, unique_rel_indices_.toType(at::kLong), unique_src_rel_gradients_);
                dst_rel_embedding.index_add_(0, unique_rel_indices_.toType(at::kLong), unique_dst_rel_gradients_);
                src_rel_grad_state.index_add_(0, unique_rel_indices_.toType(at::kLong), unique_src_rel_gradients2_);
                dst_rel_grad_state.index_add_(0, unique_rel_indices_.toType(at::kLong), unique_dst_rel_gradients2_);
            }

            }
        // }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        cout << "Epoch time: " << elapsed.count() << "ms" << std::endl;
        if((epoch_+1)%5==0){
            evaluate(true,false,node_embedding,src_rel_embedding,dst_rel_embedding);
        }
    }

    return 0;
}

