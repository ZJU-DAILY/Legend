#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "ctrl.h"
#include "buffer.h"
#include "settings.h"
#include "event.h"
#include "queue.h"

#include <torch/extension.h>
#include <torch/torch.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <curand_kernel.h>

#include <random>

#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif
#define THREADSPERQUEUE 100000

#define WARP 2
#define WARP4MMA 8
#define ReduceWARP 2
#define EMB_DIM 100

#define BLK_H 16 
#define BLK_W 8

using namespace std;
using namespace nvcuda;

using error = std::runtime_error;
using std::string;


struct __align__(64) CmdTime
{
    size_t      size;
    uint64_t    submitTime;
    uint64_t    completeTime;
    uint64_t    moveTime;
};


__host__ static
std::shared_ptr<CmdTime> createReportingList(size_t numEntries, int device)
{
    auto err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw err;
    }

    CmdTime* list = nullptr;
    err = cudaMalloc(&list, sizeof(CmdTime) * numEntries);
    if (err != cudaSuccess)
    {
        throw err;
    }
    return std::shared_ptr<CmdTime>(list, cudaFree);
}


__host__ static
std::shared_ptr<CmdTime> createReportingList(size_t numEntries)
{
    CmdTime* list = nullptr;

    auto err = cudaHostAlloc(&list, sizeof(CmdTime) * numEntries, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw err;
    }

    return std::shared_ptr<CmdTime>(list, cudaFreeHost);
}



__device__ static
void moveBytes(const void* src, size_t srcOffset, void* dst, size_t dstOffset, size_t size)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;

    const ulong4* source = (ulong4*) (((const unsigned char*) src) + srcOffset);
    ulong4* destination = (ulong4*) (((unsigned char*) dst) + dstOffset);

    for (size_t i = 0, n = size / sizeof(ulong4); i < n; i += numThreads)
    {
        destination[i + threadNum] = source[i + threadNum];
    }
    
    // const float* source = (float*) (((const float*) src) + srcOffset);
    // float* destination = (float*) (((float*) dst) + dstOffset);

    // for (size_t i = 0, n = size / sizeof(float); i < n; i += numThreads)
    // {
    //     destination[i + threadNum] = source[i + threadNum];
    // }
}


__device__ static
void waitForIoCompletion(nvm_queue_t* cq, nvm_queue_t* sq, uint64_t* errCount)
{
    const uint16_t numThreads = blockDim.x;

    for (uint16_t i = 0; i < numThreads; ++i)
    {
        nvm_cpl_t* cpl = nullptr;
        // printf("A%d", i);
        while ((cpl = nvm_cq_dequeue(cq)) == nullptr);
        
        // printf("B%d", i);
        nvm_sq_update(sq);

        if (!NVM_ERR_OK(cpl))
        {
            *errCount = *errCount + 1;
        }
    }

    nvm_cq_update(cq);
}

__device__ static inline
bool nvm_cq_poll_our(nvm_queue_t* cq, uint32_t threadNum)
{
    unsigned long beforePerpare, beforeSubmit, afterSync;
    
    bool phase_cur = ((~((cq->head + threadNum) >> (cq->log_max_entries))) & 0x01);
    
    nvm_cpl_t* cpl = &((nvm_cpl_t*)cq->vaddr)[((cq->head + threadNum) & (cq->max_entries - 1))];
    
    // Check if new completion is ready by checking the phase tag
    // if(threadNum == 0) printf("%u ", cq->max_entries);
    // if (!!_RB(*NVM_CPL_STATUS(cpl), 0, 0) != phase_cur)
    // if(((cpl_entry & 0x00010000) >> 16) != phase_cur){
    // bool phase = (cpl_entry & 0x00010000) >> 16;
    // bool phase = ((((volatile unsigned char*) ((volatile void*) (cpl)))[14]) & 0x01);
    bool phase = ((((volatile unsigned char*) cpl)[14]) & 0x01);
    // if (!!_RB(*NVM_CPL_STATUS(cpl), 0, 0) != phase_cur){
    if(phase != phase_cur){
        return false;
    }
    // __syncthreads();
    return true;
}

__device__ static inline
void nvm_cq_sq_dequeue(nvm_queue_t* cq, nvm_queue_t* sq, uint32_t threadNum, uint16_t all_threads)
{
    // uint32_t* cpl = nullptr;
    // unsigned long beforePerpare, beforeSubmit, afterSync;
    bool flag = false;
    // printf("%u\n", threadNum);
    // __syncthreads();
    // beforeSubmit = clock();
    while ((flag = nvm_cq_poll_our(cq, threadNum)) == false);
    
    // nvm_cache_invalidate((void*) &flag, sizeof(bool));
    // printf("%d %d\n", flag, threadNum);
    // while ((cpl = nvm_cq_poll_our(cq, threadNum)) == NULL);
    // nvm_cq_poll_our(cq, threadNum);
    // __syncthreads();
    unsigned res = atomicAdd(&cq->current_count, 1);
    // __syncthreads();
    if(res + 1 == all_threads){
        // if(blockIdx.x == 0) printf("%d\n", threadNum);
        cq->head = cq->head + all_threads;
        // printf("%d\n", cq->head);
        *((volatile uint32_t*) cq->db) = (cq->head & (cq->max_entries - 1));
        sq->head = ((sq->head + all_threads) & (sq->max_entries - 1));
        cq->current_count = 0;
    }
    __syncthreads();
}

__device__ static
void waitForIoCompletionOur(nvm_queue_t* cq, nvm_queue_t* sq, uint16_t threadNum, uint16_t blockThread)
{
    // const uint16_t numThreads = blockDim.x;
    // const uint32_t threadNum = threadIdx.x;
    nvm_cq_sq_dequeue(cq, sq, threadNum, blockThread);
    // __syncthreads();
}

__device__ static
nvm_cmd_t* prepareChunk(QueuePair* qp, nvm_cmd_t* last, const uint64_t ioaddr, uint16_t offset, uint64_t blockOffset, uint32_t currChunk)
{
    nvm_cmd_t local;
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint16_t threadOffset = threadNum + numThreads * offset;

    const uint32_t pageSize = qp->pageSize;
    const uint32_t blockSize = qp->blockSize;
    const uint32_t nvmNamespace = qp->nvmNamespace;
    const uint32_t chunkPages = qp->pagesPerChunk;

    // Calculate offsets
    const uint16_t blocksPerChunk = NVM_PAGE_TO_BLOCK(pageSize, blockSize, chunkPages);
    const uint64_t currBlock = NVM_PAGE_TO_BLOCK(pageSize, blockSize, (currChunk + threadNum) * chunkPages);

    // Prepare PRP list building
    void* prpList = NVM_PTR_OFFSET(qp->prpList, pageSize, threadOffset);
    uint64_t prpListAddr = NVM_ADDR_OFFSET(qp->prpListIoAddr, pageSize, threadOffset);

    uint64_t addrs[0x1000 / sizeof(uint64_t)]; // FIXME: This assumes that page size is 4K
    for (uint32_t page = 0; page < chunkPages; ++page)
    {
        addrs[page] = NVM_ADDR_OFFSET(ioaddr, pageSize, chunkPages * threadOffset + page);
    }

    // Enqueue commands
    nvm_cmd_t* cmd = nvm_sq_enqueue_n(&qp->sq, last, numThreads, threadNum);
    // nvm_cmd_t cmd;
    // Set command fields
    nvm_cmd_header(&local, threadNum, NVM_IO_READ, nvmNamespace);
    nvm_cmd_data(&local, pageSize, chunkPages, prpList, prpListAddr, addrs);
    nvm_cmd_rw_blks(&local, currBlock + blockOffset, blocksPerChunk);
    // nvm_sq_enqueue_n_our(&qp->sq, &cmd, numThreads, threadNum);
    *cmd = local;
    __threadfence();
    // __syncthreads();
    return cmd;
}

__device__ static
void prepareChunk_Our_read(QueuePair* qp, nvm_cmd_t* last, const uint64_t ioaddr, uint16_t offset, uint64_t blockOffset, uint32_t currChunk, uint64_t* prp1, uint64_t* prp2, unsigned vaddr_offset, uint16_t blockThread)
{
    // nvm_cmd_t local;
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint16_t tblockNum = blockIdx.x;
    const uint16_t threadOffset = threadNum + numThreads * offset;

    const uint32_t pageSize = qp->pageSize;
    const uint32_t blockSize = qp->blockSize;
    const uint32_t nvmNamespace = qp->nvmNamespace;
    const uint32_t chunkPages = qp->pagesPerChunk;

    // Calculate offsets
    const uint16_t blocksPerChunk = NVM_PAGE_TO_BLOCK(pageSize, blockSize, chunkPages);
    const uint64_t currBlock = NVM_PAGE_TO_BLOCK(pageSize, blockSize, (currChunk + threadNum + tblockNum * numThreads) * chunkPages);

    nvm_cmd_t cmd;
    // Set command fields
    nvm_cmd_header(&cmd, threadNum+currChunk, NVM_IO_READ, nvmNamespace);
    // nvm_cmd_data(&cmd, pageSize, chunkPages, prpList, prpListAddr, addrs);
    nvm_cmd_data_our(&cmd, prp1, prp2, (uint32_t)(tblockNum*numThreads+threadNum)+currChunk+vaddr_offset);
    // printf("%lx %d\n", prp2[threadNum+currChunk], threadNum);
    nvm_cmd_rw_blks(&cmd, currBlock + blockOffset, blocksPerChunk);
    nvm_sq_enqueue_n_our(&qp->sq, &cmd, blockThread, threadNum);
    // nvm_sq_enqueue_n_our_mblock(&qp->sq, &cmd, numThreads, threadNum, blockNum);
    // *cmd = local;
    // __threadfence();
    __syncthreads();
    // return cmd;
}

__device__ static
void prepareChunk_Our_write(QueuePair* qp, nvm_cmd_t* last, const uint64_t ioaddr, uint16_t offset, uint64_t blockOffset, uint32_t currChunk, uint64_t* prp1, uint64_t* prp2, unsigned vaddr_offset, uint16_t blockThread)
{
    // nvm_cmd_t local;
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint16_t tblockNum = blockIdx.x;
    const uint16_t threadOffset = threadNum + numThreads * offset;

    const uint32_t pageSize = qp->pageSize;
    const uint32_t blockSize = qp->blockSize;
    const uint32_t nvmNamespace = qp->nvmNamespace;
    const uint32_t chunkPages = qp->pagesPerChunk;

    // Calculate offsets
    const uint16_t blocksPerChunk = NVM_PAGE_TO_BLOCK(pageSize, blockSize, chunkPages);
    const uint64_t currBlock = NVM_PAGE_TO_BLOCK(pageSize, blockSize, (currChunk + threadNum + tblockNum * numThreads) * chunkPages);

    nvm_cmd_t cmd;
    // Set command fields
    nvm_cmd_header(&cmd, threadNum+currChunk, NVM_IO_WRITE, nvmNamespace);
    // nvm_cmd_data(&cmd, pageSize, chunkPages, prpList, prpListAddr, addrs);
    nvm_cmd_data_our(&cmd, prp1, prp2, tblockNum*numThreads+threadNum+currChunk+vaddr_offset);
    // printf("%lx %d\n", prp2[threadNum+currChunk], threadNum);
    nvm_cmd_rw_blks(&cmd, currBlock + blockOffset, blocksPerChunk);
    nvm_sq_enqueue_n_our(&qp->sq, &cmd, blockThread, threadNum);
    // nvm_sq_enqueue_n_our_mblock(&qp->sq, &cmd, numThreads, threadNum, blockNum);
    // *cmd = local;
    // __threadfence();
    __syncthreads();
    // return cmd;
}

__global__ static 
void moveKernel(void* src, void* dst, size_t chunkSize)
{
    const uint16_t numThreads = blockDim.x;
    moveBytes(src, 0, dst, 0, chunkSize * numThreads);
}



__host__ static inline
void launchMoveKernel(size_t pageSize, void* input, void* src, void* dst, size_t currChunk, const Settings& settings)
{
    const auto numPages = settings.numPages;
    const auto numThreads = settings.numThreads;
    const auto chunkSize = pageSize * numPages;

    void* dstPtr = (void*) (((unsigned char*) dst) + chunkSize * currChunk);
    void* inputPtr = (void*) (((unsigned char*) input) + chunkSize * currChunk);

    cudaError_t err = cudaMemcpyAsync(src, inputPtr, chunkSize * numThreads, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    moveKernel<<<1, numThreads>>>(src, dstPtr, chunkSize);
}



static double launchMoveKernelLoop(void* fileMap, BufferPtr destination, size_t pageSize, const Settings& settings)
{
    const size_t chunkSize = pageSize * settings.numPages;
    const size_t numThreads = settings.numThreads;
    const size_t totalChunks = settings.numChunks * numThreads;

    const size_t sourceBufferSize = NVM_PAGE_ALIGN(chunkSize * numThreads, 1UL << 16);
    auto source = createBuffer(sourceBufferSize, settings.cudaDevice);

    auto err = cudaSetDevice(settings.cudaDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    Event before;
    for (size_t currChunk = 0; currChunk < totalChunks; currChunk += numThreads)
    {
        launchMoveKernel(pageSize, fileMap, source.get(), destination.get(), currChunk, settings);
    }
    Event after;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw err;
    }

    return after - before;
}



__global__ static
void readDoubleBuffered(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount, CmdTime* times)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t pageSize = qp->pageSize;
    const size_t chunkSize = qp->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &qp->sq;

    uint64_t blockOffset = startBlock;

    uint32_t currChunk = 0;
    bool bufferOffset = false;
    uint32_t i = 0;

    nvm_cmd_t* last = prepareChunk(qp, nullptr, ioaddr, bufferOffset, blockOffset, currChunk);

    auto beforeSubmit = clock();
    if (threadNum == 0)
    {
        *errCount = 0;
        nvm_sq_submit(sq);
    }
    __syncthreads();

    while (currChunk + numThreads < numChunks)
    {
        // Prepare in advance next chunk
        last = prepareChunk(qp, last, ioaddr, !bufferOffset, blockOffset, currChunk + numThreads);

        // Consume completions for the previous window
        beforeSubmit = clock();
        if (threadNum == 0)
        {
            waitForIoCompletion(&qp->cq, sq, errCount);
            nvm_sq_submit(sq);
        }
        __syncthreads();
        auto afterSync = clock();

        // Move received chunk
        moveBytes(src, bufferOffset * numThreads * chunkSize, dst, currChunk * chunkSize, chunkSize * numThreads);
        auto afterMove = clock();

        // Record statistics
        if (times != nullptr && threadNum == 0)
        {
            CmdTime* t = &times[i];
            t->size = chunkSize * numThreads;
            t->submitTime = beforeSubmit;
            t->completeTime = afterSync;
            t->moveTime = afterMove;
        }
        __syncthreads();
    
        // Update position and input buffer
        bufferOffset = !bufferOffset;
        currChunk += numThreads;
        ++i;
    }

    // Wait for final buffer to complete
    if (threadNum == 0)
    {
        waitForIoCompletion(&qp->cq, sq, errCount);
    }
    __syncthreads();
    auto afterSync = clock();

    moveBytes(src, bufferOffset * numThreads * chunkSize, dst, currChunk * chunkSize, chunkSize * numThreads);
    auto afterMove = clock();

    // Record statistics
    if (times != nullptr && threadNum == 0)
    {
        CmdTime* t = &times[i];
        t->size = chunkSize * numThreads;
        t->submitTime = beforeSubmit;
        t->completeTime = afterSync;
        t->moveTime = afterMove;
    }
}

torch::Tensor ComplexHadamardOperator(const torch::Tensor &embs, const torch::Tensor &rels) {
    if (!rels.defined()) {
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
    torch::Tensor neg_scores = adjusted_src.bmm(negs.transpose(-1, -2)).flatten(0, 1);
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

void forward_eva(
    torch::Tensor &src_pos_embeddings_, 
    torch::Tensor &src_relation_emebeddings_, 
    torch::Tensor &dst_pos_embeddings_, 
    torch::Tensor &dst_relation_emebeddings_, 
    torch::Tensor &src_all_neg_embeddings_, 
    torch::Tensor &dst_all_neg_embeddings_, int batch_size, float &total_auc, torch::Tensor &all_ranks) 
{
    torch::Tensor lhs_neg_scores;
    torch::Tensor rhs_neg_scores;
    torch::Tensor lhs_pos_scores;
    torch::Tensor rhs_pos_scores;
    torch::Tensor adjusted_src_pos, adjusted_dst_pos;
    
    torch::Tensor lhs_ranks;
    torch::Tensor rhs_ranks;
    torch::Tensor auc;
    torch::Tensor lhs_auc;
    torch::Tensor rhs_auc;
    
    adjusted_src_pos = ComplexHadamardOperator(src_pos_embeddings_, src_relation_emebeddings_);
    tie(rhs_pos_scores, rhs_neg_scores) = DotCompare(adjusted_src_pos, dst_pos_embeddings_, dst_all_neg_embeddings_);
    
    // corrupt source
    adjusted_dst_pos = ComplexHadamardOperator(dst_pos_embeddings_, dst_relation_emebeddings_);
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
        // #pragma unroll
        // for (unsigned t = 0; t < acc_frag.num_elements; t++) {
        //     acc_frag.x[t] = expf(acc_frag.x[t]);
        // }
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
        // #pragma unroll
        // for (unsigned t = 0; t < acc_frag.num_elements; t++) {
        //     acc_frag.x[t] = expf(acc_frag.x[t]);
        // }
        wmma::store_matrix_sync(rhs_neg_scores + blockIdx.x * BLK_H * (neg_num+8) + warp_id * BLK_H, acc_frag, neg_num+8, wmma::mem_row_major);
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
        // #pragma unroll
        // for (unsigned t = 0; t < acc_frag.num_elements; t++) {
        //     acc_frag.x[t] = expf(acc_frag.x[t]);
        // }
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
        // #pragma unroll
        // for (unsigned t = 0; t < acc_frag.num_elements; t++) {
        //     acc_frag.x[t] = expf(acc_frag.x[t]);
        // }
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



void forward_our_front_mma(torch::Tensor &src_pos_embeddings_, torch::Tensor &src_relation_emebeddings_, torch::Tensor &dst_pos_embeddings_, torch::Tensor &dst_relation_emebeddings_, torch::Tensor &src_all_neg_embeddings_, torch::Tensor &dst_all_neg_embeddings_, bool train, int pos_num, int neg_num, int emb_dim, int chunk_num, unsigned rel_num, torch::Tensor &grad_src, torch::Tensor &grad_dst, torch::Tensor &grad_src_rel, torch::Tensor &grad_dst_rel, torch::Tensor &grad_src_neg, torch::Tensor &grad_dst_neg, torch::Tensor &lhs_score_exp, torch::Tensor &rhs_score_exp, torch::Tensor &lhs_pos_scores, torch::Tensor &rhs_pos_scores, torch::Tensor &adjusted_src_pos, torch::Tensor &adjusted_dst_pos, torch::Tensor &new_src_neg_c, torch::Tensor &new_dst_neg_c, cudaStream_t &stream2) {
    // torch::Tensor lhs_pos_scores;
    // torch::Tensor lhs_neg_scores;
    // torch::Tensor rhs_pos_scores;
    // torch::Tensor rhs_neg_scores;

    torch::Tensor loss;
    torch::Tensor lhs_loss;
    torch::Tensor rhs_loss;

    int regularization_coef = 1;
    int regularization_norm = 3;

    
    // corrupt destination
    // auto start = std::chrono::high_resolution_clock::now();
    
    dim3 block(32, WARP4MMA, 1);
    dim3 grid(625*chunk_num, 1, 1);
    if(rel_num > 1){
        forward_ComplEx_kernel_mma<<<grid, block, 0, stream2>>>(src_pos_embeddings_.data<float>(),
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
        }
    else{
        forward_ComplEx_kernel_mma_no_rel<<<grid, block, 0, stream2>>>(src_pos_embeddings_.flatten().data<float>(),
                                                    dst_pos_embeddings_.flatten().data<float>(),
                                                    new_src_neg_c.flatten().data<float>(),
                                                    new_dst_neg_c.flatten().data<float>(),
                                                    pos_num,
                                                    neg_num,
                                                    emb_dim,
                                                    chunk_num,
                                                    rhs_pos_scores.data<float>(),
                                                    lhs_pos_scores.data<float>(),
                                                    rhs_score_exp.data<float>(),
                                                    lhs_score_exp.data<float>(),
                                                    adjusted_src_pos.flatten().data<float>(),
                                                    adjusted_dst_pos.flatten().data<float>());
    }
    // cudaDeviceSynchronize();
    cudaStreamSynchronize(stream2);

    auto rhs_score_exp_nar = rhs_score_exp.view({pos_num, neg_num+8}).narrow(1, 0, neg_num);
    auto lhs_score_exp_nar = lhs_score_exp.view({pos_num, neg_num+8}).narrow(1, 0, neg_num);
    auto lhs_max = get<0>(lhs_score_exp_nar.max(1, true)), rhs_max = get<0>(rhs_score_exp_nar.max(1, true));
    lhs_score_exp_nar = torch::exp(lhs_score_exp_nar - lhs_max);
    rhs_score_exp_nar = torch::exp(rhs_score_exp_nar - rhs_max);

    if (train) {
        adjusted_src_pos = adjusted_src_pos.view({pos_num, emb_dim});
        adjusted_dst_pos = adjusted_dst_pos.view({pos_num, emb_dim});
    
        auto lhs_score_exp_sum = (lhs_score_exp_nar.sum(-1)).unsqueeze(1);

        auto rhs_score_exp_sum = (rhs_score_exp_nar.sum(-1)).unsqueeze(1);

        auto tmp1 = 1 / (torch::exp((lhs_pos_scores.unsqueeze(1) - torch::log(lhs_score_exp_sum)) - lhs_max) + 1);
        auto tmp2 = 1 / (torch::exp((rhs_pos_scores.unsqueeze(1) - torch::log(rhs_score_exp_sum)) - rhs_max) + 1);
        
        lhs_score_exp_sum = lhs_score_exp_sum.repeat({1,emb_dim});
        rhs_score_exp_sum = rhs_score_exp_sum.repeat({1,emb_dim});
        
        torch::Tensor op1 = (adjusted_dst_pos * tmp1.repeat({1,emb_dim}) / (lhs_score_exp_sum)).view({chunk_num, pos_num / chunk_num, emb_dim});
        torch::Tensor op2 = (adjusted_src_pos * tmp2.repeat({1,emb_dim}) / rhs_score_exp_sum).view({chunk_num, pos_num / chunk_num, emb_dim});    
        
        grad_src_neg = (lhs_score_exp_nar).view({chunk_num, pos_num / chunk_num, neg_num}).transpose(-1, -2).bmm(op1);
        grad_dst_neg = (rhs_score_exp_nar).view({chunk_num, pos_num / chunk_num, neg_num}).transpose(-1, -2).bmm(op2);
        
        torch::Tensor tmp_grad_src = rhs_score_exp_nar.view({chunk_num, pos_num / chunk_num, neg_num}).bmm(dst_all_neg_embeddings_);
        torch::Tensor tmp_grad_dst = lhs_score_exp_nar.view({chunk_num, pos_num / chunk_num, neg_num}).bmm(src_all_neg_embeddings_);
        
        if(rel_num > 1)
            grad_cal_fused<<<pos_num, WARP*32, 0, stream2>>>(src_pos_embeddings_.flatten().data<float>(),dst_pos_embeddings_.flatten().data<float>(),src_relation_emebeddings_.flatten().data<float>(),dst_relation_emebeddings_.flatten().data<float>(), (tmp_grad_src.view({pos_num, emb_dim})/rhs_score_exp_sum).flatten().data<float>(), (tmp_grad_dst.view({pos_num, emb_dim})/lhs_score_exp_sum).flatten().data<float>(), tmp2.flatten().data<float>(), tmp1.flatten().data<float>(),emb_dim,grad_src.flatten().data<float>(),grad_dst.flatten().data<float>(),grad_src_rel.flatten().data<float>(),grad_dst_rel.flatten().data<float>());
        else
            grad_cal_fused_no_rel<<<pos_num, WARP*32, 0, stream2>>>(src_pos_embeddings_.flatten().data<float>(),dst_pos_embeddings_.flatten().data<float>(), (tmp_grad_src.view({pos_num, emb_dim})/rhs_score_exp_sum).flatten().data<float>(), (tmp_grad_dst.view({pos_num, emb_dim})/lhs_score_exp_sum).flatten().data<float>(), tmp2.flatten().data<float>(), tmp1.flatten().data<float>(),emb_dim,grad_src.flatten().data<float>(),grad_dst.flatten().data<float>(),grad_src_rel.flatten().data<float>(),grad_dst_rel.flatten().data<float>());
        
        cudaStreamSynchronize(stream2);
    }
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

__global__ static
void readSingleBuffered(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount, CmdTime* times)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t pageSize = qp->pageSize;
    const size_t chunkSize = qp->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &qp->sq;

    uint64_t blockOffset = startBlock;

    uint32_t currChunk = 0;
    uint32_t i = 0;

    nvm_cmd_t* cmd = nullptr;

    if (threadNum == 0)
    {
        *errCount = 0;
    }
    __syncthreads();

    while (currChunk < numChunks)
    {
        // Prepare in advance next chunk
        cmd = prepareChunk(qp, cmd, ioaddr, 0, blockOffset, currChunk);
        // prepareChunk_Our(qp, cmd, ioaddr, 0, blockOffset, currChunk, prp1, prp2);
        // Consume completions for the previous window
        auto beforeSubmit = clock();
        if (threadNum == 0)
        {
            nvm_sq_submit(sq);

            // waitForIoCompletion(&qp->cq, sq, errCount);
        }
        __syncthreads();
        waitForIoCompletionOur(&qp->cq, sq, threadNum, 0);
        // __syncthreads();
        auto afterSync = clock();
        // Move received chunk
        moveBytes(src, 0, dst, currChunk * chunkSize, chunkSize * numThreads);
        auto afterMove = clock();

        // Record statistics
        if (times != nullptr && threadNum == 0)
        {
            CmdTime* t = &times[i];
            t->size = chunkSize * numThreads;
            t->submitTime = beforeSubmit;
            t->completeTime = afterSync;
            t->moveTime = afterMove;
        }
        __syncthreads();
    
        // Update position and input buffer
        currChunk += numThreads;
        ++i;
    }
}

__global__ static
void readSingleBuffered_our(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount, CmdTime* times, uint64_t* prp1, uint64_t* prp2, unsigned que_num, unsigned vaddr_offset)
{
    const uint16_t numThreads = blockDim.x * gridDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t queue_id = blockIdx.x; // 1 block 1 queue
    const uint32_t pageSize = (qp+queue_id)->pageSize;
    const size_t chunkSize = (qp+queue_id)->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &(qp+queue_id)->sq;
    uint64_t blockOffset = startBlock;
    size_t numChunksCopy = numChunks;
    uint32_t currChunk = 0;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t i = 0;
    nvm_cmd_t* cmd = nullptr;

    // unsigned long beforePerpare, beforeSubmit, afterSync;

    if (threadNum == 0)
    {
        *errCount = 0;
    }
    __syncthreads();
    while (currChunk < numChunks)
    {
        // Prepare in advance next chunk
        // cmd = prepareChunk(qp, cmd, ioaddr, 0, blockOffset, currChunk);
        if(tid < numChunks){
            if(numChunksCopy / blockDim.x == blockIdx.x){
                prepareChunk_Our_read(qp+queue_id, cmd, ioaddr, 0, blockOffset, currChunk, prp1, prp2, vaddr_offset, numChunksCopy % blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, numChunksCopy % blockDim.x);
            }
            else{
                prepareChunk_Our_read(qp+queue_id, cmd, ioaddr, 0, blockOffset, currChunk, prp1, prp2, vaddr_offset, blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, blockDim.x);
            }
            numChunksCopy -= (numThreads);
        }
        // Update position and input buffer
        currChunk += numThreads;
        tid += numThreads;
        ++i;
    }
}

__global__ static
void readSingleBuffered_our_origin(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount, CmdTime* times, uint64_t* prp1, uint64_t* prp2, unsigned que_num, unsigned vaddr_offset)
{
    const uint16_t numThreads = blockDim.x * gridDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t queue_id = blockIdx.x; // 1 block 1 queue
    const uint32_t pageSize = (qp+queue_id)->pageSize;
    const size_t chunkSize = (qp+queue_id)->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &(qp+queue_id)->sq;
    uint64_t blockOffset = startBlock;
    size_t numChunksCopy = numChunks;
    uint32_t currChunk = 0;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t i = 0;
    nvm_cmd_t* cmd = nullptr;

    while (currChunk < numChunks)
    {
        prepareChunk_Our_read(qp+queue_id, cmd, ioaddr, 0, blockOffset, currChunk, prp1, prp2, vaddr_offset, blockDim.x);

        waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, blockDim.x);

        // Update position and input buffer
        currChunk += numThreads;
    }

}

__global__ static
void writeSingleBuffered_our(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount, CmdTime* times, uint64_t* prp1, uint64_t* prp2, unsigned que_num, unsigned vaddr_offset)
{
    const uint16_t numThreads = blockDim.x * gridDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t queue_id = blockIdx.x; // 1 block 1 queue
    const uint32_t pageSize = (qp+queue_id)->pageSize;
    const size_t chunkSize = (qp+queue_id)->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &(qp+queue_id)->sq;
    uint64_t blockOffset = startBlock;
    size_t numChunksCopy = numChunks;
    uint32_t currChunk = 0;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t i = 0;
    nvm_cmd_t* cmd = nullptr;

    // unsigned long beforePerpare, beforeSubmit, afterSync;

    if (threadNum == 0)
    {
        *errCount = 0;
        // printf("AA %lx\n", prp2[0]);
    }
    __syncthreads();

    while (currChunk < numChunks)
    {
        // cmd = prepareChunk(qp, cmd, ioaddr, 0, blockOffset, currChunk);
        // beforePerpare = clock();
        if(tid < numChunks){
            if(numChunksCopy / blockDim.x == blockIdx.x){
                prepareChunk_Our_write(qp+queue_id, cmd, ioaddr, 0, blockOffset, currChunk, prp1, prp2, vaddr_offset, numChunksCopy % blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, numChunksCopy % blockDim.x);
            }
            else{
                prepareChunk_Our_write(qp+queue_id, cmd, ioaddr, 0, blockOffset, currChunk, prp1, prp2, vaddr_offset, blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, blockDim.x);
            }
            numChunksCopy -= (numThreads);
        }
        // Update position and input buffer
        currChunk += numThreads;
        tid += numThreads;
        ++i;
    }
}

__global__ static
void readSingleBuffered_our_merge(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount, CmdTime* times, uint64_t* prp1, uint64_t* prp2, unsigned que_num, unsigned vaddr_offset, long partition_block_size, long buffer_block_size)
{
    const uint16_t numThreads = blockDim.x * gridDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t queue_id = blockIdx.x; // 1 block 1 queue
    const uint32_t pageSize = (qp+queue_id)->pageSize;
    const size_t chunkSize = (qp+queue_id)->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &(qp+queue_id)->sq;
    // printf("%d\n", prp1[threadNum]);
    uint64_t blockOffset = startBlock;
    size_t numChunksCopy = numChunks;
    uint32_t currChunk = 0;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    nvm_cmd_t* cmd = nullptr;

    // unsigned long beforePerpare, beforeSubmit, afterSync;

    if (threadNum == 0)
    {
        *errCount = 0;
    }
    __syncthreads();

    while (currChunk < numChunks)
    {
        if(tid < numChunks){
            if(numChunksCopy / blockDim.x == blockIdx.x){
                prepareChunk_Our_read(qp+queue_id, cmd, ioaddr, 0, blockOffset + partition_block_size * (tid >= (numChunks / 2)), currChunk - (numChunks/2) * (tid >= (numChunks / 2)), prp1, prp2, vaddr_offset + 3 * (uint32_t)buffer_block_size * (tid >= (numChunks / 2)), numChunksCopy % blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, numChunksCopy % blockDim.x);
            }
            else{
                prepareChunk_Our_read(qp+queue_id, cmd, ioaddr, 0, blockOffset + partition_block_size * (tid >= (numChunks / 2)), currChunk - (numChunks/2) * (tid >= (numChunks / 2)), prp1, prp2, vaddr_offset + 3 * (uint32_t)buffer_block_size * (tid >= (numChunks / 2)), blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, blockDim.x);
            }
            numChunksCopy -= (numThreads);
        }
        currChunk += numThreads;
        tid += numThreads;
    }

}

__global__ static
void writeSingleBuffered_our_merge(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount, CmdTime* times, uint64_t* prp1, uint64_t* prp2, unsigned que_num, unsigned vaddr_offset, long partition_block_size, long buffer_block_size)
{
    const uint16_t numThreads = blockDim.x * gridDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t queue_id = blockIdx.x; // 1 block 1 queue
    const uint32_t pageSize = (qp+queue_id)->pageSize;
    const size_t chunkSize = (qp+queue_id)->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &(qp+queue_id)->sq;
    uint64_t blockOffset = startBlock;
    size_t numChunksCopy = numChunks;
    uint32_t currChunk = 0;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t i = 0;
    nvm_cmd_t* cmd = nullptr;

    // unsigned long beforePerpare, beforeSubmit, afterSync;

    if (threadNum == 0)
    {
        *errCount = 0;
    }
    __syncthreads();

    while (currChunk < numChunks)
    {
        if(tid < numChunks){
            if(numChunksCopy / blockDim.x == blockIdx.x){
                prepareChunk_Our_write(qp+queue_id, cmd, ioaddr, 0, blockOffset + partition_block_size * (tid >= (numChunks / 2)), currChunk - (numChunks/2) * (tid >= (numChunks / 2)), prp1, prp2, vaddr_offset + 3 * (uint32_t)buffer_block_size * (tid >= (numChunks / 2)), numChunksCopy % blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, numChunksCopy % blockDim.x);
            }
            else{
                prepareChunk_Our_write(qp+queue_id, cmd, ioaddr, 0, blockOffset + partition_block_size * (tid >= (numChunks / 2)), currChunk - (numChunks/2) * (tid >= (numChunks / 2)), prp1, prp2, vaddr_offset + 3 * (uint32_t)buffer_block_size * (tid >= (numChunks / 2)), blockDim.x);
                waitForIoCompletionOur(&(qp+queue_id)->cq, sq, threadNum, blockDim.x);
            }
            numChunksCopy -= (numThreads);
        }
        currChunk += numThreads;
        tid += numThreads;
        ++i;
    }
}

static void printStatistics(const Settings& settings, const cudaDeviceProp& prop, const std::shared_ptr<CmdTime> gpuTimes)
{
    const size_t numChunks = settings.numChunks;
    auto hostTimes = createReportingList(numChunks);

    auto err = cudaMemcpy(hostTimes.get(), gpuTimes.get(), sizeof(CmdTime) * numChunks, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw err;
    }

    const auto* times = hostTimes.get();
    const double rate = ((double) prop.clockRate) / 1e3;

    fprintf(stdout, "#%9s; %12s; %12s; %12s; %12s; %12s; %12s;\n",
            "size", "disk_lat", "disk_bw", "mem_lat", "mem_bw", "cum_lat", "cum_bw");
    fflush(stdout);
    for (size_t i = 0; i < numChunks; ++i)
    {
        const auto& t = times[i];
        auto diskTime = (t.completeTime - t.submitTime) / rate;
        auto moveTime = (t.moveTime - t.completeTime) / rate;
        auto totalTime = (t.moveTime - t.submitTime) / rate;

        auto diskBw = times[i].size / diskTime;
        auto moveBw = times[i].size / moveTime;
        auto totalBw = times[i].size / totalTime;

        fprintf(stdout, "%10zu; %12.3f; %12.3f; %12.3f; %12.3f; %12.3f; %12.3f;\n", 
                t.size, diskTime, diskBw, moveTime, moveBw, totalTime, totalBw);
        fflush(stdout);
    }
}


static double launchNvmKernel(const Controller& ctrl, BufferPtr destination, const Settings& settings, const cudaDeviceProp& prop)
{
    QueuePair queuePair;
    DmaPtr queueMemory = prepareQueuePair(queuePair, ctrl, settings);

    // Set up and prepare queues
    auto deviceQueue = createBuffer(sizeof(QueuePair), settings.cudaDevice);
    auto err = cudaMemcpy(deviceQueue.get(), &queuePair, sizeof(QueuePair), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    // const size_t pageSize = ctrl.info.page_size;
    const size_t pageSize = settings.pageSize;
    const size_t chunkSize = pageSize * settings.numPages;
    const size_t totalChunks = settings.numChunks * settings.numThreads;
    // Create input buffer
    const size_t sourceBufferSize = (settings.doubleBuffered + 1) * chunkSize * totalChunks+ (1UL << 16);
    auto source = createDma(ctrl.ctrl, sourceBufferSize, settings.cudaDevice, settings.adapter, settings.segmentId + 1); // vaddr is a dev ptr

    std::shared_ptr<CmdTime> times;
    if (settings.stats)
    {
        times = createReportingList(settings.numChunks, settings.cudaDevice);
    }

    err = cudaSetDevice(settings.cudaDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    // We want to count number of errors
    uint64_t* ec = nullptr;
    err = cudaMalloc(&ec, sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        throw err;
    }

    // prepare prp
    uint64_t* prp1, *prp2 = NULL;
    if(pageSize <= ctrl.info.page_size){
        printf("Condition 1\n");
        prp1 = (uint64_t *)(createBuffer(totalChunks * sizeof(uint64_t), settings.cudaDevice).get());
        uint64_t* tmp = new uint64_t[totalChunks];
        std::memset(tmp, 0, totalChunks * sizeof(uint64_t));
        for(size_t i = 0; i < totalChunks; i++){
            tmp[i] = (uint64_t)source->ioaddrs[i];
        }
        cudaMemcpy(prp1, tmp, totalChunks * sizeof(uint64_t), cudaMemcpyHostToDevice);
        delete tmp;
    }
    else if(pageSize > ctrl.info.page_size && pageSize <= ctrl.info.page_size * 2){
        printf("Condition 2\n");
        prp1 = (uint64_t *)(createBuffer(totalChunks * sizeof(uint64_t), settings.cudaDevice).get());
        prp2 = (uint64_t *)(createBuffer(totalChunks * sizeof(uint64_t), settings.cudaDevice).get());
        uint64_t* tmp1 = new uint64_t[totalChunks];
        std::memset(tmp1, 0, totalChunks * sizeof(uint64_t));
        uint64_t* tmp2 = new uint64_t[totalChunks];
        std::memset(tmp2, 0, totalChunks * sizeof(uint64_t));
        for(size_t i = 0; i < totalChunks; i++){
            tmp1[i] = (uint64_t)source->ioaddrs[2*i];
            tmp2[i] = (uint64_t)source->ioaddrs[2*i+1];
        }
        cudaMemcpy(prp1, tmp1, totalChunks* sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(prp2, tmp2, totalChunks* sizeof(uint64_t), cudaMemcpyHostToDevice);
        delete tmp1;
        delete tmp2;
    }
    else{
        printf("Condition 3\n");
        prp1 = (uint64_t *)(createBuffer(totalChunks * sizeof(uint64_t), settings.cudaDevice).get());
        prp2 = (uint64_t *)(createBuffer(totalChunks * sizeof(uint64_t), settings.cudaDevice).get());
        DmaPtr prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(totalChunks * ctrl.info.page_size, 1UL << 16), settings.cudaDevice);
        uint64_t* tmp1 = new uint64_t[totalChunks];
        std::memset(tmp1, 0, totalChunks * sizeof(uint64_t));
        uint64_t* tmp2 = new uint64_t[totalChunks];
        std::memset(tmp2, 0, totalChunks * sizeof(uint64_t));
        uint64_t* tmp3 = new uint64_t[totalChunks * ctrl.info.page_size];
        std::memset(tmp3, 0, totalChunks * ctrl.info.page_size);
        const uint32_t uints_per_page = ctrl.info.page_size / sizeof(uint64_t);
        for(size_t i = 0; i < totalChunks; i++){
            tmp1[i] = (uint64_t)source->ioaddrs[i * (pageSize / ctrl.info.page_size)];
            tmp2[i] = prp_list_dma.get()->ioaddrs[i];
            for(size_t j = 0; j < (pageSize / ctrl.info.page_size); j++){
                tmp3[i * uints_per_page + j] = (uint64_t)source->ioaddrs[i * (pageSize / ctrl.info.page_size) + j + 1];
            }
        }
        cudaMemcpy(prp1, tmp1, totalChunks* sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(prp2, tmp2, totalChunks* sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(prp_list_dma.get()->vaddr, tmp3, totalChunks * ctrl.info.page_size, cudaMemcpyHostToDevice);
        delete tmp1;
        delete tmp2;
        delete tmp3;
    }
    // Launch kernel
    double elapsed = 0;
    try
    {
        Event before;
        
        // readSingleBuffered_our<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalChunks, settings.startBlock, ec, times.get(), prp1, prp2);
        readSingleBuffered<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalChunks, settings.startBlock, ec, times.get());
        Event after;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            throw err;
        }

        elapsed = after - before;
    }
    catch (const cudaError_t err)
    {
        cudaFree(ec);
        throw err;
    }
    catch (const error& e)
    {
        cudaFree(ec);
        throw e;
    }

    // Check error status
    uint64_t errorCount = 0;
    cudaMemcpy(&errorCount, ec, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(ec);

    if (errorCount != 0)
    {
        fprintf(stderr, "WARNING: There were NVM errors\n");
    }

    if (settings.stats)
    {
        printStatistics(settings, prop, times);
    }

    return elapsed;
}

static void outputFile_our(void* data, size_t size, const char* filename)
{
    auto buffer = createBuffer(size);

    cudaError_t err = cudaMemcpy(buffer.get(), data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to copy data from destination: ") + cudaGetErrorString(err));
    }

    FILE* fp = fopen(filename, "w");
    fwrite(buffer.get(), 1, size, fp);
    fclose(fp);
}

void evaluate(DmaPtr& source, const Settings& settings, QueuePair* d_qps, BufferPtr destination, const unsigned long totalChunks, uint64_t* ec, uint64_t* prp1, uint64_t* prp2, long size_per_part, long coef, unsigned num_parts, long float_number_per_part, unsigned part_size, torch::Tensor& src_rel_embedding, torch::Tensor& dst_rel_embedding, unsigned rel_num){
    cout << "Begin read data" << endl;

    vector<vector<unsigned>> partitions(num_parts * num_parts);
    ifstream input_file("freebase86m/test.txt");
    if (!input_file.is_open()) {
        cerr << "Error opening file" << endl;
        return;
    }
    long file_line = 0;
    string line;
    while (getline(input_file, line)) {
        istringstream iss(line);
        unsigned src, rel, tgt;
        if (!(iss >> src >> tgt >> rel)) {
            cerr << "Error reading line: " << line << endl;
            continue;
        }
        file_line++;
        // cout << src << "," << tgt <<"," << rel << endl;
        unsigned src_part = src / part_size;
        unsigned tgt_part = tgt / part_size;
        if (src_part >= num_parts) src_part = num_parts - 1;
        if (tgt_part >= num_parts) tgt_part = num_parts - 1;
        int part_index = src_part * num_parts + tgt_part;
        partitions[part_index].push_back(src);
        partitions[part_index].push_back(tgt);
        partitions[part_index].push_back(rel);
    }
    input_file.close();
    cout << "File line: " << file_line << endl;

    cout << "End init vector" << endl;

    int elements_per_group = 3;
    int pos_num = 1000, neg_num = 10000, embedding_dim = 100, chunk_num = 1;
    float learning_rate=0.1;
    int batch_size=pos_num;

    int num_negative_samples = neg_num;

    
    long partition_block_size = (1UL << (size_per_part - 7)) * coef * 100; 
    long buffer_block_size = (1UL << (size_per_part - 13)) * coef * 100;
    long number_of_emb = float_number_per_part / embedding_dim; 

    cout << "Size per part:" << part_size << endl;
    cout << "Size per part binary:" << size_per_part << endl;
    // size_per_part
    cout << "Float number per part:" << float_number_per_part << endl;
    cout << "Embedding number per part:" << number_of_emb << endl;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor node_embedding_buffer = torch::from_blob((float*)source->vaddr, {3 * float_number_per_part}, opts);

    torch::Tensor src_pos_c, dst_pos_c, src_rel_c, dst_rel_c, src_neg_c, dst_neg_c;
    // torch::Tensor unique_node_embedding, unique_rel_embedding;
    
    node_embedding_buffer = node_embedding_buffer.view({3 * number_of_emb, embedding_dim});
    
    cout << "End prepare node embedding" << endl;
    cout << node_embedding_buffer.sizes() << endl;

    // src_pos_c = src_pos.cuda(); dst_pos_c = dst_pos.cuda(); src_rel_c = src_rel.cuda(); dst_rel_c = dst_rel.cuda(); src_neg_c = src_neg.cuda(); dst_neg_c = dst_neg.cuda();
    torch::Tensor grad_src = torch::zeros({pos_num, embedding_dim}, opts), 
    grad_dst = torch::zeros({pos_num, embedding_dim}, opts), 
    grad_src_rel = torch::zeros({pos_num, embedding_dim}, opts), 
    grad_dst_rel = torch::zeros({pos_num, embedding_dim}, opts),
    grad_src_neg, grad_dst_neg;

    torch::Tensor unique_node_gradients_, unique_node_gradients2_;
    
    torch::Tensor unique_src_rel_gradients_, unique_src_rel_gradients2_, unique_dst_rel_gradients_, unique_dst_rel_gradients2_;

    // bool sample_cross_part = true;

    vector<vector<int>> order = {{1, 13, 12, 14, 25},{24, 2, 26},{27, 38},{36, 3, 39},{40, 51, 48, 4, 52},{53, 64, 60, 5, 65},{66, 77, 72, 6, 78},{79, 90, 84, 7, 91}};
    
    vector<vector<int>> exchange = {{1,1,3},{2,2,4},{1,3,5},{2,4,6},{1,5,7},{2,6,8},{1,7,9},{2,8,10},{1,9,11},{0,0,1}};
    
    vector<bool> is_access(num_parts);

    float total_auc = 0.0;
    unsigned all_batchs = 0;
    torch::Tensor all_ranks = torch::empty({0});
    
    map<int, int> current_buffer_ids = {{0, 0}, {1, 1}, {2, 2}};
    // cudaStream_t    stream, stream2;
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // cudaStreamCreate(&stream2);
    cout << "Begin transfor" << endl;
    readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 0*2*partition_block_size, ec, NULL, prp1, prp2, settings.queuePairs, 0*buffer_block_size);
    readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 1*2*partition_block_size, ec, NULL, prp1, prp2, settings.queuePairs, 1*buffer_block_size);
    readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 2*2*partition_block_size, ec, NULL, prp1, prp2, settings.queuePairs, 2*buffer_block_size);
    cudaDeviceSynchronize();
    cout << "Begin train" << endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < order.size(); ++i) {
        cout << "=================Order: " << i << endl;
        if (i > 0){
            cout << "Begin read and write" << endl;
            readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, exchange[i/2][2]*2*partition_block_size, ec, NULL, prp1, prp2, settings.queuePairs, exchange[i/2][0]*buffer_block_size);
            cudaDeviceSynchronize();
            current_buffer_ids.erase(exchange[i/2][1]);
            current_buffer_ids[exchange[i/2][2]] = exchange[i/2][0];
        }
        for(int part = 0; part < order[i].size(); part++){
            if(order[i][part] == -1) continue;
            cout << "=============Part:" << part << endl;
            int part_id = order[i][part];
            int part_row_id = part_id / num_parts, part_col_id = part_id % num_parts;

            int part_size_cur = partitions[part_id].size() / 3; 
            int num_batches = (part_size_cur + pos_num - 1) / pos_num;
            // cout << "Begin transfer data" << endl;
            torch::Tensor d_flattened_partitions = torch::from_blob(partitions[part_id].data(), {static_cast<int64_t>(partitions[part_id].size())}, torch::kInt32).to(torch::kCUDA);
            // cout << "Begin iterate batch" << endl;
            
            for (int j = 0; j < num_batches; ++j) {
                // auto start = std::chrono::high_resolution_clock::now();     
                // cout << "Begin batch" << endl;       
                int batch_start = j * batch_size * 3;
                if (j < num_batches - 1) {
                    batch_size = pos_num;
                }
                else {
                    batch_size = part_size_cur % pos_num;
                }
                
                torch::Tensor d_indexed_batch = d_flattened_partitions.slice(0, batch_start, batch_start + batch_size*3);
                // cout << "Begin neg sample" << endl;

                auto ind_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
                torch::Tensor rand_idx;
                vector<torch::Tensor> ret_indices(chunk_num);
                for (int ch = 0; ch < chunk_num; ch++) {
                    rand_idx = torch::randint(0, 3*number_of_emb, {neg_num}, ind_opts);
                    ret_indices[ch] = rand_idx;
                }
                torch::Tensor ret_ind = torch::stack(ret_indices);
                torch::Tensor neg_src_node_id = ret_ind.flatten(0, 1);
                // cout << "neg size:" << ret_ind.sizes() << endl;
                for (int ch = 0; ch < chunk_num; ch++) {
                    rand_idx = torch::randint(0, 3*number_of_emb, {neg_num}, ind_opts);
                    // cout << rand_idx.max() << endl;
                    ret_indices[ch] = rand_idx;
                }
                ret_ind = torch::stack(ret_indices);
                torch::Tensor neg_tgt_node_id = ret_ind.flatten(0, 1);
                // cout << "neg size flatten:" << neg_tgt_node_id.sizes() << endl;

                torch::Tensor src_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 0);
                torch::Tensor tgt_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 1);

                torch::Tensor rel_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 2);

                torch::Tensor unique_node_indices_, unique_node_states, unique_node_embedding, src_pos_indices_mapping_, dst_pos_indices_mapping_, src_neg_indices_mapping_, dst_neg_indices_mapping_;

                src_node_id = src_node_id - (((long)(part_row_id - current_buffer_ids[part_row_id])) * number_of_emb);
                tgt_node_id = tgt_node_id - (((long)(part_col_id - current_buffer_ids[part_col_id])) * number_of_emb);

                torch::Tensor emb_idx = torch::cat({src_node_id, tgt_node_id, neg_src_node_id, neg_tgt_node_id});
            
                auto unique_tup = torch::_unique2(emb_idx, true, true, false);
                
                unique_node_indices_ = get<0>(unique_tup).to(torch::kLong);
                
                torch::Tensor emb_mapping = get<1>(unique_tup).to(torch::kLong);
                int64_t curr = 0;
                int64_t size = batch_size;
                src_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
                curr += size;
                dst_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
                curr += size;
                size = neg_src_node_id.size(0);
                src_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);
                curr += size;
                dst_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);

                unique_node_embedding = node_embedding_buffer.index_select(0, unique_node_indices_);
                        
                torch::Tensor unique_rel_indices_, rel_indices_mapping_, unique_src_rel_embedding, unique_dst_rel_embedding, unique_src_rel_states, unique_dst_rel_states;
                if(rel_num > 1){
                    unique_tup = torch::_unique2(rel_id, true, true, false);
                    
                    unique_rel_indices_ = get<0>(unique_tup).to(torch::kLong);
                    rel_indices_mapping_ = get<1>(unique_tup).to(torch::kLong);
                    
                    unique_src_rel_embedding = src_rel_embedding.index_select(0, unique_rel_indices_);
                    unique_dst_rel_embedding = dst_rel_embedding.index_select(0, unique_rel_indices_);
                    
                    src_rel_c = unique_src_rel_embedding.index_select(0, rel_indices_mapping_);
                    dst_rel_c = unique_dst_rel_embedding.index_select(0, rel_indices_mapping_);
                }
                // cout << "Begin unique index select" << endl;
                src_pos_c = unique_node_embedding.index_select(0, src_pos_indices_mapping_);
                dst_pos_c = unique_node_embedding.index_select(0, dst_pos_indices_mapping_);
                src_neg_c = unique_node_embedding.index_select(0, src_neg_indices_mapping_);
                dst_neg_c = unique_node_embedding.index_select(0, dst_neg_indices_mapping_);
                
                src_neg_c = src_neg_c.view({chunk_num, neg_num, embedding_dim});
                dst_neg_c = dst_neg_c.view({chunk_num, neg_num, embedding_dim});
                
                forward_eva(
                    src_pos_c, src_rel_c, dst_pos_c, dst_rel_c, src_neg_c, dst_neg_c, 
                    batch_size, total_auc, all_ranks);
                if(all_batchs + j + 1 > 1000) break;
            }
            all_batchs += num_batches;
            if(all_batchs*1000 > 1000000) break;
        }
        if(all_batchs*1000 > 1000000) break;
        // break;
        // cout << "End exchange" << endl;
    }
    cout << "All batchs:" << all_batchs << ", All ranks: " << all_ranks.sizes() << endl;
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
    total_auc / all_batchs, avg_ranks, mrr, ranks1, ranks5, ranks10, ranks20, ranks50, ranks100);
}

__global__ void initBuffer(float* addr, long number_of_float, float* src_tensor){
    // unsigned long long seed=1234;
    // long tid = threadIdx.x + blockIdx.x * blockDim.x;
    // if(threadIdx.x + blockIdx.x * blockDim.x == 0) printf("%ld\n", blockDim.x * gridDim.x);
    // curandState state;
    // curand_init(seed, threadIdx.x + blockIdx.x * blockDim.x, 0, &state);
    for(long tid = threadIdx.x + blockIdx.x * blockDim.x; tid < number_of_float; tid += blockDim.x * gridDim.x){
        addr[tid] = src_tensor[tid];
        // addr[tid] = (float)(tid+1) / 10000000;
        // if(tid == 0) printf("%f %f\n", addr[tid], curand_normal(&state));
        // if(threadIdx.x == 0 && blockIdx.x == 0) printf("%ld\n", tid);
    }
}

__global__ void initBuffer_test(float* addr, long number_of_float){
    for(long tid = threadIdx.x + blockIdx.x * blockDim.x; tid < number_of_float; tid += blockDim.x * gridDim.x){
        addr[tid] = (float)tid / 1000000000;
    }
}

__global__ void initBuffer_test2(float* addr, long number_of_float){
    for(long tid = threadIdx.x + blockIdx.x * blockDim.x; tid < number_of_float; tid += blockDim.x * gridDim.x){
        addr[tid] = (float)tid;
    }
}

// __global__ void initBuffer_test(float* addr, long number_of_float){
//     // unsigned long long seed=1234;
//     // long tid = threadIdx.x + blockIdx.x * blockDim.x;
//     // if(threadIdx.x + blockIdx.x * blockDim.x == 0) printf("%ld\n", blockDim.x * gridDim.x);
//     // curandState state;
//     // curand_init(seed, threadIdx.x + blockIdx.x * blockDim.x, 0, &state);
//     for(long tid = threadIdx.x + blockIdx.x * blockDim.x; tid < number_of_float; tid += blockDim.x * gridDim.x){
        
//         // addr[tid] = curand_normal(&state);
//         // addr[tid] = 0.001;
//         addr[tid] = (float)(tid+1);
//         // if(tid == 0) printf("%f %f\n", addr[tid], curand_normal(&state));
//     }
// }

// __global__ void initBuffer_test_zero(float* addr, long number_of_float){
//     for(long tid = threadIdx.x + blockIdx.x * blockDim.x; tid < number_of_float; tid += blockDim.x * gridDim.x){
//         addr[tid] = 0.0;
//     }
// }

void evaluate_local(bool test, bool all, torch::Tensor &node_embedding, torch::Tensor &src_rel_embedding, torch::Tensor &dst_rel_embedding){

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
        ifstream infile("test_edges.pt", std::ios::binary);
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
        ifstream infile("valid_edges.pt", std::ios::binary);
        infile.seekg(0, std::ios::end);
        file_size = (unsigned)(infile.tellg());
        buff = new unsigned char[file_size];
        infile.seekg(0, std::ios::beg);
        infile.read((char*)buff, file_size);
        infile.close();
        cout << "End read data" << endl;
        // input_file.open("valid_edges.txt");
    }

    // vector<vector<int>> partitions(num_parts * num_parts);
    unsigned src, rel, tgt;
    for(unsigned i = 0; i < file_size / 12; i++) {
        src = ((unsigned*)buff)[i * 3];
        rel = ((unsigned*)buff)[i * 3 + 1];
        tgt = ((unsigned*)buff)[i * 3 + 2];
        // unsigned src_part = src / part_size;
        // unsigned tgt_part = tgt / part_size;
        // if (src_part >= num_parts) src_part = num_parts - 1;
        // if (tgt_part >= num_parts) tgt_part = num_parts - 1;
        // unsigned part_index = src_part * num_parts + tgt_part;
        all_batch.push_back(src);
        all_batch.push_back(tgt);
        all_batch.push_back(rel);
    }

    // int i=0;
    // while (getline(input_file, line)) {
    //     istringstream iss(line);
    //     int src, rel, tgt;
    //     if (!(iss >> src >> rel >> tgt)) {
    //         cerr << "Error reading line: " << line << endl;
    //         continue;
    //     }
    //     all_batch.push_back(src);
    //     all_batch.push_back(tgt);
    //     all_batch.push_back(rel);
    // }
    // input_file.close();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int elements_per_group = 3;
    int pos_num = 10000, neg_num = 1000, embedding_dim = 100, chunk_num = 1;
    float learning_rate=0.1;
    int batch_size=pos_num;
    // int node_num = 14951, rel_num = 1345;
    int node_num = 4847571, rel_num = 1, edge_num = 62094395;
    int num_negative_samples = neg_num;
    torch::Tensor src_pos_c, dst_pos_c, src_rel_c, dst_rel_c, src_neg_c, dst_neg_c;
    // torch::Tensor unique_node_embedding, unique_rel_embedding;
    // torch::Tensor node_embedding = torch::zeros({node_num, embedding_dim}).cuda(), 
    // src_rel_embedding = torch::zeros({rel_num, embedding_dim}).cuda(), 
    // dst_rel_embedding = torch::zeros({rel_num, embedding_dim}).cuda();
    // torch::Tensor node_grad_state = torch::zeros({node_num, embedding_dim}).cuda(), src_rel_grad_state = torch::zeros({rel_num, embedding_dim}).cuda(), dst_rel_grad_state = torch::zeros({rel_num, embedding_dim}).cuda();


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

    auto torch_rand = torch::zeros({chunk_num, neg_num, 4}).cuda();
    
    float total_auc = 0.0;

    int num_batchs = (all_batch.size()+(3 * pos_num)-1) / (3 * pos_num);
    torch::Tensor d_flattened_partitions = torch::from_blob(all_batch.data(), {static_cast<int64_t>(all_batch.size())}, torch::kInt32).to(torch::kCUDA);
    torch::Tensor all_ranks = torch::empty({0});
    // cout<<num_batchs<<endl;
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
            auto ind_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
            torch::Tensor rand_idx;
            vector<torch::Tensor> ret_indices(chunk_num);
            for (int ch = 0; ch < chunk_num; ch++) {
                rand_idx = torch::randint(0, node_num, {neg_num}, ind_opts);
                ret_indices[ch] = rand_idx;
            }
            torch::Tensor ret_ind = torch::stack(ret_indices);
            neg_src_node_id = ret_ind.flatten(0, 1);
            // cout << "neg size:" << ret_ind.sizes() << endl;
            for (int ch = 0; ch < chunk_num; ch++) {
                rand_idx = torch::randint(0, node_num, {neg_num}, ind_opts);
                // cout << rand_idx.max() << endl;
                ret_indices[ch] = rand_idx;
            }
            ret_ind = torch::stack(ret_indices);
            neg_tgt_node_id = ret_ind.flatten(0, 1);
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
        // auto new_dst_neg_c = torch::cat({dst_neg_c, torch_rand}, 2);
        // auto new_src_neg_c = torch::cat({src_neg_c, torch_rand}, 2);
        // torch::Tensor new_src_neg_c;
        // torch::Tensor new_dst_neg_c;
        forward_eva(src_pos_c, src_rel_c, dst_pos_c, dst_rel_c, src_neg_c, dst_neg_c, 
                    batch_size, total_auc, all_ranks);
    }

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
    // printf("Auc: {:.3f}, Avg Ranks: {:.3f}, MRR: {:.3f}, Hits@1: {:.3f}, Hits@5: {:.3f}, Hits@10: {:.3f}, Hits@20: {:.3f}, Hits@50: {:.3f}, Hits@100: {:.3f}", total_auc / num_batchs, avg_ranks, mrr, ranks1, ranks5, ranks10,
    //             ranks20, ranks50, ranks100);
}

static double launchNvmKernel_our(const Controller& ctrl, BufferPtr destination, const Settings& settings, const cudaDeviceProp& prop)
{
    unsigned num_parts = 12;
    // unsigned node_num = 41652230, rel_num = 1, edge_num = 1321528663;
    // unsigned node_num = 4847571, rel_num = 1, edge_num = 62094395;
    // unsigned node_num = 14951, rel_num = 1345;
    unsigned node_num = 86054151, rel_num = 14824;
    unsigned part_size = (node_num + num_parts - 1) / num_parts;
    unsigned size_per_part = 0, tmp_part_size = part_size;
    while(tmp_part_size > 0){
        tmp_part_size = tmp_part_size / 2;
        size_per_part++;
        // cout << tmp_part_size << "," << size_per_part << endl;
    }
    size_per_part = size_per_part - 3;
    long tmp_size = 1UL << (size_per_part);
    long coef = 2;
    while(tmp_size * coef < part_size){
        // tmp_size *= coef;
        coef++;
    }
    cout << "Origin part size:" << part_size << endl;
    part_size = (1UL << (size_per_part)) * coef;
    long float_number_per_part = (1UL << (size_per_part)) * coef * 100;

    uint16_t n_qps = std::min((size_t)12, settings.queuePairs);
    // QueuePair* queuePairs = (QueuePair*) malloc(sizeof(QueuePair)*n_qps);
    // printf("%d\n", sizeof(QueuePair));
    QueuePair queuePairs[12];
    QueuePair* d_qps;
    cudaMalloc((void**)&d_qps, sizeof(QueuePair)*n_qps);
    for(uint16_t i = 0; i < n_qps; i++){
        prepareQueuePair_our(queuePairs[i], ctrl, settings, i+1);
        auto err = cudaMemcpy(d_qps+i, &queuePairs[i], sizeof(QueuePair), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw err;
        }
    }
    
    // Set up and prepare queues
    // auto deviceQueue = createBuffer(sizeof(QueuePair), settings.cudaDevice);
    // auto err = cudaMemcpy(deviceQueue.get(), &queuePair, sizeof(QueuePair), cudaMemcpyHostToDevice);
    // if (err != cudaSuccess)
    // {
    //     throw err;
    // }

    // const size_t pageSize = ctrl.info.page_size;
    const size_t pageSize = settings.pageSize;
    const unsigned long chunkSize = pageSize * settings.numPages;
    // const size_t totalChunks = settings.numChunks * settings.numThreads;
    const unsigned long totalChunks = (float_number_per_part * 24) / chunkSize;
    cout << "Total chunks: " << totalChunks << ", chunk / 6: " << totalChunks / 6 << endl;
    cout << "Chunk size: " << chunkSize << endl;
    // const size_t ChunksPerThread = 
    // printf("ps: %d\n", pageSize);
    // Create input buffer
    // const size_t sourceBufferSize = (settings.doubleBuffered + 1) * chunkSize * totalChunks+ (1UL << 16);
    const size_t sourceBufferSize = chunkSize * totalChunks + (1UL << 16);
    cout << "source buffer size: " << sourceBufferSize << endl;
    auto source = createDma(ctrl.ctrl, sourceBufferSize, settings.cudaDevice, settings.adapter, settings.segmentId + 1); // vaddr is a dev ptr
    std::shared_ptr<CmdTime> times;
    if (settings.stats)
    {
        times = createReportingList(settings.numChunks, settings.cudaDevice);
    }

    auto err = cudaSetDevice(settings.cudaDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    // We want to count number of errors
    uint64_t* ec = nullptr;
    err = cudaMalloc(&ec, sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        throw err;
    }

    uint64_t* prp1, *prp2 = NULL;
    DmaPtr prp_list_dma;
    
    cudaMalloc((void**)&prp1, totalChunks * sizeof(uint64_t));
    cudaMemset(prp1, 0, totalChunks*sizeof(uint64_t));
    cudaMalloc((void**)&prp2, totalChunks * sizeof(uint64_t));
    cudaMemset(prp2, 0, totalChunks*sizeof(uint64_t));
    prp_list_dma = createDma(ctrl.ctrl, totalChunks * ctrl.info.page_size+(1UL << 16), settings.cudaDevice);
    uint64_t* tmp1 = new uint64_t[totalChunks];
    std::memset(tmp1, 0, totalChunks * sizeof(uint64_t));
    uint64_t* tmp2 = new uint64_t[totalChunks];
    std::memset(tmp2, 0, totalChunks * sizeof(uint64_t));
    uint64_t* tmp3 = new uint64_t[totalChunks * ctrl.info.page_size];
    std::memset(tmp3, 0, totalChunks * ctrl.info.page_size);
    const uint32_t uints_per_page = ctrl.info.page_size / sizeof(uint64_t);
    for(size_t i = 0; i < totalChunks; i++){
        tmp1[i] = (uint64_t)source->ioaddrs[i * (pageSize / ctrl.info.page_size)];
        tmp2[i] = prp_list_dma.get()->ioaddrs[i];
        for(size_t j = 0; j < (pageSize / ctrl.info.page_size) - 1; j++){
            tmp3[i * uints_per_page + j] = (uint64_t)source->ioaddrs[i * (pageSize / ctrl.info.page_size) + j + 1];
        }
    }
    cudaMemcpy(prp1, tmp1, totalChunks* sizeof(uint64_t), cudaMemcpyHostToDevice);
    err = cudaMemcpy(prp2, tmp2, totalChunks* sizeof(uint64_t), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ 
        throw error(string("Failed to copy data from destination: ") + cudaGetErrorString(err));
    }
    err = cudaMemcpy((uint64_t*)prp_list_dma->vaddr, tmp3, totalChunks * ctrl.info.page_size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ 
        throw error(string("Failed to copy data from destination: ") + cudaGetErrorString(err));
    }
    delete tmp1;
    delete tmp2;
    delete tmp3;

    int pos_num = 100000, neg_num = 1000, embedding_dim = 100, chunk_num = 10;
    long partition_block_size = (1UL << (size_per_part - 7)) * coef * 100; 
    long buffer_block_size = (1UL << (size_per_part - 13)) * coef * 100; 
    long number_of_emb = float_number_per_part / embedding_dim; 

    cout << "Size per part:" << part_size << endl;
    cout << "Size per part binary:" << size_per_part << endl;
    cout << "Coef: " << coef << endl;

    cout << "Partition block size: " << partition_block_size << endl;
    cout << "Buffer block size:" << buffer_block_size << endl;
    // size_per_part
    cout << "Float number per part:" << float_number_per_part << endl;
    cout << "Embedding number per part:" << number_of_emb << endl;

    cout << "Begin random" << endl;
    auto tmp_rand = torch::randn({3 * float_number_per_part}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).mul_(0.01);
    cout << "End torch random" << endl;
    initBuffer<<<128, 512>>>((float*)source->vaddr, 3 * float_number_per_part, tmp_rand.data<float>());
    // initBuffer_test<<<128, 512>>>((float*)source->vaddr, float_number_per_part);
    // initBuffer_test2<<<128, 512>>>((float*)source->vaddr + float_number_per_part * 3, float_number_per_part);
    cudaDeviceSynchronize();

    // writeSingleBuffered_our_merge<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 3, 0*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 0*buffer_block_size, partition_block_size, buffer_block_size);
    // readSingleBuffered_our_merge<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 3, 0*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 1*buffer_block_size, partition_block_size, buffer_block_size);
    // cudaDeviceSynchronize();
    // for(int i = 0; i <num_parts; i++){
    //     writeSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, i*2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 0);
    // }
    // cudaDeviceSynchronize();
    tmp_rand = torch::tensor({});
    cout << "Begin read data" << endl;

    vector<vector<unsigned>> partitions(num_parts * num_parts);
    ifstream input_file("freebase86m/train.txt");
    if (!input_file.is_open()) {
        cerr << "Error opening file" << endl;
        return 1;
    }
    long file_line = 0;
    string line;
    while (getline(input_file, line)) {
        istringstream iss(line);
        unsigned src, rel, tgt;
        if (!(iss >> src >> tgt >> rel)) {
            cerr << "Error reading line: " << line << endl;
            continue;
        }
        file_line++;
        // cout << src << "," << tgt <<"," << rel << endl;
        unsigned src_part = src / part_size;
        unsigned tgt_part = tgt / part_size;
        if (src_part >= num_parts) src_part = num_parts - 1;
        if (tgt_part >= num_parts) tgt_part = num_parts - 1;

        int part_index = src_part * num_parts + tgt_part;

        partitions[part_index].push_back(src);
        partitions[part_index].push_back(tgt);
        partitions[part_index].push_back(rel);
    }
    input_file.close();
    cout << "File line: " << file_line << endl;

    cout << "End init vector" << endl;

    int elements_per_group = 3;
    
    float learning_rate=0.1;
    int batch_size=pos_num;

    int num_negative_samples = neg_num;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor node_embedding_buffer = torch::from_blob((float*)source->vaddr, {3 * float_number_per_part}, opts), node_grad_state_buffer = torch::from_blob(((float*)source->vaddr) + (3 * float_number_per_part), {3 * float_number_per_part}, opts);
    cout << node_embedding_buffer.max() << "," << node_embedding_buffer.min() << endl;

    torch::Tensor src_pos_c, dst_pos_c, src_rel_c, dst_rel_c, src_neg_c, dst_neg_c;
    // torch::Tensor unique_node_embedding, unique_rel_embedding;
    // torch::Tensor src_rel_embedding = torch::rand({rel_num, embedding_dim}).cuda(), dst_rel_embedding = torch::rand({rel_num, embedding_dim}).cuda();
    torch::Tensor src_rel_embedding = torch::zeros({rel_num, embedding_dim}, opts);
    src_rel_embedding.narrow(1, 0, (embedding_dim / 2) - 1).fill_(1);
    torch::Tensor dst_rel_embedding = torch::zeros({rel_num, embedding_dim}, opts);
    src_rel_embedding.narrow(1, 0, (embedding_dim / 2) - 1).fill_(1);
    torch::Tensor src_rel_grad_state = torch::zeros({rel_num, embedding_dim}, opts), dst_rel_grad_state = torch::zeros({rel_num, embedding_dim}, opts);
    cout << "End init relation embedding" << endl;
    cout << src_rel_embedding[0][0] << src_rel_embedding[0][50] << src_rel_grad_state[0][0] << endl;
    
    // torch::Tensor tensor_a = torch::from_blob(source->vaddr, {1UL << 30}, opts);
    
    // emb_buffer[0] = emb_buffer[0].view({number_of_emb, embedding_dim});
    // emb_buffer[1] = emb_buffer[1].view({number_of_emb, embedding_dim});
    // emb_buffer[2] = emb_buffer[2].view({number_of_emb, embedding_dim});
    // state_buffer[0] = state_buffer[0].view({number_of_emb, embedding_dim});
    // state_buffer[1] = state_buffer[1].view({number_of_emb, embedding_dim});
    // state_buffer[2] = state_buffer[2].view({number_of_emb, embedding_dim});
    node_embedding_buffer = node_embedding_buffer.view({3 * number_of_emb, embedding_dim});
    node_grad_state_buffer = node_grad_state_buffer.view({3 * number_of_emb, embedding_dim});
    cout << "End prepare node embedding" << endl;
    cout << node_embedding_buffer.sizes() << endl;
    // auto tmp_tensor = node_embedding_buffer.index_select(0, torch::tensor({1851160}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA)));
    // cout << node_embedding_buffer.sizes() << "," << emb_buffer[0].sizes() << endl;
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

    torch::Tensor unique_node_gradients_, unique_node_gradients2_;
    
    torch::Tensor unique_src_rel_gradients_, unique_src_rel_gradients2_, unique_dst_rel_gradients_, unique_dst_rel_gradients2_;

    auto lhs_score_exp = torch::rand({chunk_num, pos_num/chunk_num, neg_num+8}).flatten().cuda();
    auto rhs_score_exp = torch::rand({chunk_num, pos_num/chunk_num, neg_num+8}).flatten().cuda();
    auto lhs_pos_scores = torch::rand({pos_num, 1}).flatten().cuda();
    auto rhs_pos_scores = torch::rand({pos_num, 1}).flatten().cuda();
    auto adjusted_dst_pos = torch::zeros({pos_num, embedding_dim}, opts).flatten();
    auto adjusted_src_pos = torch::zeros({pos_num, embedding_dim}, opts).flatten();

    auto torch_rand = torch::zeros({chunk_num, neg_num, 4}).cuda();

    auto torch_neg_pad = torch::zeros({1, neg_num, embedding_dim+4}).cuda();

    auto torch_zeros = torch::zeros({pos_num, embedding_dim}, dtype(torch::kFloat32)).cuda();

    // bool sample_cross_part = true;

    vector<vector<int>> order = {{1, 13, 12, 14, 25},{24, 2, 26},{27, 38},{36, 3, 39},{40, 51},{48, 4, 52},{53, 64},{60, 5, 65},{66, 77},{72, 6, 78},{79, 90},{84, 7, 91},{92, 103},{96, 8, 104},{105, 116},{108, 9, 117},{118, 129},{0, 10, 120},{11, 132},{130, 131, 142},{121, 22},{133, 23, 143},{135, 47},{37, 15},{41, 63},{61, 17},{67, 89},{85, 19},{93, 115},{109, 21},{112, 57},{49, 16},{54, 76},{18, 73},{20, 97},{102, 80},{98, 32},{74, 30},{81, 114},{110, 33},{113, 69},{62, 29},{70, 125},{122, 34},{124, 58},{50, 28},{55, 88},{31, 86},{35, 134},{95, 139},{87, 43},{-1},{138, 83},{75, 42},{82, 126},{123, 46},{128, 106},{44, 99},{45, 111},{-1},{-2},{56, 100},{59, 136},{107, 140},{101, 68},{71, 137},{-1},{-1},{94, 127}};
    
    vector<vector<int>> exchange = {{1,1,3},{2,2,4},{1,3,5},{2,4,6},{1,5,7},{2,6,8},{1,7,9},{2,8,10},{1,9,11},{0,0,1},{2,10,3},{1,11,5},{2,3,7},{1,5,9},{2,7,4},{1,9,6},{2,4,8},{0,1,2},{2,8,9},{1,6,5},{2,9,10},{1,5,4},{2,10,7},{1,4,11},{0,2,3},{2,7,6},{1,11,10},{2,6,8},{1,10,9},{0,3,4},{1,9,11},{0,4,5},{2,8,7},{0,5,10},{2,7,9}};
        
    
    vector<bool> is_access(num_parts);
    is_access[0] = true; is_access[1] = true; is_access[2] = true;
    
    cudaStream_t    stream, stream2;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaStreamCreate(&stream2);
    cout << "Begin train" << endl;
    for(int epc = 0; epc < 10; epc++){
        map<int, int> current_buffer_ids = {{0, 0}, {1, 1}, {2, 2}};
        if(epc > 0){
            readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 0*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 0*buffer_block_size);
            readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 1*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 3*buffer_block_size);
            readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 1*buffer_block_size);
            readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 3*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 4*buffer_block_size);
            readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 4*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 2*buffer_block_size);
            readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 5*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 5*buffer_block_size);
            cudaDeviceSynchronize();
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < order.size(); ++i) {
            cout << "=================Order: " << i << "; All: " << order.size() << endl;
            if ((i % 2 == 1 || order[i][0] == -1) && i < (order.size() - 1)){
            // if(i > 0){
                cout << "Begin read and write" << endl;
                // writeSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, exchange[i/2][1]*2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, exchange[i/2][0]*buffer_block_size);
                // writeSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, (exchange[i/2][1]*2+1)*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, (exchange[i/2][0]+3)*buffer_block_size);
                writeSingleBuffered_our_merge<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 3, exchange[i/2][1]*2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, exchange[i/2][0]*buffer_block_size, partition_block_size, buffer_block_size);
                if(is_access[exchange[i/2][2]])
                {
                    // readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, exchange[i/2][2]*2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, exchange[i/2][0]*buffer_block_size);
                    // readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, (exchange[i/2][2]*2+1)*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, (exchange[i/2][0]+3)*buffer_block_size);
                    readSingleBuffered_our_merge<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 3, exchange[i/2][2]*2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, exchange[i/2][0]*buffer_block_size, partition_block_size, buffer_block_size);
                }
                // else{
                    // emb_buffer[exchange[i/2][0]] = torch::rand({number_of_emb, embedding_dim}, opts);
                    // state_buffer[exchange[i/2][0]] = torch::rand({number_of_emb, embedding_dim}, opts);
                    // cout << "BB" << node_grad_state_buffer[0][0] << node_grad_state_buffer[number_of_emb-1][99] << node_grad_state_buffer[number_of_emb][0] << endl;
                    // cout << (((float*)source->vaddr)[0]);
                    // cout << "AA" << node_grad_state_buffer[0][0] << node_grad_state_buffer[number_of_emb-1][99] << node_grad_state_buffer[number_of_emb][0] << endl;
                    
                // }
                // cudaStreamSynchronize(stream);
                // cudaDeviceSynchronize();
                // if(is_access[exchange[i/2][2]] == false){
                //     cudaMemset((float*)source->vaddr + (exchange[i/2][0] + 3) * float_number_per_part, 0, float_number_per_part * sizeof(float));
                //     is_access[exchange[i/2][2]] = true;
                // }
                // current_buffer_ids.erase(exchange[i/2][1]);
                // current_buffer_ids[exchange[i/2][2]] = exchange[i/2][0];
            }
            // shuffle(order[i].begin(), order[i].end(), default_random_engine(time(NULL)));
            for(int part = 0; part < order[i].size(); part++){
                if (order[i][part] == -1 || order[i][part] == -2) continue;
                cout << "=============Part:" << part << " All: " << order[i].size() << endl;
                int part_id = order[i][part];
                int part_row_id = part_id / num_parts, part_col_id = part_id % num_parts;

                int part_size_cur = partitions[part_id].size() / 3; 
                int num_batches = (part_size_cur + pos_num - 1) / pos_num;
                // cout << "Begin transfer data" << endl;
                torch::Tensor d_flattened_partitions = torch::from_blob(partitions[part_id].data(), {static_cast<int64_t>(partitions[part_id].size())}, torch::kInt32).to(torch::kCUDA);
                // cout << "Begin iterate batch" << endl;
                if(epc > 0){
                    d_flattened_partitions = d_flattened_partitions.view({part_size_cur, 3});
                    auto idx = torch::randperm(part_size_cur, torch::TensorOptions().dtype(at::kLong).device(torch::kCUDA));
                    d_flattened_partitions = d_flattened_partitions.index_select(0,idx);
                    d_flattened_partitions = d_flattened_partitions.view({part_size_cur * 3});
                }
                
                for (int j = 0; j < num_batches; ++j) {  
                    // cout << "Begin batch" << endl;       
                    int batch_start = j * batch_size * 3;
                    if (j < num_batches - 1) {
                        batch_size = pos_num;
                    }
                    else {
                        batch_size = part_size_cur % pos_num;
                    }
                    
                    torch::Tensor d_indexed_batch = d_flattened_partitions.slice(0, batch_start, batch_start + batch_size*3);

                    auto ind_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
                    torch::Tensor rand_idx;
                    vector<torch::Tensor> ret_indices(chunk_num);
                    for (int ch = 0; ch < chunk_num; ch++) {
                        rand_idx = torch::randint(0, 3 * number_of_emb, {neg_num}, ind_opts);
                        ret_indices[ch] = rand_idx;
                    }
                    torch::Tensor ret_ind = torch::stack(ret_indices);
                    torch::Tensor neg_src_node_id = ret_ind.flatten(0, 1);
                    for (int ch = 0; ch < chunk_num; ch++) {
                        rand_idx = torch::randint(0, 3 * number_of_emb, {neg_num}, ind_opts);
                        // cout << rand_idx.max() << endl;
                        ret_indices[ch] = rand_idx;
                    }
                    ret_ind = torch::stack(ret_indices);
                    torch::Tensor neg_tgt_node_id = ret_ind.flatten(0, 1);
                    // cout << "neg size flatten:" << neg_tgt_node_id.sizes() << endl;

                    torch::Tensor src_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 0);
                    torch::Tensor tgt_node_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 1);

                    torch::Tensor rel_id = d_indexed_batch.view({batch_size, elements_per_group}).select(1, 2);
                    src_node_id = src_node_id - (((long)(part_row_id - current_buffer_ids[part_row_id])) * number_of_emb);
                    tgt_node_id = tgt_node_id - (((long)(part_col_id - current_buffer_ids[part_col_id])) * number_of_emb);
                    
                    torch::Tensor unique_node_indices_, unique_node_states, unique_node_embedding, src_pos_indices_mapping_, dst_pos_indices_mapping_, src_neg_indices_mapping_, dst_neg_indices_mapping_;

                    torch::Tensor emb_idx = torch::cat({src_node_id, tgt_node_id, neg_src_node_id, neg_tgt_node_id});
                
                    auto unique_tup = torch::_unique2(emb_idx, true, true, false);
                    
                    unique_node_indices_ = get<0>(unique_tup).to(torch::kLong);
                    
                    torch::Tensor emb_mapping = get<1>(unique_tup).to(torch::kLong);
                    int64_t curr = 0;
                    int64_t size = batch_size;
                    src_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
                    curr += size;
                    dst_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
                    curr += size;
                    size = neg_src_node_id.size(0);
                    src_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);
                    curr += size;
                    dst_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);

                    unique_node_embedding = node_embedding_buffer.index_select(0, unique_node_indices_);
                    unique_node_states = node_grad_state_buffer.index_select(0, unique_node_indices_);

                    torch::Tensor unique_rel_indices_, rel_indices_mapping_, unique_src_rel_embedding, unique_dst_rel_embedding, unique_src_rel_states, unique_dst_rel_states;
                    if(rel_num > 1){
                        unique_tup = torch::_unique2(rel_id, true, true, false);
                        
                        unique_rel_indices_ = get<0>(unique_tup).to(torch::kLong);
                        rel_indices_mapping_ = get<1>(unique_tup).to(torch::kLong);
                        
                        unique_src_rel_embedding = src_rel_embedding.index_select(0, unique_rel_indices_);
                        unique_dst_rel_embedding = dst_rel_embedding.index_select(0, unique_rel_indices_);
                        
                        unique_src_rel_states = src_rel_grad_state.index_select(0, unique_rel_indices_);
                        unique_dst_rel_states = dst_rel_grad_state.index_select(0, unique_rel_indices_);
                        src_rel_c = unique_src_rel_embedding.index_select(0, rel_indices_mapping_);
                        dst_rel_c = unique_dst_rel_embedding.index_select(0, rel_indices_mapping_);
                    }

                    src_pos_c = unique_node_embedding.index_select(0, src_pos_indices_mapping_);
                    dst_pos_c = unique_node_embedding.index_select(0, dst_pos_indices_mapping_);
                    src_neg_c = unique_node_embedding.index_select(0, src_neg_indices_mapping_);
                    dst_neg_c = unique_node_embedding.index_select(0, dst_neg_indices_mapping_);
                    
                    src_neg_c = src_neg_c.view({chunk_num, neg_num, embedding_dim});
                    dst_neg_c = dst_neg_c.view({chunk_num, neg_num, embedding_dim});
                    auto new_dst_neg_c = torch::cat({dst_neg_c, torch_rand}, 2);
                    auto new_src_neg_c = torch::cat({src_neg_c, torch_rand}, 2);
                    new_src_neg_c = torch::cat({new_src_neg_c, torch_neg_pad});
                    new_dst_neg_c = torch::cat({new_dst_neg_c, torch_neg_pad});
                    
                    if(pos_num > batch_size){
                        auto to_cat = torch_zeros.narrow(0, 0, pos_num - batch_size);
                        src_pos_c = torch::cat({src_pos_c, to_cat});
                        dst_pos_c = torch::cat({dst_pos_c, to_cat});
                        if(rel_num > 1){
                            src_rel_c = torch::cat({src_rel_c, to_cat});
                            dst_rel_c = torch::cat({dst_rel_c, to_cat});
                        }
                    }
                    
                    forward_our_front_mma(
                        src_pos_c, src_rel_c, dst_pos_c, dst_rel_c, src_neg_c, dst_neg_c, 
                        true, pos_num, neg_num, embedding_dim, chunk_num, rel_num,
                        grad_src, grad_dst, grad_src_rel, grad_dst_rel, grad_src_neg, grad_dst_neg, 
                        lhs_score_exp, rhs_score_exp, lhs_pos_scores, rhs_pos_scores, adjusted_dst_pos, adjusted_src_pos, 
                        new_src_neg_c, new_dst_neg_c, stream2);
                    
                    grad_src = grad_src.view({pos_num, embedding_dim});
                    grad_dst = grad_dst.view({pos_num, embedding_dim});
                    
                    grad_src_neg = grad_src_neg.view({chunk_num*neg_num, embedding_dim});
                    grad_dst_neg = grad_dst_neg.view({chunk_num*neg_num, embedding_dim});

                    unique_node_gradients_ = torch::zeros_like(unique_node_embedding);
                    if(pos_num > batch_size){
                        unique_node_gradients_.index_add_(0, src_pos_indices_mapping_, grad_src.narrow(0, 0, batch_size));
                        
                        unique_node_gradients_.index_add_(0, src_neg_indices_mapping_, grad_src_neg);
                        
                        unique_node_gradients_.index_add_(0, dst_pos_indices_mapping_, grad_dst.narrow(0, 0, batch_size));
                        
                        unique_node_gradients_.index_add_(0, dst_neg_indices_mapping_, grad_dst_neg);
                    }
                    else{
                        unique_node_gradients_.index_add_(0, src_pos_indices_mapping_, grad_src);

                        unique_node_gradients_.index_add_(0, src_neg_indices_mapping_, grad_src_neg);

                        unique_node_gradients_.index_add_(0, dst_pos_indices_mapping_, grad_dst);

                        unique_node_gradients_.index_add_(0, dst_neg_indices_mapping_, grad_dst_neg);
                    }

                    unique_node_gradients2_ = unique_node_gradients_.pow(2);
                    unique_node_states.add_(unique_node_gradients2_);
                    unique_node_gradients_ = -learning_rate * (unique_node_gradients_ / ((unique_node_states).sqrt().add_(1e-9)));
                    node_embedding_buffer.index_add_(0, unique_node_indices_, unique_node_gradients_);
                    node_grad_state_buffer.index_add_(0, unique_node_indices_, unique_node_gradients2_);
                
                    if(rel_num > 1){
                        unique_src_rel_gradients_ = torch::zeros_like(unique_src_rel_embedding);
                        unique_dst_rel_gradients_ = torch::zeros_like(unique_dst_rel_embedding);
                        if(pos_num > batch_size){
                            unique_src_rel_gradients_.index_add_(0, rel_indices_mapping_, grad_src_rel.narrow(0, 0, batch_size));
                            unique_dst_rel_gradients_.index_add_(0, rel_indices_mapping_, grad_dst_rel.narrow(0, 0, batch_size));
                        }
                        else{
                            unique_src_rel_gradients_.index_add_(0, rel_indices_mapping_, grad_src_rel);
                            unique_dst_rel_gradients_.index_add_(0, rel_indices_mapping_, grad_dst_rel);
                        }
                        
                        unique_src_rel_gradients2_ = unique_src_rel_gradients_.pow(2);
                        unique_src_rel_states.add_(unique_src_rel_gradients2_);
                        unique_src_rel_gradients_ = -learning_rate * (unique_src_rel_gradients_ / ((unique_src_rel_states).sqrt().add_(1e-9)));
                        unique_dst_rel_gradients2_ = unique_dst_rel_gradients_.pow(2);
                        unique_dst_rel_states.add_(unique_dst_rel_gradients2_);
                        unique_dst_rel_gradients_ = -learning_rate * (unique_dst_rel_gradients_ / ((unique_dst_rel_states).sqrt().add_(1e-9)));

                        src_rel_embedding.index_add_(0, unique_rel_indices_, unique_src_rel_gradients_);
                        dst_rel_embedding.index_add_(0, unique_rel_indices_, unique_dst_rel_gradients_);
                        src_rel_grad_state.index_add_(0, unique_rel_indices_, unique_src_rel_gradients2_);
                        dst_rel_grad_state.index_add_(0, unique_rel_indices_, unique_dst_rel_gradients2_);
                    }
                    
                }
            }
            if(i % 2 == 1){
                cudaStreamSynchronize(stream);
                if(is_access[exchange[i/2][2]] == false){
                    cudaMemset((float*)source->vaddr + (exchange[i/2][0] + 3) * float_number_per_part, 0, float_number_per_part * sizeof(float));
                    is_access[exchange[i/2][2]] = true;
                }
                current_buffer_ids.erase(exchange[i/2][1]);
                current_buffer_ids[exchange[i/2][2]] = exchange[i/2][0];
            }
            // cout << "End exchange" << endl;
        }
        for(auto it = current_buffer_ids.begin(); it != current_buffer_ids.end(); it++){
            writeSingleBuffered_our_merge<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 3, (it->first)*2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, (it->second)*buffer_block_size, partition_block_size, buffer_block_size);
            // writeSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, ((it->first)*2+1)*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, (it->second+3)*buffer_block_size);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end - start;
        // cout << node_embedding_buffer[number_of_emb][0] << "," << node_embedding_buffer[number_of_emb][50] << "," << node_grad_state_buffer[number_of_emb][0] << "," << node_grad_state_buffer[number_of_emb][50] << endl;
        cout << "Epoch time: " << elapsed2.count() << "ms" << std::endl;


        if((epc+1) % 1 == 0){
            evaluate(source, settings, d_qps, destination, totalChunks, ec, prp1, prp2, size_per_part, coef, num_parts, float_number_per_part, part_size, src_rel_embedding, dst_rel_embedding, rel_num);
            // readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 0, ec, times.get(), prp1, prp2, settings.queuePairs, 0);
            // readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 2*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 1*buffer_block_size);
            // readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 4*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 2*buffer_block_size);
            // readSingleBuffered_our<<<settings.queuePairs, settings.numThreads/settings.queuePairs, 0, stream>>>(d_qps, source->ioaddrs[0], source->vaddr, destination.get(), totalChunks / 6, 6*partition_block_size, ec, times.get(), prp1, prp2, settings.queuePairs, 3*buffer_block_size);
            // cudaDeviceSynchronize();
            // torch::Tensor node_embedding = torch::from_blob((float*)source->vaddr, {4 * float_number_per_part}, opts);
            // node_embedding = node_embedding.view({4 * number_of_emb, embedding_dim});
            // evaluate_local(true,false,node_embedding,src_rel_embedding,dst_rel_embedding);
        }
    }
    

    //*****************************************************************
    double elapsed = 0;

    cudaStreamDestroy(stream);
    cudaStreamDestroy(stream2);

    return elapsed;
}

static void outputFile(BufferPtr data, size_t size, const char* filename)
{
    auto buffer = createBuffer(size);

    cudaError_t err = cudaMemcpy(buffer.get(), data.get(), size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to copy data from destination: ") + cudaGetErrorString(err));
    }
    // printf("%lx\n", ((uint64_t*)buffer.get())[0]);
    FILE* fp = fopen(filename, "w");
    fwrite(buffer.get(), 1, size, fp);
    fclose(fp);
}

static int useBlockDevice(const Settings& settings, const cudaDeviceProp& properties)
{
    int fd = open(settings.blockDevicePath, O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open block device: %s\n", strerror(errno));
        return 1;
    }

    const size_t pageSize = sysconf(_SC_PAGESIZE);
    const size_t blockSize = 512;
    const size_t totalChunks = settings.numChunks * settings.numThreads;
    const size_t totalPages = totalChunks * settings.numPages;

    fprintf(stderr, "CUDA device           : %u %s (%s)\n", settings.cudaDevice, properties.name, settings.getDeviceBDF().c_str());
#ifdef __DIS_CLUSTER__
    fprintf(stderr, "CUDA device fdid      : %lx\n", settings.cudaDeviceId);
#endif
    fprintf(stderr, "Controller page size  : %zu B\n", pageSize);
    fprintf(stderr, "Assumed block size    : %zu B\n", blockSize);
    fprintf(stderr, "Number of threads     : %zu\n", settings.numThreads);
    fprintf(stderr, "Chunks per thread     : %zu\n", settings.numChunks);
    fprintf(stderr, "Pages per chunk       : %zu\n", settings.numPages);
    fprintf(stderr, "Total number of pages : %zu\n", totalPages);
    fprintf(stderr, "Double buffering      : %s\n", settings.doubleBuffered ? "yes" : "no");

    void* ptr = mmap(nullptr, totalPages * pageSize, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, settings.startBlock * blockSize);
    if (ptr == nullptr)
    {
        close(fd);
        fprintf(stderr, "Failed to memory map block device: %s\n", strerror(errno));
        return 1;
    }

    try
    {
        auto outputBuffer = createBuffer(totalPages * pageSize);

        double usecs = launchMoveKernelLoop(ptr, outputBuffer, pageSize, settings);

        fprintf(stderr, "Event time elapsed    : %.3f µs\n", usecs);
        fprintf(stderr, "Estimated bandwidth   : %.3f MiB/s\n", (totalPages * pageSize) / usecs);

        if (settings.output != nullptr)
        {
            outputFile(outputBuffer, totalPages * pageSize, settings.output);
        }
    }
    catch (const cudaError_t err)
    {
        munmap(ptr, totalPages * pageSize);
        close(fd);
        fprintf(stderr, "Unexpected CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    catch (const error& e)
    {
        munmap(ptr, totalPages * pageSize);
        close(fd);
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    munmap(ptr, totalPages * pageSize);
    close(fd);
    return 0;
}



int main(int argc, char** argv)
{
    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    if (settings.blockDevicePath != nullptr)
    {
        return useBlockDevice(settings, properties);
    }

    try
    {

        Controller ctrl(settings.controllerPath, settings.nvmNamespace);
        ctrl.reserveQueues(1);

        // const size_t pageSize = ctrl.info.page_size;
        const size_t pageSize = settings.pageSize;
        const size_t blockSize = ctrl.ns.lba_data_size;
        const size_t chunkSize = pageSize * settings.numPages;
        const size_t totalChunks = settings.numChunks * settings.numThreads;
        const size_t totalPages = totalChunks * settings.numPages;
        const size_t totalBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, totalPages);

        if (chunkSize > ctrl.info.max_data_size)
        {
            throw error("Chunk size can not be larger than controller data size");
        }
        else if (totalBlocks > ctrl.ns.size)
        {
            throw error("Requesting read size larger than disk size");
        }

        fprintf(stderr, "CUDA device           : %u %s (%s)\n", settings.cudaDevice, properties.name, settings.getDeviceBDF().c_str());
        fprintf(stderr, "Controller page size  : %zu B\n", pageSize);
        fprintf(stderr, "Namespace block size  : %zu B\n", blockSize);
        fprintf(stderr, "Number of threads     : %zu\n", settings.numThreads);
        fprintf(stderr, "Chunks per thread     : %zu\n", settings.numChunks);
        fprintf(stderr, "Pages per chunk       : %zu\n", settings.numPages);
        fprintf(stderr, "Total number of pages : %zu\n", totalPages);
        fprintf(stderr, "Total number of blocks: %zu\n", totalBlocks);
        fprintf(stderr, "Double buffering      : %s\n", settings.doubleBuffered ? "yes" : "no");

        auto outputBuffer = createBuffer(settings.pageSize * totalPages, settings.cudaDevice);

        cudaError_t err = cudaHostRegister((void*) ctrl.ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterIoMemory);
        if (err != cudaSuccess)
        {
            throw error(string("Unexpected error while mapping IO memory (cudaHostRegister): ") + cudaGetErrorString(err));
        }

        try
        {
            double usecs = launchNvmKernel_our(ctrl, outputBuffer, settings, properties);

            // fprintf(stderr, "Event time elapsed    : %.3f µs\n", usecs);
            // fprintf(stderr, "Estimated bandwidth   : %.3f MiB/s\n", (totalPages * pageSize) / usecs);

            if (settings.output != nullptr)
            {
                // outputFile(outputBuffer, totalPages * pageSize, settings.output);
                outputFile(outputBuffer, (size_t)40000, settings.output);
            }
        }
        catch (const error& e)
        {
            cudaHostUnregister((void*) ctrl.ctrl->mm_ptr);
            throw e;
        }
        catch (const cudaError_t err)
        {
            cudaHostUnregister((void*) ctrl.ctrl->mm_ptr);
            throw error(string("Unexpected CUDA error: ") + cudaGetErrorString(err));
        }
    }
    catch (const error& e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    return 0;
}
