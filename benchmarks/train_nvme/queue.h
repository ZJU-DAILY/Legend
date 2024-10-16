#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__

#include <nvm_types.h>
#include <cstdint>
#include "buffer.h"
#include "settings.h"
#include "ctrl.h"


struct __align__(64) QueuePair
{
    
    uint32_t            pageSize;
    uint32_t            blockSize;
    uint32_t            nvmNamespace;
    uint32_t            pagesPerChunk;
    bool                doubleBuffered;
    void*               prpList;
    uint64_t            prpListIoAddr;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
    uint16_t            qp_id;
    DmaPtr              sq_mem;
    DmaPtr              cq_mem;
    // DmaPtr              unused1;
    // DmaPtr              unused2;
};


__host__ DmaPtr prepareQueuePair(QueuePair& qp, const Controller& ctrl, const Settings& settings);
__host__ void prepareQueuePair_our(QueuePair& qp, const Controller& ctrl, const Settings& settings, const uint16_t qp_id);

#endif
