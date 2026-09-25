#ifndef BWDWORKLIST_H
#define BWDWORKLIST_H
#include <vector>
#include "structs.cuh"
#include <cuda_runtime.h>
typedef unsigned int Ticket;
typedef unsigned long long int HT;

typedef union {
	struct {int numWaiting; int numEnqueued;};
	unsigned long long int combined;
} Counter;

struct WorkList {
    unsigned int size;
    unsigned int threshold;

    Partition*            partition;        
    volatile int*          path_canon_code;  
    volatile unsigned int* eqlev_first;     
    TargetCell*            target_cell;      
    volatile int*          tc_elems_pool;    
    volatile Ticket*       tickets;          
    HT*                    head_tail;        
    int*                   count;            
    Counter*               counter;        
    volatile unsigned int* pool_element_vec;   
    Cell*           pool_lcs;          
    volatile unsigned int* pool_curr_seq;     
};


__device__ bool checkThreshold(WorkList workList){

    __shared__ int numEnqueued;
    if (threadIdx.x == 0){
        numEnqueued = atomicOr(&workList.counter->numEnqueued,0);
    }
    __syncthreads();

    if ( numEnqueued >= workList.threshold){
        return false;
    } else {
        return true;
    }

}


#if __CUDA_ARCH__ < 700
	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	__device__ __forceinline__ void sleepBWD(unsigned int exp)
	{
		__threadfence();
	}
#else
	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	__device__ __forceinline__ void sleepBWD(unsigned int exp)
	{
		__nanosleep(1<<exp);
	}
#endif

__device__ unsigned int* head(HT* head_tail){
	return reinterpret_cast<unsigned int*>(head_tail) + 1;
}

__device__ unsigned int* tail(HT* head_tail) {
	return reinterpret_cast<unsigned int*>(head_tail);
}

__device__ void waitForTicket(const unsigned int P, const Ticket number, WorkList workList) {
	while (workList.tickets[P] != number)
	{
		backoff();
	}
}

__device__ bool ensureDequeue(WorkList workList){
	int Num = atomicOr(workList.count,0);
	bool ensurance = false;
	while (!ensurance && Num > 0) {
		if (atomicSub(workList.count, 1) > 0) {
			ensurance = true;
		}
		else {
			Num = atomicAdd(workList.count, 1) + 1;
		}
	}

	return ensurance;
}

__device__ bool ensureEnqueue(WorkList workList){
	int Num = atomicOr(workList.count,0);
	bool ensurance = false;
	while (!ensurance && Num < (int)workList.size)
	{
		if (atomicAdd(workList.count, 1) < (int)workList.size)
		{
			ensurance = true;
		}
		else 
		{
			Num = atomicSub(workList.count, 1) - 1;
		}
	}
	
	return ensurance;
}


__device__ void readData( Partition& out_part, TargetCell& out_tc, int* out_canon_code, unsigned int* out_eqlev_first, WorkList workList, unsigned int vertexNum){	

	__shared__ unsigned int P;
	unsigned int Pos;
	if (threadIdx.x==0){
        Pos = atomicAdd(head(const_cast<HT*>(workList.head_tail)), 1);
        P = Pos % workList.size;
        waitForTicket(P, 2 * (Pos / workList.size) + 1,workList);
	}
	__syncthreads();

	
	for (unsigned i = threadIdx.x; i < vertexNum; i += blockDim.x) {
        out_part.element_vec[i] = workList.partition[P].element_vec[i];
    }
    for (unsigned i = threadIdx.x; i < workList.partition[P].lcs_size; i += blockDim.x) {
        out_part.lcs[i] = workList.partition[P].lcs[i];
    }
    for (unsigned i = threadIdx.x; i < workList.partition[P].current_vertex_sequence_size; i += blockDim.x) {
        out_part.current_vertex_sequence[i] =
            workList.partition[P].current_vertex_sequence[i];
    }

    if (threadIdx.x == 0) {
        out_part.lcs_size = workList.partition[P].lcs_size;
        out_part.level    = workList.partition[P].level;
		out_part.current_vertex_sequence_size    = workList.partition[P].current_vertex_sequence_size;
    }
    __syncthreads();


    for (unsigned i = threadIdx.x; i < vertexNum; i += blockDim.x) {
        out_tc.elems[i] = workList.tc_elems_pool[(size_t)P * vertexNum + i];
    }
    if (threadIdx.x == 0) {
        TargetCell& src_tc = workList.target_cell[P];
        out_tc.first                 = src_tc.first;
        out_tc.length                = src_tc.length;
        out_tc.counter               = src_tc.counter;
        out_tc.last_num_aut_at_level = src_tc.last_num_aut_at_level;
    }
    __syncthreads();


    for (unsigned i = threadIdx.x; i < (vertexNum + 2); i += blockDim.x) {
        out_canon_code[i] =
            workList.path_canon_code[(size_t)P * (vertexNum + 2) + i];
    }
    if (threadIdx.x == 0) {
        *out_eqlev_first = workList.eqlev_first[P];
    }
    
	__syncthreads();
	if (threadIdx.x==0){
	    workList.tickets[P] = 2 * ((Pos + workList.size) / workList.size);
	}
}

__device__ void putData( Partition& in_part,  TargetCell& in_tc, const int* in_canon_code, unsigned int* in_eqlev_first,WorkList workList, unsigned int vertexNum){
	__shared__ unsigned int P;
	unsigned int Pos;
	unsigned int B;
	if (threadIdx.x==0){
        Pos = atomicAdd(tail(const_cast<HT*>(workList.head_tail)), 1);
        P = Pos % workList.size;
        B = 2 * (Pos /workList.size);
        waitForTicket(P, B, workList);
	}

	__syncthreads();

	for (unsigned i = threadIdx.x; i < vertexNum; i += blockDim.x) {
        workList.partition[P].element_vec[i] = in_part.element_vec[i];
    }
   
    for (unsigned i = threadIdx.x; i < in_part.lcs_size; i += blockDim.x) {
        workList.partition[P].lcs[i] = in_part.lcs[i];
    }
   
    for (unsigned i = threadIdx.x; i < in_part.current_vertex_sequence_size; i += blockDim.x) {
        workList.partition[P].current_vertex_sequence[i] =
            in_part.current_vertex_sequence[i];
    }

    if (threadIdx.x == 0) {
        workList.partition[P].lcs_size                     = in_part.lcs_size;
        workList.partition[P].current_vertex_sequence_size = in_part.current_vertex_sequence_size;
        workList.partition[P].level                        = in_part.level;
    }
    __syncthreads();

    //  copy TargetCell header + elems[] into queue slot 
    for (unsigned i = threadIdx.x; i < vertexNum; i += blockDim.x) {
        workList.tc_elems_pool[(size_t)P * vertexNum + i] = in_tc.elems[i];
    }
    if (threadIdx.x == 0) {
        TargetCell& dst_tc = workList.target_cell[P];
        dst_tc.first                 = in_tc.first;
        dst_tc.length                = in_tc.length;
        dst_tc.counter               = in_tc.counter;
        dst_tc.last_num_aut_at_level = in_tc.last_num_aut_at_level;
    }
    __syncthreads();

    //  copy canon code slice + eqlev_first into slot
    for (unsigned i = threadIdx.x; i < (vertexNum + 2); i += blockDim.x) {
        workList.path_canon_code[(size_t)P * (vertexNum + 2) + i] = in_canon_code[i];
    }
    if (threadIdx.x == 0) {
        workList.eqlev_first[P] = (*in_eqlev_first);
    }

	__threadfence();
	__syncthreads();
	if (threadIdx.x==0){
		workList.tickets[P] = B + 1;
		atomicAdd(&workList.counter->numEnqueued,1);
	}
}
__device__ inline bool enqueue(
    Partition& in_part,
    TargetCell& in_tc,
    int*        in_canon_code,    
    unsigned int*      in_eqlev_first,
    WorkList          workList,
    unsigned int      vertexNum)
{
    __shared__ bool writeData;
    if (threadIdx.x == 0) {
        writeData = ensureEnqueue(workList);
    }
    __syncthreads();

    if (writeData) {
        putData(in_part, in_tc, in_canon_code, in_eqlev_first, workList, vertexNum);
    }
    return writeData;
}
__device__ inline bool dequeue(
    Partition&   out_part,
    TargetCell&   out_tc,
    int*          out_canon_code,   
    unsigned int* out_eqlev_first, 
    WorkList      workList,
    unsigned int  vertexNum)
{
    unsigned int expoBackOff = 0;

    __shared__ bool isWorkDone;
    if (threadIdx.x==0){
		isWorkDone = false;
		atomicAdd(&workList.counter->numWaiting,1);
	}
    __syncthreads();

    __shared__ bool hasData;
    while (!isWorkDone) {
        if (threadIdx.x == 0) {
            hasData = ensureDequeue(workList);
        }
        __syncthreads();

        if (hasData) {
            // COPY payload from queue slot into caller buffers
            readData(out_part, out_tc, out_canon_code, out_eqlev_first, workList, vertexNum);

            if (threadIdx.x==0){
				Counter tempCounter;
				tempCounter.numWaiting = -1;
				tempCounter.numEnqueued = -2;
				atomicAdd(&workList.counter->combined,tempCounter.combined);
			}
            return true;
        }

        if (threadIdx.x==0){
			Counter tempCounter;
			tempCounter.combined = atomicOr(&workList.counter->combined,0);
			if (tempCounter.numWaiting==gridDim.x && tempCounter.numEnqueued==0){
				isWorkDone=true;
			}
		}

        __syncthreads();
        sleepBWD(expoBackOff++);
    }
    return false;
}

WorkList allocateWorkList(unsigned int vertexNum) {
    WorkList wl{};
    wl.size      = 1 << 14;
    wl.threshold = wl.size;

    const size_t W = wl.size;
    const size_t V = vertexNum;

    CUDA_CHECK(cudaMalloc((void**)&wl.partition, sizeof(Partition) * W));


    unsigned int* pool_element_vec_raw = nullptr;
    Cell*  pool_lcs_raw         = nullptr;
    unsigned int* pool_curr_seq_raw    = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&pool_element_vec_raw, W * V * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&pool_lcs_raw,        W * (V + 1) * sizeof(Cell)));
    CUDA_CHECK(cudaMalloc((void**)&pool_curr_seq_raw,   W * V * sizeof(unsigned int)));

    wl.pool_element_vec = (volatile unsigned int*)pool_element_vec_raw;
    wl.pool_lcs         = pool_lcs_raw; 
    wl.pool_curr_seq    = (volatile unsigned int*)pool_curr_seq_raw;


    CUDA_CHECK(cudaMalloc((void**)&wl.target_cell,  W * sizeof(TargetCell)));
    int* tc_elems_pool_raw = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&tc_elems_pool_raw, W * V * sizeof(int)));
    wl.tc_elems_pool = (volatile int*)tc_elems_pool_raw;
    CUDA_CHECK(cudaMemset((void*)tc_elems_pool_raw, 0, W * V * sizeof(int))); 


    int* path_canon_code_raw = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&path_canon_code_raw, (V + 2) * sizeof(int) * W));
    wl.path_canon_code = (volatile int*)path_canon_code_raw;

    unsigned int* eqlev_first_raw = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&eqlev_first_raw, sizeof(unsigned int) * W));
    wl.eqlev_first = (volatile unsigned int*)eqlev_first_raw;

    Ticket* tickets_raw = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&tickets_raw, sizeof(Ticket) * W));
    wl.tickets = (volatile Ticket*)tickets_raw;

    CUDA_CHECK(cudaMalloc((void**)&wl.head_tail, sizeof(HT)));
    CUDA_CHECK(cudaMalloc((void**)&wl.count,     sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&wl.counter,   sizeof(Counter)));

    // host-side wiring of structs pointing into pools
    std::vector<Partition> h_parts(W);
    for (size_t i = 0; i < W; ++i) {
        Partition p{};
        p.element_vec             = wl.pool_element_vec + i * V;  
        p.lcs                     = wl.pool_lcs         + i * (V + 1);
        p.lcs_size                = 0;
        p.level                   = 0;
        p.current_vertex_sequence = wl.pool_curr_seq    + i * V; 
        p.current_vertex_sequence_size = 0;
        h_parts[i] = p;
    }
    CUDA_CHECK(cudaMemcpy(wl.partition, h_parts.data(),
                          sizeof(Partition) * W, cudaMemcpyHostToDevice));

    // TargetCell wiring
    std::vector<TargetCell> h_tcs(W);
    for (size_t w = 0; w < W; ++w) {
        TargetCell tc{};
        tc.elems                 = (int*) (tc_elems_pool_raw + w * V);
        tc.first                 = 0;
        tc.length                = 0;
        tc.counter               = 0;
        tc.last_num_aut_at_level = -1;
        h_tcs[w] = tc;
    }
    CUDA_CHECK(cudaMemcpy(wl.target_cell, h_tcs.data(),
                          sizeof(TargetCell) * W, cudaMemcpyHostToDevice));

    // small inits 
    HT head_tail = 0x0ULL;
    Counter counter{}; counter.combined = 0;
    CUDA_CHECK(cudaMemcpy(wl.head_tail, &head_tail, sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset((void*)tickets_raw, 0, sizeof(Ticket) * W));
    CUDA_CHECK(cudaMemset(wl.count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(wl.counter, &counter, sizeof(Counter), cudaMemcpyHostToDevice));

    return wl;
}



void cudaFreeWorkList(WorkList& wl) {
    // partition subfield pools
    cudaFree((void*)(const void*)wl.pool_element_vec);
    cudaFree((void*)wl.pool_lcs);
    cudaFree((void*)(const void*)wl.pool_curr_seq);

    // top-level arrays
    cudaFree(wl.partition);
    cudaFree((void*)(const void*)wl.path_canon_code);
    cudaFree((void*)(const void*)wl.eqlev_first);

    // TargetCell + pool
    cudaFree((void*)(const void*)wl.tc_elems_pool);
    cudaFree(wl.target_cell);

    // others
    cudaFree((void*)(const void*)wl.tickets);
    cudaFree(wl.head_tail);
    cudaFree(wl.count);
    cudaFree(wl.counter);
}

#endif