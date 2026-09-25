#include "nauty_parallel.h"
#include <cstdio>
#include <string>
#include "partition_refinement.cuh"
#include "helper_functions.cuh"
#include "allocate.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include "BWDWorkList.cuh"
#include <chrono>


__global__ void nauty_parallel_kernel(
    Partition* partitions,
    unsigned int numVertices,
    CSRGraph* d_graph,
    Cell* cell_buffers,
    unsigned int* aux_arrays,
    volatile int* d_found_automorphisms,
    volatile int* d_num_aut,
    Leaf* d_first_leaf,
    CanonicalLabeling* canonical_labeling,
    int* max_aut,
    volatile bool* d_hash_val,
    int* d_node_counter,
    unsigned int* d_eqlevfirst,
    TargetCell*   d_target_cells,   
    int*          d_tc_level_reached,
    int*          d_canon_code_per_block,   
    WorkList workList,
    int* top_tc_level,
    unsigned long long* NODES_PER_SM,
    volatile int*  last_num_aut,
    Counters* counters,
    int* hash_lock
) {
    unsigned int blockId = blockIdx.x;
    Counters blockCounters;
    initializeCounters(&blockCounters);

    __shared__ unsigned int sm_id;
    if (threadIdx.x==0){
        sm_id=get_smid();
    }

    Partition& part = partitions[blockId];
    CSRGraph& graph = d_graph[0];  
    int*          my_canon_code = d_canon_code_per_block + blockId * (numVertices + 2);
    unsigned int* p_eqlev_first = d_eqlevfirst + blockId;
    unsigned int*  current_orbits          = aux_arrays + blockId * 4 * numVertices;
    unsigned int*  dsu_array          = current_orbits + numVertices;
    unsigned int*  tmp_perm           = dsu_array + numVertices;
    Cell* subsequence = cell_buffers + blockId * (numVertices + 1) * 2;
    Cell* buffer = subsequence + (numVertices + 1);
    TargetCell* tc_stack = d_target_cells + blockId * (numVertices + 1);
    volatile int*        tc_stack_top = d_tc_level_reached + blockId;

    int code, target_cell_index;
    startTime(PREPARE_ROOT,&blockCounters);
    code = partition_refinement_p(part, graph, subsequence, 0, buffer , aux_arrays, counters);
    
    if(threadIdx.x == 0){
        my_canon_code[part.level - 1] = code;
        *p_eqlev_first = part.level;
    }
    
    
    if (blockId == 0 && threadIdx.x == 0) {
        d_first_leaf->invar_path_value[part.level-1] = code;
        *d_node_counter = 1;
        canonical_labeling->invar_path_value[part.level - 1]= code;
        //printf("[blk 0] firstcode[%u] = %d; node_counter=1\n", part.level-1, code);
    }

    if (is_discrete_p(part) == 1) {
        if (blockId == 0) {
            first_terminal_p(part,*d_graph,numVertices,d_first_leaf,canonical_labeling,tmp_perm,d_hash_val,
                 p_eqlev_first );
        }
        //if (threadIdx.x == 0) printf("[blk %u] EXIT: discrete\n", blockId);
        endTime(PREPARE_ROOT,&blockCounters);
        return;
    }

    
    target_cell_index = push_stack(part, tc_stack, tc_stack_top, numVertices);

    __shared__ unsigned first, len;
    if (threadIdx.x == 0) {
        part.pushNextIter = 0;
        const Cell& hdr = part.lcs[target_cell_index];
        first = hdr.first;                 // start in element_vec
        len   = hdr.length;                // number of elements in this target cell
        last_num_aut[blockIdx.x] = 0;
    }
    __syncthreads();

    
    TargetCell& tc = tc_stack[(*tc_stack_top) - 1];

    
    for (unsigned k = threadIdx.x; k < len; k += blockDim.x) {
        int v = part.element_vec[first + k];   
        tc.elems[v] = -1;    
    }
    __syncthreads();

    // 2) This block marks its lane (k = blockIdx.x, k += gridDim.x) to 2
    //    Block 0 will naturally mark the first branch (k=0) as 2.
    for (unsigned k = blockIdx.x; k < len; k += gridDim.x) {
        int v = part.element_vec[first + k];
        tc.elems[v] = v;                       // owned by this block
    }
    
    if(threadIdx.x == 0){
        top_tc_level[blockIdx.x] = 1;
    }
    

    endTime(PREPARE_ROOT,&blockCounters);


    startTime(WAIT_FIRST_LEAF,&blockCounters);

    if (blockIdx.x != 0) {
        if (threadIdx.x == 0) {
            while (d_first_leaf->discovered== 0) {
                __nanosleep(100);              
            }
        }
     __syncthreads();
    }

    endTime(WAIT_FIRST_LEAF,&blockCounters);

    __shared__ int first_leaf_flag;

   
    int vertex;

    
    __shared__ int num_aut_current;
    
    while (true) {
        
        __syncthreads();
        if (threadIdx.x == 0) {
            first_leaf_flag = d_first_leaf->discovered;
        }
        
        if((*tc_stack_top) != 0){
            //local stack is not empty.
            if (part.pushNextIter == 1) {

                // if (threadIdx.x == 0 /* && blockIdx.x == 0 */) {
                //     printf("[blk %u] L%u: pushNextIter=1, about to push; top(before)=%d\n",
                //         (unsigned)blockIdx.x, (unsigned)part.level, *tc_stack_top);
                // }
                
                startTime(PUSH_STACK,&blockCounters);
                push_stack(part, tc_stack, tc_stack_top, numVertices);
                
                endTime(PUSH_STACK,&blockCounters);
                //__syncthreads();

                startTime(POP_STACK,&blockCounters);
                vertex = pop_stack(tc_stack, tc_stack_top, numVertices); 
                endTime(POP_STACK,&blockCounters);

                __syncthreads();
               
                
                bool enqueueSuccess;
                startTime(ENQUEUE,&blockCounters);
                if(checkThreshold(workList)){
                    //startTime(ENQUEUE,&blockCounters);
                    enqueueSuccess = enqueue(part, tc_stack[(*tc_stack_top) - 1], my_canon_code, p_eqlev_first, workList, numVertices);
                    
                } else  {
                    enqueueSuccess = false;
                    // if(threadIdx.x == 0){
                    //     printf("blk %u wanted to enqueue but worklist full, enqueue failed. \n", blockIdx.x);
                    // }
                }

                __syncthreads();
                

                if(enqueueSuccess){
                    //pop from my stack.
                   
                    for (unsigned i = threadIdx.x ; i < numVertices ; i += blockDim.x) {
                        if ( tc_stack[(*tc_stack_top) - 1].elems[i] != -1) {
                             tc_stack[(*tc_stack_top) - 1].elems[i] = -1;
                        }
                    }
                }
                
                endTime(ENQUEUE,&blockCounters);
                
            }
            else{
                
                
                   
                startTime(PRUNE_STACK_ENTRY,&blockCounters);
                prune_stack_entry(part,tc_stack, 
                    tc_stack_top, numVertices, 
                    d_found_automorphisms,
                    d_num_aut,          
                    *max_aut, 
                    current_orbits,
                    dsu_array,
                    &blockCounters);

                endTime(PRUNE_STACK_ENTRY,&blockCounters);
                
                
                startTime(POP_STACK,&blockCounters);
                vertex = pop_stack(tc_stack, tc_stack_top, numVertices); 

                endTime(POP_STACK,&blockCounters);

                if(vertex == -1){ // branched/pruned all vertices at this level.
                    startTime(BACKTRACK,&blockCounters);
                    backtrack_to_p(part, part.level - 1, numVertices, buffer,
                                    p_eqlev_first, tc_stack_top,top_tc_level);
                    endTime(BACKTRACK,&blockCounters);
                    continue;
                }
                
            }
        }
        else{
            //Check global worklist
            
            startTime(DEQUEUE,&blockCounters);
            if(!dequeue(part, tc_stack[(*tc_stack_top)], my_canon_code, p_eqlev_first, workList, numVertices)) {   
                endTime(DEQUEUE,&blockCounters);
                break;
            }

            endTime(DEQUEUE,&blockCounters);
            __syncthreads();
            
            

            if(threadIdx.x == 0){
                top_tc_level[blockIdx.x] = part.level;
                last_num_aut[blockIdx.x] = 0;
                (*tc_stack_top)++;
            }

           
            __syncthreads();

                   
            startTime(PRUNE_STACK_ENTRY,&blockCounters);
            prune_stack_entry(part,tc_stack, 
                tc_stack_top, numVertices, 
                d_found_automorphisms,
                d_num_aut,          
                *max_aut, 
                current_orbits,
                dsu_array,
                &blockCounters);

            endTime(PRUNE_STACK_ENTRY,&blockCounters);
            //__syncthreads();
            
            startTime(POP_STACK,&blockCounters);

            vertex = pop_stack(tc_stack, tc_stack_top, numVertices); 

            endTime(POP_STACK,&blockCounters);

            if(vertex == -1){
                startTime(BACKTRACK,&blockCounters);
                backtrack_to_p(part, part.level - 1, numVertices, buffer,
                                p_eqlev_first, tc_stack_top, top_tc_level);
                endTime(BACKTRACK,&blockCounters);
                continue;
            }


            //enqueue again

            bool enqueueSuccess;
            startTime(ENQUEUE,&blockCounters);
            if(checkThreshold(workList)){
                //startTime(ENQUEUE,&blockCounters);
                enqueueSuccess = enqueue(part, tc_stack[(*tc_stack_top) - 1], my_canon_code, p_eqlev_first, workList, numVertices);
                
            } else  {
                enqueueSuccess = false;
                // if(threadIdx.x == 0){
                //     printf("blk %u wanted to enqueue but worklist full, enqueue failed. \n", blockIdx.x);
                // }
            }

            __syncthreads();
            

            if(enqueueSuccess){
                //pop from my stack.
                
                for (unsigned i = threadIdx.x ; i < numVertices ; i += blockDim.x) {
                    if ( tc_stack[(*tc_stack_top) - 1].elems[i] != -1) {
                            tc_stack[(*tc_stack_top) - 1].elems[i] = -1;
                    }
                }
            }

            endTime(ENQUEUE,&blockCounters);

        }
        
        //__syncthreads();
        
        if(threadIdx.x == 0){
           num_aut_current = *d_num_aut;
        }
        __syncthreads();
        
        if(num_aut_current > last_num_aut[blockIdx.x]){
            
            startTime(VERIFY_SEQUENCE,&blockCounters);
            
            int level = verify_vertex_sequence_orbit_reps(current_orbits, d_found_automorphisms,num_aut_current,
            part.current_vertex_sequence, part.current_vertex_sequence_size,numVertices,max_aut,0,dsu_array);

            if(threadIdx.x == 0){
                last_num_aut[blockIdx.x] = num_aut_current;
            }

            // if(threadIdx.x == 0){
            //     printf("Blk %u: current level is %u, backtracking to %u, root part level is %u\n",
            //     blockIdx.x, part.level, level,top_tc_level[blockIdx.x] );
            // }
            endTime(VERIFY_SEQUENCE,&blockCounters);
            

            if (level != (part.current_vertex_sequence_size + 2)) {

                
                startTime(BACKTRACK,&blockCounters);
                backtrack_to_p(part, level, numVertices, buffer,
                                        p_eqlev_first, tc_stack_top, top_tc_level);
                endTime(BACKTRACK,&blockCounters);
                continue; // skip branching
            }
            
        }

        //print_partition(part);
        if (threadIdx.x==0){
            atomicAdd(&NODES_PER_SM[sm_id],1);
            //printf("Blk %u Branching on vertex %u at level %u\n", blockIdx.x, vertex, part.level);
            
        }

        __syncthreads();
        
        startTime(SPLIT_BY_AND_REFINE,&blockCounters);

        code = split_by_and_refine_p(part, graph, (unsigned)vertex,
            subsequence,
            buffer,
            aux_arrays,
            &blockCounters
        );
        endTime(SPLIT_BY_AND_REFINE,&blockCounters);

        //__syncthreads();
        
        
        if(threadIdx.x == 0){
            atomicAdd(d_node_counter, 1);
            part.current_vertex_sequence[part.current_vertex_sequence_size++] = (unsigned)vertex;
            part.pushNextIter = 1;

            if(blockIdx.x == 0 && first_leaf_flag == 0){
                d_first_leaf->invar_path_value[part.level - 1] = code;
                
                *p_eqlev_first = part.level;
                
            }
            else {
                
                if ((*p_eqlev_first) == part.level - 1) {
                    if (code == d_first_leaf->invar_path_value[part.level - 1]) {
                        *p_eqlev_first = part.level;
                    }
                }

            }
            
            my_canon_code[part.level - 1] = code;
        }

        __syncthreads();

        if(blockIdx.x == 0 && first_leaf_flag == 0 && is_discrete_p(part)){

            first_terminal_p(part, graph, numVertices,
                    d_first_leaf, canonical_labeling,
                    tmp_perm, d_hash_val,
                    p_eqlev_first);

            if (threadIdx.x == 0) {
                d_first_leaf->discovered = 1;
                // printf("[blk %u] L%u: published first leaf (canon_level=%d, eqlev_first=%u)\n",
                //        blockIdx.x, part.level, *canon_code_global_level, *p_eqlev_first);
            }
            __threadfence();
            __syncthreads();
            backtrack_to_p(part, part.level - 1, numVertices, buffer, p_eqlev_first, tc_stack_top,top_tc_level);
            continue;
        }


       
        //__syncthreads();
        if(blockIdx.x == 0 && first_leaf_flag == 0){continue;}
        
        //startTime(CLASSIFY_NODE,&blockCounters);
        
        classify_node(
            part, graph, numVertices,
            p_eqlev_first,
            d_found_automorphisms, d_num_aut, max_aut, 
            tmp_perm, d_hash_val,
            d_first_leaf, canonical_labeling, my_canon_code,
            buffer, tc_stack_top, top_tc_level,
            dsu_array, last_num_aut,
            current_orbits, hash_lock,&blockCounters
        );

        //endTime(CLASSIFY_NODE,&blockCounters);
        
        
    }
    
    
    __syncthreads();

    #if USE_COUNTERS
        if(threadIdx.x == 0) {
            counters[blockIdx.x] = blockCounters;
        }
    #endif
    
}

int choose_num_blocks_by_memory(
    int B_occ,
    unsigned int V,
    unsigned int E,          
    unsigned int max_aut)
{
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    // ----------------------------
    // 1) Fixed memory (independent of blocks)
    // ----------------------------

    // CSR graph (allocate_csr_graph_on_device):
    // dst[E] (uint), srcPtr[V+1] (uint), degree[V] (int), color[V] (int), plus CSRGraph struct itself (tiny)
    size_t bytes_csr =
        (size_t)E * sizeof(unsigned int) +
        (size_t)(V + 1) * sizeof(unsigned int) +
        (size_t)V * sizeof(int) +
        (size_t)V * sizeof(int) +
        sizeof(CSRGraph);  // small, but included

    size_t bytes_aut =
        (size_t)V * (size_t)max_aut * sizeof(int);

    size_t bytes_hash =
        (size_t)V * (size_t)V * sizeof(bool);

    // WorkList memory
    const size_t W = 1u << 14;
    size_t bytes_worklist =
        sizeof(Partition) * W +
        W * (size_t)V * sizeof(unsigned int) +          // pool_element_vec
        W * (size_t)(V + 1) * sizeof(Cell) +            // pool_lcs
        W * (size_t)V * sizeof(unsigned int) +          // pool_curr_seq
        W * sizeof(TargetCell) +
        W * (size_t)V * sizeof(int) +                   // tc_elems_pool
        W * (size_t)(V + 2) * sizeof(int) +             // path_canon_code
        W * sizeof(unsigned int) +                      // eqlev_first
        W * sizeof(Ticket) +
        sizeof(HT) +
        sizeof(int) +
        sizeof(Counter);

    size_t fixedBytes =
        bytes_csr +
        bytes_aut +
        bytes_hash +
        bytes_worklist;

    // Safety margin
    size_t safety = freeMem / 10;

    if (freeMem <= fixedBytes + safety) {
        // not enough even for fixed allocations
        return 0;
    }

    size_t usableMem = freeMem - fixedBytes - safety;

    // ----------------------------
    // 2) Per-block memory (scales with #blocks)
    // ----------------------------
    size_t levels_per_block = (size_t)V + 1;

    // d_tc_elems_pool per block: (V+1)*V ints
    size_t perBlock_tc_pool =
        levels_per_block * (size_t)V * sizeof(int);

    // Per-block smaller allocations
    size_t perBlock_small =
        (size_t)2 * (V + 1) * sizeof(Cell) +        // cell_buffers: 2*(V+1) Cells
        (size_t)4 * V * sizeof(unsigned int) +      // aux_arrays: 7*V uints
        (size_t)(V + 2) * sizeof(int) +             // d_canon_code_per_block slice
        sizeof(unsigned int) +                      // d_eqlevfirst per block
        sizeof(int) +                               // last_num_aut per block
        sizeof(Counters);                           // counters per block

    size_t perBlockBytes = perBlock_tc_pool + perBlock_small;
    if (perBlockBytes == 0) return 0;

    int B_mem = (int)(usableMem / perBlockBytes);
    int B_final = (B_mem < B_occ) ? B_mem : B_occ;
    if (B_final < 1) return 0;

    // auto pct = [&](size_t bytes) -> double {
    //     return freeMem ? (100.0 * (double)bytes / (double)freeMem) : 0.0;
    // };

    // size_t bytes_blockdata = (size_t)B_final * perBlockBytes;
    // size_t bytes_accounted = fixedBytes + safety + bytes_blockdata;
    // size_t bytes_remaining = (freeMem > bytes_accounted) ? (freeMem - bytes_accounted) : 0;

    // printf("---- Memory estimation ----\n");
    // printf("freeMem:      %.2f GB (100%%)\n", freeMem / 1e9);

    // printf("CSR:          %.2f GB (%.2f%%)\n", bytes_csr / 1e9, pct(bytes_csr));
    // printf("AUT:          %.2f GB (%.2f%%)\n", bytes_aut / 1e9, pct(bytes_aut));
    // printf("WORKLIST:     %.2f GB (%.2f%%)\n", bytes_worklist / 1e9, pct(bytes_worklist));
    // printf("HASH (V*V):   %.2f GB (%.2f%%)\n", bytes_hash / 1e9, pct(bytes_hash));

    // printf("SAFETY (10%%): %.2f GB (%.2f%%)\n", safety / 1e9, pct(safety));

    // printf("perBlock:     %.2f MB\n", perBlockBytes / 1e6);
    // printf("BLOCK-DATA:   %.2f GB (%.2f%%)  [B_final=%d]\n",
    //        bytes_blockdata / 1e9, pct(bytes_blockdata), B_final);

    // printf("REMAINING:    %.2f GB (%.2f%%)\n", bytes_remaining / 1e9, pct(bytes_remaining));
    // printf("B_occ=%d  B_mem=%d  =>  B_final=%d\n", B_occ, B_mem, B_final);
    // printf("--------------------------------------------\n");

    return B_final;
}

static int choose_tpb_keep_fixed_blocks(
    int dev,
    int B_final,                 // fixed number of blocks
    size_t dynSmemBytes = 0,
    int start_tpb = 64,
    int max_tpb = 1042
) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    auto Bocc_for = [&](int tpb) -> int {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            nauty_parallel_kernel,
            tpb,
            dynSmemBytes
        ));
        int numSMs = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
        return blocks_per_sm * numSMs;
    };

    // clamp
    if (start_tpb < 32) start_tpb = 32;
    if (start_tpb > prop.maxThreadsPerBlock) start_tpb = prop.maxThreadsPerBlock;
    if (max_tpb   > prop.maxThreadsPerBlock) max_tpb   = prop.maxThreadsPerBlock;

    // baseline check at start_tpb
    int best_tpb  = start_tpb;
    int best_Bocc = Bocc_for(best_tpb);

    if (best_Bocc < B_final) {
        fprintf(stderr,
            "[ABORT] Fixed numBlocks=%d is NOT feasible with TPB=%d (B_occ=%d).\n",
            B_final, best_tpb, best_Bocc
        );
        return 0; // signal failure
    }

    // try powers of 2
    for (int tpb = best_tpb * 2; tpb <= max_tpb; tpb *= 2) {
        int Bocc = Bocc_for(tpb);

        // ONLY accept if occupancy still supports your fixed B_final blocks
        if (Bocc >= B_final) {
            best_tpb  = tpb;
            best_Bocc = Bocc;
        } else {
            break;
        }
    }

    // printf("[TPB] fixed blocks=%d, chosen TPB=%d (B_occ=%d)\n",
    //        B_final, best_tpb, best_Bocc);

    return best_tpb;
}
std::vector<unsigned char> Nauty_Parallel(CSRGraph& graph, int max_aut) {

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    unsigned int vertexNum   = graph.vertexNum;
    unsigned int max_levels  = vertexNum;
    const int levels_per_block = (int)vertexNum + 1;

    // ------------------------------------------------------------------
    // Device props + occupancy-based upper bound
    // ------------------------------------------------------------------
    size_t dynSmemBytes = 0;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int threads_per_block = 128;
    if (threads_per_block > prop.maxThreadsPerBlock)
        threads_per_block = prop.maxThreadsPerBlock;

    int num_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        nauty_parallel_kernel,
        threads_per_block,
        dynSmemBytes
    ));

    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));

    int B_occ = num_blocks_per_sm * numSMs;

    int total_num_blocks = choose_num_blocks_by_memory(B_occ, graph.vertexNum, graph.edgeNum, max_aut);
    if (total_num_blocks < 1) {
        fprintf(stderr, "[ERROR] Not enough GPU memory to launch even 1 block.\n");
        return {};
    }

    threads_per_block =  choose_tpb_keep_fixed_blocks(dev, total_num_blocks, /*dynSmemBytes=*/0, 64, vertexNum);

    Partition*      d_partitions = nullptr;
    int*    d_found_automorphisms = nullptr;
    int*             d_num_aut = nullptr;
    Leaf*           d_first_leaf = nullptr;
    CanonicalLabeling*           canonical_labeling = nullptr;


    CSRGraph* d_graph = allocate_csr_graph_on_device(graph);

    allocate_partitions_for_blocks(&d_partitions, total_num_blocks, vertexNum, max_levels, graph);

    int* d_max_aut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_max_aut, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_max_aut, &max_aut, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_found_automorphisms, (size_t)vertexNum * (size_t)max_aut * sizeof( int)));
    CUDA_CHECK(cudaMemset(d_found_automorphisms, -1, (size_t)vertexNum * (size_t)max_aut * sizeof( int)));

    CUDA_CHECK(cudaMalloc(&d_num_aut, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_num_aut, 0, sizeof(int)));

    allocate_leaf_on_device(&d_first_leaf, vertexNum);
    allocate_canonical_labeling_on_device(&canonical_labeling,  vertexNum);

    Cell*   cell_buffers = nullptr;
   
    unsigned int*  aux_arrays = nullptr;

    CUDA_CHECK(cudaMalloc(&cell_buffers, (size_t)total_num_blocks *2* (vertexNum + 1) * sizeof(Cell)));
    
    CUDA_CHECK(cudaMalloc(&aux_arrays, (size_t)total_num_blocks * 4 * vertexNum * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(aux_arrays, 0, (size_t)total_num_blocks * 4 * vertexNum * sizeof(unsigned int)));


    bool* d_hash_val = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_val, (size_t)vertexNum * (size_t)vertexNum * sizeof(bool)));


    int* d_node_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_node_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_node_counter, 0, sizeof(int)));

    
    int* hash_lock = nullptr;
    CUDA_CHECK(cudaMalloc(&hash_lock, sizeof(int)));
    CUDA_CHECK(cudaMemset(hash_lock, 0, sizeof(int)));

    unsigned int* d_eqlevfirst = nullptr;
   
    CUDA_CHECK(cudaMalloc(&d_eqlevfirst, (size_t)total_num_blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_eqlevfirst, 0, (size_t)total_num_blocks * sizeof(unsigned int)));

    int* d_canon_code_per_block = nullptr;


    CUDA_CHECK(cudaMalloc(&d_canon_code_per_block, (size_t)total_num_blocks * (size_t)(vertexNum + 2) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_canon_code_per_block, 0, (size_t)total_num_blocks * (size_t)(vertexNum + 2) * sizeof(int)));

    // TargetCell per (block, level)
    TargetCell* d_target_cells = nullptr;
    int* d_tc_elems_pool = nullptr;
    int* d_tc_level_reached = nullptr;

    CUDA_CHECK(cudaMalloc(&d_target_cells, (size_t)total_num_blocks * (size_t)levels_per_block * sizeof(TargetCell)));
    CUDA_CHECK(cudaMalloc(&d_tc_elems_pool, (size_t)total_num_blocks * (size_t)levels_per_block * (size_t)vertexNum * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tc_elems_pool, -1 , (size_t)total_num_blocks * (size_t)levels_per_block * (size_t)vertexNum * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_tc_level_reached, (size_t)total_num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tc_level_reached, 0, (size_t)total_num_blocks * sizeof(int)));

    // Host-side wiring of TargetCell.elems pointers
    {
        const size_t cells_per_block = (size_t)levels_per_block;
        const size_t total_cells     = (size_t)total_num_blocks * cells_per_block;

        TargetCell* h_target_cells = new TargetCell[total_cells];

        for (int b = 0; b < total_num_blocks; ++b) {
            for (int L = 0; L < levels_per_block; ++L) {
                const size_t idx = (size_t)b * cells_per_block + (size_t)L;
                const size_t off = idx * (size_t)vertexNum;

                TargetCell tc;
                tc.elems                 = d_tc_elems_pool + off;
                tc.first                 = 0;
                tc.length                = 0;
                tc.counter               = 0;
                tc.last_num_aut_at_level = 0;

                h_target_cells[idx] = tc;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_target_cells, h_target_cells,
                              total_cells * sizeof(TargetCell), cudaMemcpyHostToDevice));
        delete[] h_target_cells;
    }

    int* top_tc_level = nullptr;
    CUDA_CHECK(cudaMalloc(&top_tc_level, (size_t)total_num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemset(top_tc_level, 0, (size_t)total_num_blocks * sizeof(int)));

    // Nodes-per-SM counter array
    unsigned long long* NODES_PER_SM_d = nullptr;
    unsigned long long* NODES_PER_SM   = (unsigned long long*)malloc(sizeof(unsigned long long) * (size_t)numSMs);
    for (int i = 0; i < numSMs; ++i) NODES_PER_SM[i] = 0ULL;

    CUDA_CHECK(cudaMalloc(&NODES_PER_SM_d, (size_t)numSMs * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(NODES_PER_SM_d, NODES_PER_SM, (size_t)numSMs * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    // WorkList (fixed size, unchanged)
    WorkList workList_d = allocateWorkList(vertexNum);
    CUDA_CHECK(cudaDeviceSynchronize());
    

    int* last_num_aut = nullptr;
    CUDA_CHECK(cudaMalloc(&last_num_aut, (size_t)total_num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemset(last_num_aut, 0, (size_t)total_num_blocks * sizeof(int)));

    Counters* counters_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&counters_d, (size_t)total_num_blocks * sizeof(Counters)));

    printf("Launching kernel with %d blocks and %d threads per block\n",
           total_num_blocks, threads_per_block);

    CUDA_CHECK(cudaEventRecord(start, 0));

    nauty_parallel_kernel<<<total_num_blocks, threads_per_block, dynSmemBytes>>>(
        d_partitions,
        vertexNum,
        d_graph,
        cell_buffers, 
        aux_arrays, 
        d_found_automorphisms,
        d_num_aut,
        d_first_leaf,
        canonical_labeling,
        d_max_aut,
        d_hash_val,
        d_node_counter,
        d_eqlevfirst,
        d_target_cells,
        d_tc_level_reached,
        d_canon_code_per_block,
        workList_d,
        top_tc_level,
        NODES_PER_SM_d,
        last_num_aut,
        counters_d,
        hash_lock
    );

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());      // launch/config errors
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    long long total_ns  = (long long)(milliseconds * 1e6);
    long long total_us  = total_ns / 1000;
    long long total_ms  = total_us / 1000;
    long long total_s   = total_ms / 1000;
    long long total_min = total_s / 60;

    long long rem_s  = total_s % 60;
    long long rem_ms = total_ms % 1000;
    long long rem_us = total_us % 1000;
    long long rem_ns = total_ns % 1000;

    std::cout << "Algorithm ran in: "
              << total_min << "min, "
              << rem_s << "s, "
              << rem_ms << "ms, "
              << rem_us << "µs, "
              << rem_ns << "ns\n";

    int h_node_counter = 0;
    CUDA_CHECK(cudaMemcpy(&h_node_counter, d_node_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Total search tree nodes created: %d\n", h_node_counter);

    int h_num_aut = 0;
    CUDA_CHECK(cudaMemcpy(&h_num_aut, d_num_aut, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Total automorphisms found: %d\n", h_num_aut);
    


#if USE_COUNTERS
    {
        std::vector<Counters> counters_h((size_t)total_num_blocks);
        CUDA_CHECK(cudaMemcpy(counters_h.data(), counters_d,
                              (size_t)total_num_blocks * sizeof(Counters),
                              cudaMemcpyDeviceToHost));

        const std::string counters_filename = "kernel_counters_percent.csv";
        FILE* fp = fopen(counters_filename.c_str(), "w");
        if (!fp) {
            perror("Failed to open kernel_counters_percent.csv");
        } else {
            fprintf(fp,
                "BLOCK_NO,"
                "ENQUEUE_pct,"
                "DEQUEUE_pct,"
                "WAIT_FIRST_LEAF_pct,"
                "PREPARE_ROOT_pct,"
                "PUSH_STACK_pct,"
                "POP_STACK_pct,"
                "PRUNE_STACK_ENTRY_pct,"
                "BACKTRACK_pct,"
                "VERIFY_SEQUENCE_pct,"
                "SPLIT_BY_AND_REFINE_pct,"
                "CLASSIFY_NODE_pct,"
                "WAIT_BEST_LEAF_LOCK_pct\n"
            );

            for (unsigned int i = 0; i < (unsigned)total_num_blocks; ++i) {
                unsigned long long total_cycles = 0;
                for (int c = 0; c < NUM_COUNTERS; ++c) total_cycles += counters_h[i].totalTime[c];
                if (total_cycles == 0) total_cycles = 1;

                fprintf(fp, "%u", i);

                auto print_pct = [&](CounterName c) {
                    double pct = 100.0 * (double)counters_h[i].totalTime[c] / (double)total_cycles;
                    fprintf(fp, ",%.2f", pct);
                };

                print_pct(ENQUEUE);
                print_pct(DEQUEUE);
                print_pct(WAIT_FIRST_LEAF);
                print_pct(PREPARE_ROOT);
                print_pct(PUSH_STACK);
                print_pct(POP_STACK);
                print_pct(PRUNE_STACK_ENTRY);
                print_pct(BACKTRACK);
                print_pct(VERIFY_SEQUENCE);
                print_pct(SPLIT_BY_AND_REFINE);
                print_pct(CLASSIFY_NODE);
                print_pct(WAIT_BEST_LEAF_LOCK);

                fprintf(fp, "\n");
            }

            fclose(fp);
            printf("[INFO] Counter percentage CSV written to %s\n", counters_filename.c_str());
        }
    }
#endif

    CUDA_CHECK(cudaMemcpy(NODES_PER_SM, NODES_PER_SM_d, (size_t)numSMs * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    FILE* f = fopen("nodes_per_sm.csv", "w");
    fprintf(f, "sm,nodes\n");
    unsigned long long total = 0ULL;
    for (int sm = 0; sm < numSMs; ++sm) {
        fprintf(f, "%d,%llu\n", sm, (unsigned long long)NODES_PER_SM[sm]);
        total += NODES_PER_SM[sm];
    }
    fclose(f);

    //  SAVE CANONICAL FORM TO FILE

    // 1) Copy the CanonicalLabeling struct back
    CanonicalLabeling h_cl{};
    CUDA_CHECK(cudaMemcpy(&h_cl, canonical_labeling,
                          sizeof(CanonicalLabeling),
                          cudaMemcpyDeviceToHost));

    // 2) Copy the canonical adjacency "hash" (V*V bools) back to host
    const size_t N = (size_t)vertexNum * (size_t)vertexNum;
    
    std::vector<unsigned char> h_canonical_labeling(N);

    CUDA_CHECK(cudaMemcpy(h_canonical_labeling.data(),
                          (const void*)h_cl.hash_of_perm_graph,
                          N * sizeof(bool),
                          cudaMemcpyDeviceToHost));


    cudaFreeWorkList(workList_d);
    cudaFree(top_tc_level);
    free(NODES_PER_SM);
    cudaFree(NODES_PER_SM_d);
    cudaFree(last_num_aut);
    cudaFree(counters_d);
    cudaFree(hash_lock);
    
    cleanup_all(
        d_partitions, total_num_blocks,
        d_graph,
        d_found_automorphisms,
        d_num_aut,
        d_first_leaf,
        canonical_labeling,
        cell_buffers,
        aux_arrays,
        d_hash_val,
        d_node_counter,
        d_max_aut,
        d_eqlevfirst,
        d_target_cells,
        d_tc_level_reached,
        d_tc_elems_pool,
        d_canon_code_per_block
    );

    // std::cout << "[DEBUG] Device synchronized, kernel should be done." << std::endl;
    return h_canonical_labeling;
}
