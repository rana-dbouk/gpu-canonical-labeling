
#ifndef PARTITION_REFINEMENT_CUH
#define PARTITION_REFINEMENT_CUH

#include <stdio.h>
#include "structs.cuh"
#include <cuda_runtime.h>
#include <limits.h> 


 __device__ void print_partition(const Partition& partition) {
    if (threadIdx.x == 0) {
        printf("Partition LCS size: %u\n", partition.lcs_size);  
        printf("[");
        for (unsigned int i = 0; i < partition.lcs_size; ++i) {
            Cell cell = partition.lcs[i];
            printf("[");
            for (unsigned int j = 0; j < cell.length; ++j) {
                unsigned int v = partition.element_vec[cell.first + j];
                printf("%u", v);
                if (j < cell.length - 1)
                    printf(", ");
            }
            printf("]");
            if (i < partition.lcs_size - 1)
                printf(", ");
        }
        printf("]\n");
    }
    __syncthreads();  
}

__device__ int is_discrete_p(const Partition& p) {
    int found = 0;

    for (int idx = threadIdx.x; idx < (int)p.lcs_size; idx += (int)blockDim.x) {
        if (p.lcs[idx].length != 1) { found = 1; break; }
    }

    // Returns nonzero if any thread has found != 0
    int any = __syncthreads_or(found);

    return any ? 0 : 1;
}


__device__ void oddEvenSort(volatile unsigned int* arr, unsigned int size) {
    // O(n^2) 
    for (unsigned int pass = 0; pass < size; ++pass) {

        
        for (unsigned int j = (threadIdx.x << 1); j + 1 < size; j += (blockDim.x << 1)) {
            unsigned int a = arr[j], b = arr[j + 1];
            if (a > b) { arr[j] = b; arr[j + 1] = a; }
        }
        __syncthreads();

        
        for (unsigned int j = (threadIdx.x << 1) + 1; j + 1 < size; j += (blockDim.x << 1)) {
            unsigned int a = arr[j], b = arr[j + 1];
            if (a > b) { arr[j] = b; arr[j + 1] = a; }
        }
        __syncthreads();
    }
}



__device__ int partition_refinement_p(
    Partition&      partition,
    const CSRGraph& graph,
    Cell*           subsequence,
    unsigned int    subseq_size_parameter,
    Cell*           buffer,
    unsigned int*   aux_arrays,
    Counters*       counters
) {
    __shared__ unsigned int active_count;
    __shared__ unsigned long long longcode;
    __shared__ int ret_code;
    __shared__ int hint;

    const unsigned int n = graph.vertexNum;
    const unsigned int block_offset = blockIdx.x * 4 * n;

    unsigned int* all_degrees = aux_arrays + block_offset;
    unsigned int* active      = aux_arrays + block_offset + n;
    unsigned int* bucket      = aux_arrays + block_offset + 2 * n;
    unsigned int* work_cell   = aux_arrays + block_offset + 3 * n;

    /*
     * active[first] == 1 means the cell starting at position first is active.
     * subsequence is only used to initialize this active set.
     */
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        all_degrees[i] = 0;
        active[i]      = 0;
        bucket[i]      = 0;
        work_cell[i]   = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        active_count = 0;
        longcode = (unsigned long long)partition.lcs_size;
        ret_code = 0;
        hint = 0;
    }
    __syncthreads();

    /*
     * Initialize active set.
     */
    if (subsequence == nullptr || subseq_size_parameter == 0) {
        for (unsigned int i = threadIdx.x; i < partition.lcs_size; i += blockDim.x) {
            unsigned int first = partition.lcs[i].first;

            active[first] = 1;
            atomicAdd(&active_count, 1u);
        }
    } else {
        for (unsigned int i = threadIdx.x; i < subseq_size_parameter; i += blockDim.x) {
            unsigned int first = subsequence[i].first;
            active[first] = 1;
            atomicAdd(&active_count, 1u);
        }
    }
    __syncthreads();

    while ((!is_discrete_p(partition)) && active_count > 0) {
        __shared__ unsigned int split1;
        __shared__ unsigned int splitter_index;
        __shared__ Cell splitter;
        __shared__ unsigned int split_size;

        /*
         * Choose splitter using the same hint/search/wraparound rule
         * as the sequential implementation.
         */
        __shared__ unsigned int candidate;
        __shared__ unsigned int start_pos;
        __shared__ int use_hint;

        if (threadIdx.x == 0) {
            candidate = UINT_MAX;
            use_hint = 0;

            if (hint >= 0 &&
                (unsigned int)hint < n &&
                active[(unsigned int)hint] != 0)
            {
                candidate = (unsigned int)hint;
                use_hint = 1;
            }

            start_pos = (hint >= 0) ? (unsigned int)hint + 1 : 0;
            if (start_pos > n) {
                start_pos = n;
            }
        }
        __syncthreads();

        if (!use_hint) {
            for (unsigned int p = start_pos + threadIdx.x; p < n; p += blockDim.x) {
                if (active[p] != 0) {
                    atomicMin(&candidate, p);
                }
            }
        }
        __syncthreads();

        if (!use_hint && candidate == UINT_MAX) {
            for (unsigned int p = threadIdx.x; p < start_pos; p += blockDim.x) {
                if (active[p] != 0) {
                    atomicMin(&candidate, p);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            split1 = candidate;

            if (split1 < n && active[split1] != 0) {
                active[split1] = 0;
                --active_count;
            }
        }
        __syncthreads();

        /*
         * Find the LCS cell whose first position is split1.
         */
        if (threadIdx.x == 0) {
            splitter_index = UINT_MAX;
        }
        __syncthreads();

        for (unsigned int i = threadIdx.x; i < partition.lcs_size; i += blockDim.x) {
            if (partition.lcs[i].first == split1) {
                atomicMin(&splitter_index, i);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            splitter = partition.lcs[splitter_index];
            split_size = splitter.length;

            unsigned int split2 = split1 + split_size - 1;
            longcode = MASH(longcode, (unsigned long long)(split1 + split2));

            if (split_size > 1) {
                longcode = MASH(longcode, (unsigned long long)split_size);
            }
        }
        __syncthreads();

        /*
         * Clear degrees.
         */
        for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
            all_degrees[i] = 0;
        }
        __syncthreads();

        /*
         * all_degrees[v] = number of neighbors of v in the splitter cell.
         */
        for (unsigned int i = threadIdx.x; i < split_size; i += blockDim.x) {
            unsigned int w = partition.element_vec[splitter.first + i];

            for (unsigned int e = graph.srcPtr[w]; e < graph.srcPtr[w + 1]; ++e) {
                unsigned int nb = graph.dst[e];
                atomicAdd(&all_degrees[nb], 1u);
            }
        }
        __syncthreads();

        /*
         * Scan LCS cells in order.
         */
        for (int index = 0; index < (int)partition.lcs_size; ++index) {
            __syncthreads();
            __shared__ Cell cell_hdr;
            __shared__ unsigned int cell_first;
            __shared__ unsigned int cell_size;

            __shared__ unsigned int common_degree;
            __shared__ unsigned int bmin_sh;
            __shared__ unsigned int bmax_sh;
            __shared__ int any_mismatch;

            if (threadIdx.x == 0) {
                cell_hdr = partition.lcs[index];
                cell_first = cell_hdr.first;
                cell_size = cell_hdr.length;
            }
            __syncthreads();

            if (cell_size <= 1) {
                continue;
            }

            /*
             * Uniformity test and bmin/bmax computation.
             */
            if (threadIdx.x == 0) {
                common_degree = all_degrees[partition.element_vec[cell_first]];
                bmin_sh = UINT_MAX;
                bmax_sh = 0;
                any_mismatch = 0;
            }
            __syncthreads();

            int mismatch = 0;

            for (unsigned int j = threadIdx.x; j < cell_size; j += blockDim.x) {
                unsigned int v = partition.element_vec[cell_first + j];
                unsigned int d = all_degrees[v];

                atomicMin(&bmin_sh, d);
                atomicMax(&bmax_sh, d);

                if (d != common_degree) {
                    mismatch = 1;
                }
            }

            if (mismatch) {
                atomicOr(&any_mismatch, 1);
            }
            __syncthreads();

            if (!any_mismatch) {
                if (threadIdx.x == 0 && split_size > 1) {
                    longcode = MASH(
                        longcode,
                        (unsigned long long)(common_degree + cell_hdr.first)
                    );
                }
                __syncthreads();
                continue;
            }

            /*
             * Split the cell following the sequential implementation.
             *
             * bucket[d] is first used as the count of vertices with degree d.
             * Later, bucket[d] is converted into the write position for degree d.
             */

            for (unsigned int d = threadIdx.x; d < n; d += blockDim.x) {
                bucket[d] = 0;
            }
            __syncthreads();

            /*
             * Count bucket[d].
             */
            for (unsigned int j = threadIdx.x; j < cell_size; j += blockDim.x) {
                unsigned int v = partition.element_vec[cell_first + j];
                unsigned int d = all_degrees[v];
                atomicAdd(&bucket[d], 1u);
            }
            __syncthreads();

            __shared__ unsigned int num_new_cells;
            __shared__ unsigned int max_size;
            __shared__ unsigned int max_index;
            __shared__ unsigned int first_pos;
            __shared__ unsigned int fidx;

            if (threadIdx.x == 0) {
                num_new_cells = 0;

                if (split_size == 1) {
                    num_new_cells += (bucket[1] > 0);
                    num_new_cells += (bucket[0] > 0);
                } else {
                    for (unsigned int d = bmin_sh; d <= bmax_sh; ++d) {
                        if (bucket[d] > 0) {
                            ++num_new_cells;
                        }
                    }
                }
            }
            __syncthreads();

            __shared__ unsigned int old_lcs_size;
            __shared__ unsigned int old_cell_was_active;

            if (threadIdx.x == 0) {
                old_lcs_size = partition.lcs_size;
                old_cell_was_active = active[cell_first];
            }
            __syncthreads();

            /*
             * Shift lcs once to make room for the new fragments.
             * The old cell at index is overwritten by the first fragment.
             */
            if (num_new_cells > 1) {
                int src_first = index + 1;
                int src_last  = (int)old_lcs_size - 1;
                int count     = (src_last >= src_first) ? (src_last - src_first + 1) : 0;

                for (int t = threadIdx.x; t < count; t += (int)blockDim.x) {
                    buffer[t] = partition.lcs[src_first + t];
                }
                __syncthreads();

                for (int t = threadIdx.x; t < count; t += (int)blockDim.x) {
                    partition.lcs[src_first + (int)num_new_cells - 1 + t] = buffer[t];
                }
            }
            __syncthreads();

            /*
             * Write fragment descriptors directly into partition.lcs.
             * This is the same logic as add_fragment() in the sequential code.
             * It also converts bucket[d] into the write position for degree d.
             */
            if (threadIdx.x == 0) {
                fidx = 0;
                first_pos = cell_first;
                max_size = 0;
                max_index = 0;

                if (split_size == 1) {
                    /*
                     * Singleton splitter: adjacent vertices first, then non-adjacent.
                     */
                    if (bmin_sh <= 1 && bmax_sh >= 1) {
                        unsigned int degree_value = 1;
                        unsigned int size = bucket[degree_value];

                        if (size > 0) {
                            Cell& new_cell = partition.lcs[index + fidx];

                            new_cell.first = first_pos;
                            new_cell.length = size;
                            new_cell.in_level = partition.level + 1;

                            if (size > max_size) {
                                max_size = size;
                                max_index = fidx;
                            }

                            bucket[degree_value] = first_pos - cell_first;
                            first_pos += size;
                            ++fidx;
                        }
                    }

                    // if (bmin_sh <= 0 && bmax_sh >= 0) {
                    if (bmin_sh == 0) {
                        unsigned int degree_value = 0;
                        unsigned int size = bucket[degree_value];

                        if (size > 0) {
                            Cell& new_cell = partition.lcs[index + fidx];

                            new_cell.first = first_pos;
                            new_cell.length = size;
                            new_cell.in_level = partition.level + 1;

                            if (size > max_size) {
                                max_size = size;
                                max_index = fidx;
                            }

                            bucket[degree_value] = first_pos - cell_first;
                            first_pos += size;
                            ++fidx;
                        }
                    }
                } else {
                    /*
                     * Non-singleton splitter: fragments ordered by increasing degree.
                     */
                    for (unsigned int degree_value = bmin_sh; degree_value <= bmax_sh; ++degree_value) {
                        unsigned int size = bucket[degree_value];

                        if (size == 0) {
                            continue;
                        }

                        Cell& new_cell = partition.lcs[index + fidx];

                        new_cell.first = first_pos;
                        new_cell.length = size;
                        new_cell.in_level = partition.level + 1;

                        longcode = MASH(
                            longcode,
                            (unsigned long long)(degree_value + new_cell.first)
                        );

                        if (size > max_size) {
                            max_size = size;
                            max_index = fidx;
                        }

                        bucket[degree_value] = first_pos - cell_first;
                        first_pos += size;
                        ++fidx;
                    }
                }

                partition.lcs_size += num_new_cells - 1;
            }
            __syncthreads();

            /*
             * Reorder vertices into work_cell.
             */
            for (unsigned int j = threadIdx.x; j < cell_size; j += blockDim.x) {
                unsigned int v = partition.element_vec[cell_first + j];
                unsigned int d = all_degrees[v];

                unsigned int rank = 0;

                for (unsigned int q = 0; q < j; ++q) {
                    unsigned int u = partition.element_vec[cell_first + q];

                    if (all_degrees[u] == d) {
                        ++rank;
                    }
                }

                work_cell[bucket[d] + rank] = v;
            }
            __syncthreads();

            /*
             * Copy reordered vertices back into partition.element_vec.
             */
            for (unsigned int j = threadIdx.x; j < cell_size; j += blockDim.x) {
                partition.element_vec[cell_first + j] = work_cell[j];
            }
            __syncthreads();

            /*
             * Singleton-splitter longcode update.
             * Non-singleton longcode was already updated while writing fragments.
             */
            if (threadIdx.x == 0) {
                if (split_size == 1) {
                    unsigned int neighbors_size = partition.lcs[index].length;
                    unsigned int non_neighbors_size =
                        (num_new_cells > 1) ? partition.lcs[index + 1].length : 0;

                    if (neighbors_size > 0 && non_neighbors_size > 0) {
                        unsigned long long boundary =
                            (unsigned long long)(cell_first + neighbors_size - 1);

                        longcode = MASH(longcode, boundary);
                    }
                }
            }
            __syncthreads();

            /*
             * Update active set using the same rules as the sequential code.
             */
            if (threadIdx.x == 0) {
                if (split_size == 1) {
                    unsigned int left_size = partition.lcs[index].length;
                    unsigned int right_size =
                        (num_new_cells > 1) ? partition.lcs[index + 1].length : 0;

                    if (old_cell_was_active || left_size >= right_size) {
                        if (num_new_cells > 1) {
                            unsigned int fstart = partition.lcs[index + 1].first;

                            if (fstart < n && active[fstart] == 0) {
                                active[fstart] = 1;
                                ++active_count;
                            }

                            if (partition.lcs[index + 1].length == 1) {
                                hint = (int)fstart;
                            }
                        }
                    } else {
                        unsigned int fstart = partition.lcs[index].first;

                        if (fstart < n && active[fstart] == 0) {
                            active[fstart] = 1;
                            ++active_count;
                        }

                        if (partition.lcs[index].length == 1) {
                            hint = (int)fstart;
                        }
                    }
                } else {
                    /*
                     * add every fragment except the first.
                     */
                    for (unsigned int f = 0; f < num_new_cells; ++f) {
                        Cell frag = partition.lcs[index + f];

                        if (frag.first != cell_first) {
                            if (frag.first < n && active[frag.first] == 0) {
                                active[frag.first] = 1;
                                ++active_count;
                            }

                            if (frag.length == 1) {
                                hint = (int)frag.first;
                            }
                        }
                    }

                    /*
                     * If the old cell was not active, add the first fragment
                     * and remove the largest fragment.
                     */
                    if (!old_cell_was_active) {
                        if (cell_first < n && active[cell_first] == 0) {
                            active[cell_first] = 1;
                            ++active_count;
                        }

                        unsigned int largest_first = partition.lcs[index + max_index].first;

                        if (largest_first < n && active[largest_first] != 0) {
                            active[largest_first] = 0;
                            --active_count;
                        }
                    }
                }

                /*
                 * Preserve the backtracking marker behavior.
                 */
                if (cell_hdr.in_level != partition.level + 1 && num_new_cells > 0) {
                    partition.lcs[index + num_new_cells - 1].in_level = cell_hdr.in_level;
                }
            }
            __syncthreads();

            /*
             * Continue after the fragments created from this cell.
             */
            index = index + (int)num_new_cells - 1;
            __syncthreads();
        }

        /*
         * Clear degrees for the next splitter.
         */
        for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
            all_degrees[i] = 0;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        unsigned int prev_level = partition.level;

        longcode = MASH(longcode, (unsigned long long)partition.lcs_size);
        ret_code = CLEANUP(longcode);

        partition.level = prev_level + 1;
    }
    __syncthreads();

    return ret_code;
}


__device__ int first_non_singleton(Partition& current_partition, int n)
{
    
    __shared__ int target_index;       // start offset in element_vec
   
    const unsigned tid = threadIdx.x;

    if (tid == 0) target_index = INT_MAX;
    __syncthreads();

    for (unsigned i = tid; i < current_partition.lcs_size; i += blockDim.x) {
        if (current_partition.lcs[i].length > 1) {
            atomicMin(&target_index, (int)i);
        }
    }
    __syncthreads();

    // no suitable cell
    if (target_index == INT_MAX) return -1;

    return target_index;
}

 __device__ int split_by_and_refine_p(Partition& partition, const CSRGraph& graph, unsigned int vertex, Cell* subsequence, Cell* buffer, unsigned int* aux_arrays, Counters* counters) {   
    unsigned int tid = threadIdx.x;

    //startTime(PRE_REFINE, counters);
    unsigned int numVertices = graph.vertexNum;

    int cell_index = first_non_singleton(partition, numVertices); // Position of original cell

   
    __syncthreads();
    Cell& cell = partition.lcs[cell_index];
    unsigned int start = cell.first;
    unsigned int end = cell.first + cell.length;

    // Step 1: Find the index of the vertex
    __shared__ unsigned int vertex_index;
    

    if (threadIdx.x == 0) {
        vertex_index = end; 
    }
    __syncthreads();

    for (unsigned int i = start + threadIdx.x; i < end; i += blockDim.x) {
       
        if (partition.element_vec[i] == vertex) {
            atomicMin(&vertex_index, i);
            break;
        }
    }
    __syncthreads();

    
    // Step 2: Shift all elements before vertex_index to the right
    if (tid == 0 && vertex_index > start) {

        // Shift elements to the right
        for (int i = vertex_index; i > (int)start; --i) {
            partition.element_vec[i] = partition.element_vec[i - 1];
        }

        // Place the vertex at the beginning
        partition.element_vec[start] = vertex;

       
    }

    __syncthreads();
    //create a new cell for all the vertices in th old cell minus the vertex
    __shared__ Cell non_trivial_cell;

    // Prepare non-trivial cell
    if (tid == 0) {
        non_trivial_cell.first = cell.first + 1;
        non_trivial_cell.length = cell.length - 1;
        non_trivial_cell.in_level = cell.in_level;
    }
    
    for(int i = tid + cell_index + 1; i < partition.lcs_size; i+= blockDim.x){
        buffer[i] =  partition.lcs[i];
    }
    __syncthreads();

    for(int i = tid + cell_index + 1; i < partition.lcs_size; i+= blockDim.x){
        partition.lcs[i+1] =  buffer[i];
    }
    __syncthreads();

    if(tid == 0){
        partition.lcs[cell_index + 1] = non_trivial_cell;  // Insert the new cell
        partition.lcs_size++; //update lcs size
        cell.length=1; //only contains the vertex now
        cell.in_level=partition.level+1; //considered newly created
        partition.lcs[cell_index]=cell; //since we modified it previously

        subsequence[0] = cell;
    }
    
    
    unsigned int subSize=1;
    __syncthreads();

    //endTime(PRE_REFINE, counters);
    

    //startTime(REFINE, counters);
    int curr_code = partition_refinement_p(partition, graph, subsequence, subSize , buffer, aux_arrays, counters);
    //endTime(REFINE, counters);
    
    __syncthreads();

    return curr_code;

}

__device__ void reconstruct_at_level_p(Partition& partition,
                                       unsigned int return_level,
                                       unsigned int numVertices,
                                       unsigned int* p_eqlev_first,
                                       Cell* buffer)
{
    if (return_level < 1) return;

    

    __shared__ unsigned i_sh;
    if (threadIdx.x == 0) i_sh = 0;
    __syncthreads();

    // Linear scan over lcs: collapse every maximal run of cells created after return_level.
    // Invariant required: descendants of any pre-split cell are contiguous in lcs.
    for (;;)
    {
        // snapshot shared i
        unsigned i = i_sh;
        if (i >= partition.lcs_size) break;

        __syncthreads();

        if (partition.lcs[i].in_level <= return_level) {
            if (threadIdx.x == 0) ++i_sh;   // advance shared i
            __syncthreads();
            continue;
        }

        // Merge a maximal run [i .. j-1] where in_level > return_level
        const unsigned first = partition.lcs[i].first;
        unsigned       j     = i+1;
        unsigned       len   = partition.lcs[i].length;

        while (j < partition.lcs_size && partition.lcs[j].in_level > return_level) {
            len += partition.lcs[j].length;
            ++j;
        }
        if (j < partition.lcs_size) {
            len += partition.lcs[j].length;
            //++j;
        }
        __syncthreads();

        oddEvenSort(partition.element_vec + first, len);

        if (threadIdx.x == 0) {
            partition.lcs[i].first    = first;
            partition.lcs[i].length   = len;
            partition.lcs[i].in_level = partition.lcs[j].in_level;
        }

        // Remove the extra (j - i - 1) cells by shifting the tail left.
        const unsigned drop = (j - i);
        if (drop > 0) {
            for (unsigned int k = j + 1 + threadIdx.x; k < partition.lcs_size; k += blockDim.x) {
                buffer[k] = partition.lcs[k];
            }
            __syncthreads();

            for (unsigned int k = j + 1 + threadIdx.x; k < partition.lcs_size; k += blockDim.x) {
                partition.lcs[k - drop] = buffer[k];
            }
            __syncthreads();

        }
        //__syncthreads();

        if (threadIdx.x == 0){
            partition.lcs_size -= drop;
            ++i_sh;  // advance shared i
        } 
        __syncthreads();
    }

    //__syncthreads();
    if (threadIdx.x == 0) {
        if (return_level < (*p_eqlev_first)) (*p_eqlev_first) = return_level;
    }
    __syncthreads();
}


#endif