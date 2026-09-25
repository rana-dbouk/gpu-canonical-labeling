#include "partition_refinement.cuh"
#include "CSRGraph.h"
#include <algorithm>

#define PRUNED_MARK (2)

__device__ int get_smid(void) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}



struct DSU_dev {
    unsigned int n;
    unsigned int* parent; 

    __device__ void bind(unsigned int n_, unsigned int* parent_buf) {
        
        n = n_;
        parent = parent_buf;

        
    }

    __device__ unsigned int find_root(unsigned int x) const {
        
        unsigned int p = parent[x];
        
        while (p != x) {
            
            x = p;
            p = parent[x];
        }
        return x;
    }

    // Atomic union-by-min: hook larger root under smaller root
    __device__ void unite_atomic(unsigned int a, unsigned int b) {
        while (true) {
            a = find_root(a);
            b = find_root(b);
            if (a == b) return;

            unsigned int hi = a > b ? a : b;
            unsigned int lo = a ^ b ^ hi; // min(a,b)

            // Try to attach hi -> lo. Succeeds only if hi is still a root.
            if (atomicCAS(&parent[hi], hi, lo) == hi) return;

            // else some other thread changed it; retry
        }
    }

    __device__ unsigned int rep(unsigned int x) const { return find_root(x); }
};


__device__ int push_stack(Partition& current_partition,
                                    TargetCell* stack,
                                      volatile int*        stack_top, int n)
{
    __shared__ int       target_index;
    __shared__ int       slot;        // slot in target_cell_at_level we will fill
    __shared__ unsigned  first;       // start offset in element_vec
    __shared__ unsigned  len;         // cell length

    const unsigned tid = threadIdx.x;

    if (tid == 0) target_index = INT_MAX;
    __syncthreads();

    // pick first non-singleton cell (min index)
    for (unsigned i = tid; i < current_partition.lcs_size; i += blockDim.x) {
        if (current_partition.lcs[i].length > 1) {
            atomicMin(&target_index, (int)i);
        }
    }
    __syncthreads();

    // no suitable cell
    if (target_index == INT_MAX) return -1;

    if (tid == 0) {
        slot = *stack_top;

        const Cell& c = current_partition.lcs[target_index];
        first = c.first;
        len   = c.length;

        TargetCell& out = stack[slot];
        out.first                 = (int)first;
        out.length                = len;
        out.counter               = 0;
        out.last_num_aut_at_level = 0;

        
        (*stack_top)++;
    }
    __syncthreads();

    // Build a vertex-mark array: 0 = not in TC, 2 = in TC (branchable)
    TargetCell& out = stack[slot];

    // 1) Zero all entries (size = num_vertices)
    
    for (unsigned v = tid; v < n; v += blockDim.x) {
        out.elems[v] = -1;
    }
    __syncthreads();

    // 2) Mark members of the selected cell as 2
    //    Members are contiguous in element_vec[first .. first+len)
    for (unsigned k = tid; k < len; k += blockDim.x) {
        const int vtx = current_partition.element_vec[first + k];
        out.elems[vtx] = vtx;
    }
    __syncthreads();
    

    return target_index;
}

__device__ int pop_stack(TargetCell* stack, volatile int* stack_top, int n){
   

    TargetCell& tc = stack[(*stack_top) - 1];
    
    __shared__ int counter_sh;
    __shared__ int result;

    if (threadIdx.x == 0) {
        while (tc.counter < n && tc.elems[tc.counter] != tc.counter)
            ++tc.counter;

        __threadfence_block();          // make global writes visible within the block
        counter_sh = tc.counter;        // take a shared snapshot
        result = (counter_sh >= (unsigned)n) ? -1 : counter_sh;
        if (result != -1) tc.counter++;
        
    }
    
    
    __syncthreads();
    return result;
    
}

__device__ int compare_adj_upper(volatile bool* A, volatile bool* B, unsigned n)
{
    __shared__ int result;
    __shared__ int found;

    if (threadIdx.x == 0) { result = 0; found = 0; }
    __syncthreads();

    for (unsigned i = 0; i < n; ++i) {
        __syncthreads();
        if (found) break;

        __shared__ int row_best_j;
        if (threadIdx.x == 0) row_best_j = 0x7fffffff;
        __syncthreads();

        unsigned base = i * n;

        
        for (unsigned j = i + 1 + threadIdx.x; j < n; j += blockDim.x) {
            bool a = A[base + j];
            bool b = B[base + j];
            if (a != b) {
                atomicMin(&row_best_j, (int)j);  
                break;                          
            }
        }
        __syncthreads();

        
        if (threadIdx.x == 0 && row_best_j != 0x7fffffff) {
            bool a = A[base + (unsigned)row_best_j]; 
            result = a ? 1 : -1;
            found  = 1;
        }
    }

    __syncthreads();
    return result;
}


// Lex compare two int arrays
// Returns: +1 if A>B, -1 if A<B, 0 if equal (prefix equal; caller decides shorter-wins).
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

__device__ int lex_compare_codes(const int* A, int lenA,
                                           volatile int* B, int lenB,
                                           int* first_mismatch_idx)
{
    const int minLen = (lenA < lenB) ? lenA : lenB;

    __shared__ int s_best;   // earliest mismatch index, or minLen if none
    __shared__ int s_res;

    if (threadIdx.x == 0) { s_best = minLen; s_res = 0; }
    __syncthreads();

    // 1) Each thread finds its earliest mismatch index in its strided sequence
    int local = minLen;
    for (int i = threadIdx.x; i < minLen; i += blockDim.x) {
        int a = A[i];
        int b = B[i];           
        if (a != b) { local = i; break; }
    }

    // 2) Warp-level min reduction on "local"
    // Reduce within each warp to get warp_min
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(mask, local, offset);
        local = imin(local, other);
    }

    // 3) One atomic per warp into shared s_best
    if ((threadIdx.x & 31) == 0) {
        atomicMin(&s_best, local);
    }

    __syncthreads();

    // 4) Finalize result + mismatch index
    if (threadIdx.x == 0) {
        if (s_best < minLen) {
            int a = A[s_best];
            int b = B[s_best];
            s_res = (a > b) - (a < b);
            *first_mismatch_idx = s_best;
        } else {
            s_res = 0;
            *first_mismatch_idx = minLen;
        }
    }

    __syncthreads();
    return s_res;
}
__device__ int get_gca_level_p(volatile unsigned int* first_sequence, unsigned int first_size,
                               volatile unsigned int* second_sequence, unsigned int second_size) {
    __shared__ int first_diff_index;

    if (threadIdx.x == 0) {
        first_diff_index = INT_MAX; 
    }
    __syncthreads();

    int min_len = min(first_size, second_size);

    for (int i = threadIdx.x; i < min_len; i += blockDim.x) {
        if (first_sequence[i] != second_sequence[i]) {
            atomicMin(&first_diff_index, i); 
        }
    }

    __syncthreads();

    if (first_diff_index != INT_MAX)
        return first_diff_index + 1;
    else
        return -1;
}




__device__ void buildPermutedFlatAdjacencyMatrixArray_p(CSRGraph& graph, unsigned int* perm, volatile bool* adjMatrix){
    unsigned int n = graph.vertexNum;
    for (unsigned long long i = threadIdx.x; i < n * n; i += blockDim.x) {
        adjMatrix[i] = false;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned int pi = perm[i];  

        unsigned int start = graph.srcPtr[i];
        unsigned int end = graph.srcPtr[i + 1];

        for (unsigned int k = start; k < end; ++k) {
            unsigned int j = graph.dst[k];        
            unsigned int pj = perm[j];      
            adjMatrix[pi * n + pj] = true;
            adjMatrix[pj * n + pi] = true;
        }
    }

    __syncthreads();

}


// This method is returning current permutation based on the partiton we reached (we are at a leaf).
 __device__ void discrete_partition_to_perm_p(const Partition& P,
                                                    unsigned n,
                                                    unsigned int* out_perm)
{
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned oldv = P.element_vec[i];  
        out_perm[oldv] = i;               
    }
    __syncthreads();
}

 __device__ void first_terminal_p(
    Partition&           current_partition,
    CSRGraph&       graph,
    unsigned              num_vertices,
    Leaf*                d_first_leaf,
    CanonicalLabeling*                canonical_labeling,
    unsigned int*         tmp_leaf_perm,   
    volatile bool*                 tmp_hash,           
    unsigned int*         p_eqlevfirst
)
{

    // Build first-leaf permutation and its permuted adjacency (hash)
    discrete_partition_to_perm_p(current_partition, num_vertices, tmp_leaf_perm);
    buildPermutedFlatAdjacencyMatrixArray_p(graph, tmp_leaf_perm, tmp_hash);

    // Hash -> first & GLOBAL best 
    const unsigned nn = num_vertices * num_vertices;
    for (unsigned idx = threadIdx.x; idx < nn; idx += blockDim.x) {
        bool h = tmp_hash[idx];
        d_first_leaf->hash_of_perm_graph[idx] = h;
        canonical_labeling->hash_of_perm_graph[idx]  = h;
    }


    // Sequence sizes
    if (threadIdx.x == 0) {
        unsigned sz = current_partition.current_vertex_sequence_size;
        d_first_leaf->vertex_sequence_size = sz;
        canonical_labeling ->vertex_sequence_size = sz;
       
    }

    // Sequence payload
    for (unsigned i = threadIdx.x; i < current_partition.current_vertex_sequence_size; i += blockDim.x) {
        unsigned v = current_partition.current_vertex_sequence[i];
        d_first_leaf->vertex_sequence[i] = v;
        canonical_labeling ->vertex_sequence[i] = v;
        
    }
    
    // Labels
    for (unsigned j = threadIdx.x; j < num_vertices; j += blockDim.x) {
        unsigned lbl = current_partition.element_vec[j];
        d_first_leaf->label[j] = lbl;
        canonical_labeling ->label[j] = lbl;
        
    }
    

    // Canonical bookkeeping — GLOBAL
    if (threadIdx.x == 0) {
        *p_eqlevfirst = current_partition.level;
        d_first_leaf->invar_path_value[current_partition.level] = 077777;
        canonical_labeling->level = current_partition.level;
        canonical_labeling->is_leaf = 1;
        canonical_labeling->invar_path_value[current_partition.level] = 077777;
    }

    for (unsigned i = threadIdx.x; i < (unsigned)current_partition.level; i += blockDim.x)
        canonical_labeling->invar_path_value[i] = (int)d_first_leaf->invar_path_value[i];


    __syncthreads();
    
   
}


// Binary search inside one CSR row
__device__ bool row_has_neighbor(const CSRGraph& g, unsigned row, unsigned needle) {
    unsigned lo = g.srcPtr[row], hi = g.srcPtr[row + 1];
    while (lo < hi) {
        unsigned mid = (lo + hi) >> 1;
        unsigned val = g.dst[mid];
        if (val < needle)      lo = mid + 1;
        else if (val > needle) hi = mid;
        else                   return true;
    }
    return false;
}



__device__ bool isAutomorphism_block_coop(const CSRGraph& g,
                               const unsigned int* __restrict__ perm)
{
    const unsigned n = g.vertexNum;
    int fail = 0;

    // 1) Quick Degree check (parallel over vertices)
    for (unsigned u = threadIdx.x; u < n; u += blockDim.x) {
        unsigned pu     = perm[u];
        unsigned deg_u  = g.srcPtr[u + 1]  - g.srcPtr[u];
        unsigned deg_pu = g.srcPtr[pu + 1] - g.srcPtr[pu];
        if (deg_u != deg_pu) { fail = 1; break; }
    }

    // Broadcast failure; early exit safely if any thread failed.
    if (__syncthreads_or(fail)) return false;

    // 2) Edge mapping check (parallel over vertices/edges)
    for (unsigned u = threadIdx.x; u < n; u += blockDim.x) {
        unsigned pu = perm[u];
        for (unsigned j = g.srcPtr[u]; j < g.srcPtr[u + 1]; ++j) {
            unsigned v  = g.dst[j];
            unsigned pv = perm[v];
            if (!row_has_neighbor(g, pu, pv)) { fail = 1; break; }
        }
        if (fail) break; // stop this thread's work; others continue to barrier
    }

    // Final block-wide result
    return __syncthreads_or(fail) == 0;
}



__device__ void backtrack_to_p(Partition& current_partition, unsigned int level, unsigned int num_vertices, Cell* buffer, unsigned int* p_eqlev_first,
                                            volatile int*         tc_level_reached, int* tc_level_top) {
   
    if(level == current_partition.level){
        return;
    }
    if(threadIdx.x == 0){
        current_partition.pushNextIter = 0;
    }
    
    
    __syncthreads();
    reconstruct_at_level_p(current_partition, level, num_vertices,
                                              p_eqlev_first,
                                              buffer);
    
    
    if(threadIdx.x == 0){

        int start   = tc_level_top[blockIdx.x];   // stack start level for this block
        int top     = *tc_level_reached;          // current stack top (#slots)
        int desired = (int)level - start + 1;     // keep TCs up to & including target level

        if (desired < 0) desired = 0;
        if (desired > top) desired = top;

        // printf("[blk %u] backtrack: cur_level=%u, stack_top=%d, target=%u, start=%d -> new_top=%d\n",
        //     (unsigned)blockIdx.x,
        //     (unsigned)current_partition.level,
        //     top,
        //     (unsigned)level,
        //     start,
        //     desired);

        *tc_level_reached = desired;

        
        current_partition.current_vertex_sequence_size = level - 1;

        
        current_partition.level = level;

    }

    
    __syncthreads();
    
}



__device__ int verify_vertex_sequence_orbit_reps(
    unsigned int* orbit,                    
    volatile int* found_automorphisms,  
    int           num_aut,
    volatile unsigned int* current_vertex_sequence,
    int           sequence_size,
    unsigned int  num_vertices, 
    int*          max_aut,
    int           last_num_aut_at_level,
    unsigned int* dsu_array
) {
    
    DSU_dev dsu;
    dsu.bind(num_vertices, dsu_array);
    
    if(last_num_aut_at_level < 0){
        last_num_aut_at_level = 0;
    }
    
    for (int i = 0; i < sequence_size; ++i) {
        
        for (unsigned int v = threadIdx.x; v < num_vertices; v += blockDim.x)
            dsu_array[v] = v;
        __syncthreads();
        
        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("[DEBUG] max_aut = %d, num_aut = %d, num_aut = %d\n", *max_aut, *num_aut, num_aut);
        // }
        // if (threadIdx.x == 0) {
        //     printf("[Block %u] >> subgroup_fixing_sequence_p: num_aut = %d, sequence_size = %d\n",
        //            blockIdx.x, num_aut, sequence_size);
        // }
   
        for (int ii = threadIdx.x + last_num_aut_at_level ; ii < num_aut; ii += blockDim.x) {
            
            
            while (found_automorphisms[ii * num_vertices + num_vertices - 1] == -1) { __nanosleep(100);    }
            
           
            bool fixes_all = true;
            volatile int* perm = &found_automorphisms[ii * num_vertices];
            
            for (int j = 0; j < i; ++j) {
                if (perm[current_vertex_sequence[j]] != current_vertex_sequence[j]) {
                    fixes_all = false;
                    break;
                }
            }
            if (fixes_all) {
                 for (unsigned int v = 0; v < num_vertices; v ++) {
                    unsigned int img = perm[v];
                    
                    
                    if (v != img) dsu.unite_atomic(v, img);
                }
                
            }
        }
        __syncthreads();


        // Now check if the last vertex in prefix is rep of its orbit
       unsigned int v = current_vertex_sequence[i];

        __shared__ int is_rep;
        if (threadIdx.x == 0) {
           
            const unsigned int rep_v  = dsu.rep(v);
            if(rep_v != v){
                is_rep = 0;
                // printf("Blk %u: backtracking to %u, orbit rep is %u, my vertex is %u \n",
                //     blockIdx.x, i + 1, rep_v, v);
            }
            else{
                is_rep = 1;
            }
            
        }
        __syncthreads();

        if (!is_rep) {
            return i + 1;  // first invalid vertex => backtrack to level i
        }
    }

    return sequence_size + 2;  // all vertices orbit representatives
}




__device__ void compute_orbits_fast_p(
    unsigned int* orbit,                
    volatile int* found_automorphisms,  
    int           num_aut,              
    volatile unsigned int* current_vertex_sequence,
    int           sequence_size,
    unsigned int  num_vertices,
    volatile int*          target_cell,          
    unsigned int  target_cell_length,
    int*          max_aut,
    int last_num_aut_at_level,
    unsigned int* dsu_array
    
) {

   
    DSU_dev dsu;
    dsu.bind(num_vertices, dsu_array);

    // init parent[i] = i in parallel
    for (unsigned int v = threadIdx.x; v < num_vertices; v += blockDim.x){
        if(target_cell[v] != -1){
            dsu.parent[v] = target_cell[v];
        }
        else{
            dsu.parent[v] = v;
        }
    }
        
    __syncthreads();

    
    for (int ii = threadIdx.x + last_num_aut_at_level ; ii < num_aut; ii += blockDim.x) {
            
        while (found_automorphisms[ii * num_vertices + num_vertices - 1] == -1) { 
            __nanosleep(100); 
         }

        
    
        bool fixes_all = true;
        volatile int* perm = &found_automorphisms[ii * num_vertices];
        for (int j = 0; j < sequence_size; ++j) {
            if (perm[current_vertex_sequence[j]] != current_vertex_sequence[j]) {
                fixes_all = false;
                break;
            }
        }
        if (fixes_all) {
            for (unsigned int v = 0; v < num_vertices; v ++) {
                
                unsigned int img = perm[v];
                //printf("vertex %u mapped to %u\n",v,img);
                if (v != img) dsu.unite_atomic(v, img);
            }
            
        }
    }
    __syncthreads();

    for (unsigned int v = threadIdx.x; v < num_vertices; v += blockDim.x){
        if(target_cell[v] != -1){
            target_cell[v] = dsu.rep((unsigned int)v);
        }
        
    }
    
   
    __syncthreads();


}




__device__ void prune_stack_entry(Partition& current_partition, volatile TargetCell* stack, 
    volatile int* stack_top, int n, 
    volatile int*   d_found_automorphisms,
    volatile int*            d_num_aut,          
    int             max_aut,      
    unsigned int*   current_orbits,
    unsigned int*   dsu_array,
    Counters* counters
    ){
    
    volatile TargetCell& tc = stack[(*stack_top) - 1];

    // PRUNE if new automorphisms
    __shared__ int curr_num_aut ;
    if(threadIdx.x == 0){
        curr_num_aut = *d_num_aut;
    }
    __syncthreads();
    if (curr_num_aut > 0 && curr_num_aut > tc.last_num_aut_at_level)
    {
        // if (threadIdx.x == 0) {
        //     printf("[blk %u] other_path_node_p: PRUNE begin  curr_num_aut=%d  last_at_lvl=%d  tc.counter=%u len=%u\n",
        //            blockIdx.x, curr_num_aut, tc.last_num_aut_at_level, tc.counter, tc.length);
        // }
        //startTime(COMPUTE_ORBITS_FAST, counters);
        compute_orbits_fast_p(
            current_orbits,
            d_found_automorphisms,
            curr_num_aut,
            current_partition.current_vertex_sequence,
            current_partition.current_vertex_sequence_size,
            n,
            tc.elems,
            (unsigned)tc.length,
            &max_aut,
            tc.last_num_aut_at_level,
            dsu_array
        );

        //endTime(COMPUTE_ORBITS_FAST, counters);
        __threadfence();

        if(threadIdx.x == 0){
            tc.last_num_aut_at_level = curr_num_aut;
        }
        __syncthreads();
        

        
    }


    // Find next index >= tc.counter with tc.elems[i] == 2, in parallel
    __shared__ unsigned int next2;
    if (threadIdx.x == 0) next2 = (unsigned)n;
    __syncthreads();

    unsigned int start = (unsigned)tc.counter;


    for (unsigned int i = start + (unsigned)threadIdx.x; i < (unsigned)n; i += (unsigned)blockDim.x) {
        if (tc.elems[i] == i) {
            atomicMin(&next2, i);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        tc.counter = (int)next2; 
    }
    __syncthreads();
    

    
}

__device__ void classify_node(
    Partition&       part,
    CSRGraph&         graph,
    unsigned          n,
    unsigned int* p_eqlev_first,
    volatile int*     d_found_automorphisms,
    volatile int*              d_num_aut,
    int*               max_aut,
    unsigned int*     tmp_perm,
    volatile bool*             tmp_hash,
    Leaf*            d_first_leaf,
    CanonicalLabeling*            canonical_labeling,
    int*              my_canon_code,
    Cell*      buffer,
    volatile int*              stack_top,
    int* tc_level_top,
    unsigned int*   dsu_array,
    volatile int* last_num_aut,
    unsigned int*  current_orbits ,
    int* hash_lock,
    Counters* blockCounters

){


    startTime(CLASSIFY_NODE,blockCounters);
    int is_disc = is_discrete_p(part);
    

    
        __shared__ int decision;  
        __shared__ int is_aut_first_leaf;

        if(threadIdx.x == 0){
            is_aut_first_leaf = 0;
            decision = 0;
        }
        __syncthreads();

        if ((*p_eqlev_first) == part.level && is_disc) {
            // automorphism[first_leaf.label[j]] = part.element_vec[j]
            for (unsigned j = threadIdx.x; j < n; j += blockDim.x) {
                unsigned u = d_first_leaf->label[j];
                unsigned v = part.element_vec[j];
                tmp_perm[u] = v;
            }
            __syncthreads();
            
            
            int is_aut = isAutomorphism_block_coop(graph, tmp_perm) ? 1 : 0;
            if (threadIdx.x == 0) {
                is_aut_first_leaf = is_aut;
            }
            __syncthreads();

            
        }
        __shared__ unsigned int raw;
        if(is_aut_first_leaf == 1){

            if (threadIdx.x == 0) {
                raw =  atomicAdd((int*)d_num_aut, 1);
                //printf("blockid: %u level: %u found aut wrt first leaf at index %u and root part level %u\n",blockIdx.x, part.level, raw, tc_level_top[blockIdx.x]);
                
            }
            __syncthreads();
            for (unsigned i = threadIdx.x; i < n - 1; i += blockDim.x){
                d_found_automorphisms[raw * n + i] = tmp_perm[i];
                
            }
            __threadfence();
            __syncthreads();

            if(threadIdx.x == 0){
                atomicExch((int*)&d_found_automorphisms[raw * n + n - 1] , tmp_perm[n - 1]);
                __threadfence();
            }

            

            int g = get_gca_level_p(d_first_leaf->vertex_sequence, d_first_leaf->vertex_sequence_size,
                                    part.current_vertex_sequence, part.current_vertex_sequence_size);
            
            if (g < 0) g = (int)part.level - 1;

            backtrack_to_p(part, (unsigned)g, n, buffer,
                        p_eqlev_first,  stack_top,tc_level_top);
            
            endTime(CLASSIFY_NODE,blockCounters);
            
            return;
        }
        endTime(CLASSIFY_NODE,blockCounters);
        //AQUIRE LOCK to best leaf

        startTime(WAIT_BEST_LEAF_LOCK,blockCounters);
        if (threadIdx.x == 0) {
            while (atomicCAS((int*)&canonical_labeling->lock, 0, 1) != 0) { /* spin */ }
        }
        __threadfence();
        __syncthreads();
        endTime(WAIT_BEST_LEAF_LOCK,blockCounters);
        
        startTime(CLASSIFY_NODE,blockCounters);
        __shared__ int pos;
        
        int lex = lex_compare_codes(my_canon_code, part.level, canonical_labeling->invar_path_value, canonical_labeling->level, &pos);

        // DEBUG: show lex inputs/outputs
        // if (threadIdx.x == 0) {
        //     printf("[blk %u] L%u lex=%d pos=%d | my_len=%u best_len=%d\n",
        //         (unsigned)blockIdx.x, (unsigned)part.level, lex, pos,
        //         (unsigned)part.level, *canon_code_level);

        //     // my path
        //     printf("  my:   ");
        //     for (int i = 0; i < (int)part.level; ++i) printf("%d ", my_canon_code[i]);
        //     printf("\n");

        //     // best path
        //     printf("  best: ");
        //     for (int i = 0; i < *canon_code_level; ++i) printf("%d ", d_canon_code[i]);
        //     printf("\n");
        // }


        // decision = 0 : proceed normally with node
        // decision = 1 : update global best , and it will be a full path (update hash)
        // decision = 2 : update global best, but it will not be a full path (no update on hash)
        // decision = 3 : backtrack one level up.
        // decision = 4 : backtrack to "pos".

        if (lex > 0) {
            if(threadIdx.x == 0){
                decision = 1;     // or 2 doesnt matter
            }
                
                
            
        } else if (lex == 0) {
            // 2) equal canon trails → shorter level wins
            if (part.level <  canonical_labeling->level) {

                if(is_disc){  //must later check if canon code is a complete path or not
                    if(threadIdx.x == 0){
                        decision = 1;
                    }
                }
                else{
                    if(threadIdx.x == 0){
                        decision = 0;  // cant take a decision yet. Procced with this path normally
                    }
                }
                
                
            } 
            else if (part.level ==  canonical_labeling->level) {
                // 3) still equal → compare hashes (upper-triangular)
                __syncthreads();
            
                
                if((canonical_labeling->is_leaf)== 1 && is_disc){
                    discrete_partition_to_perm_p(part, n, tmp_perm);

                    if(threadIdx.x == 0){
                        while (atomicCAS(hash_lock, 0, 1) != 0) { /* spin */ }
                    }
                    __threadfence();
                    __syncthreads();
                    buildPermutedFlatAdjacencyMatrixArray_p(graph, tmp_perm, tmp_hash);
                    int cmp = compare_adj_upper(tmp_hash, canonical_labeling->hash_of_perm_graph, n);
                    
                    if (threadIdx.x == 0){
                        atomicExch(hash_lock, 0);
                    } 
                    __threadfence();

                    if(cmp == 0){

                        //automorphism discovered w.r.t current best leaf
                

                        // if(threadIdx.x == 0){
                        //     printf("block %u discovred aut at level %u\n",blockIdx.x,part.level);
                        //     printf("current path vertex sequence:\n");
                        //     for(int i = 0; i < part.current_vertex_sequence_size; i++){
                        //         printf("%u ,",part.current_vertex_sequence[i]);
                        //     }
                        //     printf("\n");
                        //     printf("best path vertex sequence:\n");
                        //     for(int i = 0; i < part.current_vertex_sequence_size; i++){
                        //         printf("%u ,",d_best_leaf->vertex_sequence[i]);
                        //     }
                        //     printf("\n");

                        // }
                        
                        for (unsigned j = threadIdx.x; j < n; j += blockDim.x) {
                            unsigned u = canonical_labeling->label[j];
                            unsigned v = part.element_vec[j];
                            tmp_perm[u] = v;
                        }
                        __syncthreads();

                        
                        if (threadIdx.x == 0) {
                            
                            raw =  atomicAdd((int*)d_num_aut, 1);
                            //printf("blockid: %u level: %u found aut wrt best leaf at index %u and root part level %u \n",blockIdx.x, part.level, raw, tc_level_top[blockIdx.x]);
                        }
                        

                        __syncthreads();
                       for (unsigned i = threadIdx.x; i < n - 1; i += blockDim.x){
                            d_found_automorphisms[raw * n + i] = tmp_perm[i];
                            
                        }
                        __threadfence();
                        __syncthreads();

                        if(threadIdx.x == 0){
                            atomicExch((int*)&d_found_automorphisms[raw * n + n - 1] , tmp_perm[n-1]);
                            __threadfence();
                        }
                           
                        if (threadIdx.x == 0){
                    
                            atomicExch((int*)&canonical_labeling->lock, 0);
                        } 
                        __threadfence();

                        
                        int g = (int)part.level - 1;
                        backtrack_to_p(part, (unsigned)g, n, buffer,
                                    p_eqlev_first, stack_top,tc_level_top);
                        
                        __syncthreads();
                        
                        
                        endTime(CLASSIFY_NODE,blockCounters);
                        return;

                    }
                    else if(cmp == -1){
                        if(threadIdx.x == 0){
                            //printf("blk %u equal to canon best path but lex hash worse, backtracking by one\n",blockIdx.x);
                            decision = 3;
                        }
                    }
                    else{
                        if(threadIdx.x == 0){
                            decision = 1;
                        }
                    }


                }

                else if((canonical_labeling->is_leaf)== 1 && !is_disc){
                    
                    if(threadIdx.x == 0){
                        //printf("blk %u equal to canon best path but not discrete and best path is discrete, backtracking by one\n",blockIdx.x);
                        decision = 3; 
                    }

                    
                }

                else if ((canonical_labeling->is_leaf)== 0 && is_disc){

                    if(threadIdx.x == 0){
                        decision = 1; 
                    }

                }
            
            }
            else{

                if((canonical_labeling->is_leaf)== 1){
                    if(threadIdx.x == 0){
                        //printf("blk %u equal to canon best path but longer than best path, backtracking by one\n",blockIdx.x);
                        decision = 3;
                    }
                }

                else if ((canonical_labeling->is_leaf)== 0){
                    if (threadIdx.x == 0){
                        decision = 1;
                    }
                    
                }
                
            }
        }
        else{

            if(threadIdx.x == 0){
                
                decision = 4; // backtrack up to pos.
            }
        }
        
        __syncthreads();

        if(decision == 0){
            
            if (threadIdx.x == 0) atomicExch((int*)&canonical_labeling->lock, 0);
            //__syncthreads();
            endTime(CLASSIFY_NODE,blockCounters);
            
            return;
        }
        else if(decision == 1 || decision == 2){
        
            // overwrite global canon trail and level
            
            for (int i = threadIdx.x; i < part.level; i+= blockDim.x)
                canonical_labeling->invar_path_value[i] = my_canon_code[i];
            if (threadIdx.x == 0) {
                
                 canonical_labeling->level = part.level;

                
            }
            // if(threadIdx.x == 0){
            //   printf("blockID %u is overrideing best path \n", blockIdx.x);
            // }  

            if(is_disc){
                discrete_partition_to_perm_p(part, n, tmp_perm);
                 if(threadIdx.x == 0){
                    while (atomicCAS(hash_lock, 0, 1) != 0) { /* spin */ }
                }
                __threadfence();
                __syncthreads();

                buildPermutedFlatAdjacencyMatrixArray_p(graph, tmp_perm, tmp_hash);
                const unsigned nn = n * n;
                for (unsigned idx = threadIdx.x; idx < nn; idx += blockDim.x)
                    canonical_labeling->hash_of_perm_graph[idx] = tmp_hash[idx];
                
                __syncthreads();
                if (threadIdx.x == 0){
                    atomicExch(hash_lock, 0);
                } 
                __threadfence();
                

                for (unsigned j = threadIdx.x; j < n; j += blockDim.x)
                    canonical_labeling->label[j] = part.element_vec[j];
                
                if (threadIdx.x == 0) canonical_labeling->vertex_sequence_size = part.current_vertex_sequence_size;
                
                for (unsigned i = threadIdx.x; i < part.current_vertex_sequence_size; i += blockDim.x)
                    canonical_labeling->vertex_sequence[i] = part.current_vertex_sequence[i];

                
                if(threadIdx.x == 0){
                    canonical_labeling->is_leaf = 1; 
                    //printf("is canon full path = 1\n");
                    canonical_labeling->invar_path_value[part.level] = 077777;
                    atomicExch((int*)&canonical_labeling->lock, 0);
                   
                }

                __syncthreads();
                

                backtrack_to_p(part, part.level - 1, n, buffer,
                            p_eqlev_first, stack_top,tc_level_top);

            
                __threadfence();
                endTime(CLASSIFY_NODE,blockCounters);
                return;

            }
            else{
                if(threadIdx.x == 0){
                    canonical_labeling->is_leaf = 0; 
                    //printf("is canon full path = 0\n");
                    atomicExch((int*)&canonical_labeling->lock, 0);
                    
                    __threadfence();

                    
                }
                __syncthreads();
                endTime(CLASSIFY_NODE,blockCounters);
                return;
            }

        }
    
        else if(decision == 3){
            // check if eqlevfirst = part.level, if not, backtrack by one.

            int backtrack_level = max(*p_eqlev_first, (canonical_labeling->level) - 1);
            //int backtrack_level = part.level - 1;
            __syncthreads();
            
            if (threadIdx.x == 0) atomicExch((int*)&canonical_labeling->lock, 0);
            
            // if(threadIdx.x == 0){
            //     printf("value at level %u is %u\n",part.level, backtrack_level);
            // }
            if(backtrack_level == part.level && is_disc){
                backtrack_level--;
            }
            

            backtrack_to_p(part,backtrack_level, n, buffer,
                        p_eqlev_first,  stack_top,tc_level_top);
            

            //__syncthreads();
            endTime(CLASSIFY_NODE,blockCounters);
            return;
            
            
        }

        else if(decision == 4){
            
            
            if (threadIdx.x == 0) atomicExch((int*)&canonical_labeling->lock, 0);
            int backtrack_level = max(*p_eqlev_first, pos);
            //backtrack_level = min(backtrack_level,level);
            
            // if(threadIdx.x == 0){
            //     if(part.level > backtrack_level +1){
            //         printf("blockID %u is no longer on best path, backtracking from %u to %u\n", blockIdx.x, part.level, backtrack_level);
            //     }
            // }

            if ((*p_eqlev_first) != part.level) {

            
                // if(threadIdx.x == 0){
                //     printf("blockID %u is no longer on best path, backtracking to %u\n", blockIdx.x, backtrack_level);
                // }
                __syncthreads();
                backtrack_to_p(part, backtrack_level, n, buffer,
                            p_eqlev_first, stack_top, tc_level_top);
            }
            else{

                if(is_disc){
                    
                    //we might be able to backtrack more here, keep as side note
                    // if(threadIdx.x == 0){
                    //     printf("blockID %u is no longer on best path, backtracking to %u\n", blockIdx.x, part.level - 1);
                    // }
                    backtrack_to_p(part, part.level - 1, n, buffer,
                                p_eqlev_first, stack_top,tc_level_top);

                }
               

            }

            __syncthreads();
            endTime(CLASSIFY_NODE,blockCounters);
            return;
        }

        // release lock
        
        if (threadIdx.x == 0) atomicExch((int*)&canonical_labeling->lock, 0);
        endTime(CLASSIFY_NODE,blockCounters);
        __threadfence();

    
    

}
