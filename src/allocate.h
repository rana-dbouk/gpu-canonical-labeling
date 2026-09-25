#ifndef ALLOCATE_H
#define ALLOCATE_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "structs.cuh"
#include <vector>
#include "CSRGraph.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort);


void allocate_leaf_on_device(Leaf** device_leaf, unsigned int numVertices);

void allocate_partitions_for_blocks(Partition** d_partitions, unsigned int num_blocks, unsigned int numVertices, unsigned int max_levels, const CSRGraph& graph);



CSRGraph* allocate_csr_graph_on_device(CSRGraph& graph);
void cleanup_all(
    Partition* d_partitions, unsigned int num_blocks,
    CSRGraph* d_graph,
    int* d_found_automorphisms,
    int* d_num_aut,
    Leaf* d_first_leaf,
    CanonicalLabeling* canonical_labeling,
    Cell* cell_buffers,
    unsigned int* aux_arrays,
    bool* d_hash_val,
    int* d_node_counter,
    int* d_max_aut,
    unsigned int* d_eqlevfirst,
    TargetCell*   d_target_cells,
    int*          d_tc_level_reached,
    int*          d_tc_elems_pool,
    int*          d_canon_code_per_block 
    
);

#endif // ALLOCATE_H
