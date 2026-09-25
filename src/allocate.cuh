#include "allocate.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

void allocate_leaf_on_device(Leaf** device_leaf, unsigned int numVertices) {
    Leaf temp_leaf;

    cudaMalloc(&(temp_leaf.vertex_sequence), numVertices * sizeof(unsigned int));
    cudaMalloc(&(temp_leaf.label), numVertices * sizeof(unsigned int));
    cudaMalloc(&(temp_leaf.hash_of_perm_graph), numVertices * numVertices * sizeof(bool));
    cudaMalloc(&temp_leaf.invar_path_value,(numVertices + 2) * sizeof(int));


    temp_leaf.vertex_sequence_size = 0;
    temp_leaf.discovered = false;

    cudaMalloc(device_leaf, sizeof(Leaf));
    cudaMemcpy(*device_leaf, &temp_leaf, sizeof(Leaf), cudaMemcpyHostToDevice);
}


void allocate_canonical_labeling_on_device(
    CanonicalLabeling** device_obj,
    unsigned int numVertices
) {
    CanonicalLabeling temp{};

    // internal arrays
    cudaMalloc(&temp.vertex_sequence, numVertices * sizeof(unsigned int));
    cudaMalloc(&temp.label,numVertices * sizeof(int));  // label is int*
    cudaMalloc(&temp.hash_of_perm_graph,numVertices * numVertices * sizeof(bool));
    cudaMalloc(&temp.invar_path_value,(numVertices + 2) * sizeof(int));

    // initialize scalar fields
    temp.vertex_sequence_size = 0;
    temp.discovered = false;
    temp.level = 0;
    temp.is_leaf = 0;
    temp.lock = 0;

   
    // allocate the struct itself on device + copy the host temp struct into it
    cudaMalloc(device_obj, sizeof(CanonicalLabeling));
    cudaMemcpy(*device_obj, &temp, sizeof(CanonicalLabeling),
                          cudaMemcpyHostToDevice);
}

void allocate_partitions_for_blocks(
    Partition** d_partitions,
    unsigned int num_blocks,
    unsigned int numVertices,
    unsigned int max_levels,
    const CSRGraph& graph 
) {
    CUDA_CHECK(cudaMalloc(d_partitions, num_blocks * sizeof(Partition)));

   
    unsigned int* h_element_vec = new unsigned int[numVertices];
    // Worst case: each vertex has its own color => up to numVertices cells
    Cell* h_lcs = new Cell[numVertices + 1];
    unsigned int h_lcs_size = 0;

    // Colored => sort vertices by (color, vertex_id)
    std::vector<std::pair<int, unsigned int>> items;
    items.reserve(numVertices);
    for (unsigned int v = 0; v < numVertices; ++v) {
        items.emplace_back(graph.color[v], v);
    }

    std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                });

    // Fill element_vec with vertices grouped by color
    for (unsigned int i = 0; i < numVertices; ++i) {
        h_element_vec[i] = items[i].second;
    }

    // Build one cell per color block
    unsigned int start = 0;
    while (start < numVertices) {
        int c = items[start].first;
        unsigned int end = start + 1;
        while (end < numVertices && items[end].first == c) ++end;

        h_lcs[h_lcs_size].first    = start;       
        h_lcs[h_lcs_size].length   = end - start;
        h_lcs[h_lcs_size].in_level = 0;
        ++h_lcs_size;

        start = end;
    }

    // Allocate & initialize each block's Partition
    for (unsigned int i = 0; i < num_blocks; ++i) {
        Partition temp_partition;

        // Allocate device memory for each field
        CUDA_CHECK(cudaMalloc(&temp_partition.element_vec, numVertices * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&temp_partition.lcs,        (numVertices + 1) * sizeof(Cell)));
        CUDA_CHECK(cudaMalloc(&temp_partition.current_vertex_sequence, max_levels * sizeof(unsigned int)));

        // Copy element_vec
        cudaMemcpy((void*)temp_partition.element_vec,
                h_element_vec,
                numVertices * sizeof(unsigned int),
                cudaMemcpyHostToDevice);

        // Copy lcs[0..h_lcs_size-1]
        CUDA_CHECK(cudaMemcpy(temp_partition.lcs,
                              h_lcs,
                              h_lcs_size * sizeof(Cell),
                              cudaMemcpyHostToDevice));

        temp_partition.lcs_size = h_lcs_size;
        temp_partition.current_vertex_sequence_size = 0;
        temp_partition.level = 0;

        CUDA_CHECK(cudaMemcpy(&(*d_partitions)[i],
                              &temp_partition,
                              sizeof(Partition),
                              cudaMemcpyHostToDevice));
    }

    delete[] h_element_vec;
    delete[] h_lcs;
}


CSRGraph* allocate_csr_graph_on_device(CSRGraph& graph) {
    
    
    unsigned int* d_dst;
    unsigned int* d_srcPtr;
    int* d_degree;
    int* d_color;

    CUDA_CHECK(cudaMalloc(&d_dst, graph.edgeNum * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_srcPtr, (graph.vertexNum + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_degree, graph.vertexNum * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dst, graph.dst, graph.edgeNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_srcPtr, graph.srcPtr, (graph.vertexNum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_degree, graph.degree, graph.vertexNum * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_color, graph.vertexNum * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_color, graph.color, graph.vertexNum * sizeof(int), cudaMemcpyHostToDevice));
    
    CSRGraph h_graph_device;
    h_graph_device.dst = d_dst;
    h_graph_device.srcPtr = d_srcPtr;
    h_graph_device.degree = d_degree;
    h_graph_device.vertexNum = graph.vertexNum;
    h_graph_device.edgeNum = graph.edgeNum;
    h_graph_device.color     = d_color;    
    
    CSRGraph* d_graph;
    CUDA_CHECK(cudaMalloc(&d_graph, sizeof(CSRGraph)));
    CUDA_CHECK(cudaMemcpy(d_graph, &h_graph_device, sizeof(CSRGraph), cudaMemcpyHostToDevice));

    return d_graph;
}
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
   
){
    // 1) CSR graph
    CSRGraph h_graph;
    cudaMemcpy(&h_graph, d_graph, sizeof(CSRGraph), cudaMemcpyDeviceToHost);
    cudaFree(h_graph.dst);
    cudaFree(h_graph.srcPtr);
    cudaFree(h_graph.degree);
    cudaFree(h_graph.color);
    cudaFree(d_graph);

    // 2) Partitions (free nested device buffers per block)
    Partition* h_partitions = new Partition[num_blocks];
    cudaMemcpy(h_partitions, d_partitions, num_blocks * sizeof(Partition), cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < num_blocks; ++i) {
        cudaFree((void*)h_partitions[i].element_vec);
        cudaFree(h_partitions[i].lcs);
        cudaFree((void*)h_partitions[i].current_vertex_sequence);
    }
    delete[] h_partitions;
    cudaFree(d_partitions);

    // 3) Automorphism storage
    cudaFree(d_found_automorphisms);
    cudaFree(d_num_aut);

    // 4) Leaves
    Leaf h_first;
    CanonicalLabeling canon_lab;
    cudaMemcpy(&h_first, d_first_leaf, sizeof(Leaf), cudaMemcpyDeviceToHost);
    cudaFree((void*) h_first.vertex_sequence);
    cudaFree((void*) h_first.label);
    cudaFree((void*) h_first.hash_of_perm_graph);
    cudaFree((void*) h_first.invar_path_value);
    cudaFree(d_first_leaf);

    cudaMemcpy(&canon_lab, canonical_labeling, sizeof(CanonicalLabeling), cudaMemcpyDeviceToHost);
    cudaFree((void*) canon_lab.vertex_sequence);
    cudaFree((void*) canon_lab.label);
    cudaFree((void*) canon_lab.hash_of_perm_graph);
    cudaFree((void*) canon_lab.invar_path_value);
    cudaFree(canonical_labeling);

    // 5) Work buffers
    cudaFree(cell_buffers);
    cudaFree(aux_arrays);
    

    // 6) Other device buffers
    cudaFree(d_hash_val);
    cudaFree(d_node_counter);
    cudaFree(d_max_aut);

    // 7) per-block/per-level state
    
    cudaFree(d_eqlevfirst);
    
    // TargetCell structures and their elems pool + counters
    cudaFree(d_target_cells);
    cudaFree(d_tc_level_reached);
    cudaFree(d_tc_elems_pool);
    cudaFree(d_canon_code_per_block);

   
}

