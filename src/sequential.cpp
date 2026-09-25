#include "sequential.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip> 
#include <chrono>
#include <iostream>
#include <climits>
#include <stdexcept>
#include <string>
static const int PRUNED_MARK = -1;

Nauty_Sequential::Nauty_Sequential(CSRGraph& input_graph, unsigned int max_automorphisms)
    : graph(input_graph), max_aut(max_automorphisms), num_aut(0)
{

    num_vertices = graph.getVertexNum();
    found_automorphisms = new unsigned int[max_aut * num_vertices];
    canon_code = new int[num_vertices+2];
    //max_invar_at_level_sizes = new unsigned int[num_vertices];
    firstcode = new int[num_vertices+2];
    // firstcode_sizes = new unsigned int[num_vertices];
    first_path_tc = new int[num_vertices+1];
    for (unsigned int i = 0; i <= num_vertices; i++) {
        first_path_tc[i] = -1;  
    }
    eqlevfirst = 0;
    canupdates=0;
    //max_invar_at_level_size=0;
    first_leaf.init(num_vertices);
    best_leaf.init(num_vertices);

    auto start = std::chrono::high_resolution_clock::now();

    initializePartition_s(current_partition, graph);

    //printPartition(current_partition);
    
    target_cell_at_level = new TargetCell[graph.getVertexNum()+1];

    for (unsigned i = 0; i <= num_vertices; ++i) {
        target_cell_at_level[i].elems   = new int[num_vertices];
        target_cell_at_level[i].length  = 0;
        target_cell_at_level[i].counter = 0;
        target_cell_at_level[i].last_num_aut_at_level = 0;
    }
    tc_level_reached = 0;
    
    search_tree_traversal();

    

    
    //print_canonical_labeling(best_leaf.hash_of_perm_graph,num_vertices) ;


    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    auto duration_min = std::chrono::duration_cast<std::chrono::minutes>(duration_ns);
    auto remaining_ns_after_min = duration_ns - duration_min;

    auto duration_s = std::chrono::duration_cast<std::chrono::seconds>(remaining_ns_after_min);
    auto remaining_ns_after_s = remaining_ns_after_min - duration_s;

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining_ns_after_s);
    auto remaining_ns_after_ms = remaining_ns_after_s - duration_ms;

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(remaining_ns_after_ms);
    auto remaining_ns_after_us = remaining_ns_after_ms - duration_us;
    
    // Print the runtime in min, s, ms, µs, ns
    std::cout << "Algorithm ran in: "
            << duration_min.count() << "min, "
            << duration_s.count() << "s, "
            << duration_ms.count() << "ms, "
            << duration_us.count() << "µs, "
            << remaining_ns_after_us.count() << "ns\n";




    std::cout << "Total automorphisms discovered (sequential): " << num_aut << std::endl;

    std::cout << "Total search tree nodes created (sequential): " << node_counter << std::endl;


}


Nauty_Sequential::~Nauty_Sequential() {
    // std::cout << "Destructor called\n";
    delete[] found_automorphisms;
    delete[] canon_code;
    delete[] firstcode;
    delete[] first_path_tc;
    first_leaf.cleanup();
    best_leaf.cleanup();
    current_partition.cleanup();
    graph.cleanup();
}

void Nauty_Sequential::print_canonical_labeling(const bool* hash_of_perm_graph, unsigned int num_vertices) {
    std::cout << "Printing canonical labeling\n";
    for (unsigned int i = 0; i < num_vertices; ++i) {
        for (unsigned int j = 0; j < num_vertices; ++j) {
            std::cout << hash_of_perm_graph[i * num_vertices + j] << " ";
        }
        std::cout << "\n";
    }
}

// A and B are flat n×n row-major boolean matrices: A[i*n + j]
int Nauty_Sequential::compare_adj_upper(const bool* A, const bool* B, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        const bool* rowA = A + i * n;
        const bool* rowB = B + i * n;
        for (unsigned j = i + 1; j < n; ++j) {
            bool a = rowA[j], b = rowB[j];
            if (a != b) return a ? 1 : -1;
        }
    }
    return 0;
}


bool Nauty_Sequential::isAutomorphism(CSRGraph& graph, unsigned int* perm) {
    unsigned int vertexNum = graph.vertexNum;

    // quick degree check
    for (unsigned int u = 0; u < vertexNum; ++u)
    {
        unsigned int pu = perm[u];
        unsigned int deg_u  = graph.srcPtr[u + 1]  - graph.srcPtr[u];
        unsigned int deg_pu = graph.srcPtr[pu + 1] - graph.srcPtr[pu];
        if (deg_u != deg_pu) return false;
    }

    for (unsigned int u = 0; u < vertexNum; ++u) {
        for (unsigned int v = 0; v < vertexNum; ++v) {
            bool original_adj = false;
            bool permuted_adj = false;

            // Check if edge (u,v) exists
            for (unsigned int j = graph.srcPtr[u]; j < graph.srcPtr[u + 1]; ++j) {
                if (graph.dst[j] == v) {
                    original_adj = true;
                    break;
                }
            }

            // Check if edge (perm[u], perm[v]) exists
            unsigned int pu = perm[u], pv = perm[v];
            for (unsigned int j = graph.srcPtr[pu]; j < graph.srcPtr[pu + 1]; ++j) {
                if (graph.dst[j] == pv) {
                    permuted_adj = true;
                    break;
                }
            }

            if (original_adj != permuted_adj) {
                return false;
            }
        }
    }
    return true;
}


void Nauty_Sequential::write_automorphism(CSRGraph& graph, unsigned int* perm) {
    unsigned int n = graph.vertexNum;
    for (unsigned int u = 0; u < n; ++u) {
        unsigned int v = perm[u];
        if (u == v) continue;                 
        if (perm[v] == u && v < u) continue;  
        std::cout << '(' << u << ' ' << v << ')';
    }
    std::cout << '\n';
}



int Nauty_Sequential::get_gca_level(unsigned int* first_sequence, unsigned int first_size,
                                             unsigned int* second_sequence, unsigned int second_size) {
    // [DEBUG]
    const bool DBG = false;
    if (DBG) {
        std::cout << "[DEBUG][get_gca_level] first_size=" << first_size
                  << " second_size=" << second_size << "\n";
        std::cout << "[DEBUG][get_gca_level] first seq:  ";
        for (unsigned int i = 0; i < first_size; ++i) std::cout << first_sequence[i] << (i+1<first_size?' ':'\n');
        std::cout << "[DEBUG][get_gca_level] second seq: ";
        for (unsigned int i = 0; i < second_size; ++i) std::cout << second_sequence[i] << (i+1<second_size?' ':'\n');
    }

    int max = (first_size < second_size) ? first_size : second_size;
    for (int same_until = 0; same_until < max; ++same_until) {
        if (first_sequence[same_until] != second_sequence[same_until]) {
            // std::cout << "Mismatch at index " << same_until << " ("
            //           << first_sequence[same_until] << " != " << second_sequence[same_until] << ")\n";
            // std::cout << "Returning GCA level: " << same_until + 1 << "\n";

            if (DBG) {
                std::cout << "[DEBUG][get_gca_level] mismatch at idx=" << same_until
                          << " -> GCA level=" << (same_until + 1) << "\n";
            }
            return same_until+ 1; //+ 1 because levels in tree are ahead sequence length by 1 (root is at level 1)
        }
    }

    if (DBG) std::cout << "[DEBUG][get_gca_level] sequences share full common prefix -> return -1\n";
    return -1;
}


void Nauty_Sequential::first_path_node() {
    //printPartition(current_partition);
    // [DEBUG]
    const bool DBG = false;

    int index;
    if(!is_discrete(current_partition)){
        index = target_cell_selector(current_partition,-1,target_cell_at_level,tc_level_reached, num_vertices);
        first_path_tc[current_partition.level] = index;
       
    }
    else{
        first_path_tc[current_partition.level] = -1;
        first_terminal();
        backtrack_to(current_partition.level-1); 
        return;
    }

    TargetCell& tc = target_cell_at_level[tc_level_reached-1];

    while (tc.counter < num_vertices && tc.elems[tc.counter] != tc.counter)
        ++tc.counter;


    int vertex = tc.counter++;
    
    //new tree node
    ++node_counter;
    //printPartition(current_partition);
    
    //std::cout << "Branching on vertex " << vertex << " at level " << current_partition.level << "\n";
    
    current_partition.current_vertex_sequence[current_partition.current_vertex_sequence_size++] = vertex;

    int level_before = current_partition.level;
    split_by_and_refine(current_partition, graph, vertex,code, tc.first);

    //printPartition(current_partition);
    firstcode[current_partition.level-1] = code;
    refinements_made++;

}

void Nauty_Sequential::first_terminal() {
    //printPartition(current_partition);
    // std::cout << "First leaf discovered\n";
    // [DEBUG]
    const bool DBG = false;
    if (DBG) std::cout << "\n[DEBUG][first_terminal] ENTER at level=" << current_partition.level << "\n";

    max_level = current_partition.level;
    eqlevfirst = current_partition.level;
    firstcode[current_partition.level] = 077777;
    unsigned int* leaf_perm = discrete_partition_to_perm(current_partition, num_vertices);
   
    bool* hash_val = graph.buildPermutedFlatAdjacencyMatrixArray(leaf_perm);
            
    for (unsigned int i = 0; i < num_vertices * num_vertices; ++i)
        best_leaf.hash_of_perm_graph[i] = first_leaf.hash_of_perm_graph[i] = hash_val[i];

    delete[] hash_val;

    // Copy vertex sequence
    first_leaf.vertex_sequence_size = current_partition.current_vertex_sequence_size;
    for (unsigned int i = 0; i < current_partition.current_vertex_sequence_size; ++i)
        first_leaf.vertex_sequence[i] = current_partition.current_vertex_sequence[i];

    // Copy everything to best_leaf too
   
    best_leaf.vertex_sequence_size = first_leaf.vertex_sequence_size;
    for (unsigned int i = 0; i < best_leaf.vertex_sequence_size; ++i)
        best_leaf.vertex_sequence[i] = first_leaf.vertex_sequence[i];

    for(int j=0;j<num_vertices;j++){
        first_leaf.label[j] = best_leaf.label[j] = current_partition.element_vec[j];
    }


    first_leaf.discovered = true;
    delete[] leaf_perm;
    canon_level = eqlevcanon = current_partition.level;
    comp_canon = 0;
    
    for (int i = 0; i < current_partition.level; ++i) canon_code[i] = firstcode[i];
    canon_code[current_partition.level] = 077777;
    canupdates = 1;

}
void Nauty_Sequential::other_path_node() {

    
    const bool DBG = false;
    //  Initialize recorded target cell at this level if needed 
    //printPartition(current_partition);

    if (!is_discrete(current_partition)
        && ((eqlevfirst == current_partition.level || comp_canon >= 0)))
    {

        // Ensure storage for this level
        if (tc_level_reached < current_partition.level){
            target_cell_selector(current_partition, -1, target_cell_at_level,tc_level_reached,num_vertices);
            
        }

        TargetCell& tc = target_cell_at_level[current_partition.level-1];
        
        if (tc.counter >= num_vertices)
        {
            backtrack_to(current_partition.level - 1); 
            return; 
        }


        // prune only if automorphisms increased
        if (num_aut > 0 && num_aut > tc.last_num_aut_at_level )
        {
            // fprintf(stderr,
            // "[PRUNE] entering compute_orbits: L=%u num_aut=%d last_at_lvl=%d tc.len=%u tc.counter=%u\n",
            // current_partition.level, num_aut, tc.last_num_aut_at_level, tc.length, tc.counter);

            auto __t0 = std::chrono::steady_clock::now();
            compute_orbits_fast(
                found_automorphisms, num_aut,
                current_partition.current_vertex_sequence,
                current_partition.current_vertex_sequence_size,
                num_vertices,
                tc.elems,     // subset we're working with
                tc.length,
                max_aut,
                tc.last_num_aut_at_level
            );
            auto __t1 = std::chrono::steady_clock::now();
            compute_orbits_ns_total +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(__t1 - __t0).count();

           
            tc.last_num_aut_at_level = num_aut;

            
            
        }

        while (tc.counter < num_vertices && tc.elems[tc.counter] != tc.counter)
            ++tc.counter;

        if (tc.counter >= num_vertices) {
            backtrack_to(current_partition.level - 1);
            return;
        }
        
        int vertex = tc.counter++;
        

        // New tree node
        ++node_counter;
        
        //std::cout << "Branching on vertex " << vertex << " at level " << current_partition.level << "\n";
        current_partition.current_vertex_sequence[current_partition.current_vertex_sequence_size++] = vertex;

        int level_before = current_partition.level;
        split_by_and_refine(current_partition, graph, vertex, code,tc.first);
        // std::cout << "[DEBUG][refine] code=" << code
        //           << " after split_by_and_refine; level=" << current_partition.level
        //           << " vertex=" << vertex << "\n";

        
        if (DBG) {
            std::cout << "[DEBUG][other_path_node] split_by_and_refine returned code="
                    << code << " level(before)=" << level_before
                    << " level(after)=" << current_partition.level << "\n";
        }
    }

    //printPartition(current_partition);
    classify_node();
}



void Nauty_Sequential::classify_node(){

    if (eqlevfirst == current_partition.level - 1 && code == firstcode[current_partition.level - 1]) {
        eqlevfirst = current_partition.level;
    }
    
    if (eqlevcanon == current_partition.level - 1) {
        if (code < canon_code[current_partition.level - 1]) {
            comp_canon = -1;
        } else if (code > canon_code[current_partition.level - 1]) {
            // std::printf("[DEBUG][canon][CPU] L=%u node_code=%d canon_code[%u]=%d  -> comp_canon=+1\n",
            //             current_partition.level,
            //             code,
            //             current_partition.level - 1,
            //             canon_code[current_partition.level - 1]);
            comp_canon = 1;

        } else {
            comp_canon = 0;
            eqlevcanon = current_partition.level;
        }
    }
    if (comp_canon > 0) {
        //std::cout << "[DEBUG][canon] canon_code[" << (current_partition.level - 1) << "] = "<< canon_code[current_partition.level - 1] << "\n";
        //std::cout << "[DEBUG][other_path_node] comp_canon>0 -> canon_code[level]:=" << code << "\n";
        canon_code[current_partition.level - 1] = code;
    }

    int i,code; 

    // [DEBUG]
    const bool DBG = false;
    if (DBG) {
        std::cout << "\n[DEBUG][classify_node] ENTER: level=" << current_partition.level
                << " is_discrete=" << is_discrete(current_partition)
                << " eqlevfirst=" << eqlevfirst
                << " comp_canon=" << comp_canon
                << " canon_level=" << canon_level
                << "\n";
    }

    code = 0;
    if (eqlevfirst != current_partition.level && comp_canon < 0){
        code = 4;
        //printPartition(current_partition);
    }
    
    
    else if (is_discrete(current_partition))
    {
        if (eqlevfirst == current_partition.level)
        {
            

            unsigned int* automorphism = new unsigned int[num_vertices]();

            for (int j = 0; j < (int)num_vertices; j++) {
                automorphism[first_leaf.label[j]] = current_partition.element_vec[j];
            }
            if(isAutomorphism(graph,automorphism)==true){
                code = 1; //automorphism discovered, equivalent to first leaf
            }
            delete[] automorphism;
        }
        if (code == 0)
        {
            if (comp_canon == 0)
            {
                if (current_partition.level < canon_level)
                
                    comp_canon = 1; //shorter path is solution
                else
                {
                    
                    unsigned int* curr_perm = discrete_partition_to_perm(current_partition, num_vertices);
                    bool* curr_hash = graph.buildPermutedFlatAdjacencyMatrixArray(curr_perm);
                    //break tie using permuted graph's adjacency matrix
                    comp_canon = compare_adj_upper(curr_hash, best_leaf.hash_of_perm_graph, num_vertices);
                    //std::cout << "compare_hashes -> comp_canon=" << comp_canon << "\n";
                
                    delete[] curr_hash;
                    delete[] curr_perm;
                }
            }
            if (comp_canon == 0)
            {
                code = 2;
            }
            else if (comp_canon > 0)
                code = 3;
            else
                code = 4;
        }
    }

    // std::printf("[DEBUG][classify_node][CPU] L=%u  decision=%d  eqlev_first=%d  eqlev_canon=%d  comp_canon=%d\n",
    //         current_partition.level,
    //         code,
    //         eqlevfirst,
    //         eqlevcanon,
    //         comp_canon);

    if (DBG) std::cout << "[DEBUG][classify_node] CASE code=" << code << "\n";

    switch (code){
        case 0:                 /* normal node */
            break;
        case 1: { //
            
            unsigned int* automorphism = new unsigned int[num_vertices]();

            for (int j = 0; j < (int)num_vertices; j++) {
                automorphism[first_leaf.label[j]] = current_partition.element_vec[j];
            }
            //printf("aut wrt first leaf\n");
        
            //write_automorphism(graph, automorphism);
                

            unsigned int index = num_aut % max_aut;
            for (unsigned int i = 0; i < num_vertices; ++i) {
                found_automorphisms[index * num_vertices + i] = automorphism[i];
            }
            
            num_aut++;
        
            
            int g_seq = get_gca_level(first_leaf.vertex_sequence, first_leaf.vertex_sequence_size,
                                    current_partition.current_vertex_sequence, current_partition.current_vertex_sequence_size);
            
            //backtrack_to(current_partition.level - 1);
            if (g_seq >= 0) {
                    
                    backtrack_to((unsigned)g_seq);

            } else {

                backtrack_to(current_partition.level-1);
            }
            
            delete[] automorphism;
            
            break;
        }

        case 2: {

            
            unsigned int* automorphism = new unsigned int[num_vertices]();

            

            for (int j = 0; j < (int)num_vertices; j++) {
                automorphism[best_leaf.label[j]] = current_partition.element_vec[j];
            }
            
            unsigned int index = num_aut % max_aut;
            for (unsigned int i = 0; i < num_vertices; ++i) {
                found_automorphisms[index * num_vertices + i] = automorphism[i];
            }

            //printf("aut wrt best leaf\n");
            num_aut++;
            if (DBG) std::cout << "[DEBUG][classify_node] stored generator #" << num_aut << " at slot " << index << "\n";
        
            
            delete[] automorphism;
        
            int g = get_gca_level(best_leaf.vertex_sequence, best_leaf.vertex_sequence_size,current_partition.current_vertex_sequence, current_partition.current_vertex_sequence_size);
            
            //backtrack_to(current_partition.level - 1);
            if (g >= 0) 
                backtrack_to(g); 
            else        
                backtrack_to(current_partition.level - 1);
        
            
            break;
        }

        case 3:{
            //This is a new bsf node, probably better than the previous bsf node
            if (DBG) std::cout << "[DEBUG][classify_node] CASE 3: new best-so-far -> update canon state\n";


            unsigned int* curr_perm = discrete_partition_to_perm(current_partition, num_vertices);
            bool* curr_hash = graph.buildPermutedFlatAdjacencyMatrixArray(curr_perm);
            //break tie using permuted graph's adjacency matrix
            int comp = compare_adj_upper(curr_hash, best_leaf.hash_of_perm_graph, num_vertices);
            //std::cout << "compare_hashes -> comp_canon=" << comp << "\n";

            delete[] curr_perm;
            delete[] curr_hash;
            

            canon_level = eqlevcanon = current_partition.level;
            comp_canon = 0;
            canon_code[current_partition.level] = 077777;

            for (int j = 0; j < num_vertices; j++){
                best_leaf.label[j] = current_partition.element_vec[j];
            }
            unsigned int* leaf_perm = discrete_partition_to_perm(current_partition, num_vertices);
            bool* hash_val = graph.buildPermutedFlatAdjacencyMatrixArray(leaf_perm);

        
            for (unsigned int i = 0; i < current_partition.current_vertex_sequence_size; ++i)
                best_leaf.vertex_sequence[i] = current_partition.current_vertex_sequence[i];

            best_leaf.vertex_sequence_size = current_partition.current_vertex_sequence_size;
            for (unsigned int i = 0; i < num_vertices * num_vertices; ++i)
                best_leaf.hash_of_perm_graph[i] = hash_val[i];

            delete[] hash_val;
            delete[] leaf_perm;
            backtrack_to(current_partition.level-1);
            break;
        }
        case 4:
            // bad leaf.
            if (DBG) std::cout << "[DEBUG][classify_node] CASE 4: bad leaf -> backtrack\n";
            backtrack_to(current_partition.level-1);
            break;

    }


    
}

void Nauty_Sequential::search_tree_traversal() {
    
    code = Partition_s_refinement(current_partition, graph, nullptr, 0);
    // std::cout << "refinement code = " << code << '\n';
    firstcode[current_partition.level-1] = code; // level is 1
    node_counter++;
    refinements_made++;
    while (current_partition.level >= 1 ) {
        //printPartition(current_partition) ;
        if(!first_leaf.discovered){ 
            first_path_node();
        }
        else{
            other_path_node();
            
        }
    }

    
    // std::cout << "[SEQ] canon_code (len=" << canon_level << "): ";
    // for (unsigned i = 0; i < canon_level; ++i) std::cout << canon_code[i] << ' ';
    
}

void Nauty_Sequential::backtrack_to(unsigned int level) {
    //std::cout << "Backtracking to level " << level << "\n";
    times_backtracked++;
    //printPartitionDetail(current_partition);
    const bool DBG = false;
    if (level == 0) {
        current_partition.level = level;
        if (DBG) std::cout << "[DEBUG][backtrack_to] set level=0 and RETURN\n";
        return;
    }


    reconstruct_at_level(current_partition, level, num_vertices, eqlevfirst,eqlevcanon,comp_canon);
    current_partition.current_vertex_sequence_size = level - 1;
    tc_level_reached = level;
    current_partition.level = level;
    //printf("backtracked to level %u",level);

    //printPartitionDetail(current_partition);

}
