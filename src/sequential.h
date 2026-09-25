#ifndef NAUTY_SEQUENTIAL_H
#define NAUTY_SEQUENTIAL_H

#include "structs.h"
#include "partition_refinement.h"
#include "perm_group.h"
#include <iostream>
#include <chrono>


#include <vector>
class Nauty_Sequential {
public:
    Nauty_Sequential(CSRGraph& input_graph, unsigned int max_aut);
    ~Nauty_Sequential();
    int compare_adj_upper(const bool* A, const bool* B, unsigned n);
    int compare_hashes(bool* a, bool* b, unsigned int size);
    int compare_invariants(const unsigned int* a, unsigned int size_a,
                       const unsigned int* b, unsigned int size_b);
    void first_path_node();
    void first_terminal();
    void other_path_node();
    void classify_node();
    int canon_level;
    int eqlevcanon;
    int comp_canon;
    int canupdates;
    int code; 
    Partition_s current_partition;
    Leaf_s best_leaf;
    unsigned int refinements_made = 0;
    unsigned int leaves_visited = 0;
    unsigned int max_level = 0;
    unsigned int times_backtracked = 0;
    unsigned int total_target_cells = 0;
    unsigned int num_bad_leaves = 0;
    unsigned int branched_vertices_level_one=0;
    int* canon_code;
    int* first_path_tc ;
    std::uint64_t compute_orbits_ns_total = 0;
    std::uint64_t shifting_cells_time = 0;
    unsigned int eqlevfirst; /* level to which codes for this node match those for first leaf */
    int* firstcode;
    TargetCell* target_cell_at_level; 
    int tc_level_reached;
    


    
private:

    unsigned int node_counter = 0;
    CSRGraph& graph;
    unsigned int num_vertices;
    unsigned int max_aut;
    unsigned int* found_automorphisms;
    int num_aut;
    Leaf_s first_leaf;
    

    void print_canonical_labeling(const bool* hash_of_perm_graph, unsigned int num_vertices);
    int get_gca_level(unsigned int* first_sequence, unsigned int first_size,unsigned int* second_sequence, unsigned int second_size);
    void search_tree_traversal();
    void process_node();
    void process_leaf();
    void backtrack_to(unsigned int level);
    bool isAutomorphism(CSRGraph& graph, unsigned int* perm);
    void write_automorphism(CSRGraph& graph, unsigned int* perm);
    void prune_by_invar();
    
};

#endif // NAUTY_SEQUENTIAL_H
