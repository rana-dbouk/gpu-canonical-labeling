#include "perm_group.h"
#include "partition_refinement.h"


int* subgroup_fixing_sequence(unsigned int* found_automorphisms, int num_autt,
                               unsigned int* current_vertex_sequence, int sequence_size,
                              unsigned int num_vertices, int& num_subgroup, int max_aut, int new_discovered_aut) {
    
    int num_aut = (num_autt < max_aut ? num_autt : max_aut);
    int* subgroup_indices = new int[num_aut]();
    num_subgroup = 0;

    if(new_discovered_aut == -1){
        new_discovered_aut = 0;
    }

    for (int i = new_discovered_aut; i < num_aut; ++i) { 
        bool fixes_all = true;
        unsigned int* perm = &found_automorphisms[i * num_vertices];
        
        for (int j = 0; j < sequence_size; ++j) {
            if (perm[current_vertex_sequence[j]] != current_vertex_sequence[j]) {
                fixes_all = false;
                break;
            }
        }
        if (fixes_all) {
            subgroup_indices[num_subgroup++] = i;
        }
    }
    

    return subgroup_indices; 
}


//  DSU with "smallest-id root" policy
struct DSU {
    unsigned int n;
    unsigned int* parent;

    explicit DSU(unsigned int n_) : n(n_) {
        parent = new unsigned int[n];
        for (unsigned int i = 0; i < n; ++i) parent[i] = i;
    }
    ~DSU() { delete[] parent; }

    
    inline unsigned int find(unsigned int x) {
        while (parent[x] != x) {
            x = parent[x];
        }
        return x; // this will be the smallest vertex in the set
    }

    // attach larger root under smaller root so the root stays the minimum element
    inline void unite(unsigned int a, unsigned int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (a < b) parent[b] = a;
        else       parent[a] = b;
    }

    inline unsigned int rep(unsigned int x) { return find(x); }
};

void compute_orbits_fast(unsigned int* found_automorphisms, int num_aut,
                              unsigned int* current_vertex_sequence, int sequence_size,
                             unsigned int num_vertices,   int* target_cell,
                             unsigned int target_cell_length, int max_aut, int new_discovered_aut)
{
    int num_subgroup = 0;
    int* subgroup_indices = subgroup_fixing_sequence(found_automorphisms, num_aut,
                                                     current_vertex_sequence, sequence_size,
                                                     num_vertices, num_subgroup, max_aut, new_discovered_aut);


    // fast path: no subgroup => identity orbits
    if (num_subgroup == 0) {
        delete[] subgroup_indices;
        return;
    }

    DSU dsu(num_vertices);

    for (unsigned int v = 0; v < num_vertices; v ++){
        if(target_cell[v] != -1){
            dsu.parent[v] = target_cell[v];
        }
        else{
            dsu.parent[v] = v;
        }
    }

    

    for (int s = 0; s < num_subgroup; ++s) {
        unsigned int* perm = &found_automorphisms[subgroup_indices[s] * num_vertices];

        for (unsigned int v = 0; v < num_vertices; ++v) {
            unsigned int img = perm[v];
            if (v != img) dsu.unite(v, img);
        }
    }

    for (unsigned int v = 0; v < num_vertices; ++v){
        if(target_cell[v] != -1){
            target_cell[v] = dsu.rep((unsigned int)v);
        }
    }
        

    delete[] subgroup_indices;
    
}

unsigned int* discrete_partition_to_perm(Partition_s& partition, unsigned int n) {
    if (!is_discrete(partition)) return nullptr; 

    auto* workperm = new unsigned int[n];
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int oldv = partition.element_vec[i]; 
        workperm[oldv] = i;                          
    }
    return workperm; 
}

