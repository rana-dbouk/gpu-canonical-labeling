#ifndef NAUTY_PERM_GROUP_H
#define NAUTY_PERM_GROUP_H

#include "structs.h"

// Subgroup generators that fix the current stabilizer sequence
int* subgroup_fixing_sequence(unsigned int* found_automorphisms, int num_aut,
                              volatile unsigned int* current_vertex_sequence, int sequence_size,
                              unsigned int num_vertices, int& num_subgroup, int max_aut,int new_discovered_aut);


unsigned int* discrete_partition_to_perm(Partition_s& partition, unsigned int num_vertices);
unsigned int* perm_inverse(unsigned int* permutaion, unsigned int vertex_num);
unsigned int* perm_composition(unsigned int* first_perm, unsigned int* second_perm, unsigned int vertex_num);


// Fast orbit computation using DSU (path compression + union by rank).
// Returns orbit[i] = min representative of i’s orbit.
void compute_orbits_fast(unsigned int* found_automorphisms, int num_aut,
                                  unsigned int* current_vertex_sequence, int sequence_size,
                                  unsigned int num_vertices,
                                   int* target_cell, unsigned int target_cell_length,
                                  int max_aut, int new_discovered_aut);

#endif // NAUTY_PERM_GROUP_H
