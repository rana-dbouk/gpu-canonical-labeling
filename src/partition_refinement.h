// partition_refinement.h
#ifndef PARTITION_REFINEMENT_H
#define PARTITION_REFINEMENT_H
#include "structs.h"
#include "CSRGraph.h"
#include <vector>

/*  Nauty-compatible hash macros */
#ifndef MASH
#define MASH(l,i) ((((l) ^ 065435) + (i)) & 077777)
/* expression whose long value depends only on long l and int/long i.
   Anything goes, preferably non-commutative. */
#endif

#ifndef CLEANUP
#define CLEANUP(l) ((int)((l) % 077777))
/* expression whose value depends on long l and is less than 077777
   when converted to int then short.  Anything goes. */
#endif



int cell_index_by_first(const Partition_s& P, unsigned first);
bool check_in_cell_indices(Partition_s& partition, unsigned int numVertices);
void initializePartition_s(Partition_s& partition, const CSRGraph& graph);
bool is_discrete(Partition_s& partition);
bool is_singleton(CellStruct_s cell);
unsigned int* decode(Partition_s& partition, CellStruct_s subsequence, unsigned int starting_index);
unsigned int find_index_in_subsequence(CellStruct_s* subsequence, unsigned int subseq_size, CellStruct_s target);
int Partition_s_refinement(Partition_s& partition, const CSRGraph& graph, CellStruct_s* subsequence, unsigned int subseq_size);
void printPartition(Partition_s& partition);
void printPartitionDetail(Partition_s& partition);
void split_by_and_refine(Partition_s& partition, const CSRGraph& graph, unsigned int vertex, int& code, int tc_first);
void reconstruct_at_level(Partition_s& partition, unsigned int return_level, unsigned int numVertices, unsigned int& eqlev_first,int& eqlev_canon, int& comp_canon);
void merge_cells(Partition_s& partition, unsigned int start_cell, unsigned int end_cell);
int target_cell_selector(const Partition_s& current_partition, int hint,TargetCell* target_cells_at_level, int& tc_level_reached, unsigned int numVertices);
int target_cell_selector_largest(const Partition_s& current_partition,std::vector<std::vector<unsigned int>>& target_cells_per_level);
unsigned int degree_in_subset(const CSRGraph& graph, unsigned int v, const std::vector<int>& subset) ;
int target_cell_selector_non_trivial(const Partition_s& current_partition, const CSRGraph& graph);
void partition_shape_invar(Partition_s& partition, unsigned int* invar_array, unsigned int& new_invar_len);

#endif  // PARTITION_REFINEMENT_H
