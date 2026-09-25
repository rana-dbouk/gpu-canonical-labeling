#include <stdio.h>
#include <stdlib.h>
#include "CSRGraph.h"
#include <stdexcept>
#include "partition_refinement.h"
#include <iterator> 
#include <iostream>
#include <random>
#include <cstdint>
#include <vector>
#include <algorithm>
int cell_index_by_first(const Partition_s& P, unsigned first_pos) {

    int lo = 0, hi = (int)P.lcs_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        unsigned f = P.lcs[mid].first;
        if (first_pos < f) hi = mid - 1;
        else if (first_pos > f) lo = mid + 1;
        else {
            return mid;
        }
    }
    return -1;
}


void initializePartition_s(Partition_s& partition, const CSRGraph& graph) {
    const unsigned int n = graph.getVertexNum();

    partition.element_vec = new unsigned int[n];
    partition.lcs = new CellStruct_s[n + 1];
    partition.lcs_size = 0;
    partition.level = 0;
    partition.current_vertex_sequence = new unsigned int[n];
    partition.current_vertex_sequence_size = 0;


    //  COLORED 
    // Sort vertices by (color, vertex_id) so equal colors become contiguous in element_vec
    std::vector<std::pair<int, unsigned int>> items;
    items.reserve(n);
    for (unsigned int v = 0; v < n; ++v) items.emplace_back(graph.color[v], v);

    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
              });

    for (unsigned int i = 0; i < n; ++i) partition.element_vec[i] = items[i].second;

    // One initial cell per color block
    unsigned int start = 0;
    while (start < n) {
        const int c = items[start].first;
        unsigned int end = start + 1;
        while (end < n && items[end].first == c) ++end;

        CellStruct_s cell;
        cell.first    = start;
        cell.length   = end - start;
        cell.in_level = 0;

        partition.lcs[partition.lcs_size++] = cell;
        start = end;
    }
}

bool is_discrete(Partition_s& partition) {
    for (unsigned int i = 0; i < partition.lcs_size; ++i) {
        if (partition.lcs[i].length != 1) {
            return false;
        }
    }
  
    return true;
}
bool is_singleton(CellStruct_s cell){
    return (cell.length==1?true:false);
}
unsigned int* decode(Partition_s& partition, CellStruct_s subsequence, unsigned int starting_index) {
    unsigned int first = subsequence.first;
    unsigned int length = subsequence.length;

    // Adjust length based on starting index
    unsigned int adjusted_length = length - starting_index;
    unsigned int* decoded_values = new unsigned int[adjusted_length];

    for (unsigned int i = 0; i < adjusted_length; ++i) {
        decoded_values[i] = partition.element_vec[first + starting_index + i];
    }

    return decoded_values;
}

void printPartitionDetail(Partition_s& partition) {
    std::cout << "Current Partition:\n";
    
    for (unsigned int i = 0; i < partition.lcs_size; ++i) {
        CellStruct_s cell = partition.lcs[i];

        unsigned int* decoded_cell = decode(partition, cell, 0);
        
        std::cout << "Cell " << i << " (length " << cell.length 
                  << ", in_level " << cell.in_level << "): ";

        
        std::cout << "\n";

        delete[] decoded_cell;
    }
}

void printPartition(Partition_s& partition) {
    std::cout  << "Level: " << partition.level <<".";
    std::cout << "[";

    for (unsigned int i = 0; i < partition.lcs_size; ++i) {
        CellStruct_s cell = partition.lcs[i];
        unsigned int* decoded_cell = decode(partition, cell,0);
        
        std::cout << "[";
        for (unsigned int j = 0; j < cell.length; ++j) {
            std::cout << decoded_cell[j];
            if (j < cell.length - 1) {
                std::cout << ",";
            }
        }
        std::cout << "]";

        delete[] decoded_cell;
    }

    std::cout << "]\n";
}

/*
* We do not sort vertices.
*
* For non-singleton splitters:
*   - group by increasing degree/count value
*   - preserve the original order of vertices inside each group
*
* For singleton splitters:
*   - group as degree 1 first, then degree 0
*   - preserve the original order of vertices inside each group
*/

int Partition_s_refinement(
    Partition_s& partition,
    const CSRGraph& graph,
    CellStruct_s* subsequence,
    unsigned int subseq_size
) {
    const unsigned int num_vertices = graph.getVertexNum();
    /*
     * Nauty-based active set implementation:
     * active[first] == 1 means the cell starting at position first is active.
     * subsequence is only used to initialize this active set.
     */
    unsigned char* active = new unsigned char[num_vertices + 1]();
    unsigned int active_count = 0;

    auto add_active = [&](unsigned int first) {
        if (first < num_vertices && active[first] == 0) {
            active[first] = 1;
            ++active_count;
        }
    };

    auto remove_active = [&](unsigned int first) {
        if (first < num_vertices && active[first] != 0) {
            active[first] = 0;
            --active_count;
        }
    };

    if (subsequence == nullptr) {
        for (unsigned int i = 0; i < partition.lcs_size; ++i) {
            add_active(partition.lcs[i].first);
        }
    } else {
        for (unsigned int i = 0; i < subseq_size; ++i) {
            add_active(subsequence[i].first);
        }
    }

    uint64_t longcode = (uint64_t)partition.lcs_size;

    unsigned int* all_degrees = new unsigned int[num_vertices]();
    unsigned int* work_cell   = new unsigned int[num_vertices];
    unsigned int* bucket      = new unsigned int[num_vertices + 1]();

    int hint = 0;

    while ((!is_discrete(partition)) && active_count > 0) {
        /*
         * Choose splitter like nauty:
         * try hint, then search after hint, then wrap around.
         */
        int split1_int = -1;

        if (hint >= 0 &&
            (unsigned int)hint < num_vertices &&
            active[(unsigned int)hint] != 0) {
            split1_int = hint;
        } else {
            unsigned int start_pos = (hint >= 0) ? (unsigned int)hint + 1 : 0;

            for (unsigned int p = start_pos; p < num_vertices; ++p) {
                if (active[p] != 0) {
                    split1_int = (int)p;
                    break;
                }
            }

            if (split1_int < 0) {
                for (unsigned int p = 0; p < num_vertices; ++p) {
                    if (active[p] != 0) {
                        split1_int = (int)p;
                        break;
                    }
                }
            }
        }
        unsigned int split1 = (unsigned int)split1_int;
        remove_active(split1);

        int splitter_index = cell_index_by_first(partition, split1);

        CellStruct_s splitter = partition.lcs[splitter_index];
        unsigned int* decoded_cell = &partition.element_vec[splitter.first];

        unsigned int split2 = split1 + splitter.length - 1;

        longcode = MASH(longcode, (uint64_t)(split1 + split2));

        if (splitter.length > 1) {
            longcode = MASH(longcode, (uint64_t)splitter.length);
        }


        for (unsigned int i = 0; i < splitter.length; ++i) {
            unsigned int w = decoded_cell[i];

            for (unsigned int e = graph.srcPtr[w]; e < graph.srcPtr[w + 1]; ++e) {
                unsigned int neighbor = graph.dst[e];
                ++all_degrees[neighbor];
            }
        }

        for (int i = 0; i < (int)partition.lcs_size; ++i) {
            int cell_index = i;
            CellStruct_s cell = partition.lcs[cell_index];

            if (is_singleton(cell)) {
                continue;
            }

            unsigned int* cell_vertices = &partition.element_vec[cell.first];
            unsigned int cell_size = cell.length;

            unsigned int common_degree = all_degrees[cell_vertices[0]];
            unsigned int bmin = common_degree;
            unsigned int bmax = common_degree;
            bool is_common = true;

            for (unsigned int j = 1; j < cell_size; ++j) {
                unsigned int d = all_degrees[cell_vertices[j]];

                if (d != common_degree) {
                    is_common = false;
                }

                if (d < bmin) bmin = d;
                if (d > bmax) bmax = d;
            }

            if (is_common) {
                if (splitter.length > 1) {
                    longcode = MASH(longcode, (uint64_t)(common_degree + cell.first));
                }
                continue;
            }
            for (unsigned int d = bmin; d <= bmax; ++d) {
                bucket[d] = 0;
            }

            for (unsigned int j = 0; j < cell_size; ++j) {
                ++bucket[all_degrees[cell_vertices[j]]];
            }

            unsigned int num_new_cells = 0;

            if (splitter.length == 1) {
                if (bmin <= 1 && bmax >= 1 && bucket[1] > 0) {
                    ++num_new_cells;
                }

                if (bmin <= 0 && bmax >= 0 && bucket[0] > 0) {
                    ++num_new_cells;
                }
            } else {
                for (unsigned int d = bmin; d <= bmax; ++d) {
                    if (bucket[d] > 0) {
                        ++num_new_cells;
                    }
                }
            }

            /*
             * Shift lcs once to make room for the new fragments.
             * The old cell at cell_index will be overwritten by the first fragment.
             */
            if (num_new_cells > 1) {
                for (int j = (int)partition.lcs_size - 1; j > cell_index; --j) {
                    partition.lcs[j + (int)num_new_cells - 1] = partition.lcs[j];
                }
            }

            /*
             * Second pass: write fragment descriptors directly into partition.lcs.
             * Also convert bucket[d] into the write position for that degree.
             */
            unsigned int fidx = 0;
            unsigned int first = cell.first;
            unsigned int max_size = 0;
            unsigned int max_index = 0;

            auto add_fragment = [&](unsigned int degree_value) {
                unsigned int size = bucket[degree_value];

                if (size == 0) {
                    return;
                }

                CellStruct_s& new_cell = partition.lcs[cell_index + fidx];

                new_cell.first = first;
                new_cell.length = size;
                new_cell.in_level = partition.level + 1;

                if (splitter.length > 1) {
                    longcode = MASH(
                        longcode,
                        (uint64_t)(degree_value + new_cell.first)
                    );
                }

                if (size > max_size) {
                    max_size = size;
                    max_index = fidx;
                }

                bucket[degree_value] = first - cell.first;  // write position
                first += size;
                ++fidx;
            };

            if (splitter.length == 1) {
                if (bmin <= 1 && bmax >= 1) {
                    add_fragment(1);
                }

                if (bmin <= 0 && bmax >= 0) {
                    add_fragment(0);
                }
            } else {
                for (unsigned int d = bmin; d <= bmax; ++d) {
                    add_fragment(d);
                }
            }

            partition.lcs_size += num_new_cells - 1;

            /*
             * Reorder vertices into work_cell.
             * This preserves order inside each degree group.
             */
            for (unsigned int j = 0; j < cell_size; ++j) {
                unsigned int v = cell_vertices[j];
                unsigned int d = all_degrees[v];

                unsigned int& pos = bucket[d];
                work_cell[pos++] = v;
            }

            for (unsigned int j = 0; j < cell_size; ++j) {
                partition.element_vec[cell.first + j] = work_cell[j];
            }

            bool old_cell_was_active = (active[cell.first] != 0);

            /*
             * Singleton-splitter longcode update.
             * Non-singleton longcode was already updated in add_fragment().
             */
            if (splitter.length == 1) {
                unsigned int neighbors_size = partition.lcs[cell_index].length;
                unsigned int non_neighbors_size =
                    (num_new_cells > 1) ? partition.lcs[cell_index + 1].length : 0;

                if (neighbors_size > 0 && non_neighbors_size > 0) {
                    uint64_t boundary_c2 =
                        (uint64_t)(cell.first + neighbors_size - 1);
                    longcode = MASH(longcode, boundary_c2);
                }
            }

            /*
             * Update active set
             */
            if (splitter.length == 1) {
                unsigned int left_size = partition.lcs[cell_index].length;
                unsigned int right_size =
                    (num_new_cells > 1) ? partition.lcs[cell_index + 1].length : 0;

                if (old_cell_was_active || left_size >= right_size) {
                    if (num_new_cells > 1) {
                        unsigned int fstart = partition.lcs[cell_index + 1].first;
                        add_active(fstart);

                        if (partition.lcs[cell_index + 1].length == 1) {
                            hint = (int)fstart;
                        }
                    }
                } else {
                    unsigned int fstart = partition.lcs[cell_index].first;
                    add_active(fstart);

                    if (partition.lcs[cell_index].length == 1) {
                        hint = (int)fstart;
                    }
                }
            } else {
                /*
                 *  Add every fragment except the first
                 */
                for (unsigned int f = 0; f < num_new_cells; ++f) {
                    CellStruct_s& frag = partition.lcs[cell_index + f];

                    if (frag.first != cell.first) {
                        add_active(frag.first);

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
                    add_active(cell.first);
                    remove_active(partition.lcs[cell_index + max_index].first);
                }
            }

           
            if (cell.in_level != partition.level + 1 && num_new_cells > 0) {
                partition.lcs[cell_index + num_new_cells - 1].in_level = cell.in_level;
            }

            i = cell_index + (int)num_new_cells - 1;
        }

        for (unsigned int t = 0; t < num_vertices; ++t) {
            all_degrees[t] = 0;
        }
    }

    longcode = MASH(longcode, (uint64_t)partition.lcs_size);
    int codee = CLEANUP(longcode);

    delete[] active;
    delete[] all_degrees;
    delete[] work_cell;
    delete[] bucket;

    partition.level++;
    return codee;
}
void split_by_and_refine(Partition_s& partition, const CSRGraph& graph, unsigned int vertex, int& code, int tc_first) {   
    
    
    unsigned int cell_index = cell_index_by_first(partition,tc_first);
    CellStruct_s& cell = partition.lcs[cell_index];
    if(cell.length==1){
        return;
    }
    unsigned int start = cell.first;
    unsigned int end = cell.first + cell.length;

    // Step 1: Find the index of the vertex
    unsigned int vertex_index = start;
    while (vertex_index < end && partition.element_vec[vertex_index] != vertex) {
        vertex_index++;
    }
    
    // Step 2: Shift all elements before vertex_index to the right
    if (vertex_index > start) { 
        // Shift elements to the right
        for (int i = vertex_index; i > start; --i) {
            partition.element_vec[i] = partition.element_vec[i - 1];
        }
        // Step 3: Place the vertex at the beginning
        partition.element_vec[start] = vertex;
    }
    //create a new cell for all the vertices in th old cell minus the vertex
    CellStruct_s non_trivial_cell;
    non_trivial_cell.first=cell.first+1;
    non_trivial_cell.length=cell.length-1;
    non_trivial_cell.in_level=cell.in_level; //keeps the old in level value for backtracking purposes,since its considered the last cell in the newly created cells(trivial and non trivial)
    //insert after the original cell, now singleton cell comes first and then the rest of the vertices in the other cell
    for (int i = partition.lcs_size - 1; i > cell_index; --i) {
        partition.lcs[i + 1] = partition.lcs[i];  // Shift cells to the right
    }
    partition.lcs[cell_index + 1] = non_trivial_cell;  // Insert the new cell
    unsigned int numVertices = graph.getVertexNum();
    //Step 6: Update lcs_size
    partition.lcs_size++;
    cell.length=1; //only contains the vertex now
    cell.in_level=partition.level+1; //considered newly created
    partition.lcs[cell_index]=cell; //since we modified it previously
    
   
    //Call Refinement
    CellStruct_s* cellArray = new CellStruct_s[numVertices*2];
    cellArray[0] = cell;
    unsigned int subSize=1;
    //printPartition(partition);
    code = Partition_s_refinement(partition, graph, cellArray, subSize);
    
    delete[] cellArray;
    
}

void reconstruct_at_level(Partition_s& partition,
                          unsigned int return_level,
                          unsigned int /*numVertices*/,
                          unsigned int& eqlev_first,
                          int& eqlev_canon,
                          int& comp_canon)
{
    if (return_level < 1) return;

    // Linear scan over lcs: collapse every maximal run of cells created after return_level.
    // Invariant required: descendants of any pre-split cell are contiguous in lcs.
    for (unsigned i = 0; i < partition.lcs_size; )
    {
        if (partition.lcs[i].in_level <= return_level) {
            ++i; // this cell already existed at or before return_level → keep as-is
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

        std::sort(partition.element_vec +first, partition.element_vec + first + len);

        // Collapse the run into a single cell at position i.
        // The elements are already contiguous in element_vec[first .. first+len-1], so no move needed.
        partition.lcs[i].first    = first;
        partition.lcs[i].length   = len;
        partition.lcs[i].in_level = partition.lcs[j].in_level;


        // Remove the extra (j - i - 1) cells by shifting the tail left.
        const unsigned drop = (j - i);
        if (drop > 0) {
            for (unsigned k = j+1; k < partition.lcs_size; ++k)
                partition.lcs[k - drop] = partition.lcs[k];
            partition.lcs_size -= drop;
        }

        ++i; // advance past the merged cell
    }

    
    if (return_level < eqlev_first) eqlev_first = return_level;

    if ((int)return_level <= eqlev_canon) {
        eqlev_canon = (int)return_level;
        comp_canon  = 0;
    }

    partition.level = return_level;

    
}


int target_cell_selector(const Partition_s& current_partition, int hint,TargetCell* target_cell_at_level,int& tc_level_reached, unsigned int num_vertices) {
    
    int tc = -1;
    if (hint >= 0 && hint<current_partition.lcs_size && !is_singleton(current_partition.lcs[hint])) {
        tc = hint;
    }
    else{
        for (unsigned int i = 0; i < current_partition.lcs_size; ++i) {
            const CellStruct_s& cell = current_partition.lcs[i];
            if (cell.length > 1) {
                tc = i; // Found the first non-singleton cell
                break;
            }
        }
    }
    

    const bool DBG_TC = true;

    TargetCell& out = target_cell_at_level[tc_level_reached];

    if (tc >= 0) {
        const CellStruct_s& c = current_partition.lcs[tc];
        const unsigned first = c.first;
        const unsigned len   = c.length;


        for (unsigned v = 0; v < num_vertices; v ++) {
            out.elems[v] = -1;
        }
        

        // 2) Mark members of the selected cell as 2
        //    Members are contiguous in element_vec[first .. first+len)
        for (unsigned k = 0; k < len; k ++) {
            const int vtx = current_partition.element_vec[first + k];
            out.elems[vtx] = vtx;
        }

        out.length  = len;
        out.counter = 0;
        out.last_num_aut_at_level = 0;
        out.first=c.first;
        tc_level_reached++;

    }

    return tc; // All cells are singleton, nothing to branch on
}
int target_cell_selector_largest(const Partition_s& current_partition,std::vector<std::vector<unsigned int>>& target_cells_per_level) {
    int target_index = 0;
    unsigned int max_length = 0 ; // We only consider cells with length > 1

    for (unsigned int i = 0; i < current_partition.lcs_size; ++i) {
        const CellStruct_s& cell = current_partition.lcs[i];
        if (cell.length > max_length) {
            max_length = cell.length;
            target_index = i;
        }
    }

    

    //printf("[Level %u] Largest cell size: %u (Index %d)\n", current_partition.level, max_length, target_index);
    return target_index; // Returns -1 if no suitable cell is found
}

unsigned int degree_in_subset(const CSRGraph& graph, unsigned int v, const std::vector<int>& subset) {
    unsigned int start = graph.srcPtr[v];
    unsigned int end = graph.srcPtr[v + 1];

    unsigned int count = 0;
    for (unsigned int i = start; i < end; ++i) {
        unsigned int neighbor = graph.dst[i];
        if (std::find(subset.begin(), subset.end(), neighbor) != subset.end()) {
            ++count;
        }
    }
    return count;
}

int target_cell_selector_non_trivial(const Partition_s& current_partition, const CSRGraph& graph) {
    const CellStruct_s* lcs = current_partition.lcs;
    const unsigned int* element_vec = current_partition.element_vec;
    unsigned int num_cells = current_partition.lcs_size;

    if (num_cells == 0) return -1;

    std::vector<int> count(num_cells, 0);

    for (unsigned int i = 0; i < num_cells; ++i) {
        if (lcs[i].length <= 1) continue;

        unsigned int v = element_vec[lcs[i].first];

        for (unsigned int j = i + 1; j < num_cells; ++j) {
            if (lcs[j].length <= 1) continue;

            std::vector<int> other_cell(lcs[j].length);
            for (unsigned int k = 0; k < lcs[j].length; ++k) {
                other_cell[k] = element_vec[lcs[j].first + k];
            }

            unsigned int deg = degree_in_subset(graph, v, other_cell);
            if (deg > 0 && deg < lcs[j].length) {
                count[i]++;
                count[j]++;
            }
        }
    }

    // If all counts are 0, fallback to first non-singleton cell
    auto max_it = std::max_element(count.begin(), count.end());
    if (*max_it == 0) {
        std::cout << "[Fallback] No non-trivial joins found. Selecting first non-singleton cell." << std::endl;
        for (unsigned int i = 0; i < num_cells; ++i) {
            if (lcs[i].length > 1) return i;
        }
        return -1;
    }

    return static_cast<int>(std::distance(count.begin(), max_it));
}