#ifndef NAUTY_PARALLEL_H
#define NAUTY_PARALLEL_H
#include <vector>
#include <iostream>

#include "CSRGraph.h"

// This function initializes and starts the parallel Nauty code.
std::vector<unsigned char> Nauty_Parallel(
    CSRGraph& graph,
    int max_aut,
    unsigned int* node_count = nullptr,
    const std::string& output_prefix = "");

#endif // NAUTY_PARALLEL_H
