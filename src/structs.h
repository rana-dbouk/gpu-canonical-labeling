#ifndef NAUTY_STRUCTS_H
#define NAUTY_STRUCTS_H

#include <numeric>
#include <iostream>
#include <algorithm>
#include <stdexcept>


struct TargetCell {
    int* elems;  
    int first;
    unsigned  length;  
    unsigned  counter; // next index to branch from
    int  last_num_aut_at_level; // the last time we pruned this tc , num_ had this value, if not pruned yet, -1
    TargetCell() : elems(nullptr), first(0), length(0), counter(0),last_num_aut_at_level(-1) {}
};


struct CellStruct_s{
     unsigned int first;
     unsigned int length;
     unsigned int in_level;
    

    // Equality operator to compare two CellStruct_s objects
    bool operator==(const CellStruct_s& other) const {
        return first == other.first &&
               in_level == other.in_level &&
               length == other.length;
    }
};

struct Leaf_s {
    
    unsigned int* vertex_sequence;  // Raw array for GPU allocation
    unsigned int vertex_sequence_size;  
    bool* hash_of_perm_graph;       // The hash value of the graph it is called on. This is simply given as n^2 bit string concatenating the rows of an adjacency matrix
    bool discovered;
    int* label;
    
    void init(unsigned int numVertices) {
        vertex_sequence = new unsigned int[numVertices]();
        hash_of_perm_graph = new bool[numVertices*numVertices]();          
        vertex_sequence_size=0;
        discovered=false;
        label = new int[numVertices]();

    }

   
    void cleanup() {
        delete[] vertex_sequence;
        delete[] hash_of_perm_graph;
        delete[] label;

        vertex_sequence = nullptr;
        hash_of_perm_graph = nullptr;
        label = nullptr;
    }
};


    
struct Partition_s {
   
     unsigned int* element_vec;  
    CellStruct_s* lcs;     
     unsigned int lcs_size;
     unsigned int level;   
     unsigned int* current_vertex_sequence;
     unsigned int current_vertex_sequence_size;
    

    void cleanup() {
        delete[] element_vec;
        delete[] lcs;
        delete[] current_vertex_sequence;
        

        element_vec = nullptr;
        lcs = nullptr;
        current_vertex_sequence = nullptr;
        
    }
};

#endif