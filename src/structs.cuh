#ifndef STRUCTS_CUH
#define STRUCTS_CUH


#include <numeric>
#include <iostream>
#include <algorithm>
#include <stdexcept>


/*  Nauty-compatible hash macros */
#ifndef MASH
#define MASH(l,i) ((((l) ^ 065435) + (i)) & 077777)

#endif

#ifndef CLEANUP
#define CLEANUP(l) ((int)((l) % 077777))
/* expression whose value depends on long l and is less than 077777
   when converted to int then short.  Anything goes. */
#endif
/* ----------------------------------------------------------------------- */


#define NUM_COUNTERS 12
#define USE_COUNTERS 0




enum CounterName { ENQUEUE=0, DEQUEUE, WAIT_FIRST_LEAF,  PREPARE_ROOT, 
    PUSH_STACK, POP_STACK, PRUNE_STACK_ENTRY, BACKTRACK, VERIFY_SEQUENCE, CLASSIFY_NODE,
    SPLIT_BY_AND_REFINE, WAIT_BEST_LEAF_LOCK
};



struct Counters {
    unsigned long long tmp[NUM_COUNTERS];
    unsigned long long totalTime[NUM_COUNTERS];
};

static __device__ void initializeCounters(Counters* counters) {
    #if USE_COUNTERS
    if(threadIdx.x == 0) {
        
        for(unsigned int i = 0; i < NUM_COUNTERS; ++i) {
            counters->totalTime[i] = 0;
            counters->tmp[i] = 0ULL;
        }
    }
    #endif
}



static __device__ void startTime(CounterName counterName, Counters* counters) {
#if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters->tmp[counterName] = clock64();
    }
#endif
}

static __device__ void endTime(CounterName counterName, Counters* counters) {
#if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters->totalTime[counterName] += clock64() - counters->tmp[counterName];
    }
#endif
}


struct TargetCell {
    volatile int* elems; 
    volatile int first; //where this cell starts in the element vector
    volatile unsigned  length;  
    volatile int  counter; // next index to branch from
    volatile int  last_num_aut_at_level; // the last time we pruned this tc , num_ had this value, if not pruned yet, -1
    TargetCell() : elems(nullptr), first(0), length(0), counter(0),last_num_aut_at_level(-1) {}
};


struct Cell{
    volatile unsigned int first;
    volatile unsigned int length;
    volatile unsigned int in_level;
    
    bool operator==(const Cell& other) const {
        return first == other.first &&
               in_level == other.in_level &&
               length == other.length;
    }
};

struct Leaf {
    
    volatile unsigned int* vertex_sequence; 
    volatile unsigned int vertex_sequence_size;  
    volatile bool* hash_of_perm_graph;       // The hash value of the graph it is called on. This is simply given as n^2 bit string concatenating the rows of an adjacency matrix
    volatile int discovered;
    volatile int* label;
    volatile int* invar_path_value;
    
    void init(unsigned int numVertices) {
        vertex_sequence = new unsigned int[numVertices]();  
        hash_of_perm_graph = new bool[numVertices*numVertices]();     
        vertex_sequence_size=0;
        discovered=false;
        label = new int[numVertices]();
        invar_path_value = new int[numVertices + 2]();

    }

    
    void cleanup() {
        delete[] vertex_sequence;
        delete[] hash_of_perm_graph;
        delete[] label;
        delete[] invar_path_value;

        vertex_sequence = nullptr;
        hash_of_perm_graph = nullptr;
        label = nullptr;
        invar_path_value = nullptr;
    }
};

struct CanonicalLabeling {
    
    volatile unsigned int* vertex_sequence; 
    volatile unsigned int vertex_sequence_size;  
    volatile bool* hash_of_perm_graph;
    volatile bool discovered;
    volatile int* label;
    volatile int level;
    volatile int is_leaf;
    volatile int lock;
    volatile int* invar_path_value; 
 
    
    void init(unsigned int numVertices) {
        vertex_sequence = new unsigned int[numVertices]();  
        hash_of_perm_graph = new bool[numVertices*numVertices]();     
        vertex_sequence_size=0;
        discovered=false;
        label = new int[numVertices]();
        invar_path_value = new int[numVertices + 2]();

    }

    
    void cleanup() {
        delete[] vertex_sequence;
        delete[] hash_of_perm_graph;
        delete[] label;
        delete[] invar_path_value;

        vertex_sequence = nullptr;
        hash_of_perm_graph = nullptr;
        label = nullptr;
        invar_path_value = nullptr;
    }
};

    
struct Partition {
    
    volatile unsigned int* element_vec;  
    Cell* lcs;    
    volatile unsigned int lcs_size;
    volatile unsigned int level;   
    volatile unsigned int* current_vertex_sequence;
    volatile unsigned int current_vertex_sequence_size;
    volatile int pushNextIter;
    

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