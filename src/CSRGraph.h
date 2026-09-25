#ifndef CSRGRAPH_H
#define CSRGRAPH_H

#include <iostream>

class CSRGraph {


public:
    unsigned int vertexNum;
    unsigned int edgeNum;
    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    int* color;
    CSRGraph();
    CSRGraph(unsigned int vertices, unsigned int edges);
    //~CSRGraph();
    void cleanup();
    static CSRGraph fromDIMACS(const std::string& filename);
    void addEdge(unsigned int src, unsigned int dest);
    int getDegree(unsigned int vertex) const;
    void getNeighbors(unsigned int vertex) const;
    void printGraph() const;
    bool* buildFlatAdjacencyMatrixArray() const;
    bool* buildPermutedFlatAdjacencyMatrixArray(unsigned int* perm) const;
    unsigned int* generate_random_permutation(unsigned int n);
    CSRGraph randomly_permute();
    CSRGraph permute_graph(unsigned int* perm);
    bool hasEdgeSortedRow(unsigned int u,unsigned int v) const;
    bool isAutomorphism(const unsigned int* perm) const;
    void printAdjacency();
    unsigned int getVertexNum() const;
    unsigned int getEdgeNum() const;
    unsigned int* getDst() const;
    unsigned int* getSrcPtr() const;
    int* getDegree() const;
    int* getColor() const;
};

#endif // CSRGRAPH_H
