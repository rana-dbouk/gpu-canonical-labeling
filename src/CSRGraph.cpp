#include "CSRGraph.h"
#include <cstdlib> 
#include <ctime>   
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iostream>

CSRGraph::CSRGraph(unsigned int vertices, unsigned int edges)
    : vertexNum(vertices), edgeNum(edges) {
    dst = new unsigned int[edgeNum];
    srcPtr = new unsigned int[vertexNum + 1]; 
    degree = new int[vertexNum]();
    color = new int[vertexNum];
    for (unsigned int i = 0; i < vertexNum; ++i) color[i] = 0;
}
CSRGraph::CSRGraph(): dst(nullptr), srcPtr(nullptr), degree(nullptr),color(nullptr), vertexNum(0), edgeNum(0) {};

void CSRGraph::cleanup() {
    if (dst) {
        delete[] dst;
        dst = nullptr;
    }
    if (srcPtr) {
        delete[] srcPtr;
        srcPtr = nullptr;
    }
    if (degree) {
        delete[] degree;
        degree = nullptr;
    }
    if (color) { 
        delete[] color; 
        color = nullptr; 
    }
}

CSRGraph CSRGraph::fromDIMACS(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open DIMACS file: " + filename);
    }

    unsigned int num_vertices = 0, num_edges = 0;
    std::vector<std::pair<unsigned int, unsigned int>> edges;
    std::vector<std::pair<unsigned int, int>> node_colors;

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == 'c') continue; 

        std::istringstream iss(line);
        char type;
        iss >> type;

        if (type == 'p') {
            std::string tmp;
            iss >> tmp >> num_vertices >> num_edges;

        } else if (type == 'n') {
            unsigned int vid;
            int col;
            iss >> vid >> col;
            if (vid == 0) continue;
            vid--; // convert to 0-based
            node_colors.push_back({vid, col});

        } else if (type == 'e') {
            unsigned int u, v;
            iss >> u >> v;
            if (u == v) continue;
            u--; v--;             // convert to 0-based indexing
            edges.push_back(std::make_pair(u, v));
            edges.push_back(std::make_pair(v, u)); // undirected
        }
    }

    infile.close();
    std::sort(edges.begin(), edges.end());

    CSRGraph graph(num_vertices, edges.size());

    // apply colors found in the file (overwrites default)
    for (auto &kv : node_colors) {
        if (kv.first < graph.vertexNum) {
            graph.color[kv.first] = kv.second;
        }
    }

    std::vector<unsigned int> degrees(num_vertices, 0);
    for (size_t i = 0; i < edges.size(); ++i) {
        unsigned int u = edges[i].first;
        degrees[u]++;
    }

    graph.srcPtr[0] = 0;
    for (unsigned int i = 1; i <= num_vertices; ++i) {
        graph.srcPtr[i] = graph.srcPtr[i - 1] + degrees[i - 1];
    }

    
    std::fill(degrees.begin(), degrees.end(), 0);

    for (size_t i = 0; i < edges.size(); ++i) {
        unsigned int u = edges[i].first;
        unsigned int v = edges[i].second;
        unsigned int pos = graph.srcPtr[u] + degrees[u];
        graph.dst[pos] = v;
        degrees[u]++;
    }

    for (unsigned int i = 0; i < num_vertices; ++i) {
        graph.degree[i] = graph.srcPtr[i + 1] - graph.srcPtr[i];
    }

    return graph;
}

unsigned int* CSRGraph::generate_random_permutation(unsigned int n) {
    unsigned int* perm = new unsigned int[n];
    for (unsigned int i = 0; i < n; ++i) perm[i] = i;

    for (unsigned int i = n - 1; i > 0; --i) {
        unsigned int j = rand() % (i + 1);
        unsigned int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    std::cout << "Generated permutation: [";
    for (unsigned int i = 0; i < n; ++i) {
        std::cout << perm[i];
        if (i < n - 1) std::cout << ", "; 
    }
    std::cout << "]" << std::endl;
    return perm;
}
CSRGraph CSRGraph::randomly_permute() {
   
    unsigned int* perm = generate_random_permutation(vertexNum);
    CSRGraph permutedGraph(vertexNum, edgeNum);
    unsigned int* newDegree = new unsigned int[vertexNum]();

    for (unsigned int i = 0; i < vertexNum; ++i) {
        unsigned int oldVertex = i;  
        unsigned int newVertex = perm[oldVertex];  

        for (unsigned int j = srcPtr[oldVertex]; j < srcPtr[oldVertex + 1]; ++j) {
            unsigned int oldNeighbor = dst[j]; 
            unsigned int newNeighbor = perm[oldNeighbor]; 
            ++newDegree[newVertex];
        }
    }

    
    permutedGraph.srcPtr[0] = 0;
    for (unsigned int i = 0; i < vertexNum; ++i) {
        permutedGraph.srcPtr[i + 1] = permutedGraph.srcPtr[i] + newDegree[i];
    }

  
    for (unsigned int i = 0; i < vertexNum; ++i) {
        unsigned int oldVertex = i; 
        unsigned int newVertex = perm[oldVertex]; 

        unsigned int start = srcPtr[oldVertex]; 
        unsigned int end = srcPtr[oldVertex + 1]; 

        unsigned int insertPos = permutedGraph.srcPtr[newVertex];  
        for (unsigned int j = start; j < end; ++j) {
            unsigned int oldNeighbor = dst[j];  
            unsigned int newNeighbor = perm[oldNeighbor]; 

            permutedGraph.dst[insertPos++] = newNeighbor;
        }
    }

    for (unsigned int i = 0; i < vertexNum; ++i) {
        permutedGraph.degree[i] = newDegree[i];
    }

   
    for (unsigned int oldV = 0; oldV < vertexNum; ++oldV) {
        unsigned int newV = perm[oldV];
        permutedGraph.color[newV] = this->color[oldV];
    }

   
    for (unsigned int i = 0; i < vertexNum; ++i) {
        unsigned int start = permutedGraph.srcPtr[i];
        unsigned int end = permutedGraph.srcPtr[i + 1];
        std::sort(permutedGraph.dst + start, permutedGraph.dst + end);  
    }

   
    delete[] newDegree;
    delete[] perm;

    return permutedGraph;
}

CSRGraph CSRGraph::permute_graph(unsigned int* perm) {

   
    CSRGraph permutedGraph(vertexNum, edgeNum);
    unsigned int* newDegree = new unsigned int[vertexNum]();

   
    for (unsigned int i = 0; i < vertexNum; ++i) {
        unsigned int oldVertex = i; 
        unsigned int newVertex = perm[oldVertex]; 

       
        for (unsigned int j = srcPtr[oldVertex]; j < srcPtr[oldVertex + 1]; ++j) {
            unsigned int oldNeighbor = dst[j]; 
            unsigned int newNeighbor = perm[oldNeighbor]; 
            ++newDegree[newVertex];
        }
    }

   
    permutedGraph.srcPtr[0] = 0;
    for (unsigned int i = 0; i < vertexNum; ++i) {
        permutedGraph.srcPtr[i + 1] = permutedGraph.srcPtr[i] + newDegree[i];
    }

   
    for (unsigned int i = 0; i < vertexNum; ++i) {
        unsigned int oldVertex = i; 
        unsigned int newVertex = perm[oldVertex]; 
        unsigned int start = srcPtr[oldVertex]; 
        unsigned int end = srcPtr[oldVertex + 1]; 

        unsigned int insertPos = permutedGraph.srcPtr[newVertex];  
        for (unsigned int j = start; j < end; ++j) {
            unsigned int oldNeighbor = dst[j];  
            unsigned int newNeighbor = perm[oldNeighbor];  
            permutedGraph.dst[insertPos++] = newNeighbor; 
        }
    }

    for (unsigned int i = 0; i < vertexNum; ++i) {
        permutedGraph.degree[i] = newDegree[i];
    }


    for (unsigned int oldV = 0; oldV < vertexNum; ++oldV) {
        unsigned int newV = perm[oldV];
        permutedGraph.color[newV] = this->color[oldV];
    }

    for (unsigned int i = 0; i < vertexNum; ++i) {
        unsigned int start = permutedGraph.srcPtr[i];
        unsigned int end = permutedGraph.srcPtr[i + 1];
        std::sort(permutedGraph.dst + start, permutedGraph.dst + end); 
    }

   
    delete[] newDegree;
   
    return permutedGraph; 
}


void CSRGraph::addEdge(unsigned int src, unsigned int dest) {
    if (src >= vertexNum || dest >= vertexNum) return;
    dst[edgeNum] = dest;
    srcPtr[src + 1] = edgeNum + 1;
    degree[src]++;
}


int CSRGraph::getDegree(unsigned int vertex) const {
    if (vertex >= vertexNum) return -1;  
    return degree[vertex];
}
int* CSRGraph::getColor() const {
    return color;
}


void CSRGraph::getNeighbors(unsigned int vertex) const {
    if (vertex >= vertexNum) return;

    unsigned int startIdx = srcPtr[vertex];
    unsigned int endIdx = srcPtr[vertex + 1];

    for (unsigned int i = startIdx; i < endIdx; ++i) {
        std::cout << dst[i] << " ";
    }
    std::cout << std::endl;
}


void CSRGraph::printGraph() const {
    for (unsigned int i = 0; i < vertexNum; ++i) {
        std::cout << "Vertex " << i << ": ";
        getNeighbors(i);
    }
}


unsigned int CSRGraph::getVertexNum() const {
    return vertexNum;
}

unsigned int CSRGraph::getEdgeNum() const {
    return edgeNum;
}


bool* CSRGraph::buildFlatAdjacencyMatrixArray() const {
    unsigned int size = vertexNum * vertexNum;
    bool* adjMatrix = new bool[size](); 

    for (unsigned int i = 0; i < vertexNum; ++i) {
        unsigned int start = srcPtr[i];
        unsigned int end = srcPtr[i + 1];

        for (unsigned int j = start; j < end; ++j) {
            unsigned int neighbor = dst[j];
            adjMatrix[i * vertexNum + neighbor] = true;
            adjMatrix[neighbor * vertexNum + i] = true;
        }
    }

    return adjMatrix; 
}


bool* CSRGraph::buildPermutedFlatAdjacencyMatrixArray(unsigned int* perm) const {
    unsigned int n = vertexNum;
    bool* adjMatrix = new bool[n * n]();  

    for (unsigned int i = 0; i < n; ++i) {
        unsigned int pi = perm[i]; 

        unsigned int start = srcPtr[i];
        unsigned int end = srcPtr[i + 1];

        for (unsigned int k = start; k < end; ++k) {
            unsigned int j = dst[k];         
            unsigned int pj = perm[j];       
            adjMatrix[pi * n + pj] = true;
            adjMatrix[pj * n + pi] = true;
        }
    }

    return adjMatrix; 
}



void CSRGraph::printAdjacency() {
    const unsigned int n = getVertexNum();
    for (unsigned int u = 0; u < n; ++u) {
       
        std::vector<unsigned int> nbrs;
        nbrs.reserve(srcPtr[u+1] - srcPtr[u]);
        for (unsigned int j = srcPtr[u]; j < srcPtr[u + 1]; ++j) {
            nbrs.push_back(dst[j]);
        }
        
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

        std::cout << u << " : ";
        std::cout << " ";
        for (size_t k = 0; k < nbrs.size(); ++k) {
            if (k) std::cout << " ";
            std::cout << nbrs[k];
        }
        std::cout << ";\n";
    }
}



unsigned int* CSRGraph::getDst() const { return dst; }
unsigned int* CSRGraph::getSrcPtr() const { return srcPtr; }
int* CSRGraph::getDegree() const { return degree; }