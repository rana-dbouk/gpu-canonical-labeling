#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "CSRGraph.h"
#include "sequential.h"
#include "nauty_parallel.h"


static void print_help(const char* program_name) {
    std::cout
        << "Usage:\n"
        << "  " << program_name << " <graph.txt> [--sequential|--parallel]\n"
        << "  " << program_name << " <graph1.txt> <graph2.txt> [--sequential|--parallel]\n\n"
        << "Modes:\n"
        << "  One graph file         Compute the canonical labeling of the graph.\n"
        << "                         The canonical adjacency list is written to a file.\n"
        << "  Two graph files        Compute canonical labels for both graphs and check\n"
        << "                         whether they are isomorphic.\n\n"
        << "Options:\n"
        << "  --max-aut=N            Maximum number of automorphisms stored.\n"
        << "                         Default: 10000\n"
        << "  --sequential           Use the sequential canonical labeling algorithm.\n"
        << "  --parallel             Use the parallel canonical labeling algorithm. Default.\n"
        << "  -h, --help             Print this help message.\n";
}


template <typename T>
static void print_canonical_adjacency_list(const T* canon_adj,
                                           unsigned int n,
                                           std::ostream& out,
                                           int max_width = 80)
{
    for (unsigned int i = 0; i < n; ++i) {
        std::ostringstream prefix_stream;
        prefix_stream << std::setw(3) << i << " : ";
        std::string prefix = prefix_stream.str();

        out << prefix;
        int current_width = static_cast<int>(prefix.size());

        bool first_neighbor = true;

        for (unsigned int j = 0; j < n; ++j) {
            if (!canon_adj[i * n + j]) {
                continue;
            }

            std::string token = " " + std::to_string(j);

            if (!first_neighbor &&
                current_width + static_cast<int>(token.size()) + 1 > max_width) {
                out << "\n    ";
                current_width = 4;
                token = " " + std::to_string(j);
            }

            out << token;
            current_width += static_cast<int>(token.size());
            first_neighbor = false;
        }

        out << ";\n";
    }
}


template <typename AType, typename BType>
static int compare_canonical_adjacency_upper(const AType* A,
                                             const BType* B,
                                             unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i + 1; j < n; ++j) {
            bool a = static_cast<bool>(A[i * n + j]);
            bool b = static_cast<bool>(B[i * n + j]);

            if (a != b) {
                return a ? 1 : -1;
            }
        }
    }

    return 0;
}


int main(int argc, char* argv[]) {
    bool use_parallel = true;
    unsigned int max_aut = 10000;

    const std::string output_file = "output";
    std::vector<std::string> graph_files;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg.rfind("--max-aut=", 0) == 0) {
            max_aut = static_cast<unsigned int>(std::stoul(arg.substr(10)));
        } else if (arg == "--parallel") {
            use_parallel = true;
        } else if (arg == "--sequential") {
            use_parallel = false;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "[ERROR] Unknown option: " << arg << "\n";
            return 1;
        } else {
            graph_files.push_back(arg);
        }
    }

    if (graph_files.size() != 1 && graph_files.size() != 2) {
        std::cerr << "[ERROR] Invalid arguments.\n";
        print_help(argv[0]);
        return 1;
    }

    try {
        auto canonicalize = [&](CSRGraph& graph) {
            if (use_parallel) {
                return Nauty_Parallel(graph, max_aut);
            }

            Nauty_Sequential nauty(graph, max_aut);

            unsigned int n = graph.getVertexNum();
            std::vector<unsigned char> canon(n * n);

            for (unsigned int i = 0; i < n * n; ++i) {
                canon[i] = static_cast<unsigned char>(
                    nauty.best_leaf.hash_of_perm_graph[i]
                );
            }

            return canon;
        };

        if (graph_files.size() == 1) {
            CSRGraph graph = CSRGraph::fromDIMACS(graph_files[0]);
            std::vector<unsigned char> canon = canonicalize(graph);

            if (canon.empty()) {
                std::cerr << "[ERROR] Canonical labeling failed.\n";
                return 1;
            }

            std::ofstream out(output_file);

            if (!out.is_open()) {
                std::cerr << "[ERROR] Could not open output file: "
                          << output_file << "\n";
                return 1;
            }

            print_canonical_adjacency_list(
                canon.data(),
                graph.getVertexNum(),
                out
            );

            return 0;
        }

        CSRGraph graph1 = CSRGraph::fromDIMACS(graph_files[0]);
        CSRGraph graph2 = CSRGraph::fromDIMACS(graph_files[1]);

        std::cout << "\nGraph 1: " << graph_files[0] << "\n";
        std::cout << "Computing canonical form...\n";
        std::vector<unsigned char> canon1 = canonicalize(graph1);

        std::cout << "\nGraph 2: " << graph_files[1] << "\n";
        std::cout << "Computing canonical form...\n";
        std::vector<unsigned char> canon2 = canonicalize(graph2);

        if (canon1.empty() || canon2.empty()) {
            std::cerr << "[ERROR] Canonical labeling failed.\n";
            return 1;
        }

        int comp = compare_canonical_adjacency_upper(
            canon1.data(),
            canon2.data(),
            graph1.getVertexNum()
        );

        std::cout << "\nResult: "
          << (comp == 0 ? "ISOMORPHIC" : "NOT ISOMORPHIC")
          << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}