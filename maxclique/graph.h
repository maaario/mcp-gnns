#ifndef MAXCLIQUE_GRAPH_H
#define MAXCLIQUE_GRAPH_H

#include <cstdio>
#include <vector>

using namespace std;

// Representation of undirected, simple input graph.
struct Graph {
    int num_vertices = 0, num_edges = 0;
    
    // Graph representation as lists of neighbours.
    vector<vector<int> > neighbours;
    
    // When a smaller graph is constructed from original graph, its vertices are renamed to 1 .. n.
    // To report clique vertices of the original graph, a backward index is needed.
    vector<int> names_of_original_vertices;
    
    Graph deep_copy() const;

    void remove_vertex(int v);

    // A constant used to mark vertices which will not be selected for subgraph.
    static const int VERTEX_NOT_USED;

    Graph create_graph_from_neighbours_of_vertex(int v) const;
};

Graph read_graph(FILE* file);

void write_graph(const Graph &graph, FILE* file);


#endif // MAXCLIQUE_GRAPH_H
