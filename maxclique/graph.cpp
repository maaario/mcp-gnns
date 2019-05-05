#include "graph.h"

#include <cassert>

using namespace std;

Graph Graph::deep_copy() const {
    Graph graph;
    graph.num_vertices = num_vertices;
    graph.num_edges = num_edges;
    graph.neighbours = neighbours;
    graph.names_of_original_vertices = names_of_original_vertices;
    return graph;
}

void Graph::remove_vertex(int v) {
    for (int neighbour : neighbours[v]) {
        int neighbour_count = neighbours[neighbour].size();
        for (int j = 0; j < neighbour_count; j++) {
            if (neighbours[neighbour][j] == v) {
                neighbours[neighbour][j] = neighbours[neighbour][neighbour_count-1];
                neighbours[neighbour].pop_back();
                break;
            }
        }
    }
    neighbours[v].clear();
}

Graph Graph::create_graph_from_neighbours_of_vertex(int v) const {
    Graph graph;
    graph.num_vertices = neighbours[v].size();
    graph.neighbours.resize(graph.num_vertices, vector<int>());

    // Set new vertex names and backward index.
    vector<int> new_vertex_names(num_vertices, VERTEX_NOT_USED);
    for (int i = 0; i < int(neighbours[v].size()); i++) {
        int w = neighbours[v][i];
        new_vertex_names[w] = i;
        graph.names_of_original_vertices.push_back(names_of_original_vertices[w]);
    }

    // Add edges to new graph.
    for (int w : neighbours[v]) {
        int new_w = new_vertex_names[w];
        for (int x : neighbours[w]) {
            int new_x = new_vertex_names[x];
            if (new_x != VERTEX_NOT_USED) {
                graph.neighbours[new_w].push_back(new_x);
            }
        }
    }

    // Set the number of edges.
    for (int new_w = 0; new_w < graph.num_vertices; new_w++) {
        graph.num_edges += graph.neighbours[new_w].size();
    }
    graph.num_edges /= 2;

    return graph;
}

const int Graph::VERTEX_NOT_USED = -1;

Graph read_graph(FILE* file) {
    Graph graph;
    fscanf(file, " %d %d", &(graph.num_vertices), &(graph.num_edges));
    
    graph.neighbours.resize(graph.num_vertices);
    for (int i=0; i<graph.num_edges; i++) {
        int a, b;
        fscanf(file, " %d %d", &a, &b);
        a--; b--;
        graph.neighbours[a].push_back(b);
        graph.neighbours[b].push_back(a);
    }

    graph.names_of_original_vertices.resize(graph.num_vertices);
    for (int v = 0; v < graph.num_vertices; v++) {
        graph.names_of_original_vertices[v] = v;
    }

    return graph;
}

void write_graph(const Graph &graph, FILE* file) {
    fprintf(file, "%d %d\n", graph.num_vertices, graph.num_edges);
    
    int num_edges_written = 0;
    for (int v = 0; v < graph.num_vertices; v++) {
        for (int w : graph.neighbours[v]) {
            if (v < w) {
                fprintf(file, "%d %d\n", v + 1, w + 1);
                num_edges_written++;
            }
        }
    }
    
    assert(num_edges_written == graph.num_edges);
}
