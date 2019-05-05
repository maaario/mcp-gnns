#include "maxclique_utils.h"

#include <algorithm>
#include <cstdio>

using namespace std;

FindCliques::FindCliques(Settings settings): settings(settings) {
    strategy = make_unique<BranchBoundStrategy>(settings);
    strategy->initialize();
    metrics_manager = make_unique<MetricsManager>(settings.metrics_file_path);
}

FindCliques::~FindCliques() {
    strategy->finish();
}

// Branch and bound for finding max. clique: selects one vertex c.vertex that is 
// added to clique and removed from candidates. For a next recursion step
// only candidates neighbouring with c.vertex are considered.
void FindCliques::find_max_clique_recurse(Graph &graph, int depth, int parent_call_id) {
    if (graph.num_vertices == 0) {
        return;
    }
    RecursionCallMetrics metrics = metrics_manager->get_new_call_metrics(parent_call_id);
    if (settings.max_calls > 0 && metrics.call_id > settings.max_calls) {
        return;
    }
    metrics.recursion_depth = depth;
    metrics.num_candidates = graph.num_vertices;
    metrics.num_graph_edges = graph.num_edges;
    metrics.start_call_stopwatch();

    metrics.start_heuristic_stopwatch();
    vector<Candidate> candidates = strategy->compute_scores(graph);
    metrics.stop_heuristic_stopwatch();

    sort(candidates.begin(), candidates.end());
    reverse(candidates.begin(), candidates.end());
    
    for (Candidate& c : candidates) {
        // Only continue if better clique can be found.
        if (clique.size() + c.bounding_score <= max_clique.size()) {
            if (strategy->equal_bounding_and_branching_scores()) {
                break;
            } else {
                // If different strategy is used for bounding than for branching (used for sorting),
                // we have to iterate through all candidates to visit those with higher bounds.
                continue;
            }
        }
        
        metrics.num_candidates_visited++;

        // Add vertex to clique and update max_clique.
        clique.push_back(graph.names_of_original_vertices[c.vertex]);
        if (clique.size() > max_clique.size()) {
            metrics.larger_clique_discovered = clique.size();
            max_clique = clique;
        }

        // Select new candidate set/graph: all vertices that neighbour with vertex of 
        // candidate c.
        Graph subgraph = graph.create_graph_from_neighbours_of_vertex(c.vertex);

        // Remove the vertex from candidates forever.
        graph.remove_vertex(c.vertex);
        
        // Continue with building larger clique.
        metrics.stop_call_stopwatch();
        find_max_clique_recurse(subgraph, depth + 1, metrics.call_id);   
        metrics.start_call_stopwatch();

        clique.pop_back();
    }

    metrics.stop_call_stopwatch();
    metrics_manager->save_call_metrics(metrics);
}

void FindCliques::test_clique(const Graph& graph, const vector<int>& clique) {
    vector<vector<bool> > adjacency_matrix(
        graph.num_vertices, vector<bool>(graph.num_vertices, false));
    for (int i = 0; i < graph.num_vertices; i++)
        for (int j : graph.neighbours[i])
            adjacency_matrix[i][j] = true;

    for (int i : clique)
        for (int j : clique) 
            if (i != j && !adjacency_matrix[i][j])
                fprintf(stderr, "Vertices %d and %d are said to be in clique but they are not neighbours!\n", i, j);
}

void FindCliques::find_max_clique(const Graph& graph) {
    clique.clear();
    max_clique.clear();

    Graph copied_graph = graph.deep_copy();
    
    find_max_clique_recurse(copied_graph);
    test_clique(graph, max_clique);
    sort(max_clique.begin(), max_clique.end());
}

const int FindCliques::GENERIC_VERTEX = -1;

void FindCliques::find_max_clique_for_each_vertex(const Graph& graph) {
    max_clique_for_each_vertex.clear();
    max_clique_for_each_vertex.resize(graph.num_vertices, 1);

    for (int i = 0; i < graph.num_vertices; i++) {
        clique.clear();
        max_clique.clear();
        // We reuse information from previously explored cliques as the lower bound.
        max_clique.resize(max_clique_for_each_vertex[i] - 1, GENERIC_VERTEX);

        Graph subgraph = graph.create_graph_from_neighbours_of_vertex(i);

        find_max_clique_recurse(subgraph);    
        test_clique(graph, max_clique);

        int max_clique_size = int(max_clique.size()) + 1;
        max_clique_for_each_vertex[i] = max_clique_size;
        // Apart from vertex i we also update all vertices in clique with new lower bounds.
        for (int v : max_clique) {
            if (v != GENERIC_VERTEX) {
                max_clique_for_each_vertex[v] = 
                    max(max_clique_for_each_vertex[v], max_clique_size);
            }
        }
    }
}
