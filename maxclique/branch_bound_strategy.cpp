#include "branch_bound_strategy.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <Python.h>

#include "graph.h"
#include "maxclique_utils.h"
#include "settings.h"

using namespace std;

bool Candidate::operator< (const Candidate &other) { 
    return branching_score < other.branching_score; 
}

template<typename T> Strategy * createStrategyInstance() { return new T; }
typedef map<string, Strategy*(*)()> strategy_map_type;

const int BranchBoundStrategy::MAX_BRANCH_SCORE = 1000;

BranchBoundStrategy::BranchBoundStrategy(Settings settings) {
    pair_branch = settings.pair_branch;

    if (settings.branch != settings.bound) {
        equal_bounding_and_branching_methods = false;
    } else if (settings.branch == "neural") {
        if(settings.bound_trained_model_dir != settings.branch_trained_model_dir) {
            equal_bounding_and_branching_methods = false;
        } else {
            equal_bounding_and_branching_methods = true;
        }
    }

    strategy_map_type strategy_map;
    strategy_map["no"] = &createStrategyInstance<NoStrategy>;
    strategy_map["degree"] = &createStrategyInstance<DegreeStrategy>;
    strategy_map["coloring"] = &createStrategyInstance<ColoringStrategy>;
    strategy_map["coloring_feasible"] = &createStrategyInstance<ColoringFeasibleStrategy>;
    strategy_map["optimal"] = &createStrategyInstance<OptimalStrategy>;
    strategy_map["neural"] = &createStrategyInstance<NeuralStrategy>;
    
    if (strategy_map.find(settings.branch) == strategy_map.end()) {
        fprintf(stderr, "Unknown branch strategy %s.", settings.branch.c_str());
    }
    if (strategy_map.find(settings.bound) == strategy_map.end()) {
        fprintf(stderr, "Unknown bound strategy %s.", settings.bound.c_str());
    }

    branch = unique_ptr<Strategy>(strategy_map[settings.branch]());
    if (settings.branch_trained_model_dir != "") {
        branch->trained_model_dir = settings.branch_trained_model_dir;
    }
    bound = unique_ptr<Strategy>(strategy_map[settings.bound]());
    if (settings.bound_trained_model_dir != "") {
        bound->trained_model_dir = settings.bound_trained_model_dir;
    }
}

vector<Candidate> BranchBoundStrategy::compute_scores(const Graph& graph) {
    if (equal_bounding_and_branching_methods) {
        return branch->compute_scores(graph);
    }
    vector<Candidate> branch_candidates = branch->compute_scores(graph);
    vector<Candidate> bound_candidates = bound->compute_scores(graph);
    
    double min_branch = 1e100, max_branch = 1e-100;
    for (int v = 0; v < graph.num_vertices; v++) {
        min_branch = min(min_branch, branch_candidates[v].branching_score);
        max_branch = max(max_branch, branch_candidates[v].branching_score);
    }

    for (int v = 0; v < graph.num_vertices; v++) {
        double branch_score = branch_candidates[v].branching_score;
        if (pair_branch) {
            branch_score = ceil(
                ((branch_score - min_branch) / (max_branch - min_branch + 1))
                * MAX_BRANCH_SCORE
            );
            branch_score += 2 * MAX_BRANCH_SCORE * bound_candidates[v].bounding_score;
        }
        bound_candidates[v].branching_score = branch_score;
    }
    return bound_candidates;
}

bool BranchBoundStrategy::equal_bounding_and_branching_scores() {
    if (pair_branch || equal_bounding_and_branching_methods) {
        return true;
    }
    return false;
}

const int NoStrategy::INF = 1234567890;

vector<Candidate> NoStrategy::compute_scores(const Graph& graph) {
    vector<Candidate> candidates;
    for (int v = 0; v < graph.num_vertices; v++) {
        candidates.emplace_back(v, INF, INF);
    }
    return candidates;
}

vector<Candidate> DegreeStrategy::compute_scores(const Graph& graph) {
    vector<Candidate> candidates;
    for (int v = 0; v < graph.num_vertices; v++) {
        candidates.emplace_back(v, graph.neighbours[v].size() + 1, graph.neighbours[v].size() + 1);
    }
    return candidates;
}

const int ColoringStrategy::UNCOLORED = -1;    

vector<Candidate> ColoringStrategy::compute_scores(const Graph& graph) {
    vector<Candidate> candidates = DegreeStrategy().compute_scores(graph);
    vector<Candidate> new_candidates = candidates;

    sort(candidates.begin(), candidates.end());
    reverse(candidates.begin(), candidates.end());
    const int max_degree = candidates[0].bounding_score;

    vector<int> coloring(graph.num_vertices, ColoringStrategy::UNCOLORED);

    for (Candidate& c : candidates) {
        // Set colors used by neighbours as available.
        vector<int> available_colors(max_degree + 1, 1);
        for (int neighbour : graph.neighbours[c.vertex]) {
            if (coloring[neighbour] != ColoringStrategy::UNCOLORED) {
                available_colors[coloring[neighbour]] = 0;
            }
        }
        // Find the lowest unused color.
        for (int color = 0; color < max_degree + 2; color++) {
            if (available_colors[color]) {
                new_candidates[c.vertex].bounding_score = color + 1;
                new_candidates[c.vertex].branching_score = color + 1;
                coloring[c.vertex] = color;
                break;
            }
        }
    }

    return new_candidates;
}

vector<Candidate> ColoringFeasibleStrategy::compute_scores(const Graph& graph) {
    vector<Candidate> candidates;
    for (int v = 0; v < graph.num_vertices; v++) {
        Graph subgraph = graph.create_graph_from_neighbours_of_vertex(v);
        int max_color = 0;
        
        if (subgraph.num_vertices > 0) {
            vector<Candidate> colors = ColoringStrategy().compute_scores(subgraph);
            for (const Candidate& c : colors) {
                max_color = max(max_color, c.bounding_score);
            }
        }
        
        candidates.emplace_back(v, max_color + 1, max_color + 1);
    }
    return candidates;
}

vector<Candidate> OptimalStrategy::compute_scores(const Graph& graph) {
    Graph graph_copy = graph.deep_copy();
    graph_copy.names_of_original_vertices.clear();
    graph_copy.names_of_original_vertices.resize(graph_copy.num_vertices);
    for (int v = 0; v < graph.num_vertices; v++) {
        graph_copy.names_of_original_vertices[v] = v;
    }

    Settings settings;
    FindCliques find_cliques(settings);
    find_cliques.find_max_clique_for_each_vertex(graph_copy);
    vector<Candidate> candidates;
    for (int v = 0; v < graph_copy.num_vertices; v++) {
        int mc = find_cliques.max_clique_for_each_vertex[v];
        candidates.emplace_back(v, mc, mc);
    }
    return candidates;
}

void NeuralStrategy::initialize() {
    Py_Initialize();
    PyRun_SimpleString("import clique_finding_models.predict as predict\n");
    string call_init = "predict.initialize(\"";
    call_init += trained_model_dir;
    call_init += "\")\n";
    PyRun_SimpleString(call_init.c_str());
}

vector<Candidate> NeuralStrategy::compute_scores(const Graph& graph) {
    vector<Candidate> candidates;
    for (int v = 0; v < graph.num_vertices; v++) {
        candidates.emplace_back(v, 1, 1);
    }

    if (graph.num_edges == 0) {
        return candidates;
    }

    FILE *temp_input_graph, *temp_clique_sizes;
    temp_input_graph = fopen("temp_input_graph.in", "w");
    write_graph(graph, temp_input_graph);
    fclose(temp_input_graph);

    PyRun_SimpleString("predict.predict(\"temp_input_graph.in\", \"temp_clique_sizes.out\")\n");

    temp_clique_sizes = fopen("temp_clique_sizes.out", "r");
    for (int v = 0; v < graph.num_vertices; v++) {
        double prediction = 0;
        fscanf(temp_clique_sizes, " %lf", &prediction);
        candidates[v].branching_score = prediction;
        candidates[v].bounding_score = ceil(prediction);
    }
    fclose(temp_clique_sizes);

    return candidates;
}

void NeuralStrategy::finish() {
    Py_Finalize();
}
