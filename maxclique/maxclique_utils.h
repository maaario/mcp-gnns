#ifndef MAXCLIQUE_UTILS_H
#define MAXCLIQUE_UTILS_H
/*
 * Implementation of a simple branch-and-bound algorithm to find maximal
 * cliques as described in Approach described in the paper 
 * "An Efficient Branch-and-Bound Algorithm for Finding a Maximum Clique" 
 * by Etsuji Tomita and Tomokazu Seki:
 * 
 * Input:
 * <num. of vertices> <num. of edges>
 * <vertex1 id> <vertex2 id>            // Vertices are indexed from 1.
 * <vertex3 id> <vertex4 id>
 * ...
 * 
 * Output:
 * <size of max. clique>
 * <list of space separated vertex ids in clique>
 */

#include <memory>
#include <vector>

#include "branch_bound_strategy.h"
#include "graph.h"
#include "metrics.cpp"
#include "settings.h"

using namespace std;

struct FindCliques {
    
    unique_ptr<BranchBoundStrategy> strategy;
    unique_ptr<MetricsManager> metrics_manager;

    // The current and the largest cliques  will be stored here. 
    // Initial max. clique has size of 0.
    vector<int> clique, max_clique;

    // A copy of settings of the run.
    Settings settings;

    FindCliques(Settings settings);
    ~FindCliques();
    
    // Branch and bound for finding max. clique: selects one vertex c.vertex that is 
    // added to clique and removed from candidates. For a next recursion step
    // only candidates neighbouring with c.vertex are considered.
    void find_max_clique_recurse(Graph &graph, int depth=1, int parent_call_id=0);

    static void test_clique(const Graph& graph, const vector<int>& clique);

    void find_max_clique(const Graph& graph);

    vector<int> max_clique_for_each_vertex;

    // A placeholder for a vertex of maximum clique if we only remember the clique size.
    static const int GENERIC_VERTEX;

    void find_max_clique_for_each_vertex(const Graph& graph);
};

#endif