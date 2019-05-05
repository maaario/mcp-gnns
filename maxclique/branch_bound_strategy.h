#ifndef MAXCLIQUE_BRANCH_BOUND_STRATEGY_H
#define MAXCLIQUE_BRANCH_BOUND_STRATEGY_H

#include <memory>
#include <vector>

#include "graph.h"
#include "settings.h"

using namespace std;

// A container to hold vertex with assigned branch and bound scores, used as a result of 
// BranchBoundStrategy.
struct Candidate {
    // ID of candidate vertex.
    int vertex = 0;
    
    // The heuristic score of vertex. Vertices with higher scores will be explored sooner than 
    // those with lower scores.
    double branching_score = 0;

    // The upper bound for the max. number of vertices that can be potentially 
    // added to the clique from the candidate set. Equivalently, upper bound  on max. clique size 
    // induced by vertex and its neighbours.
    int bounding_score = 0;
    
    Candidate(int vertex, double branching_score, int bounding_score)
        : vertex(vertex), branching_score(branching_score), bounding_score(bounding_score) {}
    
    // Better candidate is the one with higher branchings score.
    bool operator< (const Candidate &other);
};

// Abstract class defining branch / bound heuristic interface.
struct Strategy {
    // A path from which trained model will be loaded. Supply before calling `initialize()`.
    string trained_model_dir;
    // This function is run in the start of clique finder.
    virtual void initialize() {};
    // This function is run when a new candidate set is constructed (possibly less often).
    virtual vector<Candidate> compute_scores(const Graph& graph) = 0;
    // This function is run in the destructor of clique finder.
    virtual void finish() {};
};  

// Based on settings, encapsulates branch strategy and bound strategy and combines their results
// to candidate set.
struct BranchBoundStrategy : Strategy {
    unique_ptr<Strategy> branch, bound;
    
    // If set use pair (bound, branch) as branching heuristic.
    bool pair_branch = false;

    // When using branch pair, normalize branch scores to range (0, MAX_BRANCH_SCORE).
    static const int MAX_BRANCH_SCORE;

    // If the same method is used for branch and bound, only one heuristic needs to be computed.
    bool equal_bounding_and_branching_methods = false;
    
    BranchBoundStrategy(Settings settings);

    // Run initialize functions of the encapsulated strategies.    
    void initialize() { branch->initialize(); bound->initialize(); }

    // Combine candidate scores gained from the encapsulated strategies.    
    vector<Candidate> compute_scores(const Graph& graph);

    // Run finish functions of the encapsulated strategies.    
    void finish()  { branch->finish(); bound->finish(); }

    // If candidates sorted by branching scores are also sorted by bounding scores,
    // the first candidate with bounding score too low terminates the recursive step.
    bool equal_bounding_and_branching_scores();
};

// All vertices are considered as equal to each other. The whole recursion tree should be visited.
struct NoStrategy : Strategy {
    static const int INF;
    
    vector<Candidate> compute_scores(const Graph& graph);
};

// Simple strategy which assigns scores as vertex degrees.
struct DegreeStrategy : Strategy {
    vector<Candidate> compute_scores(const Graph& graph);
};

// Sort vertices by degree descendingly, then assign the lowest possible color
// from \Delta + 1 colors to each vertex. Tomita 2003.
struct ColoringStrategy : Strategy {
    static const int UNCOLORED;
    
    vector<Candidate> compute_scores(const Graph& graph);
};

// Compute bounds for each vertex $v$ as max color of graph induced by $\{v\} \cup N(v)$.
struct ColoringFeasibleStrategy : Strategy {
    static const int UNCOLORED;
    
    vector<Candidate> compute_scores(const Graph& graph);
};


// Find max. clique sizes for all vertices and use this as "heuristic" function.
struct OptimalStrategy : Strategy {
    vector<Candidate> compute_scores(const Graph& graph);
};

// Score is computed with use of prototype 4 neural network.
struct NeuralStrategy : Strategy {
    // A Python interpreter is initialized and a trained model is loaded.
    void initialize();
    // At each call, a graph is written into file, model in python interpreter
    // predicts clique sizes and reports them into another file which is read to set scores.
    vector<Candidate> compute_scores(const Graph& graph);
    // Python interpreter is closed.
    virtual void finish();
};

#endif  // MAXCLIQUE_BRANCH_BOUND_STRATEGY_H
