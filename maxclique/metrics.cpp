#ifndef MAXCLIQUE_METRICS_H
#define MAXCLIQUE_METRICS_H

#include <chrono>
#include <cstdio>

using namespace std;
using namespace std::chrono;

// In this container, metrics from one recursion call are collected and then can be
// processed / written into file in the end of the call.
struct RecursionCallMetrics {
    int call_id = 0;
    int parent_call_id = 0;
    int recursion_depth = 0;
    int num_candidates = 0;
    int num_candidates_visited = 0;
    // Number of edges as a proxy of the work needed to compute the heuristic.
    int num_graph_edges = 0;
    // 0 or the size of the new max. clique.
    int larger_clique_discovered = 0;
    
    // Seconds (or fractions of second) spent during one heuristic computation or in one call.
    double seconds_in_heuristic = 0;
    double seconds_in_call = 0;

    high_resolution_clock::time_point heuristic_start;
    
    void start_heuristic_stopwatch() {
        heuristic_start = high_resolution_clock::now();
    }

    void stop_heuristic_stopwatch() {
        high_resolution_clock::time_point heuristic_stop = high_resolution_clock::now();
        seconds_in_heuristic = duration_cast<duration<double>>(
            heuristic_stop - heuristic_start).count();
    }

    high_resolution_clock::time_point last_call_start;
    
    void start_call_stopwatch() {
        last_call_start = high_resolution_clock::now();
    }
    
    void stop_call_stopwatch() {
        high_resolution_clock::time_point last_call_stop = high_resolution_clock::now();
        seconds_in_call += duration_cast<duration<double>>(
            last_call_stop - last_call_start).count();
    }
};

// Manages a file / files to which branch and bound metrics are written.
// Counts the number of function calls and thus keep tracks of call ids.
// Enables to create a "RecursionCallMetrics" which collects data from one recursion call
// and then to write these data in the end of the call.
struct MetricsManager {
    int next_call_id = 1;

    FILE* file = nullptr;

    MetricsManager(string file_path) {
        if (file_path != "") {
            file = fopen(file_path.c_str(), "w");
        }
        if (file) {
            fprintf(file, "call_id,parent_call_id,recursion_depth," 
                        "num_candidates,num_candidates_visited,num_graph_edges,"
                        "larger_clique_discovered,seconds_in_call,seconds_in_heuristic\n"
            );
        }
    }

    RecursionCallMetrics get_new_call_metrics(int parent_call_id) {
        RecursionCallMetrics metrics;
        metrics.call_id = next_call_id++;
        metrics.parent_call_id = parent_call_id;
        return metrics;
    }

    void save_call_metrics(RecursionCallMetrics &metrics) {
        if (file) {
            fprintf(file, "%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", 
                metrics.call_id, metrics.parent_call_id, metrics.recursion_depth, 
                metrics.num_candidates, metrics.num_candidates_visited, metrics.num_graph_edges,
                metrics.larger_clique_discovered, metrics.seconds_in_call, 
                metrics.seconds_in_heuristic
            );
        }
    }
};

#endif  // MAXCLIQUE_METRICS_H
