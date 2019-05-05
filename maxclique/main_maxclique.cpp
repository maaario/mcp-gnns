#include <cstdio>

#include "cxxopts.hpp"

#include "maxclique_utils.h"
#include "settings.h"

using namespace std;

Settings parse(int argc, char* argv[]) {
    cxxopts::Options options(argv[0], " - find the maximum clique with branch and bound.");

    options
      .add_options()
      ("branch", "branching strategy", cxxopts::value<string>()->default_value("coloring"))
      ("bound", "bounding strategy", cxxopts::value<string>()->default_value("coloring"))
      ("m,metrics", "metrics file path", cxxopts::value<string>()->default_value(""))
      ("branch-dir", "directory of a trained neural model to use for branching", 
      cxxopts::value<string>()->default_value(""))
      ("bound-dir", "directory of a trained neural model to use for bounding", 
      cxxopts::value<string>()->default_value(""))
      ("pair-branch", "for branch score use pair (bound, branch)")
      ("max-calls", "kill after 'max-calls' recursive calls", 
      cxxopts::value<int>()->default_value("0"));

    auto arguments = options.parse(argc, argv);
    
    Settings settings;
    settings.branch = arguments["branch"].as<string>();
    settings.bound = arguments["bound"].as<string>();
    settings.metrics_file_path = arguments["metrics"].as<string>();
    settings.branch_trained_model_dir = arguments["branch-dir"].as<string>();
    settings.bound_trained_model_dir = arguments["bound-dir"].as<string>();
    settings.pair_branch = arguments["pair-branch"].as<bool>();
    settings.max_calls = arguments["max-calls"].as<int>();

    return settings;
}

int main(int argc, char* argv[]) {
    Settings settings = parse(argc, argv);

    const Graph graph = read_graph(stdin);
    FindCliques find_cliques(settings);
    find_cliques.find_max_clique(graph);

    // Print out the result as the list of vertices in clique.
    printf("%d\n", int(find_cliques.max_clique.size()));
    for(int v : find_cliques.max_clique)
        printf("%d ", v);
    printf("\n");
}
