#include <cstdio>

#include "maxclique_utils.h"
#include "settings.h"

using namespace std;

int main() {
    const Graph graph = read_graph(stdin);
    Settings settings;
    FindCliques find_cliques(settings);
    find_cliques.find_max_clique_for_each_vertex(graph);

    for(int mc : find_cliques.max_clique_for_each_vertex)
        printf("%d ", mc);
    printf("\n");
}
