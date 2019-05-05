#ifndef MAXCLIQUE_SETTINGS_H
#define MAXCLIQUE_SETTINGS_H

#include <string>

using namespace std;

struct Settings {
    string metrics_file_path = "";
    string branch = "coloring";
    string bound = "coloring";
    string branch_trained_model_dir = "";
    string bound_trained_model_dir = "";
    bool pair_branch = false;
    int max_calls = 0;
};

#endif  // MAXCLIQUE_SETTINGS_H