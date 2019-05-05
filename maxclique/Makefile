LIBRARIES_CPP = graph.cpp branch_bound_strategy.cpp maxclique_utils.cpp cxxopts.hpp

PYTHON_LINKS = -I/usr/include/python3.5m -I/usr/include/x86_64-linux-gnu/python3.5m -lpython3.5m

maxclique : main_maxclique.cpp main_clique_for_each_vertex.cpp maxclique_utils.cpp
	g++ $(LIBRARIES_CPP) main_maxclique.cpp -o main_maxclique -std=c++14 -W -Wall $(PYTHON_LINKS)
	g++ $(LIBRARIES_CPP) main_clique_for_each_vertex.cpp -o main_clique_for_each_vertex -std=c++14 -W -Wall $(PYTHON_LINKS)

# Convert .clq (graph format provided by DIAMACS) to .in (simple text graph representation)
%.in : %.clq 
	cat $*.clq | grep -v '^c' | sed 's/p [^0-9]*//' | sed 's/e //' > $*.in

test: maxclique
	for f in 00000 00001 00002 ; do \
		./main_clique_for_each_vertex < test/$$f.in > test/$$f.out_test; \
		diff test/$$f.out test/$$f.out_test; \
	done
	for strategy in no degree coloring coloring_feasible optimal; do \
		for f in 00000 00001 00002 ; do \
			./main_maxclique --branch=$$strategy --bound=$$strategy < test/$$f.in > test/$$f.max_out_test; \
			diff test/$$f.max_out test/$$f.max_out_test; \
		done; \
	done
	for f in 00000 00001 00002 ; do \
		./main_maxclique --branch=neural --bound=coloring \
		--branch-dir=../experiments/02_tune_models/selected_trained_models/chebnet_rank \
		< test/$$f.in > test/$$f.max_out_test; \
		diff test/$$f.max_out test/$$f.max_out_test; \
	done;

install:
	# Download cxxopts lightwight commandline arguments library header.
	wget https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp
