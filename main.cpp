#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include <unordered_map>

namespace std {

  template< typename T >
  typename vector<T>::iterator 
  insert_sorted( vector<T> & vec, T const& item )
  {
    return vec.insert
      ( 
       upper_bound( vec.begin(), vec.end(), item ),
       item 
        );
  }


  template <>
  struct hash<std::vector<int>>
  {
    std::size_t operator()(const std::vector<int>& v) const
    {
      using std::size_t;
      using std::hash;
      using std::string;

      // Compute individual hash values for first,
      // second and third and combine them using XOR
      // and bit shifting:
      std::size_t result = 0;
      for(auto i: v) {
	result = (result << 1) ^ (hash<int>()(i));
      }
    }
  };

}

typedef std::vector<std::vector<int> > Parents;
typedef std::vector<std::vector<bool> > Rows;

class Graph {
public:
  int n;
  Parents parents;

  Graph(int _n) : n(_n) {
    parents = Parents(n);
  }
};

class Util {
public:
  static void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    elems.clear();
    while (std::getline(ss, item, delim)) {
      elems.push_back(item);
    }
  }    


  static std::vector<bool> one_hot_vector(int len, int index) {
    std::vector<bool> v(len, false);
    v[index] = true;
    return v;
  }

  static int lookup_or_populate(std::unordered_map<std::vector<int>,int>& map, std::vector<int>& key, int val) {
    std::unordered_map<std::vector<int>,int>::const_iterator i = map.find(key);
    if(i==map.end()) {
      map[key] = val;
      return val;
    }
    else {
      return i->second;
    }
  }

  static void read_rows(Rows& rows, std::string file_name) {
    rows.clear();
    
    std::ifstream infile(file_name);

    std::string line;
    std::getline(infile, line);

    std::vector<std::string> tokens;

    while (std::getline(infile, line))
      {
	std::vector<bool> row;
        split(line, ',', tokens);
	for(auto tok: tokens) {
	  if(tok == "\"true\"") {
	    row.push_back(true);
	  }
	  else if(tok == "\"false\"") {
	    row.push_back(false);
	  }
	  else {
	    exit(1);
	  }
	}
	rows.push_back(row);
      }
  }
};

// class generator:
class RangeGen {
  int current;
  RangeGen() {current=0;}
  int operator()() {return ++current;}
};


class Scorer {
public:

  Scorer(double alpha) : _alpha(alpha) {
  }

  void compute_node_scores(Graph& graph, Rows& rows, std::vector<int>& nodes_to_update, std::vector<double>& node_scores) {
    node_scores.clear();

    std::vector<std::unordered_map<std::vector<int>, int > > observed_parent_configurations;
    std::vector<std::vector<std::vector<int> > > m; // node -> pconfindex -> nodeval -> count
    int n = rows[0].size();

    observed_parent_configurations.resize(n);
    m.resize(n);

    for(auto r: rows) {
      for(auto i: nodes_to_update) {
	int val = r[i] ? 1 : 0;

	// make pconf
	std::vector<int> pconf;
	for(auto pnode: graph.parents[i]) {
	  pconf.push_back(r[pnode]);
	}

	// get index of pconf (make index if necessary)

	int pconfindex = Util::lookup_or_populate(observed_parent_configurations[i], pconf, observed_parent_configurations[i].size());
	if(pconfindex >= m[i].size()) {
	  m[i].push_back(std::vector<int>(2,0));	
	}
	m[i][pconfindex][val]++;
      }
    }


    double a_ij0 = _alpha * 2;

    for(auto i: nodes_to_update) {
      double node_score = 0.0;
      for(int j=0; j<m[i].size(); j++) {
	int m_ij0 = std::accumulate(m[i][j].begin(), m[i][j].end(), 0);
	node_score += lgamma(a_ij0) - lgamma(a_ij0 + m_ij0);
	for(int k=0; k<2; k++) {
	  if(m[i][j][k]>0) {
	    node_score += lgamma(_alpha + m[i][j][k]) - lgamma(_alpha);
	  }
	}
      }
      node_scores.push_back(node_score);
    }
  }

  void compute_all_node_scores(Graph& graph, Rows& rows, std::vector<double>& node_scores) {
    std::vector<int> nodes_to_update(graph.n);
    std::generate(nodes_to_update.begin(), nodes_to_update.end(), RangeGen());
    compute_node_scores(graph, rows, nodes_to_update, node_scores);
  }
  
private:
  double _alpha;
};

struct Action {
public:
  enum Type {ADD,REMOVE,REVERSE};
  Type type;
  int parent;
  int child;
};

struct SearchState {
public:
  struct ActionSet {
    std::vector<Action> actions;
    std::vector<double> score_deltas;
    std::vector<bool> valid;
    std::vector<bool> dirty;
    void init(Graph) {
    }
  };
  struct Score {
    std::vector<double> node_scores;
    std::vector<bool> dirty;
    double total_score;
  };

  Graph& graph;
  Scorer& scorer;
  Rows& rows;

  Score score;
  ActionSet actions;

  SearchState(Graph& g, Scorer& s, Rows& r) : graph(g), scorer(s), rows(r) {
    scorer.compute_all_node_scores(graph, rows, score.node_scores);
    score.total_score = std::accumulate(score.node_scores.begin(), score.node_scores.end(), 0);
    actions.init(g.n);
  }

  void update(Graph& graph, Scorer& scorer, Rows& rows, Action action) {
    // TODO updates the graph, action set and scores
  }
};

class LocalSearch {
public:

  struct Params {
    int n_iters;
    bool strict;
    float anneal_start;
    float anneal_slope;
  };

  LocalSearch() {
  }

  void run(Scorer& scorer, Graph& graph, Rows& rows, Params& params) {
    Scorer::ScoreInfo score(graph.n);

    ActionSet actions(graph, scorer, rows);
    
    std::cout << "Total Score: " << score.total_score << std::endl;
    
    Action action;

    for(int i=0; i<params.n_iters; i++) {
      bool better_action_exists = decide(params, actions, action, score);
      if(better_action_exists) {
	actions.update(graph, scorer, rows, action, score);
      }
      else break;
    }


  }

  Action decide(Params& params, ActionSet& actions, Action& action, double& score) {
    // Depending on policy, choose next action
  }
};

int main(int argc, char* argv[]) {
  Rows rows;
  Scorer scorer(1.0);
  Util::read_rows(rows, std::string(argv[1]));
  LocalSearch search;
  Graph graph(rows[0].size());
  search.run(scorer, graph, rows);
}
