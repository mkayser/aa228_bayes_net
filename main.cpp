#include <stdio>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

class Util {
public:
  std::vector<bool> one_hot_vector(int len, int index) {
    std::vector<bool> v(len, false);
    v[index] = true;
    return v;
  }

  template <typename K, typename V> 
  V lookup_or_populate(std::unordered_map<K,V>& map, K& key, V& val) {
    auto i = map.find(key);
    if(i==map.end()) {
      map[key] = val;
      return val;
    }
    else {
      return *i;
    }
  }

  void read_rows(std::vector<std::vector<bool> >& rows, std::string file_name) {
    rows.clear();
    // TODO read lines
  }
};

class Scorer {
public:
  struct ScoreInfo {
    double total_score;
    std::vector<double> node_scores;
    ScoreInfo(int n) : node_scores(n,0), total_score(0) {
    }
  };

  Scorer(double alpha) : _alpha(alpha) {
  }

  void update_score(ScoreInfo& info, std::vector<bool>& nodes_to_update, std::vector<std::vector<int> >& parents, std::vector<std::vector<bool> >& rows) {
    std::vector<std::unordered_map<std::vector<int>, int > > observed_parent_configurations;
    std::vector<std::vector<std::vector<int> > > m; // node -> pconfindex -> nodeval -> count
    int n = rows[0].size();

    for(auto r: rows) {
      for(int i=0; i<n; i++) {
	val = r[i] ? 1 : 0;

	if(nodes_to_update[i]) {
	  // make pconf
	  std::vector<int> pconf;
	  for(auto pnode: parents[i]) {
	    pconf.push_back(r[pnode]);
	  }

	  // get index of pconf (make index if necessary)
	  int pconfindex = Util::lookup_or_populate(observed_parent_configurations[i], pconf, observed_parent_configurations[i].size());

	  // 
	  if(pconfindex >= m[i].size()) {
	    m[i].push_back(std::vector<int>(2,0));
	  }
	  m[i][pconfindex][val]++;
	}
      }
    }

    a_ij0 = _alpha * 2;

    info.total_score = 0.0;

    for(int i=0; i<n; i++) {
      if(nodes_to_update[i]) {
	info.node_scores[i] = 0;
	for(j=0; j<m[i].size(); j++) {
	  m_ij0 = std::accumulate(m[i][j].begin(); m[i][j].end(), 0);
	  info.node_scores[i] += lgamma(a_ij0) - lgamma(a_ij0 + m_ij0);
	  for(k=0; k<2; k++) {
	    if(m[i][j][k]>0) {
	      info.node_scores[i] += lgamma(_alpha + m[i][j][k]) - lgamma(_alpha);
	    }
	  }
	}
      }
      info.total_score += info.node_scores[i];
    }
  }

  void update_score(ScoreInfo& info, int node_to_update, std::vector<std::vector<int> >& parents, std::vector<std::vector<bool> >& rows) {
    std::vector<bool> nodes_to_update = Util::one_hot_vector(parents.size(), node_to_update);
    update_score(score, nodes_to_update, parents, rows);
  }

  void compute_score(ScoreInfo& score, std::vector<std::vector<int> >& parents, std::vector<std::vector<bool> >& rows) {
    std::vector<bool> nodes_to_update(parents.size(), true);
    update_score(score, nodes_to_update, parents, rows);
  }
  
private:
  double _alpha;
};

class LocalSearch {
public:
  LocalSearch()
private:

};

int main(int argc, char* argv[]) {
  std::vector<std::vector<bool> > rows;
  Util::read_rows(rows, std::string<argv[1]>);
  
}
