import Queue
import sys
import argparse
import numpy as np
import random
import math
import itertools
import collections

ALPHA = 1

class GraphState(object):
    nodes=None
    indices=None
    score=None

    def __init__(self,ns):
        self.nodes = ns
        self.indices = range(len(ns))
        self.score = sum(n.score for n in self.nodes)

    def parents_of(self,i):
        return self.nodes[i].parents

    def __repr__(self):
        return "\n".join([n.__repr__() for n in self.nodes, self.score])

class NodeState(object):
    index=None
    parents=None
    score=None

    def __init__(self,i,p,scorer,rows):
        self.index = i
        self.parents = p
        #print ("Creating node with index={} and parents={}".format(self.index,self.parents))
        self.score = scorer.score(self.index, self.parents, rows)

    def __str__(self):
        return "{}_{}_{}".format(self.index,self.parents,self.score)

    def __repr__(self):
        return "i={}: p={} (s={})".format(self.index,self.parents,self.score)

class NodeQueue(object):
    queue=None
    history=None

    def __init__(self):
        self.queue = Queue.PriorityQueue()
        self.history = set()

    def put(self, priority, node):
        signature = (node.index,) + tuple(sorted(node.parents))
        if signature not in self.history:
            self.queue.put((priority,node))
            self.history.add(signature)
    
    def get(self):
        try:
            return self.queue.get()[1]
        except Queue.Empty:
            return None

    def empty(self):
        return self.queue.empty()

class Rows(object):
    headers=None
    data=None
    nvars=None

    def parse(self,line):
        return [i.strip().strip("\"") for i in line.strip().split(",")]

    def __init__(self, file_name):
        with open(file_name) as fin:
            header_line = next(fin)
            self.headers = self.parse(header_line)
            self.nvars = len(self.headers)

            print("nvars={}".format(self.nvars))

            rows = []
            for line in fin:
                row = np.array(self.parse(line))
                assert(len(row)==len(self.headers))
                rows.append(row=="true")
            self.data = np.vstack(rows)

class Scorer(object):
    def __init__(self):
        pass

    def score(self, index, parent_indices, rows):
        indices = parent_indices + [index]
        subdata = rows.data[:,indices]
        tuplecounts = collections.Counter([tuple(row) for row in subdata])

        lga = math.lgamma(ALPHA)

        total = 0.0
        for t,c in tuplecounts.iteritems():
            total += math.lgamma(ALPHA+c)-lga
        for p,g in itertools.groupby(sorted(tuplecounts.items()), lambda t: t[0][:-1]):
            lg = list(g)
            mij0 = sum(i[1] for i in lg)
            aij0 = ALPHA * len(list(lg))
            total += math.lgamma(aij0) - math.lgamma(aij0+mij0)
        #print("Score: {}".format(total))
        return total

class RandomNodeStateGen(object):
    def __init__(self):
        pass

    def generate(self, i, scorer, rows, n):
        arities = range(0,rows.nvars,max(1,int(rows.nvars/n)))
        nodes = []
        otherindices = [j for j in range(rows.nvars) if j!=i]
        
        while len(nodes) < n:
            for a in arities:
                if len(nodes)==n:
                    break
                ns = NodeState(i,sorted(random.sample(otherindices,a)),scorer,rows)
                nodes.append(ns)
        return nodes
                
            

class Search(object):
    n_iters=None
    n_init=None
    limit=None
    gen=None
    
    def __init__(self, _n_iters, _n_init, _limit):
        self.n_iters = _n_iters
        self.n_init = _n_init
        self.limit = _limit
        self.gen = RandomNodeStateGen()

    def search(self, i, scorer, rows):
        
        q = NodeQueue()
        initial_nodes = self.gen.generate(i,scorer,rows,self.n_init)

        final_list = []

        for n in initial_nodes:
            q.put(-n.score, n)
        for iter_num in range(self.n_iters):
            n = q.get()
            final_list.append(n)
            #print("Score: {}".format(n.score))
            for neighbor in self.neighbors(n,scorer,rows):
                q.put(-neighbor.score, neighbor)
        while not q.empty():
            final_list.append(q.get())
        return sorted(final_list, lambda x,y: cmp(-x.score,-y.score))

    def neighbors(self, n, scorer, rows):
        # three operations: remove parent, add parent, swap parent with new parent        
        results = []

        # Remove
        for i in range(len(n.parents)):
            newparents = list(n.parents)
            del newparents[i]
            results.append(NodeState(n.index,newparents,scorer,rows))

        # Swap
        for i in range(len(n.parents)):
            for j in range(rows.nvars):
                if j not in n.parents and j!=n.index:
                    newparents = list(n.parents + [j])
                    del newparents[i]
                    newparents = sorted(newparents)
                    results.append(NodeState(n.index,newparents,scorer,rows))

        # Add
        for j in range(rows.nvars):
            if j not in n.parents and j!=n.index:
                newparents = sorted(list(n.parents + [j]))
                results.append(NodeState(n.index,newparents,scorer,rows))
        
        return results
        

# The code for "cyclic" is taken from a Stack Overflow post: http://codereview.stackexchange.com/questions/86021/check-if-a-directed-graph-contains-a-cycle
def cyclic(g):
    """Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False

    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.parents_of(vertex):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g.indices)

class GraphQueue(object):
    queue=None
    history=None

    def __init__(self):
        self.queue = Queue.PriorityQueue()
        self.history = set()

    def put(self, priority, indexing):
        signature = tuple(indexing)
        if signature not in self.history:
            self.queue.put((priority,indexing))
            self.history.add(signature)
    
    def get(self):
        try:
            return self.queue.get()[1]
        except Queue.Empty:
            return None
            
    def empty(self):
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()

class GraphSearch(object):
    node_state_seqs=None
    q=None

    def __init__(self,state_seqs):
        self.node_state_seqs = state_seqs
        self.q = GraphQueue()
        initial_state = (0,) * len(self.node_state_seqs)
        self.q.put(-1*self.score(initial_state), initial_state)

    def extract_best_acyclic(self):
        print("Q size: {}".format(self.q.size()))
        while self.q.size() < 10000:
            indexing = self.q.get()
            if indexing is None:
                return None

            g = self.make_graph_from_indexing(indexing)
            print("Try: {}".format(g))
            if not cyclic(g):
                return g
            else:
                for idx in self.neighbors(indexing):
                    self.q.put(-1*self.score(idx), idx)
        while not self.q.empty():
            indexing = self.q.get()
            if indexing is None:
                return None

            g = self.make_graph_from_indexing(indexing)
            print("Try: {}".format(g))
            if not cyclic(g):
                return g
        return None

    def neighbors(self, indexing):
        for i in range(len(indexing)):
            new_ind = indexing[:i] + (indexing[i]+1,) + indexing[i+1:]
            if self.indexing_is_valid(new_ind):
                yield new_ind

    def indexing_is_valid(self, idx):
        return (not any(j >= len(self.node_state_seqs[i]) for i,j in enumerate(idx)))

    def make_node_list_from_indexing(self, idx):
        return [self.node_state_seqs[i][j] for i,j in enumerate(idx)]

    def make_graph_from_indexing(self, idx):
        return GraphState(self.make_node_list_from_indexing(idx))

    def score(self,indexing):
        assert (self.indexing_is_valid(indexing))
        nodes = self.make_node_list_from_indexing(indexing)
        scores = [n.score for n in nodes]
        return sum(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("-n_init", type=int, default=5, help="Number of initial random nodestates")
    parser.add_argument("-n_iter", type=int, default=5, help="Number of iterations")
    parser.add_argument("-limit", type=int, default=100, help="Hard limit on number of states evaluated per node")
    
    args = parser.parse_args()

    rows = Rows(args.csv)

    searcher = Search(args.n_iter, args.n_init, args.limit)
    scorer = Scorer()

    best_node_states = [searcher.search(i,scorer,rows) for i in range(rows.nvars)]
    for i,s in enumerate(best_node_states):
        print("Node #{}".format(i))
        for n in s:
            print("  {}".format(n))
    print("Lengths={}".format([len(i) for i in best_node_states]))

    gs = GraphSearch(best_node_states)
    g = gs.extract_best_acyclic()
    print("{}\nCyclic={}".format(g, cyclic(g)))
    

main()
