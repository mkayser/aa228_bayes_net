import Queue
import sys
import argparse
import numpy as np
import random
import math
import itertools

ALPHA = 1

class NodeState(object):
    index=None
    parents=None
    score=None

    def __init__(self,i,p,scorer,rows):
        self.index = i
        self.parents = p
        self.score = scorer.score(self.index, self.parents, rows)

class NodeQueue(object):
    queue=None
    history=None

    def __init__(self):
        self.queue = Queue.PriorityQueue()
        history = set()

    def put(self, priority, node):
        signature = [node.index] + sorted(node.parents)
        if signature not in history:
            self.queue.put((priority,node))
            history.add(signature)
    
    def get(self):
        try:
            return self.queue.get()[1]
        except Queue.Empty:
            return None

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
        tuples, counts = np.unique([tuple(row) for row in subdata], return_counts=True)

        lga = math.lgamma(ALPHA)

        total = 0.0
        for t,c in zip(tuples,counts):
            total += math.lgamma(ALPHA+c)-lga
        for p,g in itertools.groupby(zip(tuples,counts), lambda t: t[0][:-1]):
            mij0 = sum(i[1] for i in g)
            aij0 = ALPHA * len(g)
            total += math.lgamma(aij0) - math.lgamma(aij0+mij0)
        return score

class RandomNodeStateGen(object):
    def __init__(self):
        pass

    def generate(self, i, scorer, rows, n):
        arities = range(0,rows.nvars,max(1,int(rows.nvars/n)))
        nodes = []
        otherindices = [j for j in range(n) if j!=i]
        
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
    gen=None
    
    def __init__(self, _n_iters, _n_init):
        self.n_iters = _n_iters
        self.n_init = _n_init
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
            print("Score: {}".format(n.score))
            for neighbor in self.neighbors(n,scorer,rows):
                q.put(-neighbor.score, neighbor)
        return sorted(final_list, lambda x,y: cmp(-x.score,-y.score))

    def neighbors(n, scorer, rows):
        # three operations: remove parent, add parent, swap parent with new parent        
        results = []

        # Remove
        for i in range(len(n.parents)):
            newparents = list(n.parents)
            del newparents[i]
            results.add(NodeState(n.index,newparents,scorer,rows))

        # Swap
        for i in range(len(n.parents)):
            for j in range(rows.nvars):
                if j not in parents:
                    newparents = list(n.parents + [j])
                    del newparents[i]
                    newparents = sorted(newparents)
                    results.add(NodeState(n.index,newparents,scorer,rows))

        # Add
        for j in range(rows.nvars):
            if j not in parents:
                newparents = sorted(list(n.parents + [j]))
                results.add(NodeState(n.index,newparents,scorer,rows))
        
        return results
        
            


def main():
    rows_file = sys.argv[1]
    rows = Rows(rows_file)

    searcher = Search(10, 10)
    scorer = Scorer()

    best_node_states = [searcher.search(i,scorer,rows) for i in range(rows.nvars)]
    print("Lengths=".format([len(i) for i in best_node_states]))
    

main()
