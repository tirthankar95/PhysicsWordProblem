import numpy as np
import logging
from collections import defaultdict
operators = ['+', '*', '=', '/']

def parse(equation):
    elements_ = [var for var in equation.split(' ') if var not in operators]
    elements = []
    for x in elements_:
        try: var = float(x)
        except: elements.append(x)
    return elements

class GraphEquation:
    def __init__(self, equations):
        self.equation_element = []
        for equation in equations:
            self.equation_element.append(parse(equation))
        N = len(self.equation_element)
        self.adj = defaultdict(list)
        for i in range(N):
            for j in range(N):
                if i == j: continue 
                allEdges = [(j, ch) for ch in set(self.equation_element[i]).intersection(set(self.equation_element[j]))]
                self.adj[i].extend(allEdges)
    def getEquation(self):
        qid = np.random.randint(len(self.adj))
        threshold, eqn = 0.5, [qid]
        self.vis, unk = defaultdict(bool), defaultdict(bool)
        self.qu = [qid]
        self.vis[qid] = True 
        while len(self.qu):
            src = self.qu.pop(0)
            # PICK or don't PICK
            if 0.5 + np.random.normal() >= threshold:
                edgeId = np.random.randint(len(self.adj[src]))
                edge = self.adj[src][edgeId]
                unk[edge[-1]] = True 
                if edge not in self.vis:
                    eqn.append(edge[0])
                    self.qu.append(edge[0])
                    self.vis[edge[0]]
        unk[np.random.choice(self.equation_element[eqn[-1]])] = True 
        known = defaultdict(bool)
        for eId in eqn:
            for ch in self.equation_element[eId]:
                if ch not in unk:
                    known[ch] = True 
        unk, known = [k for k in unk.keys()], [k for k in known.keys()]
        logging.debug(f'{unk}, {known}')
        return unk, known, eqn
    


    