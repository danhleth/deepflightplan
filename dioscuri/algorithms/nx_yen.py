import networkx as nx

class nxYen():
    def __init__(self, top_k=1, retrieve_j = 50, weights=''):
        self.top_k = top_k
        self.retrieve_j = retrieve_j
        self.weights = weights

    def _choose_potential_weight(self):
        if self.weights == '':
            return None
        if len(self.weights) == 1:
            return self.weights[0]
        return self.random.choice(self.weights)
    
    def retrieve_multiple_shortest_path(self, G, source, target ):
        if not nx.algorithms.has_path(G, source, target):
            return None
        
        weight = self._choose_potential_weight()
        top_j_shortest_path = nx.algorithms.simple_paths.shortest_simple_paths(G, source, target, weight=weight)
        # paths = [next(paths) for _ in range(self.top_k)]
        paths = []
        for i in range(self.retrieve_j):
            try:
                path = next(top_j_shortest_path)
                paths.append(path)
            except StopIteration:
                break
        return paths