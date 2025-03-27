import networkx as nx

class nxYen():
    def __init__(self, top_k=1, weights=''):
        self.top_k = top_k
        self.weights = weights

    def _choose_potential_weight(self):
        if self.weights == '':
            return None
        if len(self.weights) == 1:
            return self.weights[0]
        return self.random.choice(self.weights)
    
    def top_k_shortest_path(self, G, source, target):
        if not nx.algorithms.has_path(G, source, target):
            return None
        
        weight = self._choose_potential_weight()
        top_k_shortest_path = nx.algorithms.simple_paths.shortest_simple_paths(G, source, target, weight=weight)
        # paths = [next(paths) for _ in range(self.top_k)]
        paths = []
        for i in range(self.top_k):
            try:
                path = next(top_k_shortest_path)
                paths.append(path)
            except StopIteration:
                break
        return paths