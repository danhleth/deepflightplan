import networkx as nx

class nxYen():
    def __init__(self, top_k=1):
        self.top_k = top_k

    def top_k_shortest_path(self, G, source, target, weight=None):
        if not nx.algorithms.has_path(G, source, target):
            return None
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