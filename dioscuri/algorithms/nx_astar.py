import networkx as nx
import numpy as np
import time

class nxAstar():
    def __init__(self, top_k=1):
        self.top_k = top_k
        pass

    def top_k_shortest_path(self, graph, source, target, weight=None):
        if not nx.algorithms.has_path(graph, source, target):
            return None
        return self.astar_path(graph, source, target, weight=weight)
    
    def astar_path(self, graph, source, target, weight=None):
        if not nx.algorithms.has_path(graph, source, target):
            return None
        path = nx.astar_path(graph, source, target, weight=weight)
        return path
    