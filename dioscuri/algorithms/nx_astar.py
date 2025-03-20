import networkx as nx
import numpy as np
import time

class nxAstar():
    def __init__(self):
        pass

    def astar_path(self, graph, source, target, weight=None):
        if not nx.algorithms.has_path(graph, source, target):
            return None
        if weight:
            path = nx.astar_path(graph, source, target, weight=weight)
        else:
            path = nx.astar_path(graph, source, target)
        return path
    