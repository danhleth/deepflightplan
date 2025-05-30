from geopy import distance
import numpy as np
from dioscuri.datasets.enroute_graph_wrapper import WaypointNode

class GeopyGreatCircleDistance:
    def __init__(self):
        pass

    def compute_distance(self,NodeA, NodeB):
        """
        Calculate the great circle distance between two nodes
        """
        if type(NodeA) == type(NodeB) == WaypointNode:
            x1, y1 = NodeA.lat, NodeA.long
            x2, y2 = NodeB.lat, NodeB.long
            return distance.great_circle((x1, y1), (x2, y2)).miles 

        if type(NodeA) == type(NodeB) == np.ndarray:
            x1, y1 = NodeA[0], NodeA[1]
            x2, y2 = NodeB[0], NodeB[1]
            return distance.great_circle((x1, y1), (x2, y2)).miles
        
        return distance.great_circle(NodeA, NodeB).miles
    