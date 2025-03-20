from geopy import distance

class GeopyGreatCircleDistance:
    def __init__(self):
        pass

    def compute_distance(self,NodeA, NodeB):
        """
        Calculate the great circle distance between two nodes
        """

        x1, y1 = NodeA.lat, NodeA.long
        x2, y2 = NodeB.lat, NodeB.long
        return distance.great_circle((x1, y1), (x2, y2)).nm   