import numpy as np

class Hausdorff:
    """
    Class to calculate the Hausdorff distance between two sets of points.
    """

    def __init__(self):
        pass

    
    def d_hausdorff_distance(self, set_a, set_b, distance_function=None):
        """
        Calculate the directed Hausdorff distance from set_a to set_b.
        """
        # Implementation of directed Hausdorff distance calculation
        max_distance = 0.0
        for point_a in set_a:
            min_distance = float('inf')
            for point_b in set_b:
                distance = distance_function(point_a, point_b) if distance_function else 0
                if distance < min_distance:
                    min_distance = distance
            if min_distance > max_distance:
                max_distance = min_distance
        
        return max_distance

    def calculate(self, XA, XB, distance_function=None):
        """
        Calculate the Bi-Hausdorff distance from set_a to set_b.
        """
        nA = XA.shape[0]
        nB = XB.shape[0]
        cmax = 0.

        # Calculate the directed Hausdorff distance from A to B
        for i in range(nA):
            cmin = np.inf
            for j in range(nB):
                d = distance_function(XA[i,:], XB[j,:])
                if d<cmin:
                    cmin = d
                if cmin<cmax:
                    break
            if cmin>cmax and np.inf>cmin:
                cmax = cmin

        # Calculate the directed Hausdorff distance from B to A
        for j in range(nB):
            cmin = np.inf
            for i in range(nA):
                d = distance_function(XA[i,:], XB[j,:])
                if d<cmin:
                    cmin = d
                if cmin<cmax:
                    break
            if cmin>cmax and np.inf>cmin:
                cmax = cmin

        # Return the maximum of the two minimal directed distances       
        return cmax
    
    def __call__(self, *args, **kwds):
        self.calculate(*args, **kwds)