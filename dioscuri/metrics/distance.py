import numpy as np
import pandas as pd

class HausdorffDistance:
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
        Calculate the Hausdorff distance between two sets of points, handling NaN values.

        Parameters:
        - XA: NumPy array of shape (n, 2) with points from set A.
        - XB: NumPy array of shape (m, 2) with points from set B.
        - distance_function: Function to compute distance between two points.

        Returns:
        - The Hausdorff distance, or inf if no valid points exist.
        """
        try:
            # Filter out rows with NaN values
            mask_A = ~pd.isnull(XA).any(axis=1)  # True for rows without NaN
            mask_B = ~pd.isnull(XB).any(axis=1)
            XA_valid = XA[mask_A]  # Select only valid points from XA
            XB_valid = XB[mask_B]  # Select only valid points from XB

            # Check if either set is empty after filtering
            if XA_valid.size == 0 or XB_valid.size == 0:
                return np.inf

            # Update the number of points
            nA = XA_valid.shape[0]
            nB = XB_valid.shape[0]

            # Initialize the maximum distance
            cmax = 0.0

            # Directed Hausdorff distance from A to B
            for i in range(nA):
                cmin = np.inf
                for j in range(nB):
                    d = distance_function(XA_valid[i, :], XB_valid[j, :])
                    if d < cmin:
                        cmin = d
                if cmin > cmax and np.inf > cmin:
                    cmax = cmin

            # Directed Hausdorff distance from B to A
            for j in range(nB):
                cmin = np.inf
                for i in range(nA):
                    d = distance_function(XA_valid[i, :], XB_valid[j, :])
                    if d < cmin:
                        cmin = d
                if cmin > cmax and np.inf > cmin:
                    cmax = cmin

            return cmax

        except Exception as e:
            print(f"Error: {e}")
            return np.inf
    
    def __call__(self, *args, **kwds):
        self.calculate(*args, **kwds)

    def __str__(self):
        return self.__class__.__name__

class DiffTotalDistance:
    """
    Class to calculate the total distance between two sets of points.
    """

    def __init__(self):
        pass

    def calculate(self, XA, XB, distance_function=None):
        """
        Calculate the total distance between two sets of points.

        Parameters:
        - XA: NumPy array of shape (n, 2) with points from set A.
        - XB: NumPy array of shape (m, 2) with points from set B.
        - distance_function: Function to compute distance between two points.

        Returns:
        - The total distance.
        """
        try:
            # Filter out rows with NaN values
            mask_A = ~pd.isnull(XA).any(axis=1)  # True for rows without NaN
            mask_B = ~pd.isnull(XB).any(axis=1)
            XA_valid = XA[mask_A]  # Select only valid points from XA
            XB_valid = XB[mask_B]  # Select only valid points from XB

            # Check if either set is empty after filtering
            if XA_valid.size == 0 or XB_valid.size == 0:
                return np.inf

            # Update the number of points
            nA = XA_valid.shape[0]
            nB = XB_valid.shape[0]

            # Initialize the total distance
            total_distanceA = 0.0
            total_distanceB = 0.0

            # Calculate the total distance
            for i in range(nA-1):
                d = distance_function(XA_valid[i, :], XA_valid[i+1, :])
                total_distanceA += d
            
            for i in range(nB-1):
                d = distance_function(XB_valid[i, :], XB_valid[i+1, :])
                total_distanceB += d
            return np.abs(total_distanceA - total_distanceB)

        except Exception as e:
            print(f"Error: {e}")
            exit(0)
            return np.inf
    
    def __call__(self, *args, **kwds):
        self.calculate(*args, **kwds)

    def __str__(self):
        return self.__class__.__name__
    