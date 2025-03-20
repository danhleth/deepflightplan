import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import networkx as nx
from geopy import distance
import geojson


class Lido21EnrouteAirwayDataset():
    def __init__(self, file_path: str):
        # Initialize directed graph
        self.G = None
        self.G = self.load_geojson(file_path)

    def load_geojson(self, geojson_data):
        """
        Create a directed graph from GeoJSON route data
        Returns: NetworkX DiGraph object
        """

        if self.G is None:
            self.G = nx.DiGraph()
        else:
            self.G.clear()
        
        # If input is a string, parse it as JSON
        if isinstance(geojson_data, str):
            # data = json.loads(geojson_data)
            with open(geojson_data) as f:
                data = geojson.load(f)
        else:
            data = geojson_data
        
        # Process each feature in the collection
        for feature in data['features']:
            props = feature['properties']
            
            # Get node identifiers
            start_node = f"{props['fix0_icao']}-{props['fix0_ident']}"
            end_node = f"{props['fix1_icao']}-{props['fix1_ident']}"
            
            # Parse coordinates
            geometry = feature['geometry']
            start_long, start_lat  = geometry['coordinates'][0]
            end_long, end_lat = geometry['coordinates'][1]
            
            # Add nodes with attributes if they don't exist
            if not self.G.has_node(start_node):
                self.G.add_node(start_node, 
                        lat=start_lat,
                        long=start_long)
                
            if not self.G.has_node(end_node):
                self.G.add_node(end_node,
                        lat=end_lat,
                        long=end_long)
            
            # Add edge with attributes
            geodistance = distance.great_circle((start_lat, start_long), (end_lat, end_long)).nm
            self.G.add_edge(start_node, end_node,
                    route_ident=props['route_ident'],
                    name=props['name'],
                    distance=geodistance,
                    direction=props['direction'],
                    route_indent=props['route_ident'])
        return self.G
    

class ODAirportDataset():
    def __init__(self, file_path: str):
        self.df = self.load_csv(file_path)

    def load_csv(self, csv_file):
        self.df = pd.read_csv(csv_file)
        return self.df