import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import networkx as nx
from geopy import distance
import geojson


class Lido21EnrouteAirwayDataset():
    def __init__(self, file_path: str, hiden_sectors: bool = False):
        # Initialize directed graph
        self.G = None
        self.G = self.load_geojson(file_path, hiden_sectors)

    def load_geojson(self, geojson_data, hiden_sectors: bool = False):
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
            if hiden_sectors == False:
                start_node = f"{props['fix0_icao']}-{props['fix0_ident']}"
                end_node = f"{props['fix1_icao']}-{props['fix1_ident']}"
            else:
                start_node = props['fix0_ident']
                end_node = props['fix1_ident']
            
            # Parse coordinates
            geometry = feature['geometry']
            start_long, start_lat  = geometry['coordinates'][0]
            end_long, end_lat = geometry['coordinates'][1]
            
            # Add nodes with attributes if they don't exist
            if not self.G.has_node(start_node):
                self.G.add_node(start_node, 
                        lat=start_lat,
                        long=start_long,
                        fix_icao=props['fix0_icao'])
                
            if not self.G.has_node(end_node):
                self.G.add_node(end_node,
                        lat=end_lat,
                        long=end_long,
                        fix_icao=props['fix1_icao'])
            
            # Add edge with attributes
            geodistance = distance.great_circle((start_lat, start_long), (end_lat, end_long)).nm
            self.G.add_edge(start_node, end_node,
                    route_ident=props['route_ident'],
                    name=props['name'],
                    distance=geodistance,
                    direction=props['direction'])
        return self.G
    
    def get_nodes(self, node_name):
        """
        Get specific node attributes
        """
        if self.G is None:
            raise ValueError("Graph not initialized. Please load the graph first.")
        
        if self.G.has_node(node_name):
            node_data = self.G.nodes[node_name]
            return node_data
        return None
    
    def get_list_nodes_np(self, node_names: List[str]) -> Iterable:
        """
        Get a list of nodes from the graph
        """
        if self.G is None:
            raise ValueError("Graph not initialized. Please load the graph first.")
        
        nodes = []
        for node_name in node_names.split():
            if self.G.has_node(node_name):
                node_data = self.G.nodes[node_name]
                nodes.append([node_data['lat'], node_data['long']])
        return np.array(nodes)

class ODAirportDataset():
    def __init__(self, file_path: str):
        self.df = self.load_csv(file_path)

    def load_csv(self, csv_file):
        self.df = pd.read_csv(csv_file)
        return self.df