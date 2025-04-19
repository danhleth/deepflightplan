import networkx as nx
from geopy import distance


class WaypointType():
    AIRPORT = "airport"
    ENROUTE = "enroute"
    SID = "sid"
    STAR = "star"

class WaypointNode():
    def __init__(self, lat=None, long=None, name=None, icao=None, type=None):
        self.lat = lat
        self.long = long
        self.icao = icao
        self.type = type
        self.name = name

    def __str__(self):
        return f"{self.name} {self.icao} ({self.lat}, {self.long}) {self.type}"

class EnrouteAirway():
    def __init__(self, start_node=None, 
                        end_node=None, 
                        route_ident=None, 
                        name=None):
        self.start_node = start_node
        self.end_node = end_node
        self.route_ident = route_ident
        self.name = name


class Route():
    def __init__(self,  start_node=None, 
                        end_node=None, 
                        route_ident=None, 
                        name=None,
                        list_of_nodes=None):
        self.start_node = start_node
        self.end_node = end_node
        self.route_ident = route_ident
        self.name = name
        self.list_of_nodes = list_of_nodes
    

    def get_route(self):
        return self.list_of_nodes
    

def get_path_cost(graph, calculate_path):
    """
    Calculate the total distance of the shortest path between two nodes
    Returns: tuple of (path, total_distance) or (None, None) if no path exists
    """
    try:
        # Calculate total distance
        total_distance = 0
        for i in range(len(calculate_path) - 1):
            total_distance += graph[calculate_path[i]][calculate_path[i+1]]['distance']
            
        return total_distance
    
    except nx.NetworkXNoPath:
        return None
    

def find_closest_node(graph, src_node: WaypointNode, criterion, threshold=float('inf')):
    """
    Find the closest node in the graph to the source node
    """
    closest_node = None
    min_distance = float('inf')
    for node in graph:
        tmp_node = WaypointNode(lat=graph.nodes[node]['lat'], 
                                long=graph.nodes[node]['long'],
                                name=node)
        distance = criterion.compute_distance(src_node, tmp_node)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_node = tmp_node
    return [closest_node, min_distance]


def find_k_closest_node(graph, src_node: WaypointNode, criterion, k=1, threshold=float('inf')):
    """
    Find the k closest node in the graph to the source node
    """
    rs = []
    for node in graph:
        tmp_node = WaypointNode(lat=graph.nodes[node]['lat'], 
                                long=graph.nodes[node]['long'],
                                name=node)
        distance = criterion.compute_distance(src_node, tmp_node)
        if distance < threshold:
            rs.append([tmp_node, distance])
    rs = sorted(rs, key=lambda x: x[1])
    return rs[:k]