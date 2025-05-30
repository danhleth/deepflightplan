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
    