
import itertools
import random
import pandas as pd
from tqdm import tqdm
from dioscuri.datasets.enroute_graph_wrapper import WaypointNode, WaypointType
from dioscuri.distance.gcd_distance import GeopyGreatCircleDistance
from dioscuri.metrics.distance import DiffTotalDistance
import networkx as nx
import numpy as np
import re
import os
import folium
import math
import json

def route_to_sector_with_count_waypoint(route: str) -> str:
    """ 
    Example:
    "VV-TSN VV-LATHA VV-NIXUP VV-CN WS-ESPOB WS-ENREP WS-VEPLI WM-EGOLO WM-ROBMO WM-VMR WS-PU20 WS-VTK" -> "VV4WS3WM3WS2"
    """
    # Split the route into segments; handle empty input
    segments = route.split()
    if not segments:
        return ""
    
    # Define a key function to extract the two-letter code from each segment
    def key_func(segment):
        return segment.split('-')[0]
    
    # Initialize result list to store code-count pairs
    result = []
    
    # Group consecutive segments by their code and count each group
    for code, group in itertools.groupby(segments, key=key_func):
        count = sum(1 for _ in group)  # Count the number of segments in the group
        result.append(code + str(count))  # Append code followed by count
    
    # Join all parts into a single string without separators
    return '-'.join(result)


def route_to_sector_unique(route: str, graph_datasource) -> str:
    # Split the route into segments
    nodes = route.split()
    # Handle empty input
    if not nodes:
        return ""
    
    nodes = [graph_datasource.get_nodes(node) for node in nodes]
    
    # Extract the two-letter code from each segment
    codes = [segment['fix_icao'] for segment in nodes]
    
    # Remove consecutive duplicates using groupby
    unique_codes = [key for key, _ in itertools.groupby(codes)]
    
    # Join the unique codes with hyphens
    return '-'.join(unique_codes)


def filter_routes(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    idx = df.groupby('sector_with_count_waypoint')['total_distances'].idxmin()
    df_final = df.loc[idx].reset_index(drop=True)
    
    return df_final

def calculate_bearing(coord1, coord2):
    """Calculate bearing between two [lat, lon] coordinates in degrees."""
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def find_corresponding_airport(airport_df, airport_code):
    """
    Find the corresponding airport in the airport DataFrame based on the airport code.
    """
    if airport_code in airport_df['iata'].values:
        return airport_df[airport_df['iata'] == airport_code].iloc[0]
    elif airport_code in airport_df['icao'].values:
        return airport_df[airport_df['icao'] == airport_code].iloc[0]
    else:
        raise ValueError(f"Airport code {airport_code} not found in the airport DataFrame.")


def filter_sector_by_name(geojson_data, sector_names: list):
    """
    Filter GeoJSON data to return only the sector with the specified name.

    Parameters:
    - geojson_data (dict): GeoJSON data containing multiple sectors.
    - sector_name (str): Name of the sector to filter.

    Returns:
    - dict: Filtered GeoJSON data containing only the specified sector.
    """
    if not isinstance(geojson_data, dict) or geojson_data.get('type') != 'FeatureCollection':
        raise ValueError("Invalid GeoJSON data format. Expected a FeatureCollection.")

    filtered_features = []
    for feature in geojson_data['features']:
        for sector_name in sector_names:
            if feature['properties'].get('index') == sector_name:
                filtered_features.append(feature)

    if not filtered_features:
        raise ValueError(f"Sector with name '{sector_names}' not found in the GeoJSON data.")

    return {
        'type': 'FeatureCollection',
        'features': filtered_features
    }
    
<<<<<<< HEAD
    
=======
>>>>>>> 4fa38b0577075bdf3296de5dfde71b1fafed314a
def load_geojson(file_path):
    try:
        with open(file_path, 'r') as f:
            geojson_data = json.load(f)
        if geojson_data['type'] != 'FeatureCollection':
            raise ValueError("GeoJSON must be a FeatureCollection")
        return geojson_data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Error loading GeoJSON: {str(e)}")
        return None
    

def visualize_routes(df, enroute_graph):
    origin = df['origin'].iloc[0]
    destination = df['destination'].iloc[0]

    origin_lat, origin_lon = df.iloc[0][['origin_lat', 'origin_long']]
    destination_lat, destination_lon = df.iloc[0][['destination_lat', 'destination_long']]


    origin_coords = [origin_lat, origin_lon]
    destination_coords = [destination_lat, destination_lon]
    # Initialize a folium map centered approximately between Origin and Destination
    center_lat = (origin_lat + destination_lat) / 2
    center_lon = (origin_lon + destination_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

<<<<<<< HEAD
    folium.GeoJson('/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/lido21/enroute_airway.geojson', 
                   name='EnrouteAirway').add_to(m)
    
    sector_data = load_geojson("/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/facilities/sectors.geojson")
    # names = ["WSJCE", "WSJCH", "WSJCG", "WSJCF", "WSJCC", "WSJCD", "WSJCB", "WSJCA"]

    # sector_data = filter_sector_by_name(sector_data, names)
=======
    folium.GeoJson('/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/deepflightplan/datasets/lido21/enroute_airway.geojson', 
                   name='EnrouteAirway').add_to(m)
    
    sector_data = load_geojson("/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/deepflightplan/datasets/facilities/sectors.geojson")
    names = ["WSJCE", "WSJCH", "WSJCG", "WSJCF", "WSJCC", "WSJCD", "WSJCB", "WSJCA"]

    sector_data = filter_sector_by_name(sector_data, names)
>>>>>>> 4fa38b0577075bdf3296de5dfde71b1fafed314a
    folium.GeoJson(sector_data, 
                   style_function=lambda feature: {
                        "fillColor": "#ffff00",
                        "color": "black",
                        "weight": 2,
                        "dashArray": "5, 5",
                    },
                   name='Sector').add_to(m)
<<<<<<< HEAD
   
=======

>>>>>>> 4fa38b0577075bdf3296de5dfde71b1fafed314a
    folium.Marker(
        location=origin_coords,
        popup=origin,
        icon=folium.Icon(color='green', icon='plane-departure', prefix='fa')
    ).add_to(m)

    folium.Marker(
        location=destination_coords,
        popup=destination,
        icon=folium.Icon(color='red', icon='plane-arrival', prefix='fa')
    ).add_to(m)

    for idx, row in df.iterrows():

        synth_coords = np.array(row['route_np'])
        origin = row['origin']
        destination = row['destination']

        synth_coords = np.insert(synth_coords, 0, np.array([origin_lat, origin_lon]), axis=0)  # Insert origin at the start
        synth_coords = np.insert(synth_coords, len(synth_coords), np.array([destination_lat, destination_lon]), axis=0)  # Append destination at the end

        # remove nan values in synth_coords
        synth_coords = synth_coords[~np.isnan(synth_coords).any(axis=1)]
        if len(synth_coords) < 2:
            print(f"Skipping route {idx+1} for {origin} to {destination} due to insufficient coordinates.")
            continue
        
        # Plot real route (blue)
        folium.PolyLine(
            locations=synth_coords,
            color='red',
            weight=3,
            opacity=0.8,
            popup=f'Synthesized Route {idx+1}',
            dash_array='5'  # Dashed line for real route
        ).add_to(m)

        # Add directional triangles for synthesized route
        for i in range(len(synth_coords) - 1):
            start = synth_coords[i]
            end = synth_coords[i + 1]
            mid_lat = (start[0] + end[0]) / 2
            mid_lon = (start[1] + end[1]) / 2
            bearing = calculate_bearing(start, end)
            folium.RegularPolygonMarker(
                location=[mid_lat, mid_lon],
                number_of_sides=3,
                radius=5,
                rotation=bearing - 90,
                color='red',
                fill_color='red',
                fill_opacity=0.6,
                popup=f'Synthesized Route {idx+1} Direction'
            ).add_to(m)


        # Add markers for waypoints in synthesized route
        for coord in synth_coords:
            node = enroute_graph.get_node_from_coordinates(coord[0], coord[1])
            folium.CircleMarker(
                location=coord,
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                popup=f"Synthesized Waypoint {node if node else idx+1}"
            ).add_to(m)

    # Add a legend
    legend_html = '''
        <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 200px; height: 90px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white; opacity:0.8;">
        &nbsp; <b>Route Legend</b><br>
        &nbsp; <i style="color:blue">———</i> Real Route<br>
        &nbsp; <i style="color:red">— —</i> Synthesized Route<br>
        </div>
        '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def process_single_od(args):
    """
    Process a single origin-destination pair
    Returns a DataFrame with results
    """
    # Unpack arguments
    index, row, algorithm, distance, graph_datasource, opt, logger = args
    carrier_code, flight_number, specific_aircraft_code, origin, destination, org_lat, org_long, dest_lat, dest_long, utc_dep_time, flying_time, aircraft_range, aircraft_speed = row
    
    # Create waypoint nodes
    org_node = WaypointNode(lat=org_lat, long=org_long, name=origin, type=WaypointType.AIRPORT)
    dest_node = WaypointNode(lat=dest_lat, long=dest_long, name=destination, type=WaypointType.AIRPORT)

    # Find closest nodes
    # k_closest_nodes_origin = find_k_closest_node(graph_datasource.G, org_node, distance,
    #                                             k=opt['k_closest_node_from_airport'], threshold=60)
    # k_closest_nodes_destination = find_k_closest_node(graph_datasource.G, dest_node, distance,
    #                                                 k=opt['k_closest_node_from_airport'], threshold=60)
    k_closest_nodes_origin = find_k_suitable_nodes(graph=graph_datasource.G, 
                                                   src_node=org_node, 
                                                   dest_node=dest_node,
                                                   criterion=distance,
                                                    k=opt['k_closest_node_from_airport'], threshold=47) # 47 mile = 40 nautical mile

    k_closest_nodes_destination = find_k_suitable_nodes(graph=graph_datasource.G, 
                                                        src_node=dest_node, 
                                                        dest_node=org_node,
                                                        criterion=distance,
                                                        k=opt['k_closest_node_from_airport'], threshold=47) # 47 mile = 40 nautical mile

    if len(k_closest_nodes_origin) == 0 or len(k_closest_nodes_destination) == 0:
        logger.info(f"No closest nodes found for {origin} to {destination}.")
        return pd.DataFrame()
    
    # Generate combinations
    combinations = list(itertools.product(k_closest_nodes_origin, k_closest_nodes_destination))
    
    # Initialize result lists
    results_df  = {
        "carrier_code":[], "flight_number": [],
        "specific_aircraft_code":[],
        "origin": [], "destination": [],
        "origin_lat": [], "origin_long": [],
        "destination_lat": [], "destination_long": [],
        "min_dist_origin": [], "min_dist_dest": [],
        "route_distances": [], "total_distances": [],
        "utc_dep_time": [],
        "route_flying_time": [],  "total_flying_time": [],
        "aircraft_speed_mph": [],
        'aircraft_range': [],
        "route": [],
        "reformatted_route": [],
        "route_np": []
    }
    
    logger.info(f"Processing OD pair {index}: {origin} to {destination} with {len(combinations)} combinations.")
    i = 0
    # Process each combination
    while (len(results_df['route']) < algorithm.retrieve_j) and (i < len(combinations)):
        (min_node_origin, min_dist_origin, _), (min_node_dest, min_dist_dest, _) = random.choice(combinations)
        # RETRIEVE MULTIPLE SHORTEST PATHS
        multiple_shortest_routes = algorithm.retrieve_multiple_shortest_path(graph_datasource.G, 
                                                      min_node_origin.name, 
                                                      min_node_dest.name)
        i += 1
        if multiple_shortest_routes is None:
            logger.info(f"No routes found for {origin} to {destination} with origin node {min_node_origin.name} and destination node {min_node_dest.name}.")
            continue
        
        for route in multiple_shortest_routes:
            distance = get_path_distance(graph_datasource.G, route)
            total_distance = distance + min_dist_origin + min_dist_dest
            if (total_distance > aircraft_range):
                logger.info(f"{distance} {min_dist_origin} {min_dist_dest} {aircraft_range} - Skipping route from {origin} to {destination} with origin node {min_node_origin.name} and destination node {min_node_dest.name}.")
                continue
            
            route_str = ' '.join(route)
            route_np = graph_datasource.get_list_nodes_np(route_str)
            route_flying_time = distance / aircraft_speed
            total_flying_time = total_distance / aircraft_speed

            results_df['carrier_code'].append(carrier_code)
            results_df['flight_number'].append(flight_number)
            results_df['specific_aircraft_code'].append(specific_aircraft_code)
            results_df['utc_dep_time'].append(utc_dep_time)
            results_df['origin'].append(origin)
            results_df['origin_lat'].append(org_lat)
            results_df['origin_long'].append(org_long)
            results_df['destination'].append(destination)
            results_df['destination_lat'].append(dest_lat)
            results_df['destination_long'].append(dest_long)
            results_df['min_dist_origin'].append(min_dist_origin)
            results_df['min_dist_dest'].append(min_dist_dest)
            results_df['route_distances'].append(distance)
            results_df['total_distances'].append(total_distance)
            results_df['aircraft_speed_mph'].append(aircraft_speed)
            results_df['aircraft_range'].append(aircraft_range)
            results_df['route_flying_time'].append(route_flying_time)
            results_df['total_flying_time'].append(total_flying_time)
            results_df['route'].append(route_str)  
            results_df['reformatted_route'].append(reformat_synthesized_route(route_str, graph_datasource))
            results_df['route_np'].append(route_np.tolist())

        
    # Create DataFrame for this OD pair
    tmp_df = pd.DataFrame(results_df)
    tmp_df = tmp_df[:algorithm.retrieve_j]
    saved_synthesized_route_path = opt['save_dir']/ 'saved_synthesized_routes'
    saved_synthesized_route_path.mkdir(parents=True, exist_ok=True)
    if tmp_df.empty:
        return tmp_df
    tmp_df = tmp_df.drop_duplicates(subset=['carrier_code', 'flight_number', 'utc_dep_time', 'origin', 'destination', \
                                            'route_distances', 'total_distances', 'aircraft_speed_mph', 'aircraft_range', \
                                            'route_flying_time', 'total_flying_time'], keep='first')
    origin = tmp_df['origin'].iloc[0]
    destination = tmp_df['destination'].iloc[0]
    m = visualize_routes(tmp_df, graph_datasource)
    m.save(os.path.join(saved_synthesized_route_path, f"{origin}_{destination}.html"))
    tmp_df.to_csv(os.path.join(saved_synthesized_route_path, f"{origin}-{destination}.csv"), index=False)
    return tmp_df


def route_to_list_waypoint(route, graph_datasource):
    """
    Convert a route string to a list of WaypointNode objects
    """
    route_list = []
    str_waypoints = route.split()
    for waypoint_name in str_waypoints:
        # Find the closest node in the graph
        node = graph_datasource.get_nodes(waypoint_name)
        if node is not None:
            # Create a WaypointNode object
            waypoint_node = WaypointNode(lat=node["lat"], long=node["long"], name=waypoint_name, type=WaypointType.ENROUTE)
            route_list.append(waypoint_node)
    return route_list

def route_to_numpy_coordinates(route, enroute_graph):
    """
    Convert a route to numpy coordinates.
    """
    if isinstance(route, str):
        route = route.split()

    if isinstance(route, float):
        raise ValueError("Route is a float, not a string or list.")
        return np.array([[None, None]])

    coordinates = []
    for node_name in route:
        if "/" in node_name:
            node_name = node_name.split("/")[0]
        if "-" in node_name:
            node_name = node_name.split("-")[1]
        
        data = enroute_graph.get_nodes(node_name)
        if data is not None:
            coordinates.append([data['lat'], data['long']])
        else:
            coordinates.append([None,None])
            # print(f"Node {node_name} not found in graph.")
    return np.array(coordinates) if len(coordinates) > 0 else np.array([[None, None]])

# Define the function to evaluate a single origin-destination group
def calculate_distance_od(od_group, graph_datasource, distance_fn, metric_fns):
    _, od_df = od_group
    top_1 = od_df.iloc[0]
    # trajectory_A = route_to_list_waypoint(top_1["route"], graph_datasource)

    trajectory_A = route_to_numpy_coordinates(top_1["route"], graph_datasource)
    trajectory_A = np.array(trajectory_A)
    trajectory_A = trajectory_A[~pd.isnull(trajectory_A).any(axis=1)]

    rs = []
    route_nps = []
    total_distances = []
    for i, row in tqdm(od_df.iterrows(), leave=False):
        route = row["route"]
        trajectory_B = route_to_numpy_coordinates(route, graph_datasource)
        trajectory_B = np.array(trajectory_B)
        trajectory_B = trajectory_B[~pd.isnull(trajectory_B).any(axis=1)]
        trajectory_B_distance = DiffTotalDistance().calculate(trajectory_B, np.array([[0,0]]), GeopyGreatCircleDistance().compute_distance)
        metric_dementions = []
        for metric_fn in metric_fns:
            distance = metric_fn.calculate(trajectory_A, trajectory_B, distance_fn)
            if distance is None:
                continue
            metric_dementions.append(distance)
        route_nps.append(trajectory_B.tolist())
        total_distances.append(trajectory_B_distance)
        rs.append(metric_dementions)

    # Convert the list of results to a DataFrame
    rs = pd.DataFrame(rs, columns=[str(metric_fn) for metric_fn in metric_fns])
    rs.insert(len(rs.columns)-1, "total_distances", total_distances)
    rs.insert(len(rs.columns), "route_np", route_nps)
    return rs


def reformat_synthesized_route(synthesized_route, enroute_graph):
    """
    Reformat the synthesized route to match the format of the enroute graph.
    """
    synthesized_route = synthesized_route.split()
    
    reformatted_route = []
    previous_route_ident = None
    for i in range(len(synthesized_route)-1):
        node1 = synthesized_route[i]
        node2 = synthesized_route[i+1]
        edges = enroute_graph.G.get_edge_data(node1,node2)
        if edges is None:
            continue
        
        reformatted_route.append(node1.split("-")[1])
        current_route_ident = edges['route_ident']
        if previous_route_ident != current_route_ident:
            reformatted_route.append(current_route_ident)
            previous_route_ident = current_route_ident
        reformatted_route.append(node2.split("-")[1])

    # Example ['POVOT', 'M774', 'KEONG', 'KEONG', 'W44', 'ELBAM', 'ELBAM', 'BLI', 'BLI', 'W33', 'GOMAT', 'GOMAT', 'NR', 'NR', 'W43', 'KPG']
    # Now remove the duplicates neighboring nodes
    reformatted_route = [reformatted_route[i] for i in range(len(reformatted_route)) if i == 0 or reformatted_route[i] != reformatted_route[i-1]]
    return " ".join(reformatted_route)


def get_path_distance(graph, calculate_path):
    """
    Calculate the total distance of the shortest path between two nodes
    Returns: tuple of (path, total_distance) or (None, None) if no path exists
    """
    import numpy as np
    try:
        # Calculate total distance
        total_distance = 0
        for i in range(0, len(calculate_path) - 1):
            if graph.has_edge(calculate_path[i], calculate_path[i+1]):
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
    # source: https://aviation.stackexchange.com/questions/66748/does-the-enroute-phase-begin-at-a-30nm-distance-from-the-departure-airport
    lowerbound_threshold = 35 # miles
    for node in graph:
        tmp_node = WaypointNode(lat=graph.nodes[node]['lat'], 
                                long=graph.nodes[node]['long'],
                                name=node)
        distance = criterion.compute_distance(src_node, tmp_node)
        if  lowerbound_threshold < distance and distance < threshold:
            rs.append([tmp_node, distance])
    rs = sorted(rs, key=lambda x: x[1])
    return rs[:k]

def find_k_suitable_nodes(graph, src_node, dest_node, criterion, k=1, threshold=float('inf')):
    """
    Find the k suitable nodes in the graph to the source and destination nodes
    """
    rs = []
    for node in graph:
        tmp_node = WaypointNode(lat=graph.nodes[node]['lat'], 
                                long=graph.nodes[node]['long'],
                                name=node)
        distance_src = criterion.compute_distance(src_node, tmp_node)
        distance_dest = criterion.compute_distance(dest_node, tmp_node)
        if distance_src < threshold and distance_dest:
            rs.append([tmp_node, distance_src, distance_dest])
    rs = sorted(rs, key=lambda x: x[2])
    return rs[:k]


def parse_routestr_to_np(route_str):
    """
    Parse a string or bytes-like representation of a 2D array into a NumPy array.
    
    Args:
        route_str (str or bytes): String or bytes like "[[-34.946944, 138.524444], [None, None]]"
    
    Returns:
        np.ndarray: Parsed array with None replaced by np.nan, or a default array on failure
    """
    try:
        route_str = route_str[1:-1]
        route_list = [list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", item))) for item in route_str.split("], [")]
        route_list = [x if len(x) == 2 else [None, None] for x in route_list]
        # Convert to NumPy array
        route_array = np.array(route_list)
        
        # Replace None with np.nan
        route_array[route_array == None] = np.nan
        
        return route_array
    except Exception as e:
        print(f"Error parsing route: {e}")
        print(f"Original route string: {route_str}")
        # Return a default array if parsing fails
        return np.array([[np.nan, np.nan]])