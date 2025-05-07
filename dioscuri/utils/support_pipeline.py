
import itertools
import pandas as pd
from tqdm import tqdm
from dioscuri.datasets.enroute_graph_wrapper import WaypointNode, WaypointType
from dioscuri.datasets.enroute_graph_wrapper import find_k_closest_node
from dioscuri.datasets.enroute_graph_wrapper import get_path_cost
from dioscuri.datasets.enroute_graph_wrapper import find_closest_node

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

def route_to_sector_unique(route: str) -> str:
    # Split the route into segments
    segments = route.split()
    # Handle empty input
    if not segments:
        return ""
    
    # Extract the two-letter code from each segment
    codes = [segment.split('-')[0] for segment in segments]
    
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

def process_single_od(args):
    """
    Process a single origin-destination pair
    Returns a DataFrame with results
    """
    # Unpack arguments
    index, row, algorithm, distance, graph_dataset, opt = args
    origin, destination, aircraft_range, org_lat, org_long, dest_lat, dest_long = row
    
    # Create waypoint nodes
    org_node = WaypointNode(lat=org_lat, long=org_long, name=origin, type=WaypointType.AIRPORT)
    dest_node = WaypointNode(lat=dest_lat, long=dest_long, name=destination, type=WaypointType.AIRPORT)

    # Find closest nodes
    k_closest_nodes_origin = find_k_closest_node(graph_dataset.G, org_node, distance,
                                                k=opt['k_closest_node_from_airport'], threshold=30)
    k_closest_nodes_destination = find_k_closest_node(graph_dataset.G, dest_node, distance,
                                                    k=opt['k_closest_node_from_airport'], threshold=30)
    
    if len(k_closest_nodes_origin) == 0 or len(k_closest_nodes_destination) == 0:
        return pd.DataFrame()

    # Generate combinations
    combinations = list(itertools.product(k_closest_nodes_origin, k_closest_nodes_destination))
    
    # Initialize result lists
    tmp_origin_airports = []
    tmp_destination_airports = []
    tmp_min_dist_dests = []
    tmp_min_dist_origins = []
    tmp_routes = []
    tmp_route_distances = []
    tmp_total_distances = []

    # Process each combination
    for (min_node_origin, min_dist_origin), (min_node_dest, min_dist_dest) in combinations:
        tmp_top_k_paths = algorithm.top_k_shortest_path(graph_dataset.G, 
                                                      min_node_origin.name, 
                                                      min_node_dest.name)
        if tmp_top_k_paths is None:
            continue
            
        for route in tmp_top_k_paths:
            cost = get_path_cost(graph_dataset.G, route)
            if ((cost + min_dist_origin + min_dist_dest) > aircraft_range) or \
               (len(tmp_routes) >= algorithm.top_k):
                continue

            tmp_origin_airports.append(origin)
            tmp_destination_airports.append(destination)
            tmp_min_dist_dests.append(min_dist_dest)
            tmp_min_dist_origins.append(min_dist_origin)
            tmp_routes.append(' '.join(route))
            tmp_route_distances.append(cost)
            tmp_total_distances.append(cost + min_dist_origin + min_dist_dest)

    # Create DataFrame for this OD pair
    tmp_df = pd.DataFrame(dict(
        origin=tmp_origin_airports,
        destination=tmp_destination_airports,
        min_dist_origin=tmp_min_dist_origins,
        min_dist_dest=tmp_min_dist_dests,
        route_distances=tmp_route_distances,
        total_distances=tmp_total_distances,
        route=tmp_routes
    ))
    
    # sector_with_count_waypoint = tmp_df["route"].apply(route_to_sector_with_count_waypoint)
    # tmp_df["sector_with_count_waypoint"] = sector_with_count_waypoint
    # count_unique_sectors = tmp_df["route"].apply(lambda x: len(route_to_sector_unqiue(x).split("-")))
    # tmp_df["count_unique_sector"] = count_unique_sectors

    # tmp_df = filter_routes(tmp_df)
    # Logic filter for the top k shortest path by heuristics
    # tmp_df = tmp_df.sort_values(by=["total_distances", "count_unique_sector"]).iloc[:algorithm.top_k]

    return tmp_df


def route_to_list_waypoint(route, graph_dataset):
    """
    Convert a route string to a list of WaypointNode objects
    """
    route_list = []
    str_waypoints = route.split()
    for waypoint_name in str_waypoints:
        # Find the closest node in the graph
        node = graph_dataset.get_nodes(waypoint_name)
        if node is not None:
            # Create a WaypointNode object
            waypoint_node = WaypointNode(lat=node["lat"], long=node["long"], name=waypoint_name, type=WaypointType.ENROUTE)
            route_list.append(waypoint_node)
    return route_list



# Define the function to evaluate a single origin-destination group
def eval_single_od(od_group, graph_dataset, distance_fn, metric_fn):
    _, od_df = od_group
    top_1 = od_df.iloc[0]
    # trajectory_A = route_to_list_waypoint(top_1["route"], graph_dataset)
    trajectory_A = graph_dataset.get_list_nodes_np(top_1["route"])
    
    rs = []
    for i, row in tqdm(od_df.iterrows(), leave=False):
        route = row["route"]
        trajectory_B = graph_dataset.get_list_nodes_np(route)
        haursdorff_distance = metric_fn.calculate(trajectory_A, trajectory_B, distance_fn)
        
        if haursdorff_distance is None:
            continue
        rs.append(haursdorff_distance)
    return rs