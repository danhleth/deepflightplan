import logging
from typing import Callable, Dict, Optional
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import itertools
from multiprocessing import Pool

from dioscuri.opt import Opts

from dioscuri.datasets.enroute_graph_wrapper import EnrouteAirway, WaypointNode, WaypointType

from dioscuri.datasets import DATASET_REGISTRY
from dioscuri.algorithms import ALGORITHM_REGISTRY
from dioscuri.criterion import CRITERION_REGISTRY

from dioscuri.datasets.enroute_graph_wrapper import get_path_cost, find_closest_node, find_k_closest_node
from dioscuri.utils.getter import (get_instance, 
                               get_instance_recursively)
from dioscuri.utils.loading import load_yaml



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

def route_to_sector_unqiue(route: str) -> str:
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
    index, row, algorithm, criterion, graph_dataset, opt = args
    origin, destination, aircraft_range, org_lat, org_long, dest_lat, dest_long = row
    
    # Create waypoint nodes
    org_node = WaypointNode(lat=org_lat, long=org_long, name=origin, type=WaypointType.AIRPORT)
    dest_node = WaypointNode(lat=dest_lat, long=dest_long, name=destination, type=WaypointType.AIRPORT)

    # Find closest nodes
    k_closest_nodes_origin = find_k_closest_node(graph_dataset.G, org_node, criterion,
                                                k=opt['k_closest_node_from_airport'], threshold=30)
    k_closest_nodes_destination = find_k_closest_node(graph_dataset.G, dest_node, criterion,
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
    
    sector_with_count_waypoint = tmp_df["route"].apply(route_to_sector_with_count_waypoint)
    tmp_df["sector_with_count_waypoint"] = sector_with_count_waypoint
    count_unique_sectors = tmp_df["route"].apply(lambda x: len(route_to_sector_unqiue(x).split("-")))
    tmp_df["count_unique_sector"] = count_unique_sectors

    tmp_df = filter_routes(tmp_df)
    # Logic filter for the top k shortest path by heuristics
    tmp_df = tmp_df.sort_values(by=["total_distances", "count_unique_sector"]).iloc[:algorithm.top_k]

    return tmp_df

class Pipeline:
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Opts,
    ):  
        
        self.cfg_data = opt["data"]
        self.cfg_algorithm = opt["algorithm"]
        self.cfg_criterion = opt["criterion"]
        self.opt = opt["opt"]

        self.random = random.Random(self.opt["seed"])


        # Save settings file
        save_dir = Path(self.opt["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_opt_path = save_dir / "opt.yaml"
        with open(saved_opt_path, "w") as f:
            yaml.dump(opt, f)

        self.logger = self._init_logging(log_file=save_dir / "log.txt", name="logger")
        self.logger.info(self.opt)
        self.logger.info(self.cfg_algorithm)
    def _init_logging(self, log_file='log.txt', name='logger'):
        """ Initialize logging
        """
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger
    
    def get_dataset(self, name):
        self.logger.info("Getting dataset")
        dataset = get_instance(name, registry=DATASET_REGISTRY)
        return dataset

    def sanitycheck(self):
        self.logger.info("Sanity checking before converting")

    def generate_flightroute_od(self, algorithm, criterion, od_airport_dataset, graph_dataset):
        """
        Generate flight routes using multiprocessing
        """
        df_results = pd.DataFrame()
        
        self.logger.info("Generating flight plan with multiprocessing")
        # Prepare arguments for multiprocessing
        args = [(index, row, algorithm, criterion, graph_dataset, self.opt) 
                for index, row in od_airport_dataset.df.iterrows()]
        
        # Use number of CPU cores minus 1 to leave some resources free
        num_processes = max(1, self.opt["num_processes"])
        self.logger.info(f"Using {num_processes} processes")
        
        # Process in parallel
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_single_od, args), 
                            total=len(od_airport_dataset.df)))
        
        # Combine results
        for i, tmp_df in enumerate(results):
            df_results = pd.concat([df_results, tmp_df], axis=0)


        self.logger.info(f"There are {len(df_results)} flight plans synthesized")
        # Save final results
        if not df_results.empty:
            save_dir = Path(self.opt["save_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(save_dir / "flightplan_final.csv", index=False)
            
        return df_results
    
    def fit(self):
        # Load the list_od_pair_dataset
        od_airport_dataset = self.get_dataset(self.cfg_data["od_dataset"])
        # Load the graph_dataset
        graph_dataset = self.get_dataset(self.cfg_data["datasource"])
        # Load the criterion
        criterion = get_instance(self.cfg_criterion, registry=CRITERION_REGISTRY)
        # Load the algorithm    
        algorithm = get_instance(self.cfg_algorithm, registry=ALGORITHM_REGISTRY)
        # Start generate the flight plan
        df_results = self.generate_flightroute_od(algorithm, criterion, od_airport_dataset, graph_dataset)
        
