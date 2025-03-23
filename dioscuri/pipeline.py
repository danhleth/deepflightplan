import logging
from typing import Callable, Dict, Optional
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time


from dioscuri.opt import Opts

from dioscuri.datasets.enroute_graph_wrapper import EnrouteAirway, WaypointNode, WaypointType

from dioscuri.datasets import DATASET_REGISTRY
from dioscuri.algorithms import ALGORITHM_REGISTRY
from dioscuri.criterion import CRITERION_REGISTRY

from dioscuri.datasets.enroute_graph_wrapper import get_path_cost, find_closest_node
from dioscuri.utils.getter import (get_instance, 
                               get_instance_recursively)
from dioscuri.utils.loading import load_yaml


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
        # Save settings file
        save_dir = Path(self.opt["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_opt_path = save_dir / "opt.yaml"
        with open(saved_opt_path, "w") as f:
            yaml.dump(opt, f)

        self.logger = logging.getLogger()
    
    def get_dataset(self, name):
        self.logger.info("Getting dataset")
        dataset = get_instance(name, registry=DATASET_REGISTRY)
        return dataset

    def sanitycheck(self):
        self.logger.info("Sanity checking before converting")

    def generate_flightroute_od(self, algorithm, criterion, od_airport_dataset, graph_dataset):
        self.logger.info("Generating flight plan")
        
        origin_airports = []
        min_dist_origins = []
        min_dist_dests = []
        destination_airports = []
        routes = []
        route_distances = []
        total_distances = []

        for _, (origin, destination, org_lat, org_long, dest_lat, dest_long) in tqdm(od_airport_dataset.df.iterrows(), total=len(od_airport_dataset.df)):
            
            org_node = WaypointNode(lat=org_lat, long=org_long, name=origin, type=WaypointType.AIRPORT)
            dest_node = WaypointNode(lat=dest_lat, long=dest_long, name=destination, type=WaypointType.AIRPORT)

            # Find the closest node
            closest_origin, min_dist_origin = find_closest_node(graph_dataset.G, org_node, criterion, threshold=30)
            closest_destination, min_dist_dest = find_closest_node(graph_dataset.G, dest_node, criterion, threshold=30)
            if closest_origin is None or closest_destination is None:
                continue

            # Get the top_k_shortest_path
            top_k_paths = algorithm.top_k_shortest_path(graph_dataset.G, closest_origin.name, closest_destination.name, weight="distance")
            if top_k_paths is None:
                continue
            # Get the total distance
            # total_distance = sum([graph_dataset[route[i]][route[i+1]]["distance"] for i in range(len(route)-1)])
            # total_distances.append(total_distance)
            for route in top_k_paths:
                cost = get_path_cost(graph_dataset.G, route)
                # Track results
                origin_airports.append(origin)
                destination_airports.append(destination)
                min_dist_dests.append(min_dist_dest)
                min_dist_origins.append(min_dist_origin)
                routes.append(' '.join(route))
                route_distances.append(cost)
                total_distances.append((cost+min_dist_origin+min_dist_dest))
        
        df_results = pd.DataFrame(dict(origin=origin_airports, 
                                       destination=destination_airports,
                                       min_dist_origin=min_dist_origins,
                                       min_dist_dest=min_dist_dests,
                                       route_distances=route_distances,
                                       total_distances=total_distances,
                                       route=routes 
                                       ))
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

        # Save the flight plan
        save_dir = Path(self.opt["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(save_dir / "flightplan.csv", index=False)
        self.logger.info("Sucessfully generated flight plan")
        
