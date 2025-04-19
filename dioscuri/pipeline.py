import logging
from typing import Callable, Dict, Optional
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import time
from multiprocessing import Pool
from functools import partial

from dioscuri.opt import Opts

from dioscuri.datasets.enroute_graph_wrapper import EnrouteAirway, WaypointNode, WaypointType

from dioscuri.datasets import DATASET_REGISTRY
from dioscuri.algorithms import ALGORITHM_REGISTRY
from dioscuri.distance import DISTANCE_REGISTRY
from dioscuri.metrics import METRIC_REGISTRY

from dioscuri.datasets.enroute_graph_wrapper import get_path_cost, find_closest_node, find_k_closest_node
from dioscuri.utils.getter import (get_instance, 
                               get_instance_recursively)
from dioscuri.utils.loading import load_yaml
from dioscuri.utils.support_pipeline import (process_single_od, route_to_list_waypoint, eval_single_od)


class Pipeline:
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Opts,
    ):  
        
        self.cfg_data = opt["data"]
        self.cfg_algorithm = opt["algorithm"]
        self.cfg_distance = opt["distance"]
        self.cfg_metric = opt["metric"]
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

    def generate_flightroute_od(self, algorithm, distance, od_airport_dataset, graph_dataset):
        """
        Generate flight routes using multiprocessing
        """
        df_results = pd.DataFrame()
        
        self.logger.info("Generating flight plan with multiprocessing")
        # Prepare arguments for multiprocessing
        args = [(index, row, algorithm, distance, graph_dataset, self.opt) 
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
        return df_results
    
    def evaluate(self, df, graph_dataset, distance_fn, metric_fn, num_processes=None):
        # Group the dataframe by origin and destination
        self.logger.info(f"Evaluating {len(df)} flight plans")

        # Create a partial function with fixed arguments
        eval_fn = partial(eval_single_od, graph_dataset=graph_dataset, distance_fn=distance_fn, metric_fn=metric_fn)

        # Use multiprocessing Pool with specified number of processes
        num_processes = max(self.opt["num_processes"], 1)
        self.logger.info(f"Using {num_processes} processes for evaluation")

        # Group the DataFrame by 'origin' and 'destination'
        groups_od = df.groupby(['origin', 'destination'])

        # Convert groups to a list for parallel processing
        od_groups_list = list(groups_od)

        all_results = []
        # Use Pool.imap to process groups in parallel and handle results
        with Pool(processes=num_processes) as pool:
            # Apply eval_fn to each OD group in parallel
            results_iter = pool.imap(eval_fn, (group for group in od_groups_list))

            # Process results and save to CSV
            save_dir = Path(self.opt["save_dir"]) / "Eval"
            save_dir.mkdir(parents=True, exist_ok=True)
        
            # Iterate over groups and results simultaneously
            for (od_key, od_group), rs in zip(od_groups_list, results_iter):
                od_group['hausdorff'] = rs
                od_group.to_csv(save_dir / (str("-".join(od_key)) + ".csv"), index=False)
                all_results.append(od_group)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df

    def fit(self):
        # Load the list_od_pair_dataset
        od_airport_dataset = self.get_dataset(self.cfg_data["od_dataset"])
        # Load the graph_dataset
        graph_dataset = self.get_dataset(self.cfg_data["datasource"])
        # Load the distance
        distance = get_instance(self.cfg_distance, DISTANCE_REGISTRY)
        # Load the algorithm    
        algorithm = get_instance(self.cfg_algorithm, registry=ALGORITHM_REGISTRY)
        # Start generate the flight plan
        start = time.time()
        df_results = self.generate_flightroute_od(algorithm, distance, od_airport_dataset, graph_dataset)
        end = time.time()
        self.logger.info(f"Time taken to generate flight plan: {end - start} seconds")
        # path = "/home/danhle/AIATFM/data_preparation/deepflightplan/tasks/generating_flightplan/runs/exp2/flightplan_final.csv"
        # df_results = pd.read_csv(path)
        metric_fn = get_instance(self.cfg_metric, registry=METRIC_REGISTRY)

        start = time.time()
        df_results = self.evaluate(df_results, graph_dataset, distance.compute_distance, metric_fn)
        end = time.time()
        self.logger.info(f"Time taken to evaluate flight plan: {end - start} seconds")

        # Save the final results
        if not df_results.empty:
            save_dir = Path(self.opt["save_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(save_dir / "flightplan_final.csv", index=False)
        return df_results