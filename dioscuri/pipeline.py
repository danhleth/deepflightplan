import logging
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import time
from multiprocessing import Pool
from functools import partial

from dioscuri.opt import Opts


from dioscuri.datasets import DATASET_REGISTRY
from dioscuri.algorithms import ALGORITHM_REGISTRY
from dioscuri.distance import DISTANCE_REGISTRY
from dioscuri.metrics import METRIC_REGISTRY

from dioscuri.utils.getter import (get_instance)
from dioscuri.utils.support_pipeline import (process_single_od, \
                                            calculate_distance_od, \
                                            route_to_sector_unique, \
                                            route_to_sector_with_count_waypoint)


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

    def synthesize_flightroute_od(self, algorithm, distance, od_airport_dataset, graph_datasource):
        """
        Generate flight routes using multiprocessing 
        """
        #### PROCEDURE PROCESSING - FOR DEBUGGING PURPOSE ####
        # df_results = pd.DataFrame()
        # for index, row in tqdm(od_airport_dataset.df.iterrows(), total=len(od_airport_dataset.df)):
        #     args = (index, row, algorithm, distance, graph_datasource, self.opt, self.logger)
        #     tmp_df = process_single_od(args)
            
        #     if tmp_df is not None and not tmp_df.empty:
        #         df_results = pd.concat([df_results, tmp_df], axis=0)
        # return df_results


        #### PARALLEL PROCESSING ####
        self.logger.info("Generating flight plan with multiprocessing")
        # Prepare arguments for multiprocessing
        args = [(index, row, algorithm, distance, graph_datasource, self.opt, self.logger) 
                for index, row in od_airport_dataset.df.iterrows()]
        
        # Use number of CPU cores minus 1 to leave some resources free
        num_processes = max(1, self.opt["num_processes"])
        self.logger.info(f"Using {num_processes} processes")
        
        # Process in parallel
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_single_od, args), 
                            total=len(od_airport_dataset.df)))
        
        df_results = pd.DataFrame()
        # Combine results
        for i, tmp_df in enumerate(results): 
            if tmp_df is not None and not tmp_df.empty:
                df_results = pd.concat([df_results, tmp_df], axis=0)

        # for od_key, od_group_df in df_results.groupby(['carrier_code', 'flight_number','origin', 'destination', 'utc_dep_time']):
                # save_dir = Path(self.opt["save_dir"]) / "OD-SynthesizedFlightRoute" 
                # save_dir.mkdir(parents=True, exist_ok=True)
                # filename = str("-".join(od_key))
                # od_group_df.to_csv(save_dir / f"{filename}.csv", index=False)

        self.logger.info(f"There are {len(df_results)} flight plans synthesized")
        # df_results.to_csv(Path(self.opt["save_dir"]) / "synthesized_flight_route.csv", index=False)
        return df_results
    
    def evaluate_with_shortest_route(self, df, graph_datasource, distance_fn, metric_fns, sub_save_dir="Eval"):
        # Group the dataframe by origin and destination
        self.logger.info(f"Evaluating {len(df)} flight plans")

        # Group the DataFrame by 'origin' and 'destination'
        groups_od = df.groupby(['carrier_code', 'flight_number','origin', 'destination', 'utc_dep_time'])

        ## PROCESSING - FOR DEBUGGING PURPOSE ###
        # Process results and save to CSV

        # save_dir = Path(self.opt["save_dir"]) / sub_save_dir
        # save_dir.mkdir(parents=True, exist_ok=True)

        # all_results = []
        # for od_pair, od_group_df in groups_od:
        #     df_rs = calculate_distance_od((od_pair, od_group_df), graph_datasource, distance_fn, metric_fns)
        #     # Add the results to the od_group_df
        #     for metric_fn in metric_fns:
        #         od_group_df.insert(len(od_group_df.columns), str(metric_fn), df_rs[str(metric_fn)].to_list())
        #     if "route_np" not in od_group_df.columns:
        #             # Convert the route_np column to a list
        #         od_group_df.insert(len(od_group_df.columns), "route_np", df_rs["route_np"].to_list())
        #     od_group_df.to_csv(save_dir / (str("-".join(od_pair)) + ".csv"), index=False)
        #     all_results.append(od_group_df)

        # # Concatenate all results into a single DataFrame

        # combined_df = pd.concat(all_results, ignore_index=True)

        # return combined_df

        #### PARALLEL PROCESSING ####
        # Convert groups to a list for parallel processing
        od_groups_list = list(groups_od)
        # Create a partial function with fixed arguments
        eval_fn = partial(calculate_distance_od, graph_datasource=graph_datasource, distance_fn=distance_fn, metric_fns=metric_fns)

        # Use multiprocessing Pool with specified number of processes
        num_processes = max(self.opt["num_processes"], 1)
        self.logger.info(f"Using {num_processes} processes for evaluation")

        all_results = []
        # Use Pool.imap to process groups in parallel and handle results
        with Pool(processes=num_processes) as pool:
            # Apply eval_fn to each OD group in parallel
            results_iter = pool.imap(eval_fn, (group for group in od_groups_list))

            # Process results and save to CSV
            # save_dir = Path(self.opt["save_dir"]) / sub_save_dir
            # save_dir.mkdir(parents=True, exist_ok=True)
        
            # Iterate over groups and results simultaneously
            for (od_key, od_group), rs_df in zip(od_groups_list, results_iter):
                for metric_fn in metric_fns:
                    od_group.insert(len(od_group.columns), str(metric_fn), rs_df[str(metric_fn)].to_list())
                if "route_np" not in od_group.columns:
                    # Convert the route_np column to a list
                    od_group.insert(len(od_group.columns), "route_np", rs_df["route_np"].to_list())
                if "total_distances" not in od_group.columns:
                    od_group.insert(len(od_group.columns)-3, "total_distances", rs_df["total_distances"].to_list())
                # od_group.to_csv(save_dir / (str("-".join(od_key)) + ".csv"), index=False)
                all_results.append(od_group)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df


    def filtering_logic(self, df, graph_datasource, algorithm):
        self.logger.info("Filtering flight plans based on the filtering logic")
        od_group = df.groupby(['carrier_code', 'flight_number','origin', 'destination', 'utc_dep_time'])
        top_k = algorithm.top_k if hasattr(algorithm, 'top_k') else 10
        
        combined_df = pd.DataFrame()
        for od_key, od_group_df in od_group:
            od_group_df['unique_sectors'] = od_group_df['route'].apply(lambda x: route_to_sector_unique(x, graph_datasource))
            od_group_df['len_unique_sectors'] = od_group_df['unique_sectors'].apply(lambda x: len(x.split("-")))
            od_group_df = od_group_df.sort_values(by=["total_distances", "len_unique_sectors"], ascending=True)
            # Filter the top k flight plans based on the total distances
            if len(od_group_df) > top_k:
                od_group_df = od_group_df.head(top_k)
                combined_df = pd.concat([combined_df, od_group_df], axis=0)
            else:
                combined_df = pd.concat([combined_df, od_group_df], axis=0)
        if combined_df.empty:
            self.logger.warning("No flight plans passed the filtering logic")
            return df
        self.logger.info(f"Filtered flight plans: {len(combined_df)}")
        return combined_df


    def fit(self):
        # Load the list_od_pair_dataset
        gt_airport_dataset = None
        evaled_synthesized_df = None
        od_airport_dataset = self.get_dataset(self.cfg_data["od_dataset"])
        if "ground_truth" in self.cfg_data:
            gt_airport_dataset = self.get_dataset(self.cfg_data["ground_truth"])

        # Load the graph_datasource
        graph_datasource = self.get_dataset(self.cfg_data["datasource"])
        # Load the distance
        distance = get_instance(self.cfg_distance, DISTANCE_REGISTRY)
        # Load the algorithm    
        algorithm = get_instance(self.cfg_algorithm, registry=ALGORITHM_REGISTRY)
        # Load the metric functions
        metric_fns = [get_instance(m, registry=METRIC_REGISTRY) for m in self.cfg_metric] if isinstance(self.cfg_metric, list) else [get_instance(self.cfg_metric, registry=METRIC_REGISTRY)]
        
        # Start generate the flight plan
        synthesized_df = self.synthesize_flightroute_od(algorithm, distance, od_airport_dataset, graph_datasource)

        if synthesized_df.empty:
            self.logger.warning("No flight plans synthesized. Exiting.")
            return
        
        self.logger.info("shape before filtering: {}".format(synthesized_df.shape))
        # Filter the synthesized flight plans based on the filtering logic
        synthesized_df = self.filtering_logic(synthesized_df, graph_datasource, algorithm)
        synthesized_df.to_csv(Path(self.opt["save_dir"]) / "synthesized_flight_route.csv", index=False)
        # synthesized_df = pd.read_csv(Path(self.opt["save_dir"]) / "synthesized_flight_route.csv")

        evaled_synthesized_df = self.evaluate_with_shortest_route(synthesized_df, graph_datasource, distance.compute_distance, metric_fns, sub_save_dir="OD-EvalSynthesized")
        evaled_synthesized_df.to_csv(Path(self.opt["save_dir"]) / "evaled_synthesized_flight_route.csv", index=False)
        
        if gt_airport_dataset is not None:
            self.logger.info('Ground truth dataset len: {}'.format(len(gt_airport_dataset.df)))
            evaled_gt_df = self.evaluate_with_shortest_route(gt_airport_dataset.df, graph_datasource, distance.compute_distance, metric_fns, sub_save_dir="OD-GT")
            self.logger.info(f"Evaluating GT flight plans with {len(evaled_gt_df)} flight plans")
            if not evaled_gt_df.empty:
                save_dir = Path(self.opt["save_dir"])
                # Filter the results based on the filtering logic
                evaled_gt_df.to_csv(save_dir / "evaled_ground_truth_flight_route.csv", index=False)
