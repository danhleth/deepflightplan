from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from dioscuri.datasets import Lido21EnrouteAirwayDataset
from dioscuri.metrics import Hausdorff
from dioscuri.distance import GeopyGreatCircleDistance
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import ast
import re

def parse_route_np(route_str):
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
        return np.array([[np.nan, np.nan]])


# Define the function to evaluate a single origin-destination group
def eval_single_od(od_group, graph_dataset, distance_fn, metric_fn):
    _, od_df = od_group
    top_1 = od_df.iloc[0]
    # trajectory_A = route_to_list_waypoint(top_1["route"], graph_dataset)
    trajectory_A = top_1["route_np"]
    
    trajectory_A = np.asarray(trajectory_A, dtype=np.float64)
    trajectory_A = trajectory_A[~np.isnan(trajectory_A).any(axis=1)]
    rs = []
    for i, row in tqdm(od_df.iterrows(), leave=False):
        trajectory_B = np.asarray(row['route_np'], dtype=np.float64)
        trajectory_B = trajectory_B[~np.isnan(trajectory_B).any(axis=1)]
        haursdorff_distance = metric_fn.calculate(trajectory_A, trajectory_B, distance_fn)
        if haursdorff_distance is None:
            continue
        rs.append(haursdorff_distance)
    return rs
    
def evaluate_with_shortest_route(df, graph_dataset, distance_fn, metric_fn, num_processes=None, save_dir=None):

    # Group the DataFrame by 'origin' and 'destination'
    groups_od = df.groupby(['ADEP', 'ADES'])

    # Convert groups to a list for parallel processing
    od_groups_list = list(groups_od)

    all_results = []


    # Process results and save to CSV
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over groups and results simultaneously
    for od_group in tqdm(od_groups_list):
        od_key, od_df = od_group
        rs = eval_single_od(od_group=od_group, graph_dataset=graph_dataset, distance_fn=distance_fn, metric_fn=metric_fn)
        od_df['hausdorff'] = rs
        od_df['route_np'] = od_df['route_np'].apply(lambda x: x.to_list())
        od_df.to_csv(save_dir / (str("-".join(od_key)) + ".csv"), index=False)
        all_results.append(od_df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df



def main():
    prefix_root = "/Users/danhleth/Projects/AIATFM/deepflightplan"
    enroute_graph = Lido21EnrouteAirwayDataset(file_path=f"{prefix_root}/datasets/lido21/enroute_airway.geojson", hidden_sectors=True)
    airport_df = pd.read_csv(f"{prefix_root}/datasets/airports/iata-icao.csv")

    tmp_real_flight_route_df = pd.read_csv(f"{prefix_root}/runs/exp/real_flight_route.csv")

    tmp_real_flight_route_df['route_np'] = tmp_real_flight_route_df['route_np'].apply(
        lambda x: parse_route_np(x)
    )

    rs = evaluate_with_shortest_route(
        df=tmp_real_flight_route_df,
        graph_dataset=enroute_graph,
        distance_fn=GeopyGreatCircleDistance().compute_distance,
        metric_fn=Hausdorff(),
        save_dir=Path(prefix_root) / "runs" / "exp" / "real_flight_route_eval"
    )

if __name__ == "__main__":
    main()