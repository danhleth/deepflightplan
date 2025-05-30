import pandas as pd
import numpy as np


distribution_dir = "/Users/danhleth/Projects/AIATFM/deepflightplan/runs/exp"
evaled_synthesized_flight_route_df = pd.read_csv(
    f"{distribution_dir}/evaled_synthesized_flight_route.csv"
)
evaled_ground_truth_flight_route_df = pd.read_csv(
    f"{distribution_dir}/evaled_ground_truth_flight_route.csv"
)

evaled_synthesized_flight_route_df = evaled_synthesized_flight_route_df[
   ~np.isinf(evaled_synthesized_flight_route_df['HausdorffDistance'])
]

evaled_ground_truth_flight_route_df = evaled_ground_truth_flight_route_df[
   ~np.isinf(evaled_ground_truth_flight_route_df['HausdorffDistance'])
]
evaled_synthesized_flight_route_df['ElapsedTime'] = evaled_synthesized_flight_route_df.apply(lambda row: row['total_distances'] / row['aircraft_speed_mph'], axis=1)
evaled_ground_truth_flight_route_df['ElapsedTime'] = evaled_synthesized_flight_route_df.apply(lambda row: row['total_distances'] / row['aircraft_speed_mph'], axis=1)


print("evaled_synthesized_flight_route_df: -----")
print(evaled_synthesized_flight_route_df['HausdorffDistance'].describe())
print(evaled_synthesized_flight_route_df['DiffTotalDistance'].describe())
print(evaled_synthesized_flight_route_df['ElapsedTime'].describe())

print("evaled_ground_truth_flight_route_df: -----")
print(evaled_ground_truth_flight_route_df['HausdorffDistance'].describe())
print(evaled_ground_truth_flight_route_df['DiffTotalDistance'].describe())
print(evaled_ground_truth_flight_route_df['ElapsedTime'].describe())
