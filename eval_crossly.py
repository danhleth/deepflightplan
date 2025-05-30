import pandas as pd
import numpy as np
import os
import folium
import networkx as nx
from tqdm import tqdm
from itertools import product
from dioscuri.datasets import Lido21EnrouteAirwayDataset
from dioscuri.metrics import HausdorffDistance, DiffTotalDistance
from dioscuri.distance import GeopyGreatCircleDistance
from dioscuri.datasets.dataset import Lido21EnrouteAirwayDataset
from dioscuri.utils.support_pipeline import parse_routestr_to_np
from joblib import Parallel, delayed
from pathlib import Path

import math

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

# Hausdorff distance calculation function
def compute_hausdorff(real_route_np, synthesized_route_np):
    return HausdorffDistance().calculate(synthesized_route_np, real_route_np, GeopyGreatCircleDistance().compute_distance)

def compute_total_distance(real_route_np, synthesized_route_np):
    return DiffTotalDistance().calculate(synthesized_route_np, real_route_np, GeopyGreatCircleDistance().compute_distance)

def visualize_routes(df, enroute_graph, airport_df):
    origin = df['Origin'].iloc[0]
    destination = df['Destination'].iloc[0]
    origin_lat, origin_lon = df.iloc[0][['Origin Latitude', 'Origin Longitude']]
    destination_lat, destination_lon = df.iloc[0][['Destination Latitude', 'Destination Longitude']]

    origin_coords = np.array([origin_lat, origin_lon])
    destination_coords = np.array([destination_lat, destination_lon])
    # Initialize a folium map centered approximately between Origin and Destination
    center_lat = (origin_lat + destination_lat) / 2
    center_lon = (origin_lon + destination_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

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

        # Plot each route
    for idx, row in df.iterrows():
        real_coords = row['Real Route NP']
        synth_coords = row['Synthesized Route NP']
        origin = row['Origin']
        destination = row['Destination']

        real_coords.insert(0, origin_coords)  # Insert origin at the start
        real_coords.append(destination_coords)  # Append destination at the end
        # Plot real route (blue)
        folium.PolyLine(
            locations=real_coords,
            color='blue',
            weight=3,
            opacity=0.8,
            popup=f'Real Route {idx+1}',
            dash_array='5'  # Dashed line for real route
        ).add_to(m)

        synth_coords.insert(0, origin_coords)  # Insert origin at the start
        synth_coords.append(destination_coords)  # Append destination at the end
        # Plot synthesized route (red) 
        folium.PolyLine(
            locations=synth_coords,
            color='red',
            weight=3,
            opacity=0.8,
            popup=f'Synthesized Route {idx+1}',
            dash_array='10, 5'  # Different dash pattern for synthesized route
        ).add_to(m)
        # Add directional triangles for real route
        for i in range(len(real_coords) - 1):
            start = real_coords[i]
            end = real_coords[i + 1]
            # Calculate midpoint
            mid_lat = (start[0] + end[0]) / 2
            mid_lon = (start[1] + end[1]) / 2
            # Calculate bearing for triangle orientation
            bearing = calculate_bearing(start, end)
            folium.RegularPolygonMarker(
                location=[mid_lat, mid_lon],
                number_of_sides=3,  # Triangle
                radius=5,
                rotation=bearing - 90,  # Adjust for triangle point to face direction
                color='blue',
                fill_color='blue',
                fill_opacity=0.6,
                popup=f'Real Route {idx+1} Direction'
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

        # Add markers for waypoints in real route
        for coord in real_coords:
            node = enroute_graph.get_node_from_coordinates(coord[0], coord[1])
            folium.CircleMarker(
                location=coord,
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                popup=f"Real Waypoint {node if node else idx+1}"
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

# Parallel computation of Hausdorff distances
def calculate_pairwise_distances(od_pair, od_synthesized_route_df, airport_df, save_dir=None):
    (origin, destination), real_group = od_pair
    results = []
    if (origin, destination) not in od_synthesized_route_df.groups.keys():
        print(f"Origin: {origin}, Destination: {destination} not found in synthesized route.")
        return results
    synthesized_group = od_synthesized_route_df.get_group((origin, destination))
    real_route_datas = real_group[['aircraft_speed_mph','route','route_np']].values
    synthesized_route_datas = synthesized_group[['aircraft_speed_mph','route','route_np']].values
    origin_lat, origin_lon = find_corresponding_airport(airport_df, origin)[['latitude', 'longitude']]
    destination_lat, destination_lon = find_corresponding_airport(airport_df, destination)[['latitude', 'longitude']]
    for real_route_data, synthesized_data in tqdm(product(
        real_route_datas, synthesized_route_datas
    )): 
        real_speed, real_route_str, real_route_np = real_route_data
        synthesized_speed, synthesized_route_str, synthesized_route_np = synthesized_data 
        try:
            real_route_np = np.asarray(real_route_np, dtype=float)
            synthesized_route_np = np.asarray(synthesized_route_np, dtype=float)
            # Remove rows with NaN values
            real_route_np = real_route_np[~np.isnan(real_route_np).any(axis=1)]
            synthesized_route_np = synthesized_route_np[~np.isnan(synthesized_route_np).any(axis=1)]
            hausdorff_distance = compute_hausdorff(real_route_np, synthesized_route_np)
            diff_total_distance = compute_total_distance(real_route_np, synthesized_route_np)
        except Exception as e:
            print(f"Error computing Hausdorff distance for {origin} to {destination}: {e}")
            hausdorff_distance = np.nan
            diff_total_distance = np.nan

        results.append({
            'Origin': origin,
            'Origin Latitude': origin_lat,
            'Origin Longitude': origin_lon,
            'Destination': destination,
            'Destination Latitude': destination_lat,
            'Destination Longitude': destination_lon,
            'Real Speed (MPH)': real_speed,
            'Synthesized Speed (MPH)': synthesized_speed,
            'Hausdorff Distance': hausdorff_distance,
            'Diff Total Distance': diff_total_distance,
            "Real Route": real_route_str,
            "Synthesized Route": synthesized_route_str,
            'Real Route NP': real_route_np.tolist(),
            'Synthesized Route NP': synthesized_route_np.tolist(),
        })

    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{origin}_{destination}.csv")
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Hausdorff Distance', 'Diff Total Distance'], ascending=True)
        enroute_graph = Lido21EnrouteAirwayDataset(file_path="/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/lido21/enroute_airway.geojson", hidden_sectors=True)
        m = visualize_routes(results_df, enroute_graph, airport_df)
        results_df.to_csv(save_path, index=False)
        m.save(os.path.join(save_dir, f"{origin}_{destination}.html"))
    return results 


def main():

    prefix_root = "/Users/danhleth/Projects/AIATFM/deepflightplan"
    enroute_graph = Lido21EnrouteAirwayDataset(file_path=f"{prefix_root}/datasets/lido21/enroute_airway.geojson", hidden_sectors=True)
    airport_df = pd.read_csv(f"{prefix_root}/datasets/airports/iata-icao.csv")

    # Conversion rate from nautical miles to statute miles
    NM_TO_M_RATE = 1.15078
    # Convert Mach to MPH
    MACH_TO_MPH_RATE = 767.269148

    # Use joblib for parallel processing

    prefix = "/Users/danhleth/Projects/AIATFM/deepflightplan/runs/exp"
    sub_save_dir = prefix + "/cross_compare_distance"


    tmp_synthesized_flight_route_df = pd.read_csv(f"{prefix}/evaled_synthesized_flight_route.csv")
    tmp_real_flight_route_df = pd.read_csv(f"{prefix}/evaled_ground_truth_flight_route.csv")

    # Filter out rows with nan values or infinities.
    tmp_synthesized_flight_route_df = tmp_synthesized_flight_route_df.dropna()
    tmp_real_flight_route_df = tmp_real_flight_route_df.dropna()
    tmp_synthesized_flight_route_df = tmp_synthesized_flight_route_df[~tmp_synthesized_flight_route_df.isin([np.inf, -np.inf]).any(axis=1)]
    tmp_real_flight_route_df = tmp_real_flight_route_df[~tmp_real_flight_route_df.isin([np.inf, -np.inf]).any(axis=1)]

    tmp_real_flight_route_df['route_np'] = tmp_real_flight_route_df['route_np'].apply(
        lambda x: parse_routestr_to_np(x)
    )
    tmp_synthesized_flight_route_df['route_np'] = tmp_synthesized_flight_route_df['route_np'].apply(
        lambda x: parse_routestr_to_np(x)
    )

    od_real_route_df = tmp_real_flight_route_df.groupby(['origin', 'destination'])
    od_synthesized_route_df = tmp_synthesized_flight_route_df.groupby(['origin', 'destination'])


    # for od_pair in od_real_route_df:
    #     origin, destination = od_pair[0]
    #     rs = calculate_pairwise_distances(od_pair, sub_save_dir)

    results = Parallel(n_jobs=-1)(delayed(calculate_pairwise_distances)(od_pair, od_synthesized_route_df, airport_df, sub_save_dir) 
                                for od_pair in od_real_route_df)
    rs_Df = pd.DataFrame([item for sublist in results for item in sublist])
        
    # Save results
    rs_Df.to_csv(Path(prefix)/"cross_compare_distance.csv", index=False)

if __name__ == "__main__":
    main()