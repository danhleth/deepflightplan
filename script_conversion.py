
import json
import ast
from datetime import datetime, timedelta
import math
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, LineString, Point, LinearRing, MultiPoint
from shapely import intersection, within
from dioscuri.utils.support_pipeline import parse_routestr_to_np
from dioscuri.distance import GeopyGreatCircleDistance
import folium
import os
from copy import deepcopy
from tqdm import tqdm

distance_fn = GeopyGreatCircleDistance().compute_distance

# Function to load GeoJSON from a file
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

def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon.

    Parameters:
    - point (tuple): A tuple representing the point (latitude, longitude).
    - polygon (list): A list of tuples representing the polygon vertices.

    Returns:
    - bool: True if the point is inside the polygon, False otherwise.
    """
    point = Point(point)
    polygon = Polygon(polygon)
    return within(point, polygon)

def calculate_facilities_entry_exit_per_route(sector, route_np, sector_name, row_df):
    for i in range(len(route_np) - 1):
        route_line = LineString([route_np[i], route_np[i + 1]])
        polygon = LinearRing(sector['geometry']['coordinates'][0])
        intersection_points = intersection(route_line, polygon)

        if intersection_points.is_empty:
            print(f"No intersection found for flight {row_df['carrier_code']}{row_df['flight_number']} in sector {sector_name}")
            return None, None

        print(f"Flight {row_df['carrier_code']}{row_df['flight_number']} intersects with sector {sector_name} at points:")
        for point in intersection_points.geoms:
            print(f" - Intersection at {point.x}, {point.y}")

# def append_origin_destination_coordinates(row_df, sector_name):

def search_sector_by_coordinates(sector_data, coordinates):
    rs = None
    for sector in sector_data['features']:
        if sector['geometry']['type'] == 'Polygon':
            coords = [[x, y] for (y, x) in sector['geometry']['coordinates'][0]]
            if is_point_in_polygon(coordinates, coords):
                rs =  sector['properties']['index']
        elif sector['geometry']['type'] == 'MultiPolygon':
            for polygon in sector['geometry']['coordinates']:
                coords = [[x, y] for (y, x) in polygon[0]]
                if is_point_in_polygon(coordinates, coords):
                    rs = sector['properties']['index']
        elif sector['geometry']['type'] == 'GeometryCollection':
            for geom in sector['geometry']['geometries']:
                if geom['type'] == 'Polygon':
                    coords = [[x, y] for (y,x) in geom['coordinates'][0]]
                    if is_point_in_polygon(coordinates, coords):
                        rs =  sector['properties']['index']
                elif geom['type'] == 'MultiPolygon':
                    for polygon in geom['coordinates']:
                        if is_point_in_polygon(coordinates, polygon):
                            rs =  sector['properties']['index']
                elif geom['type'] == 'LineString':
                    continue
                    # line = LineString(geom['coordinates'])
                    # point = Point(coordinates)
                    # if point.intersects(line):
                    #     return sector['properties']['index']
    return rs


def get_candidate_sector(sector_data, route_np):
    candidates = []
    if len(route_np) < 2:
        print("Route is too short to calculate intersections.")
        return candidates
    
    for i in range(len(sector_data['features'])):
        sector = sector_data['features'][i]
        if sector['geometry']['type'] == 'Polygon':
            sector_coords = [[x,y] for y,x in sector['geometry']['coordinates'][0]]
        elif sector['geometry']['type'] == 'MultiPolygon':
            sector_coords = [[x,y] for y,x in sector['geometry']['coordinates'][0][0]]
        elif sector['geometry']['type'] == 'GeometryCollection':
            full_sector_coords = []
            for geom in sector['geometry']['geometries']:
                if geom['type'] == 'Polygon':
                    sector_coords = [[x,y] for y,x in geom['coordinates'][0]]
                    full_sector_coords.extend(sector_coords)
                elif geom['type'] == 'MultiPolygon':
                    sector_coords = [[x,y] for y,x in geom['coordinates'][0][0]]
                    full_sector_coords.extend(sector_coords)
                elif geom['type'] == 'LineString':
                    sector_coords = [[x,y] for y,x in geom['coordinates']]
                    # full_sector_coords.extend(sector_coords)
            sector_coords = full_sector_coords
        polygon = Polygon(sector_coords)
        line = LineString(route_np)

        if line.intersects(polygon):
            tmp_sector = deepcopy(sector)
            tmp_sector['geometry']['coordinates'] = sector_coords
            candidates.append(tmp_sector)


    return candidates

def check_point_line_string(point, line):
    """
    Check if a point is on a line string.
    """
    point = Point(point)
    return point.intersects(line)

def calculate_facilities_entry_exit(sector_data, flight_plan_df):

    def check_point_in_np_array(point, np_array):
        """
        Check if a point is in a numpy array of coordinates.
        """
        return any(np.array_equal(point, coord) for coord in np_array)

    results_data = {"carrier_code": [], "flight_number": [],
                    "origin":[],"destination":[], 
                    "sector_name": [], "entry_time": [], "exit_time": []}
    
    used_sectors = set()  # To keep track of used sectors

    for _, row_df in tqdm(flight_plan_df.iterrows(), total=len(flight_plan_df)):
        route_np = row_df['route_np']
        org_lat, org_lon = row_df['origin_lat'], row_df['origin_long']
        dest_lat, dest_lon = row_df['destination_lat'], row_df['destination_long']
        route_np = np.insert(route_np, 0, [org_lat, org_lon], axis=0)  # Insert origin coordinates
        route_np = np.insert(route_np, len(route_np), [dest_lat, dest_lon], axis=0)  # Insert destination coordinates
        aircraft_speed  = row_df['aircraft_speed_mph']
        sector_candidates = get_candidate_sector(sector_data, route_np) # Filter sectors that intersect with the route
        
        if sector_candidates is None or len(sector_candidates) == 0:
            print(f"No intersecting sectors found for flight {row_df['carrier_code']}{row_df['flight_number']} from {row_df['origin']} to {row_df['destination']}.")
            continue
        
        data_dict = {}
        recoords = []
        sector_recoords = []
        for i in range(len(route_np)-1):
            
            first_point = np.array(route_np[i])
            second_point = np.array(route_np[i + 1])

            sector_first_point = search_sector_by_coordinates(sector_data, first_point)
            sector_second_point = search_sector_by_coordinates(sector_data, second_point)

            line = LineString([first_point, second_point])
            key = f"{i}-{i+1}"
            if key not in data_dict:
                data_dict[key] = []
            
            for sector in sector_candidates:
                used_sectors.add(sector['properties']['index'])
                sector_coords = sector['geometry']['coordinates']
                sector_linering = LinearRing(sector_coords)

                sector_polygon = Polygon(sector_coords)
                lring_intersection_points = intersection(line, sector_linering)
                polygon_intersection_points = intersection(line, sector_polygon)

                # if lring_intersection_points.is_empty and polygon_intersection_points.is_empty:
                #     # print(key, "No intersection found", line, sector['properties']['index'])
                #     if check_point_in_np_array(first_point, route_np):
                #         recoords.append(first_point)
                #     if check_point_in_np_array(second_point, route_np):
                #         recoords.append(second_point)
                
                if lring_intersection_points.is_empty and (not polygon_intersection_points.is_empty) and (sector_first_point == sector_second_point):
                    if check_point_in_np_array(first_point, route_np):
                        recoords.append(first_point)
                        sector_recoords.append(sector_first_point)

                    if check_point_in_np_array(second_point, route_np):
                        recoords.append(second_point)
                        sector_recoords.append(sector_second_point)
                
                if (not lring_intersection_points.is_empty) and (not polygon_intersection_points.is_empty):
                    if isinstance(lring_intersection_points, Point) and isinstance(polygon_intersection_points, LineString):
                        if check_point_line_string(first_point, polygon_intersection_points):
                            recoords.append(first_point)
                            recoords.append(np.array([lring_intersection_points.x, lring_intersection_points.y]))
                            sector_recoords.append(sector_first_point)
                            sector_recoords.append(sector_first_point)
                        
                        if check_point_line_string(second_point, polygon_intersection_points):
                            recoords.append(np.array([lring_intersection_points.x, lring_intersection_points.y]))
                            recoords.append(second_point)
                            sector_recoords.append(sector_second_point)
                            sector_recoords.append(sector_second_point)

        assert len(recoords) == len(sector_recoords), "Recoords and sector_recoords must have the same length"
        
        total_time = 0
        entry_time = 0
        exit_time = 0

        total_time = row_df['utc_dep_time']
        sub_travel_time = timedelta(hours=distance_fn(route_np[0], route_np[1]) / aircraft_speed)
        entry_time = total_time + sub_travel_time
        total_time = entry_time
        for i in range(1,len(recoords)-1):
            first_point = recoords[i]
            second_point = recoords[i + 1]

            sector_first_point = sector_recoords[i]
            sector_second_point = sector_recoords[i + 1]

            if (sector_first_point != sector_second_point) and (distance_fn(first_point, second_point) < 5): # point is on the sector boundary
                exit_time = total_time
                results_data["carrier_code"].append(row_df['carrier_code'])
                results_data["flight_number"].append(row_df['flight_number'])
                results_data["origin"].append(row_df['origin'])
                results_data["destination"].append(row_df['destination'])
                results_data["sector_name"].append(sector_first_point)
                results_data["entry_time"].append(entry_time)
                results_data["exit_time"].append(exit_time)
                entry_time = total_time
            
            else:
                # print(sector_first_point, sector_second_point, first_point, second_point, (sector_first_point != sector_second_point), (np.all(first_point == second_point)))
                distance_between_two_points = distance_fn(first_point, second_point)
                sub_travel_time = timedelta(hours=distance_between_two_points / aircraft_speed)
                total_time += sub_travel_time
            
            if i == len(recoords) - 1:
                exit_time = total_time
                results_data["carrier_code"].append(row_df['carrier_code'])
                results_data["flight_number"].append(row_df['flight_number'])
                results_data["origin"].append(row_df['origin'])
                results_data["destination"].append(row_df['destination'])
                results_data["sector_name"].append(sector_first_point)
                results_data["entry_time"].append(entry_time)
                results_data["exit_time"].append(exit_time)
    
    return results_data


def main():
    synthesized_flight_route_path = "/Users/danhleth/Projects/AIATFM/deepflightplan/runs/exp2/synthesized_flight_route.csv"
    # synthesized_flight_route_path = "/Users/danhleth/Projects/AIATFM/deepflightplan/runs/exp2/saved_synthesized_routes/WSSS-WMKK.csv"
    sector_path = "/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/facilities/sectors.geojson"
    sector_data = load_geojson(sector_path)

    synthesized_flight_route_df = pd.read_csv(synthesized_flight_route_path)
    synthesized_flight_route_df['route_np'] = synthesized_flight_route_df['route_np'].apply(parse_routestr_to_np)
    synthesized_flight_route_df['utc_dep_time'] = pd.to_datetime(synthesized_flight_route_df['utc_dep_time'])

    # names = ["RPHI1"] #, "WSJCA"]
    # sector_data = filter_sector_by_name(sector_data, names)

    # point = [5.93721, 116.051]
    # point = [4.0, 119.81]
    # point = [ 14.5086, 121.02]
    # found_sector = search_sector_by_coordinates(sector_data, point)
    # print("Points found in sector:", found_sector)

    rl_flight_plan = calculate_facilities_entry_exit(sector_data, synthesized_flight_route_df)
    rl_flight_plan_df = pd.DataFrame(rl_flight_plan)
    rl_flight_plan_df.to_csv("rl_flight_plan.csv", index=False)

    exit(0)
    m = folium.Map(location=point, zoom_start=6)
    folium.Marker(location=point, popup=f"{point[0]} {point[1]}").add_to(m)
    popup = folium.GeoJsonPopup(
    fields=["index"],
    aliases=["sector_name"],
    localize=True,
    labels=True,
    style="background-color: yellow;",
)
    folium.GeoJson(sector_data, popup=popup).add_to(m)
    m.save("intersections/sector_map.html")

   


if __name__ == "__main__":
    main()