
import json
import ast
from datetime import datetime, timedelta
import math
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, LineString, Point, LinearRing
from shapely import intersection
from dioscuri.utils.support_pipeline import parse_routestr_to_np
import folium
import os

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

def calculage_facilities_entry_exit_per_route(sector, route_np):
    pass

def calculate_facilities_entry_exit(sector_data, flight_plan_df):
    for sector in sector_data['features']:
        sector_name = sector['properties']['index']
        sector_polygon = sector['geometry']['coordinates'][0]
        sector_polygon = Polygon(sector_polygon)
        print(sector_polygon)
        break
        for index, row in flight_plan_df.iterrows():
            route_np = row['route_np']
            route_line = LineString(route_np)
            
            if route_line.intersects(sector_polygon):
                intersection_point = route_line.intersection(sector_polygon)
                if isinstance(intersection_point, Point):
                    print(f"Flight {row['flight_id']} intersects {sector_name} at {intersection_point.x}, {intersection_point.y}")
                elif isinstance(intersection_point, LineString):
                    print(f"Flight {row['flight_id']} intersects {sector_name} along {intersection_point.wkt}")


def main():
    synthesized_flight_route_path = "/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/deepflightplan/runs/exp/synthesized_flight_route.csv"
    sector_path = "/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/deepflightplan/datasets/facilities/sectors.geojson"
    sector_data = load_geojson(sector_path)
    synthesized_flight_route_df = pd.read_csv(synthesized_flight_route_path)
    synthesized_flight_route_df['route_np'] = synthesized_flight_route_df['route_np'].apply(parse_routestr_to_np)


    calculate_facilities_entry_exit(sector_data, synthesized_flight_route_df)
    exit(0)

    # rl_flight_plan = calculate_fir_entry_exit(sector_data, synthesized_flight_route_df)
    # rl_flight_plan_df = pd.DataFrame(rl_flight_plan)
    # rl_flight_plan_df.to_csv("rl_flight_plan.csv", index=False)

    names = ["WSJCB"] #, "WSJCA"]
    os.makedirs("intersections", exist_ok=True)
    sector_data = filter_sector_by_name(sector_data, names)

    route_np = [[0.425589, 101.441611], [0.7, 102.27], [0.996667, 103.165], [1.041667, 103.498333], [1.35019,103.994]] # 
    # route_np = [[0.425589, 101.441611], [0.7, 102.27]] # 
    # route_np = [[0.996667, 103.165], [1.041667, 103.498333]]
    # route_np = [[y,x] for x,y in route_np]  # Convert to (lat, lon) format
    
    polygon = sector_data['features'][0]['geometry']['coordinates'][0]
    polygon = [[x,y] for y,x in polygon]
    points = intersection(LineString(route_np), LinearRing(polygon))

    for point in points.geoms:
        print(point.x, point.y)

    m = folium.Map(location=[0.7, 102.27], zoom_start=6)

    # route_np = [[x,y] for y,x in route_np]  # Convert to (lat, lon) format
    # polygon = [[x,y] for y,x in polygon]  # Convert to (lat, lon) format

    folium.PolyLine(route_np, color='blue', weight=2.5, opacity=0.8).add_to(m)
    folium.Polygon(polygon, color='red', weight=2.5, opacity=0.8).add_to(m)
    for point in points.geoms:
        folium.Marker(location=[point.x, point.y], popup=f'Intersection at {point.x}lat {point.y}long', icon=folium.Icon(color='green')).add_to(m)

    os.makedirs("intersections", exist_ok=True)
    m.save("intersections/intersection_map.html")


if __name__ == "__main__":
    main()