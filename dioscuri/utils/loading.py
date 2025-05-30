from typing import Any, Dict

import yaml
import geojson
import json

def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)

# def load_geojson(filepath):
#     with open(filepath, 'r') as f:
#         data = geojson.load(f)
#     return data


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