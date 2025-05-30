from dioscuri.datasets.dataset import Lido21EnrouteAirwayDataset
import folium
import numpy as np
import geojson

def load_gepjson_data(file_path):
    with open(file_path, 'r') as f:
        data = geojson.load(f)
    return data

def visualize_sectors():
    # Create a map object
    m = folium.Map(location=[40, -100], zoom_start=4)

    # Load GeoJSON file
    # folium.GeoJson('/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/lido21/enroute_airway.geojson', name='GeoJSON').add_to(m)
    folium.GeoJson('/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/facilities/sectors.geojson', name='GeoJSON').add_to(m)

    # Save the map as an HTML file
    m.save('sectors.html')

def filter_routes_by_icao(data, icao_code):
    """
    Filter GeoJSON FeatureCollection by ICAO code and return in original format.
    
    Args:
        data (dict): GeoJSON FeatureCollection
        icao_code (str): ICAO code to filter by (e.g., 'RJ', 'RC', 'VV')
    
    Returns:
        dict: GeoJSON FeatureCollection with features where fix0_icao or fix1_icao matches icao_code
    """
    # Initialize the output FeatureCollection
    filtered_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Iterate through features and filter by ICAO code
    for feature in data['features']:
        properties = feature['properties']
        if properties['fix0_icao'] == icao_code or properties['fix1_icao'] == icao_code:
            # Append the feature as-is to maintain original format
            filtered_data['features'].append(feature)
    
    return filtered_data


def visualize_enroute_airways():
    data_path = '/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/lido21/enroute_airway.geojson'
    enroute_airway_data = load_gepjson_data(data_path)

    icao_code = 'PT'
    matched_data = filter_routes_by_icao(enroute_airway_data, icao_code)
    m = folium.Map(location=[40, -100], zoom_start=4)

    # Load GeoJSON file
    folium.GeoJson(enroute_airway_data, name='GeoJSON').add_to(m)

    # Save the map as an HTML file
    # m.save(f'{icao_code}_enroute_airwat.html')
    m.save(f'enroute_airway.html')


if __name__ == "__main__":
    visualize_enroute_airways()
    # visualize_sectors()