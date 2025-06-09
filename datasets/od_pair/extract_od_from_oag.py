import pandas as pd
from tqdm import tqdm
import timezonefinder
import pytz
from datetime import datetime
import json
from shapely.geometry import Polygon, LineString


# Constants
NM_TO_MILE = 1.15078


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

def find_data_coodinate(airport_dict, airport_iata_name):
    """Look up airport data from a dictionary for faster access."""
    airport_iata_name = airport_iata_name.upper()
    return airport_dict.get(airport_iata_name)

def local_timestr_to_utc(coordinates, local_time_str, tz_cache, time_format='%Y-%m-%d %H:%M'):
    """Converts local time at an airport to UTC with timezone caching."""
    lat, long = coordinates
    coord_key = (lat, long)

    # Check cache for timezone
    if coord_key not in tz_cache:
        tf = timezonefinder.TimezoneFinder()
        tz_name = tf.certain_timezone_at(lat=lat, lng=long)
        if tz_name is None:
            return None
        tz_cache[coord_key] = pytz.timezone(tz_name)
    
    local_timezone = tz_cache[coord_key]
    
    try:
        local_dt = datetime.strptime(local_time_str, time_format)
        local_dt_aware = local_timezone.localize(local_dt)
        utc_dt = local_dt_aware.astimezone(pytz.utc)
        return utc_dt
    except ValueError:
        print(f"Error: Invalid time format for '{local_time_str}'. Expected '{time_format}'.")
        return None

def estimate_arrival_day(dep_time_str, flying_time_str):
    """Estimate the arrival time based on departure time and flying time."""
    dep_time = pd.to_datetime(dep_time_str, format='%Y-%m-%d %H:%M')
    hours, minutes = map(int, flying_time_str.split(':'))
    flying_time = hours * 60 + minutes  # Convert to minutes
    arrival_day = dep_time + pd.Timedelta(minutes=flying_time)
    if arrival_day < dep_time:
        arrival_day += pd.Timedelta(days=1)
    return arrival_day.strftime('%Y-%m-%d')


def check_flights_crossing_sector(geojson_data, df):
    """
    Check which flights' origin-destination paths cross the Bangkok sector.

    Parameters:
    - geojson_data (dict): GeoJSON data containing the Bangkok sector polygon.
    - df.

    Returns:
    - list: List of dictionaries with flight details that cross the sector.
    """

    # Extract the Bangkok sector polygon coordinates from GeoJSON
    # sector_polygon_coords = geojson_data['features'][0]['geometry']['coordinates'][0]

    crossing_flights = []
    num_sectors = len(geojson_data['features'])
    for _, feature in enumerate(tqdm(geojson_data['features'])):
        try:
            sector_polygon_coords = feature['geometry']['coordinates'][0]
            sector_polygon = Polygon(sector_polygon_coords)
            # Parse the CSV flight data
            # Store flights that cross the sector
            
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking flights in sector {feature['properties']['index']}  {_}/{num_sectors}"):
                # Extract origin and destination coordinates
                org_long = float(row['org_long'])
                org_lat = float(row['org_lat'])
                dest_long = float(row['dest_long'])
                dest_lat = float(row['dest_lat'])
                
                # Create a LineString for the flight path (great circle approximated as straight line in lat/long)
                flight_path = LineString([(org_long, org_lat), (dest_long, dest_lat)])
                
                # Check if the flight path intersects the sector polygon
                if flight_path.intersects(sector_polygon):
                    crossing_flights.append({
                        'carrier_code': row['carrier_code'],
                        'flight_number': row['flight_number'],
                        'specific_aircraft_code': row['specific_aircraft_code'],
                        'origin': row['origin'],
                        'destination': row['destination'],
                        'org_lat': org_lat,
                        'org_long': org_long,
                        'dest_lat': dest_lat,
                        'dest_long': dest_long,
                        'utc_dep_time': row['utc_dep_time'],
                        'flying_time': row['flying_time'],
                        'aircraft_range': row['aircraft_range'],
                        'aircraft_speed': row['aircraft_speed']
                    })
        except Exception as e:
            print(f"Error processing feature {feature['properties']['index']}: {str(e)}")
            continue
    
    return crossing_flights


def extract_oag_file(file_path, airport_file):

    # Read data
    df = pd.read_csv(file_path, usecols=[
        'Carrier Code', 'Flight No', 'Specific Aircraft Code', 'Dep Airport Code', 'Arr Airport Code',
        'Local Dep Time', 'Local Arr Time', 'Flying Time',
        'Aircraft Range (NM)', 'Aircraft Cruise Speed (MPH)', 'Time series'
    ])
    airport_df = pd.read_csv(airport_file)


    if df is None or df.empty:
        print("No flights crossing the sector found.")
        return

    # Remove rows with NaN IATA codes
    initial_len = len(airport_df)
    airport_df = airport_df[airport_df['iata'].notna()]
    if len(airport_df) < initial_len:
        print(f"Removed {initial_len - len(airport_df)} rows with NaN IATA codes.")

    # Check for duplicate IATA codes
    duplicates = airport_df[airport_df['iata'].duplicated(keep=False)]
    if not duplicates.empty:
        print(f"Warning: Found {len(duplicates)} duplicate IATA codes in airport file:")
        print(duplicates[['iata', 'icao', 'latitude', 'longitude', 'airport', 'country_code']])
        print("Keeping the sectorst occurrence of each IATA code.")

    # Remove duplicates, keeping the first occurrence
    airport_df = airport_df.drop_duplicates(subset='iata', keep='first')

    # Create a dictionary for faster airport lookups
    airport_dict = airport_df.set_index('iata')[['icao', 'latitude', 'longitude']].to_dict('index')

    # Initialize timezone cache
    tz_cache = {}

    # Preprocess time strings
    df['Local Dep Time'] = df['Local Dep Time'].astype(str).str.zfill(4)
    df['Local Arr Time'] = df['Local Arr Time'].astype(str).str.zfill(4)
    df['Local Dep Time Str'] = df['Time series'] + ' ' + df['Local Dep Time'].str[:2] + ':' + df['Local Dep Time'].str[2:]
    # df['Local Arr Day'] = df.apply(lambda row: estimate_arrival_day(row['Local Dep Time Str'], row['Flying Time']), axis=1)
    # df['Local Arr Time Str'] = df['Local Arr Day'] + ' ' + df['Local Arr Time'].str[:2] + ':' + df['Local Arr Time'].str[2:]

    # Initialize result lists
    result_data = {
        'carrier_code': [], 'flight_number': [],
        'specific_aircraft_code':[],
        'origin': [], 'destination': [],
        # 'origin_iata': [], 'destination_iata': [], 'origin_icao': [], 'destination_icao': [],
        'org_lat': [], 'org_long': [], 'dest_lat': [], 'dest_long': [],
        'utc_dep_time': [], 'flying_time': [], 'aircraft_range': [], 'aircraft_speed': []
    } # 'utc_arr_time': [],

    # Process rows with tqdm for progress tracking
    for _, row in tqdm(df.iterrows(), total=len(df)):
        org = row['Dep Airport Code'].upper()
        dest = row['Arr Airport Code'].upper()

        # Lookup airport data
        org_data = find_data_coodinate(airport_dict, org)
        dest_data = find_data_coodinate(airport_dict, dest)
        
        if org_data is None or dest_data is None:
            print(f"Warning: Missing data for {org} or {dest}. Skipping this flight.")
            continue

        lat1, long1 = org_data['latitude'], org_data['longitude']
        lat2, long2 = dest_data['latitude'], dest_data['longitude']
        
        if any(pd.isna(x) for x in [lat1, long1, lat2, long2]):
            print(f"Warning: Missing coordinates for {org} or {dest}. Skipping this flight.")
            continue

        # Convert times to UTC
        utc_dep_time = local_timestr_to_utc((lat1, long1), row['Local Dep Time Str'], tz_cache)
        # utc_arr_time = local_timestr_to_utc((lat2, long2), row['Local Arr Time Str'], tz_cache)

        # Additional data
        flying_time = row['Flying Time']
        if utc_dep_time is None:
            continue
 
        # Append to results
        result_data['carrier_code'].append(row['Carrier Code'])
        result_data['flight_number'].append(row['Flight No'])
        result_data['specific_aircraft_code'].append(row['Specific Aircraft Code'])
        # result_data['origin_iata'].append(org)
        # result_data['destination_iata'].append(dest)
        # result_data['origin_icao'].append(org_data['icao'])
        # result_data['destination_icao'].append(dest_data['icao'])
        result_data['origin'].append(org_data['icao'])
        result_data['destination'].append(dest_data['icao'])
        result_data['org_lat'].append(lat1)
        result_data['org_long'].append(long1)
        result_data['dest_lat'].append(lat2)
        result_data['dest_long'].append(long2)
        result_data['utc_dep_time'].append(utc_dep_time)
        # result_data['utc_arr_time'].append(utc_arr_time)
        result_data['flying_time'].append(flying_time)
        result_data['aircraft_range'].append(float(row['Aircraft Range (NM)']) * NM_TO_MILE)
        result_data['aircraft_speed'].append(float(row['Aircraft Cruise Speed (MPH)']))

    # Create final DataFrame
    od_df = pd.DataFrame(result_data)
    od_df.drop_duplicates(inplace=True)
    od_df.to_csv('od_pair_oct_icao.csv', index=False)

def main():
    airport_file = "/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/deepflightplan/datasets/airports/iata-icao.csv"
    oag_file = "/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/oag/Sch_AS_202410_AL_JobId3339801.zip"
    sector_facilities_file = "/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/deepflightplan/datasets/facilities/sectors.geojson"

    extract_oag_file(oag_file, airport_file)

    df = pd.read_csv("/Users/danhle/Projects/ATMRI/AIATFM/data_preparation/deepflightplan/datasets/od_pair/od_pair_oct_icao.csv")
    sector_data = load_geojson(sector_facilities_file)
    crossing_flights = check_flights_crossing_sector(sector_data, df)
    if crossing_flights:
        crossing_flights = pd.DataFrame(crossing_flights)
        crossing_flights.drop_duplicates(inplace=True)
        crossing_flights.to_csv('od_oct_crossing_asean.csv', index=False)
    else:
        print("No flights crossing the FIR found.")


if __name__ == "__main__":
    main()