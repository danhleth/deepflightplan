import pandas as pd

save_dir ="/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/groundtruth"
prefix_root = "/Users/danhleth/Projects/AIATFM/deepflightplan"
airport_df = pd.read_csv(f"/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/airports/iata-icao.csv")
df = "/Users/danhleth/Projects/AIATFM/deepflightplan/datasets/groundtruth/oct_full.csv"
df = pd.read_csv(df)
df.rename(columns={'Callsign': 'callsign', 
                   'ADEP': 'origin', 
                   'ADES': 'destination', 
                   'Mach Number': 'Mach Number', 
                   'ICAO Route': 'route'}, inplace=True)

MACH_TO_MPH_RATE = 767.269148


df['origin'] = df['origin'].str.upper().str.strip()
df['destination'] = df['destination'].str.upper().str.strip()
# airport_df['icao'] = airport_df['icao'].str.upper().str.strip()
# airport_df['iata'] = airport_df['iata'].str.upper().str.strip()

# Create ICAO to IATA mapping
# icao_to_iata = airport_df.set_index('icao')['iata'].to_dict()
# df['origin'] = df['origin'].map(icao_to_iata)
# df['destination'] = df['destination'].map(icao_to_iata)

df['Mach Number'] = df['Mach Number'].apply(
    lambda x: float(x[1:]) / 100 * MACH_TO_MPH_RATE if isinstance(x, str) and x.startswith('M') else 0.0
)
df.rename(columns={'Mach Number': 'aircraft_speed_mph'}, inplace=True)


# tmp = df['Mach Number'].apply(
#     lambda x: round((float(x.replace('M', '')) / 100) * 660, 1)
# )

# df.insert(10, 'aircraft_speed_mph', tmp.to_list())

df = df[['callsign', 'origin', 'destination', 'aircraft_speed_mph', 'route']]
df = df.drop_duplicates()
df.to_csv(save_dir + "/oct_processed_full.csv", index=False)