import pandas as pd
import numpy as np
import glob

# loading the files

files = glob.glob("Datasets for features/data-2024-*-fixed.csv")
meteo = pd.read_csv("Datasets for features/meteo 2024.csv")
sites = pd.read_csv("Datasets for features/sites.csv")
stops = pd.read_excel("Datasets for features/De lijn stops.xlsx")

print(sites.columns)
print(sites.head(20))
print(stops.columns)
print(stops.columns)
print(stops.head())
sites_small = sites[['counting_station', 'lat', 'long']].copy()
stops_small = stops[['lat', 'long']].copy()

# remove duplicate stop coordinates (make unique values)
stops_small = stops_small.drop_duplicates().reset_index(drop=True)

print("Sites sample:")
print(sites_small.head())

print("Stops sample:")
print(stops_small.head())


# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# Cross join: every counting station with every De Lijn stop
sites_small['key'] = 1
stops_small['key'] = 1

pairs = sites_small.merge(
    stops_small,
    on='key',
    suffixes=('_station', '_stop')
).drop(columns='key')

print("Number of station-stop pairs:", len(pairs))


# Compute distance in meters
pairs['distance'] = haversine(
    pairs['lat_station'],
    pairs['long_station'],
    pairs['lat_stop'],
    pairs['long_stop']
)

print("Distance summary:")
print(pairs['distance'].describe())


# Keep only stops within 1 km
pairs_1km = pairs[pairs['distance'] <= 1000].copy()

print("Pairs within 1 km:", len(pairs_1km))


# Avoid division by zero if a stop is exactly at the station location
pairs_1km['distance'] = pairs_1km['distance'].replace(0, 1)

# Inverse-distance weight
pairs_1km['weight'] = 1 / pairs_1km['distance']

# Count number of nearby stops
pairs_1km['nearby_stop'] = 1


# Accessibility index per station
accessibility = (
    pairs_1km.groupby('counting_station', as_index=False)
    .agg(
        accessibility_index=('weight', 'sum'),
        n_nearby_stops=('nearby_stop', 'sum')
    )
)

# Make sure ALL stations appear, even if they have no stop within 1 km
accessibility = sites_small[['counting_station']].drop_duplicates().merge(
    accessibility,
    on='counting_station',
    how='left'
)

# Replace missing values with 0
accessibility['accessibility_index'] = accessibility['accessibility_index'].fillna(0)
accessibility['n_nearby_stops'] = accessibility['n_nearby_stops'].fillna(0)

print("Accessibility sample:")
print(accessibility.head())
print(accessibility.describe())