import pandas as pd
import numpy as np

# loading the files
col_names_sites = ["siteID", "site_nr", "long", "lat", "naam", "domein", "wegnr",
                   "district", "gemeente", "interval", "datum_van"]
sites = pd.read_csv("Datasets for features\data_raw\sites.csv", header=None, names=col_names_sites)
stops = pd.read_csv("Datasets for features\data_raw\de_lijn_stops_2024_07_01.txt")

print(sites.columns)
print(sites.head(20))
print(stops.columns)
print(stops.head())
sites_small = sites[['siteID', 'lat', 'long']].copy()
stops_small = stops[['stop_lat', 'stop_lon']].copy()

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
    pairs['lat'],
    pairs['long'],
    pairs['stop_lat'],
    pairs['stop_lon']
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
    pairs_1km.groupby('siteID', as_index=False)
    .agg(
        accessibility_index=('weight', 'sum'),
        n_nearby_stops=('nearby_stop', 'sum')
    )
)

# Make sure ALL stations appear, even if they have no stop within 1 km
accessibility = sites_small[['siteID']].drop_duplicates().merge(
    accessibility,
    on='siteID',
    how='left'
)

# Replace missing values with 0
accessibility['accessibility_index'] = accessibility['accessibility_index'].fillna(0)
accessibility['n_nearby_stops'] = accessibility['n_nearby_stops'].fillna(0)

print("Accessibility sample:")
print(accessibility.head())
print(accessibility.describe())

sites_new = sites.merge(accessibility, on="siteID", how="left")
sites_new = sites_new[["siteID", "long", "lat", "gemeente", "accessibility_index", "n_nearby_stops"]]
sites_new = sites_new.rename(columns={"gemeente": "municipality"})

print(sites_new.columns)
# siteID', 'long', 'lat', 'municipality', 'accessibility_index', 'n_nearby_stops'

# Save sites_new as the new sites csv, to be merged with the count data
sites_new.to_csv("Datasets for features\sites_x_pub_transport.csv", index=False)