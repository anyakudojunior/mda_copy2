import pandas as pd
from scipy.spatial import cKDTree


meteo = pd.read_csv("Datasets for features\meteo_features.csv")
site = pd.read_csv("Datasets for features\count_data.csv")

#dates and hours for meteo and site
meteo["date"] = pd.to_datetime(meteo["date"], format='mixed', dayfirst=True)
site["date"] = pd.to_datetime(site["date"], format='mixed', dayfirst=True)
meteo["hour"] = meteo["hour"].astype(int)
site["hour"] = site["hour"].astype(int)

#give each weather station simple id number
stations = meteo[["lat", "long"]].drop_duplicates().reset_index(drop=True)
stations["station_id"] = stations.index

#take the 3 nearest weather stations for each cycling site
tree = cKDTree(stations[["lat", "long"]].to_numpy())
d, nearbystations = tree.query(site[["lat", "long"]].to_numpy(), k=3)

#store them in seperate columns
site["station_id_1"] = nearbystations[:, 0]  # closest
site["station_id_2"] = nearbystations[:, 1]  # 2nd closest
site["station_id_3"] = nearbystations[:, 2]  # 3rd closest

# attach station_id to meteo rows
meteo = meteo.merge(stations, on=["lat", "long"], how="left")

weather_variables = ["temp_dry_shelter_avg", "rain", "sun_duration"]

#merge using closest station first
final_merged = site.merge(
    meteo.rename(columns={"station_id": "station_id_1"}),
    on=["station_id_1", "date", "hour"],
    how="left",
    suffixes=("_site", "_station")
)

#for rows still missing, try 2nd closest station
missing = final_merged[weather_variables].isna().any(axis=1)
fill = final_merged[missing][["station_id_2", "date", "hour"]].merge(
    meteo.rename(columns={"station_id": "station_id_2"}),
    on=["station_id_2", "date", "hour"],
    how="left"
)
final_merged.loc[missing, weather_variables] = fill[weather_variables].values

#for rows still missing, try 3rd closest station
missing = final_merged[weather_variables].isna().any(axis=1)
fill = final_merged[missing][["station_id_3", "date", "hour"]].merge(
    meteo.rename(columns={"station_id": "station_id_3"}),
    on=["station_id_3", "date", "hour"],
    how="left"
)
final_merged.loc[missing, weather_variables] = fill[weather_variables].values

# delete columns that are not needed for analysis
final_merged = final_merged.drop(columns=["station_id_1", "station_id_2",
                                          "station_id_3", "lat_station", "long_station"])

# checking for missing data in siteID
print(final_merged["siteID"].value_counts())
## IDs with incomplete data subsets
print(final_merged["siteID"].value_counts()[final_merged["siteID"].value_counts() != 8784].to_string())
## missing completely
print(set(range(1,145)) - set(final_merged["siteID"]))

# delete the rows belonging to sites 80, 12, 123, 144
final_merged = final_merged[~final_merged["siteID"].isin([80, 12, 123, 144])]


print(final_merged.shape)
print(final_merged[["date", "hour", "temp_dry_shelter_avg", "rain", "sun_duration","month"]].head(10))

#save as a csv
final_merged.to_csv("Datasets for features\merged_data.csv", index=False)