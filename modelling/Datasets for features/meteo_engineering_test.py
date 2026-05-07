import pandas as pd

# load data
meteo = pd.read_csv(r"Datasets for features\data_raw\aws_1hour_2025.csv")


# format the coordinates and the date into format needed
meteo = meteo[["the_geom", "timestamp", "precip_quantity", "temp_dry_shelter_avg", "sun_duration"]]
meteo[["lat","long"]] = (meteo["the_geom"].str.replace("POINT", "", regex=False)
                                       .str.replace("(", "", regex=False).str.replace(")", "", regex=False)
                                       .str.strip().str.split(expand=True).astype(float))
meteo["datetime"] = pd.to_datetime(meteo["timestamp"])

print(meteo.columns)
print(meteo.head())

# Drop bad rows
meteo = meteo.dropna(subset=['datetime'])

# Floor to hour
# It rounds the datetime to the nearest hour (weather data is at the hourly level)
meteo['datetime_hour'] = meteo['datetime'].dt.floor('h')

# Aggregate to hourly level
meteo_hourly = (
    meteo.groupby(['datetime_hour','lat','long'], as_index=False)
    .agg({
        'temp_dry_shelter_avg': 'mean',
        'precip_quantity': 'mean',
        'sun_duration': 'mean'
    })
)

# Extract date and hour
meteo_hourly['date'] = meteo_hourly['datetime_hour'].dt.date
meteo_hourly['hour'] = meteo_hourly['datetime_hour'].dt.hour

# Rain binary
# Depends on the 0.8 value
meteo_hourly['rain'] = (meteo_hourly['precip_quantity'] >= 0.8).astype(int)


# Keep only needed columns
meteo_features = meteo_hourly[
    ['date', 'hour', 'temp_dry_shelter_avg', 'rain', 'sun_duration','lat', 'long']
]

print("Final")
print(meteo_features.head())
print(meteo_features.isna().sum())

# total columns: date, hour, temp_dry_shelter, rain, sun_hours
print(len(meteo_features))


print(meteo_features.describe())

meteo_features.to_csv("Datasets for features\meteo_features_test.csv", index=False)