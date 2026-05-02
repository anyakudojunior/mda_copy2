import pandas as pd

# load data
meteo = pd.read_csv(r"data_raw\aws_1hour_2024.csv")


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
    meteo.groupby('datetime_hour', as_index=False)
    .agg({
        'temp_dry_shelter_avg': 'mean',
        'precip_quantity': 'mean',
        'sun_duration': 'sum'
    })
)

# Extract date and hour
meteo_hourly['date'] = meteo_hourly['datetime_hour'].dt.date
meteo_hourly['hour'] = meteo_hourly['datetime_hour'].dt.hour

# Rain binary 
# Depends on the 0.8 value
meteo_hourly['rain'] = (meteo_hourly['precip_quantity'] >= 0.8).astype(int)

# Daily sunshine: total minutes per day -> hours
# This adds up all sunshine minutes across whole day 

daily_sun = (
    meteo_hourly.groupby('date', as_index=False)['sun_duration']
    .sum()
)

daily_sun['sun_hours'] = daily_sun['sun_duration'] / 60
##TODO: sometimes more than 24 hours?!
# probably becaosue in lines 26-33 you don't aggregate by location and sum up for all locations
print(daily_sun['sun_hours'].describe())


# Merge daily sun back to hourly weather
meteo_features = meteo_hourly.merge(
    daily_sun[['date', 'sun_hours']],
    on='date',
    how='left'
)

# Keep only needed columns
meteo_features = meteo_features[
    ['date', 'hour', 'temp_dry_shelter_avg', 'rain', 'sun_hours']
]

print(meteo_features.head())
print(meteo_features.isna().sum())

# total columns: date, hour, temp_dry_shelter, rain, sun_hours
print(len(meteo_features))

##TODO missing the code for merging with the sites/counts data
# --> probably merging by haversine distance of the coordinates