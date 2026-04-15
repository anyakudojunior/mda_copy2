
import pandas as pd
import numpy as np
import glob

# loading the files

files = glob.glob("Datasets for features/data-2024-*-fixed.csv")

df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

meteo = pd.read_csv("Datasets for features/meteo 2024.csv")
sites = pd.read_csv("Datasets for features/sites.csv")
stops = pd.read_excel("Datasets for features/De lijn stops.xlsx")

print(meteo.columns)
print(meteo.columns)
print(meteo.head())

# Clean columns
meteo['date 1'] = meteo['date 1'].astype(str).str.strip()
meteo['hour 1'] = meteo['hour 1'].astype(str).str.strip()

# Create datetime
# Combining the date and hour colummn (ex. 07/12/2024 13:00)
meteo['datetime'] = pd.to_datetime(
    meteo['date 1'] + ' ' + meteo['hour 1'],
    dayfirst=True,
    errors='coerce'
)

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

# total coloumn : date, hour, temp_dry_shelter, rain, sun_hours
print(len(meteo_features))