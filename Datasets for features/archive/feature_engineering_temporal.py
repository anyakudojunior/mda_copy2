
import pandas as pd
import numpy as np
import glob
import os #add darya
os.chdir(r"C:\Users\Think\Documents\GitHub\MDA_course\Datasets for features") #add darya

# loading the files

files = glob.glob("archive\data-2024-*-fixed.csv") #add darya
#files = glob.glob("Datasets for features/data-2024-*-fixed.csv")

df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

meteo = pd.read_csv("Datasets for features/meteo 2024.csv")
sites = pd.read_csv("Datasets for features/sites.csv")
stops = pd.read_excel("Datasets for features/De lijn stops.xlsx")


print(df.columns)
print(df.head(10))


# Rename columns for clearlity 
df = df.rename(columns={
    'counting_station': 'station_id',
    'date 1': 'date',
    'hour 1': 'time',
    'counts': 'count'
})

# Clean text columns
# Raw datasets are required to be organized
df['IN_OUT'] = df['IN_OUT'].astype(str).str.strip().str.upper()
df['date'] = df['date'].astype(str).str.strip()
df['time'] = df['time'].astype(str).str.strip()

# Keep only needed columns
df = df[['station_id', 'IN_OUT', 'date', 'time', 'count']]

# Create full timestamp from date + time 
# day/month/year
df['datetime'] = pd.to_datetime(
    df['date'] + ' ' + df['time'],
    dayfirst=True,
    errors='coerce'
)

# Check parsing problems
bad_rows = df[df['datetime'].isna()][['date', 'time']].drop_duplicates()
print("Bad date/time rows:")
print(bad_rows.head(20))
print("Number of bad date/time rows:", len(bad_rows))

# Drop rows that could not be parsed
# some row where datetime could not be parsed cannot be used for aggregation
df = df.dropna(subset=['datetime'])

# Floor to the hour
df['datetime_hour'] = df['datetime'].dt.floor('H') #add darya: should be h?

# Aggregate 15-minute counts to hourly counts per station and direction
df_hourly = (
    df.groupby(['station_id', 'datetime_hour', 'IN_OUT'], as_index=False)['count']
      .sum()
)

# Pivot IN / OUT into columns
# It is needed to calculate total cyclists since In, out are combined in the IN_OUT
df_hourly = (
    df_hourly.pivot_table(
        index=['station_id', 'datetime_hour'],
        columns='IN_OUT',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    .reset_index()
)

# Make sure both columns exist even if one direction is missing somewhere
if 'IN' not in df_hourly.columns:
    df_hourly['IN'] = 0
if 'OUT' not in df_hourly.columns:
    df_hourly['OUT'] = 0

# Target variable
df_hourly['cyclists'] = df_hourly['IN'] + df_hourly['OUT']

# Temporal features
df_hourly['date'] = df_hourly['datetime_hour'].dt.date
df_hourly['hour'] = df_hourly['datetime_hour'].dt.hour
df_hourly['day_of_week'] = df_hourly['datetime_hour'].dt.dayofweek
df_hourly['weekday_weekend'] = df_hourly['day_of_week'].isin([5, 6]).astype(int)
df_hourly['month'] = df_hourly['datetime_hour'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

df_hourly['season'] = df_hourly['month'].apply(get_season)

# debugging
print("Columns:")
print(df_hourly.columns)
