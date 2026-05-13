import pandas as pd

df = pd.read_csv("merged_data.csv")

print(df[[
    "count",
    "hour",
    "weekend",
    "rain",
    "temp_dry_shelter_avg",
    "sun_duration"
]].describe())