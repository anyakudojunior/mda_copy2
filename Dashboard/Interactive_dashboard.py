import streamlit as st
import pandas as pd
import lightgbm as lgb
import pydeck as pdk
import numpy as np
import os

st.set_page_config(layout="wide")
st.title("Cycling Traffic Predictor")

directory = os.path.dirname(os.path.abspath(__file__))
model = lgb.Booster(model_file=os.path.join(directory, "cycling_stations.txt"))

sites = pd.read_csv(os.path.join(directory, "sites.csv"), header=None)
sites.columns = [
    "index", "siteID", "long_site", "lat_site", "name", "operator",
    "road", "district", "municipality", "interval", "date"
]


@st.cache_data
def load_training_stats():
    df = pd.read_csv(os.path.join(directory, "merged_data.csv"))
    monthly_avgs = df.groupby("month")[["temp_dry_shelter_avg", "rain", "sun_duration", "hour", "weekend"]].mean()
    site_stats = df.groupby(["siteID", "month"])["count"].agg(["mean", "std"]).reset_index()
    site_stats.columns = ["siteID", "month", "count_mean", "count_std"]
    site_stats["count_std"] = site_stats["count_std"].fillna(1)
    return monthly_avgs, site_stats

monthly_avgs, site_stats = load_training_stats()

# SIDEBARS

month   = st.sidebar.slider("Month", 1, 12, 6)
hour    = st.sidebar.slider("Hour", 0, 23, 12)
weekend = st.sidebar.selectbox("Day type", [0, 1], format_func=lambda x: "Weekday" if x == 0 else "Weekend")
temp    = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 15.0, step=0.5)
rain    = rain = st.sidebar.selectbox(
    "Rain",
    [0, 1],
    format_func=lambda x: "No rain" if x == 0 else "Rain"
)
sun     = sun = st.sidebar.slider(
    "Sun duration (min)",
    0, 60, 0,
    step=1
)

# per-site per-month stats

sites_m = sites.merge(site_stats[site_stats["month"] == month], on="siteID", how="left")
sites_m["count_mean"] = sites_m["count_mean"].fillna(sites_m["count_mean"].mean())
sites_m["count_std"]  = sites_m["count_std"].fillna(sites_m["count_std"].mean())

# feature funciton

def build_features(df, hour, weekend, rain, temp, sun, month):
    f = df[["siteID"]].copy()
    f["hour"]                 = hour
    f["weekend"]              = weekend
    f["rain"]                 = rain
    f["temp_dry_shelter_avg"] = temp
    f["sun_duration"]         = sun
    f["spring"] = int(month in [3, 4, 5])
    f["summer"] = int(month in [6, 7, 8])
    f["autumn"] = int(month in [9, 10, 11])
    f["winter"] = int(month in [12, 1, 2])
    expected = model.feature_name()
    for col in expected:
        if col not in f.columns:
            f[col] = 0
    return f[expected].to_numpy(dtype=float)

# PREDICTION

avgs     = monthly_avgs.loc[month]
baseline = model.predict(build_features(sites_m,
    hour=int(round(avgs["hour"])),
    weekend=int(round(avgs["weekend"])),
    rain=avgs["rain"],
    temp=avgs["temp_dry_shelter_avg"],
    sun=avgs["sun_duration"],
    month=month
))
scenario = model.predict(build_features(sites_m, hour, weekend, rain, temp, sun, month))

sites_m["baseline"] = np.round(baseline, 1)
sites_m["scenario"] = np.round(scenario, 1)
sites_m["delta"]    = np.round(scenario - baseline, 1)

# colour classification

sites_m["color"] = [
    [200, 30, 30] if row["scenario"] > row["count_mean"] + row["count_std"] else
    [0,  180,  0] if row["scenario"] < row["count_mean"] - row["count_std"] else
    [30, 100, 200]
    for _, row in sites_m.iterrows()
]

# map

st.write("🔴 Unusually high for this site this month  🟢 Unusually low  🔵 Normal")

st.pydeck_chart(pdk.Deck(
    layers=[pdk.Layer(
        "ScatterplotLayer",
        data=sites_m,
        get_position="[long_site, lat_site]",
        get_fill_color="color",
        get_radius=120,
        pickable=True,
    )],
    initial_view_state=pdk.ViewState(
        latitude=sites_m["lat_site"].mean(),
        longitude=sites_m["long_site"].mean(),
        zoom=8,
    ),
    tooltip={"text": "Site: {siteID}\nBaseline: {baseline}\nScenario: {scenario}\nΔ {delta}"},
))

# output table (can be deleted after)

st.dataframe(
    sites_m[["siteID", "name", "municipality", "baseline", "scenario", "delta"]]
    .sort_values("delta", ascending=False)
    .reset_index(drop=True)
)