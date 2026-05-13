import streamlit as st
import pandas as pd
import lightgbm as lgb
import pydeck as pdk
import numpy as np
import os

st.set_page_config(layout="wide")
st.title("Cycling Traffic Predictor")

directory = os.path.dirname(os.path.abspath(__file__))
model = lgb.Booster(model_file=os.path.join(directory, "cycling_model.txt"))

print(model.pandas_categorical)
pandas_cat = model.pandas_categorical

sites = pd.read_csv(os.path.join(directory, "sites.csv"), header=None)
sites.columns = [
    "index", "siteID", "long_site", "lat_site", "name", "operator",
    "road", "district", "municipality", "interval", "date"
]




@st.cache_data
def load_training_stats():
    df = pd.read_csv(os.path.join(directory, "merged_data.csv"))

    site_stats = (
        df.groupby(["siteID", "month"])["count"]
        .agg(["mean", "std"])
        .reset_index()
    )
    site_stats.columns = ["siteID", "month", "count_mean", "count_std"]
    site_stats["count_std"] = site_stats["count_std"].fillna(1)

    weather_stats = (
        df.groupby("month")[["temp_dry_shelter_avg", "sun_duration"]]
        .mean()
        .reset_index()
    )

    season_lookup = (
        df.groupby("month")["season"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    return site_stats, weather_stats, season_lookup


site_stats, weather_stats, season_lookup = load_training_stats()

# sidebars, dropdown menus
month   = st.sidebar.slider("Month", 1, 12, 6)
hour    = st.sidebar.slider("Hour", 0, 23, 12)
weekend = st.sidebar.selectbox("Day type", [0, 1],
              format_func=lambda x: "Weekday" if x == 0 else "Weekend")
temp    = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 15.0, step=0.5)
rain    = st.sidebar.selectbox("Rain", [0, 1],
              format_func=lambda x: "No rain" if x == 0 else "Rain")
sun     = st.sidebar.slider("Sun duration (min)", 0, 60, 0, step=1)

# merge
sites_m = sites.merge(
    site_stats[site_stats["month"] == month],
    on="siteID", how="left"
)
sites_m["count_mean"] = sites_m["count_mean"].fillna(0)
sites_m["count_std"]  = sites_m["count_std"].fillna(1)

# constructing the features
def build_features(df_sites, hour, weekend, rain, temp, sun, month):
    f = df_sites[["siteID", "municipality", "long_site", "lat_site"]].copy()

    season_val = season_lookup.loc[
        season_lookup["month"] == month, "season"
    ].iloc[0]
    f["season"] = season_val

    # Encode categoricals using exact levels stored in the model
    f["siteID"] = pd.Categorical(
        f["siteID"], categories=pandas_cat[0]
    ).codes.astype(int)
    f["season"] = pd.Categorical(
        f["season"], categories=pandas_cat[1]
    ).codes.astype(int)
    f["municipality"] = pd.Categorical(
        f["municipality"], categories=pandas_cat[2]
    ).codes.astype(int)

    # Cyclic time features
    f["hr_sin"]   = np.sin(hour * (2 * np.pi / 24))
    f["hr_cos"]   = np.cos(hour * (2 * np.pi / 24))
    f["mnth_sin"] = np.sin((month - 1) * (2 * np.pi / 12))
    f["mnth_cos"] = np.cos((month - 1) * (2 * np.pi / 12))

    # Scalar features
    f["weekend"]              = int(weekend)
    f["rain"]                 = int(rain)
    f["temp_dry_shelter_avg"] = float(temp)
    f["sun_duration"]         = float(sun)
    f["outlier_flag"]         = 0

    # fill any missing with 0
    expected = model.feature_name()
    for col in expected:
        if col not in f.columns:
            f[col] = 0

    return f[expected].to_numpy(dtype=float)


# baseline is average weather, hour=12, weekday, no rain ───
weather_row = weather_stats[weather_stats["month"] == month].iloc[0]

st.write("baseline temp used:", float(weather_row["temp_dry_shelter_avg"]))
st.write("baseline hour used: 12")

baseline_features = build_features(
    sites_m,
    hour=12,
    weekend=0,
    rain=0,
    temp=float(weather_row["temp_dry_shelter_avg"]),
    sun=float(weather_row["sun_duration"]),
    month=month,
)
baseline = model.predict(baseline_features)

# Scenario:
scenario_features = build_features(
    sites_m,
    hour=hour,
    weekend=weekend,
    rain=rain,
    temp=temp,
    sun=sun,
    month=month,
)
scenario = model.predict(scenario_features)

sites_m["baseline"] = np.round(baseline, 1)
sites_m["scenario"] = np.round(scenario, 1)
sites_m["delta"]    = np.round(scenario - baseline, 1)

# red if delta > 2, green if delta < -2, blue if normal
sites_m["color"] = [
    [200, 30,  30] if row["delta"] >  2 else
    [0,  180,   0] if row["delta"] < -2 else
    [30, 100, 200]
    for _, row in sites_m.iterrows()
]

# map
st.write("🔴 Much higher than baseline  🟢 Much lower than baseline  🔵 Similar to baseline")

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
    tooltip={
        "text": "Site: {siteID}\nBaseline: {baseline}\nScenario: {scenario}\nΔ {delta}"
    },
))

st.dataframe(
    sites_m[["siteID", "name", "municipality", "baseline", "scenario", "delta"]]
    .sort_values("delta", ascending=False)
    .reset_index(drop=True)
)