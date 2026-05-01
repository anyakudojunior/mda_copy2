import pandas as pd

sites = pd.read_csv("sites.csv")

# storing what we need in a new dataframe to later merge with the hourly data
sites = sites[["counting_station", "latitude", "longitude", "municipality"]]

# merge with data
finaldataset = data.merge(sites, on"counting_station", how="left")
