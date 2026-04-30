# loading packages and setting working directory
import pandas as pd
import glob
import os
os.chdir(r"C:\Users\Think\Documents\GitHub\MDA_course\data")


# save all 12 data sets names in one vector and create column labels
files = glob.glob("data-2024-*.csv")
col_names = ["siteID", "direction", "type", "start_date", "end_date", "count"]

# read in all data sets + bind the rows to create one big data set
data = pd.concat([pd.read_csv(f, header=None, names=col_names) for f in files], ignore_index=True)
# checking if data sets look correct
data.shape
data.head()

# preparing the columns, filter type to bikes only
data = data[data["type"] == "FIETSERS"]

# instead of start and end date, create one column for start year, month, day and time (especially hour)
data["start_date"] = pd.to_datetime(data["start_date"])
data["year"] = data["start_date"].dt.year
data["month"] = data["start_date"].dt.month
data["day"] = data["start_date"].dt.day
#data["time"] = data["start_date"].dt.time # do we need this?
data["hour"] = data["start_date"].dt.hour

# delete unnecessary columns (do we need end_date?)
data = data.drop(columns=["direction", "type", "start_date", "end_date"])

# check structure
data


# create one count number per hour
# for the same siteID, year, month, day, hour - add up all the counts
data_sum = data.groupby(["siteID", "year", "month", "day", "hour"],as_index=False)["count"].sum()
data_sum

data_sum.to_csv("count_data.csv", index=False)