# loading packages
import pandas as pd
import glob

### part 1: loading and formating the count data sets
# save all 12 data set names in one vector and create column labels
files = glob.glob("Datasets for features\data_raw\data-2025-*.csv")
col_names = ["siteID", "direction", "type", "start_date", "end_date", "count"]

# read in all data sets + bind the rows to create one big data set
data = pd.concat([pd.read_csv(f, header=None, names=col_names) for f in files], ignore_index=True)
# checking if data sets look correct
print(data.shape)

# preparing the columns: filter type to bikes only
data = data[data["type"] == "FIETSERS"]

# instead of start and end date, create single columns for date, hour etc.
data["start_date"] = pd.to_datetime(data["start_date"])
data["date"] = data["start_date"].dt.date # for grouping per date to sum up
data["month"] = data["start_date"].dt.month # temp: for getting season
data["hour"] = data["start_date"].dt.hour
data['day_of_week'] = data['start_date'].dt.dayofweek # temp: for getting weekend
data['weekend'] = data['day_of_week'].isin([5, 6]).astype(int) # if weekend:1, if weekday:0


# delete unnecessary columns
data = data.drop(columns=["direction", "type", "end_date", "day_of_week"])

# check structure
print(data)


# create one count number per hour
# for the same siteID, date and hour - add up all the counts (month and weekend kept for analysis)
data_sum = data.groupby(["siteID", "date", "hour", "month", "weekend"], as_index=False).agg({
    "count": "sum", "start_date": "first"})

print(data_sum)

# first create one season variable, then add dummy variable per season instead
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

data_sum['season'] = data_sum['month'].apply(get_season)

print(data_sum)


### part 2: loading and merging with the sites data, including public transport
sites = pd.read_csv("Datasets for features\sites_x_pub_transport_test.csv")

# merge with count data
data_new = data_sum.merge(sites, on="siteID", how="left")

print(data_new.iloc[12])

data_new.to_csv("Datasets for features\count_data_test.csv", index=False)
