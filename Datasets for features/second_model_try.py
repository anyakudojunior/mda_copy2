import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_poisson_deviance
import matplotlib.pyplot as plt


data_train = pd.read_csv("merged_data.csv")
data_test = pd.read_csv("merged_data_test.csv")


# create cyclic variables for both train and test
for df in [data_train, data_test]:
    df["hr_sin"] = np.sin(df["hour"] * (2 * np.pi / 24))
    df["hr_cos"] = np.cos(df["hour"] * (2 * np.pi / 24))

    df["mnth_sin"] = np.sin((df["month"] - 1) * (2 * np.pi / 12))
    df["mnth_cos"] = np.cos((df["month"] - 1) * (2 * np.pi / 12))


# define target variable
y_train = data_train["count"]
y_test = data_test["count"]


# define feature variables
# drop count = target
# drop start_date/date = string date columns
# drop season = text column
# drop hour/month = replaced by cyclic variables

X_train = data_train.drop(columns=[
    "count", "start_date", "date", 
    "hour", "month", "long", "lat"
], errors="ignore")

X_test = data_test.drop(columns=[
    "count", "start_date", "date",
    "hour", "month", "long", "lat"
], errors="ignore")


# define/prepare categorical variables
cat_feature = ["siteID", "municipality"]

X_train["siteID"] = X_train["siteID"].astype("category")
X_test["siteID"] = X_test["siteID"].astype("category")

X_train["municipality"] = X_train["municipality"].astype("category")
X_test["municipality"] = X_test["municipality"].astype("category")


# create dataset in lgb format
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_feature
)

test_data = lgb.Dataset(
    X_test,
    label=y_test,
    categorical_feature=cat_feature
)


# set lightgbm parameters
params = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,

}

# Stop training when the model no longer improves on test data. 
# The best model is around 127 
model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[test_data],
    valid_names=["test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

pred = model.predict(X_test)

pois_dev = mean_poisson_deviance(y_test, pred)
print("Poisson deviance:", pois_dev)


# Comparing with the basic model that does not count ohter variables (Always predic average count)
from sklearn.metrics import mean_poisson_deviance
import numpy as np

baseline_pred = np.repeat(y_train.mean(), len(y_test))

baseline_dev = mean_poisson_deviance(y_test, baseline_pred)

print("Baseline Poisson deviance:", baseline_dev)
print("LightGBM Poisson deviance:", pois_dev)

