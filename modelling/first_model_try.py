import pandas as pd
import lightgbm as lgb
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_poisson_deviance
import matplotlib.pyplot as plt


# load the data and define as test/train data
data_train = pd.read_csv("Datasets for features\merged_data.csv")
data_test = pd.read_csv("Datasets for features\merged_data_test.csv")

y_train = data_train["count"]
X_train = data_train.drop(columns=["count", "start_date", "date"]) # date string --> can't be analysed

y_test = data_test["count"]
X_test = data_test.drop(columns=["count", "start_date", "date"])



# define/prepare categorical variables
cat_feature = ["siteID", "municipality", "hour", "season"]

X_train["siteID"] = X_train["siteID"].astype("category")
X_test["siteID"] = X_test["siteID"].astype("category")
X_train["municipality"] = X_train["municipality"].astype("category")
X_test["municipality"] = X_test["municipality"].astype("category")
X_train["hour"] = X_train["hour"].astype("category")
X_test["hour"] = X_test["hour"].astype("category")
X_train["season"] = X_train["season"].astype("category")
X_test["season"] = X_test["season"].astype("category")


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
    "metric": "poisson", # "rmse" in try one
    "learning_rate": 0.05,
    "num_leaves": 64, # default:31
    "max_depth": -1, # default; increase if we want to limit over-fitting
    "feature_fraction": 0.8, # for speeding up training+limiting over-fitting
    #"bagging_fraction": 0.8,
    #"bagging_freq": 5,
    # "verbosity": -1 ## could use to hide messages during training
}


model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    #valid_sets=[test_data],
    #early_stopping_rounds=50
)

pred = model.predict(X_test)

# rmse = mean_squared_error(y_test, pred)
# print("RMSE:", rmse)
# # RMSE: 666.9139874297595
pois_dev = mean_poisson_deviance(y_test, pred)
print(pois_dev)
# 13.120399211179524

# old: when season had 4 dummies+with the stations containing missings
# 16.65225535215885


# feature importance plot
lgb.plot_importance(model)
plt.savefig("feature_importance_try.jpg", dpi=300, bbox_inches="tight")
plt.show()
