import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_poisson_deviance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap

data_train = pd.read_csv(r"Datasets for features\merged_data.csv")
data_test = pd.read_csv(r"Datasets for features\merged_data_test.csv")

for df in [data_train, data_test]:
    df["hr_sin"] = np.sin(df["hour"] * (2 * np.pi / 24))
    df["hr_cos"] = np.cos(df["hour"] * (2 * np.pi / 24))
    df["mnth_sin"] = np.sin((df["month"] - 1) * (2 * np.pi / 12))
    df["mnth_cos"] = np.cos((df["month"] - 1) * (2 * np.pi / 12))

y_train = data_train["count"]
y_test = data_test["count"]

X_train = data_train.drop(columns=["count", "start_date", "date", "hour", "month", "long", "lat"], errors="ignore")
X_test = data_test.drop(columns=["count", "start_date", "date", "hour", "month", "long", "lat"], errors="ignore")

cat_feature = ["siteID", "municipality", "season"]
for df in [X_train, X_test]:
    df["siteID"] = df["siteID"].astype("category")
    df["municipality"] = df["municipality"].astype("category")
    df["season"] = df["season"].astype("category")

# split 2025 into 20% validation and 80% test
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.8, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature)
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feature)

params = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.01,
    "num_leaves": 128,
    "feature_fraction": 1.0,
    "max_depth": -1,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 100,
    "seed": 42,
    "verbosity": -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=600,
    valid_sets=[val_data],
    valid_names=["val"],
    callbacks=[lgb.log_evaluation(period=100)]
)

pred = model.predict(X_test)
pois_dev = mean_poisson_deviance(y_test, pred)
train_dev = mean_poisson_deviance(y_train, model.predict(X_train))

print("Train Poisson deviance:", train_dev)
print("Test Poisson deviance:", pois_dev)
print("Gap:", pois_dev - train_dev)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.jpg", dpi=300, bbox_inches="tight")
plt.show()

lgb.plot_importance(model)
plt.savefig("feature_importance_try_cyclic.jpg", dpi=300, bbox_inches="tight")
plt.show()

baseline_pred = np.repeat(y_train.mean(), len(y_test))
baseline_dev = mean_poisson_deviance(y_test, baseline_pred)
print("Baseline Poisson deviance:", baseline_dev)
print("LightGBM Poisson deviance:", pois_dev)