import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_poisson_deviance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
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

param_grid = {
    "num_leaves": [32, 64, 128],
    "learning_rate": [0.01, 0.05, 0.1],
    "feature_fraction": [0.6, 0.8, 1.0],
}

fixed_params = {
    "objective": "poisson",
    "metric": "poisson",
    "max_depth": -1,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 30,
    "seed": 42,
    "verbosity": -1
}

best_dev = float("inf")
best_params = None
best_model = None

for grid_params in ParameterGrid(param_grid):
    params = {**fixed_params, **grid_params}
    m = lgb.train(
        params,
        train_data,
        num_boost_round=600,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    p = m.predict(X_test)
    test_dev = mean_poisson_deviance(y_test, p)
    train_dev = mean_poisson_deviance(y_train, m.predict(X_train))
    gap = test_dev - train_dev
    print(f"Params: {grid_params} -> Train: {train_dev:.4f} | Test: {test_dev:.4f} | Gap: {gap:.4f}")
    if test_dev < best_dev:
        best_dev = test_dev
        best_params = grid_params
        best_model = m

print("Best params:", best_params)
print("Best Poisson deviance:", best_dev)

pred = best_model.predict(X_test)
pois_dev = mean_poisson_deviance(y_test, pred)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.jpg", dpi=300, bbox_inches="tight")
plt.show()

lgb.plot_importance(best_model)
plt.savefig("feature_importance_try_cyclic.jpg", dpi=300, bbox_inches="tight")
plt.show()

baseline_pred = np.repeat(y_train.mean(), len(y_test))
baseline_dev = mean_poisson_deviance(y_test, baseline_pred)
print("Baseline Poisson deviance:", baseline_dev)
print("LightGBM Poisson deviance:", pois_dev)