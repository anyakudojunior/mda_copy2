param_grid = {
    "min_data_in_leaf": [30, 100, 300]
}

fixed_params = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.01,
    "num_leaves": 128,
    "feature_fraction": 1.0,
    "max_depth": -1,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
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
        num_boost_round=500,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    p = m.predict(X_test)

    test_dev = mean_poisson_deviance(y_test, p)

    train_dev = mean_poisson_deviance(
        y_train,
        m.predict(X_train)
    )

    gap = test_dev - train_dev

    print(
        f"Params: {grid_params} "
        f"-> Train: {train_dev:.4f} | "
        f"Test: {test_dev:.4f} | "
        f"Gap: {gap:.4f}"
    )

    if test_dev < best_dev:
        best_dev = test_dev
        best_params = grid_params
        best_model = m

print("Best params:", best_params)
print("Best Poisson deviance:", best_dev)