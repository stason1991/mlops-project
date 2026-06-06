config = {
    "random_state": 42,
    "data": {
        "test_size": 0.2,
    },

    "lasso_grid": {
        "alpha": [
            0.0001, 0.0005, 0.001, 0.005, 0.01,
            0.05, 0.1, 0.5, 1.0, 2.0
        ],
        "max_iter": [1000, 1500, 2000, 2500, 3000]
    },

    "rf_grid": {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [5, 8, 10, 12, 15],
        "min_samples_split": [2, 5],
        "n_jobs": [-1]  # Все доступные ядра процессора
    }
}
