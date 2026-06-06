import os
import warnings
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import config

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Wine_Quality")

LOCAL_DATA_PATH = "data/winequality-red.csv"

if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(
        f"Критическая ошибка: Файл не найден {LOCAL_DATA_PATH}. "
        f"Выполните команду 'dvc pull'."
    )

print(f"Загрузка данных из локального DVC-хранилища: {LOCAL_DATA_PATH}")
df = pd.read_csv(LOCAL_DATA_PATH, sep=',')
X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=config["random_state"]
)

best_r2 = -float("inf")
best_mse = float("inf")
best_mae = float("inf")
best_model_info = {}


def run_single_run(model_type, params, run_name):
    """Запуск одного эксперимента и логирование в MLflow."""
    global best_r2, best_mse, best_mae, best_model_info

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("model_type", model_type)
        for p_name, p_val in params.items():
            mlflow.log_param(p_name, p_val)

        if model_type == "lasso":
            model = Lasso(**params, random_state=config["random_state"])
        elif model_type == "random_forest":
            model = RandomForestRegressor(
                **params, random_state=config["random_state"]
            )

        pipeline_obj = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])

        pipeline_obj.fit(X_train, y_train)
        preds = pipeline_obj.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        if (r2 > best_r2 and mse < best_mse and mae < best_mae) or \
           not best_model_info:
            best_r2 = r2
            best_mse = mse
            best_mae = mae
            best_model_info = {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "model_type": model_type,
                "params": params,
                "r2": r2,
                "mse": mse,
                "mae": mae
            }

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_test, y=preds, alpha=0.4, color='darkblue')
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', lw=2
        )
        plt.title(f"{run_name}")
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.tight_layout()

        plot_path = "predictions_plot.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)
        mlflow.sklearn.log_model(pipeline_obj, artifact_path="model_pipeline")

        if os.path.exists(plot_path):
            os.remove(plot_path)


if __name__ == "__main__":
    lasso_grid_combinations = list(ParameterGrid(config["lasso_grid"]))
    print(f"--- Запуск {len(lasso_grid_combinations)} экспериментов Lasso ---")
    for i, params_block in enumerate(lasso_grid_combinations, 1):
        run_title = f"Lasso_Run_{i}_alpha_{params_block['alpha']}"
        run_single_run("lasso", params_block, run_title)
        if i % 10 == 0:
            print(f"Выполнено {i} из {len(lasso_grid_combinations)} Lasso")

    rf_grid_combinations = list(ParameterGrid(config["rf_grid"]))
    print(f"\n--- Запуск {len(rf_grid_combinations)} экспериментов RF ---")
    for i, params_block in enumerate(rf_grid_combinations, 1):
        run_title = (
            f"RF_Run_{i}_trees_{params_block['n_estimators']}_"
            f"depth_{params_block['max_depth']}_"
            f"split_{params_block['min_samples_split']}"
        )
        run_single_run("random_forest", params_block, run_title)
        if i % 10 == 0:
            print(f"Выполнено {i} из {len(rf_grid_combinations)} RF")

    print("\n" + "=" * 50)
    print("Эксперименты завершены успешно")
    print(f"Лучший комплексный запуск: {best_model_info['run_name']}")
    print(f"Тип модели: {best_model_info['model_type']}")
    print(f"Параметры: {best_model_info['params']}")
    print(
        f"Метрики -> R2: {best_model_info['r2']:.4f} | "
        f"MSE: {best_model_info['mse']:.4f} | "
        f"MAE: {best_model_info['mae']:.4f}"
    )
    print("=" * 50)

    print("\nРегистрация лучшего пайплайна в Mlflow Model Registry...")
    model_registry_name = "wine_quality_pipeline"
    artifact_uri = f"runs:/{best_model_info['run_id']}/model_pipeline"

    mlflow.register_model(model_uri=artifact_uri, name=model_registry_name)
    print(f"Пайплайн успешно зарегистрирован как: {model_registry_name}")
