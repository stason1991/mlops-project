import os
import sys
import subprocess
import pandas as pd
import joblib
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from config import config

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


def load_data():
    """Загрузка данных о вине через DVC API и сохранение на диск."""
    import dvc.api

    print("Загружаем данные из репозитория DVC...")
    with dvc.api.open(
        path='data/winequality-red.csv',
        repo='https://github.com',
        rev='develop'
    ) as fd:
        df = pd.read_csv(fd)

    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    csv_path = os.path.join(BASE_DIR, 'data', 'wine_data.csv')
    df.to_csv(csv_path, index=False)

    print(f"Данные успешно загружены и сохранены в: {csv_path}")
    return csv_path


def train_model(csv_path: str):
    """Обучение лучшей конфигурации модели по параметрам из конфига."""
    print(f"Считываем загруженные данные из: {csv_path}")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['quality'])
    y = df['quality']

    print("Инициализируем лучшую модель...")
    best_model = RandomForestRegressor(
        n_estimators=config["rf_grid"]["n_estimators"],
        max_depth=config["rf_grid"]["max_depth"],
        min_samples_split=config["rf_grid"]["min_samples_split"],
        random_state=config["random_state"],
        n_jobs=config["rf_grid"]["n_jobs"]
    )

    pipeline_obj = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", best_model)
    ])

    print("Запуск обучения модели на полных данных...")
    pipeline_obj.fit(X, y)

    model_obj_path = os.path.join(BASE_DIR, 'temp_trained_pipeline.pkl')
    joblib.dump(pipeline_obj, model_obj_path)

    print(f"Модель успешно обучена и сохранена в: {model_obj_path}")
    return model_obj_path


def save_model(temp_model_path: str) -> str:
    """Сохранение модели в целевую папку и автоматический пуш в DVC."""
    print(f"Получен путь к временно обученной модели: {temp_model_path}")

    pipeline_obj = joblib.load(temp_model_path)

    final_model_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, 'wine_quality_model.pkl')

    joblib.dump(pipeline_obj, final_model_path)
    print(f"Финальный Pipeline успешно сохранен по пути: {final_model_path}")

    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    print("Запуск процесса версионирования в DVC...")
    try:
        subprocess.run(
            [sys.executable, "-m", "dvc", "add",
             "models/wine_quality_model.pkl"],
            cwd=PROJECT_ROOT,
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "dvc", "push"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print("Новая версия модели успешно загружена в MinIO S3.")
    except Exception as e:
        print(f"Критическая ошибка при работе с DVC пайплайном: {e}")
        raise e

    return final_model_path


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'wine_production_pipeline',
    default_args=default_args,
    schedule_interval='0 0 * * *',
    catchup=False,
    tags=['mlops', 'wine', 'production'],
) as dag:

    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    task_train = PythonOperator(
        task_id='train_model',
        python_callable=lambda ti: train_model(
            ti.xcom_pull(task_ids='load_data')
        )
    )

    task_save = PythonOperator(
        task_id='save_model',
        python_callable=lambda ti: save_model(
            ti.xcom_pull(task_ids='train_model')
        )
    )

    task_load >> task_train >> task_save
