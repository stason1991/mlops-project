import os
import sys
import subprocess
import json
import pandas as pd
import joblib
import mlflow
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from config import config
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


# Адрес сервера mlflow
mlflow.set_tracking_uri("http://localhost:5000")


def load_data():
    """Загрузка данных через dvc pull из корня проекта."""
    print("Запускаем dvc pull для синхронизации данных...")
    try:
        subprocess.run(
            [sys.executable, "-m", "dvc", "pull"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print("Данные успешно синхронизированы.")
    except Exception as e:
        print(f"Ошибка при выполнении dvc pull: {e}")
        raise e

    csv_path = os.path.join(PROJECT_ROOT, 'data', 'winequality-red.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Критическая ошибка: файл не найден по пути {csv_path}")
        
    return csv_path


def train_model(csv_path: str):
    """Выгрузка лучших параметров из Mlflow и обучение модели"""
    print(f"Считываем данные для обучения из: {csv_path}")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['quality'])
    y = df['quality']

    print("Выгружаем лучшие параметры и метрики из MLFlow...")
    client = mlflow.tracking.MlflowClient()
    model_name = "wine_quality_pipeline"
    
    try:
        # Находим последнюю зарегистрированную версию модели
        latest_versions = client.get_latest_versions(model_name)
        if not latest_versions:
            raise RuntimeError(f"Модель {model_name} не найдена в реестре MLflow. Запустите сначала experiments.py")
            
        run_id = latest_versions[0].run_id
        run_data = client.get_run(run_id).data
        
        # Получаем параметры (приводим их к нужным типам)
        mlflow_params = run_data.params
        best_n_estimators = int(mlflow_params.get("n_estimators", 100))
        best_max_depth = mlflow_params.get("max_depth")
        best_max_depth = int(best_max_depth) if best_max_depth and best_max_depth != "None" else None
        best_min_samples_split = int(mlflow_params.get("min_samples_split", 2))
        
        # Выгружаем метрики лучшего запуска для истории
        experiment_metrics = run_data.metrics
        print(f"Успешно получены параметры из MLflow: n_estimators={best_n_estimators}, max_depth={best_max_depth}")
        
    except Exception as e:
        print(f"Не удалось связаться с MLflow ({e}). Используем резервные дефолтные параметры.")
        best_n_estimators = 100
        best_max_depth = None
        best_min_samples_split = 2
        experiment_metrics = {}

    print("Инициализируем RandomForestRegressor с точными параметрами...")
    # Передаются числа из MLflow, не списки
    best_model = RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        random_state=config["random_state"],
        n_jobs=config["rf_grid"]["n_jobs"]
    )

    pipeline_obj = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", best_model)
    ])

    print("Запуск обучения модели на полных данных...")
    pipeline_obj.fit(X, y)

    # Сохраняем временно обученную модель
    model_obj_path = os.path.join(BASE_DIR, 'temp_trained_pipeline.pkl')
    joblib.dump(pipeline_obj, model_obj_path)

    # Создаем файл метаданных (метрики, параметры, дата, описание) по ТЗ
    metadata = {
        "model_type": "RandomForestRegressor",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters_used": {
            "n_estimators": best_n_estimators,
            "max_depth": best_max_depth,
            "min_samples_split": best_min_samples_split
        },
        "experiment_best_metrics": experiment_metrics,
        "description": "Модель переобучена в Airflow на актуальных данных с использованием лучших параметров из MLflow."
    }

    # Сохраняем временно файл метаданных
    meta_obj_path = os.path.join(BASE_DIR, 'temp_metadata.json')
    with open(meta_obj_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("Обучение завершено. Файлы подготовлены.")
    return model_obj_path, meta_obj_path

def save_model(paths_tuple) -> str:
    """Сохранение модели и метаданных в целевую папку и автоматический пуш в DVC."""
    if not paths_tuple:
        raise ValueError("Данные о путях временных файлов не получены из XCom")


    temp_model_path, temp_meta_path = paths_tuple


    final_model_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(final_model_dir, exist_ok=True)
    
    final_model_path = os.path.join(final_model_dir, 'wine_quality_model.pkl')
    final_meta_path = os.path.join(final_model_dir, 'model_metadata.json')

    # Переносим файлы из временных папок в финальные структуры проекта
    pipeline_obj = joblib.load(temp_model_path)
    joblib.dump(pipeline_obj, final_model_path)
    
    with open(temp_meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(final_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # Удаляем временные локальные файлы задачи
    if os.path.exists(temp_model_path): os.remove(temp_model_path)
    if os.path.exists(temp_meta_path): os.remove(temp_meta_path)

    print("Запуск процесса версионирования в DVC...")
    try:
        # Добавляем в DVC два файла: модель и файл метаданных
        subprocess.run(
            [sys.executable, "-m", "dvc", "add",
            os.path.join('models', 'wine_quality_model.pkl'),
            os.path.join('models', 'model_metadata.json')],
            cwd=PROJECT_ROOT,
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "dvc", "push"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print("Модель и метаданные успешно версионированы и отправлены в MinIO S3.")
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
