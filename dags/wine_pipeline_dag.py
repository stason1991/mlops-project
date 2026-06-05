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

# Базовая директория для работы с файлами
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Корневая папка проекта
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

def load_data():
    """Загрузка данных о вине через DVC и сохранение на диск."""
    import dvc.api

    print("Загружаем данные из репозитория DVC...")
    with dvc.api.open(
        path='data/winequality-red.csv',
        repo='https://github.com',
        rev='develop'
    ) as fd:
        df = pd.read_csv(fd)
    
    # Создаем директорию и сохраняем датасет
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    csv_path = os.path.join(BASE_DIR, 'data', 'wine_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Данные успешно загружены и сохранены в: {csv_path}")
    return csv_path


def train_model(csv_path: str):
    """Обучение лучшей конфигурации моделей по итогам экспериментов в MLflow."""
    print(f"Считываем загруженные данные из: {csv_path}")
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=['quality'])
    y = df['quality']
    
    print("Инициализируем лучшую модель (RandomForestRegressor)...")
    # Динамически берем параметры из конфига, выбирая лучшие по итогам экспериметов
    # В rf_grid списки, так что берем финальные лучшие индексы:
    # # n_estimators: 100 (индекс 1), max_depth: 15 (индекс 4), min_samples_split: 2 (индекс 0)
    best_model = RandomForestRegressor(
        max_depth=config["rf_grid"]["max_depth"][4],
        min_samples_split=config["rf_grid"]["min_samples_split"][0],
        n_estimators=config["rf_grid"]["n_estimators"][1],
        random_state=config["random_state"],
        n_jobs=["rf_grid"]["n_jobs"][0]
    )
    
    # Заворачиваем в пайплайн вместе со скалером
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", best_model)
    ])
    
    print("Запуск обучения модели на полных данных...")
    pipeline.fit(X, y)
    
    # Временный путь для передачи объекта через XCom
    model_obj_path = os.path.join(BASE_DIR, 'temp_trained_pipeline.pkl')
    joblib.dump(pipeline, model_obj_path)
    
    print(f"Модель успешно обучена и временно сохранена в: {model_obj_path}")
    return model_obj_path


def save_model(temp_model_path: str) -> str:
    """Сохранение финального артефакта модели в целевую папку и пуш в DVC."""
    print(f"Получен путь к временно обученной модели: {temp_model_path}")
    
    # Загружаем временную модель
    pipeline = joblib.load(temp_model_path)
    
    # Путь для сохранения финального артефакта в папку models корня проекта
    final_model_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "wine_quality_model.pkl")

    joblib.dump(pipeline, final_model_path)
    print(f"Финальный pipeline успешно сохранен по пути: {final_model_path}")
    
    # Чистим временный файл
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    # Автоматизация версионирования через DVC
    print("Запуск версионирования в DVC....")
    try:
        # Выполняем 'dvc add' в корневом каталоге проекта
        subprocess.run(
            [sys.executable, "-m", "dvc", "add", "models/wine_quality_model.pkl"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print("DVC отследил новый файл модели")

        # DVC push для отправки новой модели в inIO S3
        subprocess.run(
            [sys.executable, "-m", "dvc", "push"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print("Новая версия модели успешно загружена в хранилище MinIO")

    except Exception as e:
        print(f"Критическая ошибка при работе с DVC пайплайном: {e}")
        raise e
 
    return final_model_path


# Настройки DAG по умолчанию
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Инициализируем DAG с ежедневным расписанием
with DAG(
    'wine_production_pipeline',
    default_args=default_args,
    schedule_interval='0 0 * * *',
    catchup=False,
    tags=['mlops', 'wine', 'production'],
) as dag:

    # Загружаем данные
    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    # Обучаем модели
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=lambda ti: train_model(
            ti.xcom_pull(task_ids='load_data')
        )
    )

    # Сохраненяем модель и отправляем в DVC
    task_save = PythonOperator(
        task_id='save_model',
        python_callable=lambda ti: save_model(
            ti.xcom_pull(task_ids='train_model')
        )
    )

    # Загрузка -> обучение -> сохранение
    task_load >> task_train >> task_save