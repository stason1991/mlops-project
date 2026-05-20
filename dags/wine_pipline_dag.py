import os
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Базовая директория для работы с файлами
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data() -> str:
    """Загрузка данных о вине."""
    print('Представим, что тут загрузились данные о вине...')
    df = pd.DataFrame()
    
    # Сохраняем временный датасет
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    csv_path = os.path.join(BASE_DIR, 'data', 'wine_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Данные успешно загружены и сохранены в: {csv_path}")
    return csv_path


def train_model(csv_path: str) -> str:
    """Обучение модели."""
    print(f"Считываем загруженные данные из: {csv_path}")
    print('Представим, что тут обучилась модель....')
    
    # Возвращаем условное название обученной модели
    trained_model_name = "LogisticRegression_Wine_Model"
    return trained_model_name


def save_model(model_name: str) -> str:
    """Сохранение модели."""
    print(f"Получена обученная модель: {model_name}")
    
    # Путь для сохранения артефакта модели
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    print(f"Модель успешно сохранена по пути: {model_path}")
    return model_path


# Настройки DAG по умолчанию
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Инициализируем DAG с ежедневным расписанием
with DAG(
    'wine_mock_pipeline',
    default_args=default_args,
    schedule_interval='0 0 * * *',
    catchup=False,
    tags=['mlops', 'wine'],
) as dag:

    # 1. Загрузка данных
    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    # 2. Обучение модели
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=lambda ti: train_model(
            ti.xcom_pull(task_ids='load_data')
        )
    )

    # 3. Сохранение модели
    task_save = PythonOperator(
        task_id='save_model',
        python_callable=lambda ti: save_model(
            ti.xcom_pull(task_ids='train_model')
        )
    )

    # Загрузка -> обучение -> сохранение
    task_load >> task_train >> task_save