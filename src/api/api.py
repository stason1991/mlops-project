import os
import sys
import warnings
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sklearn.exceptions import InconsistentVersionWarning

# Настройки и пути
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "wine_quality_model.pkl")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "wine_quality_pipeline" 
MODEL_VERSION = "1" 

# Переменная для хранения пайплайна в памяти
pipeline = None

# Режим 1: синхронизация (Используем для фиксации новой модели в DVC)
def export_mlflow_to_dvc():
    # Скачиваем пайплайн из MLflow, сохраняем в файл models/wine_quality_model.pkl,
    # генерируем для него .dvc слепок и пушим бинарник в MinIO.
    # Файл оставляем на диске, но благодаря DVC автоматически блокируем для Git.
    import subprocess
    print("\n[MLflow -> DVC] Старт выгрузки нового пайплайна...")
    try:
        # Скачиваем актуальный пайплайн из MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        print(f"Загрузка {model_uri} из MLflow...")
        loaded_pipeline = mlflow.sklearn.load_model(model_uri)
        
        # Сохраняем
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(loaded_pipeline, MODEL_PATH)
        print(f"Бинарник успешно сохранен на диске: {MODEL_PATH}")
        
        # Фиксируем структуру в DVC и пушим в бакет
        print("Выполнение dvc add...")
        subprocess.run([sys.executable, "-m", "dvc", "add", MODEL_PATH], check=True)
        print("Выполнение dvc push...")
        subprocess.run([sys.executable, "-m", "dvc", "push"], check=True)
        
        print("[MLflow -> DVC] Успешно! Файл затрекан в DVC и сохранен локально.")
        print("-> Сделайте коммит: git add models/wine_quality_model.pkl.dvc models/.gitignore && git commit")
    except Exception as e:
        print(f"[Ошибка выгрузки]: {e}")
        sys.exit(1)

# Режим 2: FastAPI с автоматической загрузкой из локального файла
@asynccontextmanager
async def lifespan(app: FastAPI):
    # При запуске API автоматически загружаем актуальную версию модели
    # из локального файла models/wine_quality_model.pkl.
    global pipeline
    print(f"\n[API] Автоматическая загрузка модели из локального файла: {MODEL_PATH}...")
    try:
        # Проверяем физическое наличие файла модели на диске
        if os.path.exists(MODEL_PATH):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                # Загружаем модель в оперативную память
                pipeline = joblib.load(MODEL_PATH)
            print("[API] Пайплайн успешно загружен в оперативную память. Сервис готов к работе.\n")
        else:
            raise FileNotFoundError(
                f"Критическая ошибка: файл модели не найден по пути: {MODEL_PATH}.\n"
                f"Если проект на новой машине, сначала выполните команду 'dvc pull' в терминале."
            )
    except Exception as e:
        print(f"[API Критическая ошибка при старте]: {e}")
        sys.exit(1)  # Завершаем процесс, если модель не найдена или повреждена
    yield

app = FastAPI(lifespan=lifespan)

# Класс валидации Pydantic
class WineFeatures(BaseModel):
    model_config = ConfigDict(
        # Позволяет парсить JSON как по алиасам (с пробелами), так и по именам (с подчеркиванием)
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0, 
                "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
                "sulphates": 0.56, "alcohol": 9.4
            }
        }
    )
    fixed_acidity: float = Field(alias="fixed acidity", examples=[7.4])
    volatile_acidity: float = Field(alias="volatile acidity", examples=[0.7])
    citric_acid: float = Field(alias="citric acid", examples=[0.0])
    residual_sugar: float = Field(alias="residual sugar", examples=[1.9])
    chlorides: float = Field(examples=[0.076])
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide", examples=[11.0])
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide", examples=[34.0])
    density: float = Field(examples=[0.9978])
    pH: float = Field(examples=[3.51])
    sulphates: float = Field(examples=[0.56])
    alcohol: float = Field(examples=[9.4])

    @field_validator("*")
    @classmethod
    def check_positive(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Значения свойств вина не могут быть отрицательными")
        return value

# Эндпоинты
@app.post("/predict")
async def predict(features: WineFeatures, request: Request) -> dict:
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Пайплайн моделей не загружен.")
    try:
        # WineFeatures автоматически проверяет данные, если там отрицательные числа, Pydantic выдаст ошибку до этого шага.
        # Берем оригинальный JSON из тела запроса (там пробелы)
        raw_json = await request.json()

        # Задаем правильный порядок колонок, как в исходном датасете
        REQUIRED_COLUMNS = [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]
        input_data = pd.DataFrame([raw_json])
        input_data = input_data[REQUIRED_COLUMNS]

        # Передаем в пайплайн
        prediction = pipeline.predict(input_data)
        
        if hasattr(prediction, "ndim") and prediction.ndim > 0:
            predicted_value = prediction[0]
        elif hasattr(prediction, "__len__") and len(prediction) > 0:
            predicted_value = prediction[0]
        else:
            predicted_value = prediction

        return {"predicted_quality": float(predicted_value)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}") 

@app.get("/healthcheck")
async def healthcheck() -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель недоступна")
    return {"status": "ok"}

@app.get("/model-info")
async def model_info() -> dict:
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена в память")

    try:
        if hasattr(pipeline, "named_steps"):
            regressor = pipeline.named_steps.get("regressor")
            pipeline_steps = list(pipeline.named_steps.keys())
        else:
            # Если в файле чистая модель, а не pipeline
            regressor = pipeline
            pipeline_steps = ["no_pipeline_direct_model"]

        regressor_params = regressor.get_params() if regressor else {}

        # Все параметры делаем строками, чтобы избежать падений на валидации типов в JSON
        stringified_params = {str(k): str(v) for k, v in regressor_params.items()}

        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "pipeline_steps": list(pipeline.named_steps.keys()),
            "regressor_type": type(regressor).__name__ if regressor else "Unknown",
            "hyperparameters": stringified_params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

# Точка входа (Управление режимами работы)
if __name__ == "__main__":
    if len(sys.argv) > 1 and "--sync" in sys.argv:
        export_mlflow_to_dvc()
    else:
        uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)