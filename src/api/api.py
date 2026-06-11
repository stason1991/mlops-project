import os
import sys
import warnings
import subprocess
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

print(f"[DEBUG] корень проекта определен как: {PROJECT_ROOT}")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
)
MODEL_NAME = "wine_quality_pipeline"
MODEL_VERSION = "1"

# Переменная для хранения пайплайна в памяти
pipeline = None


def export_mlflow_to_dvc():
    """Выгрузка нового пайплайна из MLflow в DVC."""
    import subprocess
    print("\n[MLflow -> DVC] Старт выгрузки нового пайплайна...")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        print(f"Загрузка {model_uri} из MLflow...")
        loaded_pipeline = mlflow.sklearn.load_model(model_uri)

        os.makedirs(os.path.join(PROJECT_ROOT, MODEL_DIR), exist_ok=True)
        absolute_model_path = os.path.join(PROJECT_ROOT, MODEL_PATH)

        joblib.dump(loaded_pipeline, absolute_model_path)
        print(f"Бинарник успешно сохранен на диске: {absolute_model_path}")

        print("Выполнение dvc add...")
        subprocess.run(
            [sys.executable, "-m", "dvc", "add", MODEL_PATH],
            cwd=PROJECT_ROOT,
            check=True
        )
        print("Выполнение dvc push...")
        subprocess.run(
            [sys.executable, "-m", "dvc", "push"],
            cwd=PROJECT_ROOT,
            check=True
        )

        print("[MLflow -> DVC] Успешно! Файл затрекан в DVC.")
    except Exception as e:
        print(f"[Ошибка выгрузки]: {e}")
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл API для автоматической загрузки модели."""
    global pipeline
    absolute_model_path = os.path.join(PROJECT_ROOT, MODEL_PATH)
    
    # ИСПРАВЛЕНО (Пункт 6): Если файла нет, пробуем dvc pull
    if not os.path.exists(absolute_model_path):
        print(f"[API] Файл {MODEL_PATH} не найден. Скачиваем через dvc pull...")
        try:
            subprocess.run(
                [sys.executable, "-m", "dvc", "pull", MODEL_PATH],
                cwd=PROJECT_ROOT,
                check=True
            )
            print("[API] dvc pull успешно выполнен.")
        except Exception as dvc_err:
            print(f"[API Ошибка] Не удалось выполнить dvc pull: {dvc_err}")

    # Пробуем загрузить модель в память
    print(f"\n[API] Загрузка модели из локального файла: {absolute_model_path}...")
    try:
        if os.path.exists(absolute_model_path):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=InconsistentVersionWarning
                )
                pipeline = joblib.load(absolute_model_path)
            print("[API] Пайплайн успешно загружен в память.\n")
        else:
            raise FileNotFoundError(
                f"Файл модели отсутствует по пути: {absolute_model_path}"
            )
    except Exception as e:
        print(f"[API КРИТИЧЕСКАЯ ОШИБКА]: Приложение запущено без модели! Ошибка: {e}")
        pipeline = None

    yield


app = FastAPI(lifespan=lifespan)


class WineFeatures(BaseModel):
    """Класс валидации входящих признаков вина через Pydantic."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "fixed acidity": 7.4, "volatile acidity": 0.7,
                "citric acid": 0.0, "residual sugar": 1.9,
                "chlorides": 0.076, "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0, "density": 0.9978,
                "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4
            }
        }
    )
    fixed_acidity: float = Field(alias="fixed acidity", examples=[7.4])
    volatile_acidity: float = Field(alias="volatile acidity", examples=[0.7])
    citric_acid: float = Field(alias="citric acid", examples=[0.0])
    residual_sugar: float = Field(alias="residual sugar", examples=[1.9])
    chlorides: float = Field(examples=[0.076])
    free_sulfur_dioxide: float = Field(
        alias="free sulfur dioxide", examples=[11.0]
    )
    total_sulfur_dioxide: float = Field(
        alias="total sulfur dioxide", examples=[34.0]
    )
    density: float = Field(examples=[0.9978])
    pH: float = Field(examples=[3.51])
    sulphates: float = Field(examples=[0.56])
    alcohol: float = Field(examples=[9.4])

    @field_validator("*")
    @classmethod
    def check_positive(cls, value: float) -> float:
        """Проверка, чтобы свойства вина не были отрицательными."""
        if value < 0:
            raise ValueError("Значения признаков не могут быть отрицательными")
        return value


@app.post("/predict")
async def predict(features: WineFeatures, request: Request) -> dict:
    """Эндпоинт инференса для предсказания качества вина."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Пайплайн не загружен.")
    try:
        raw_json = await request.json()

        required_cols = [
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates",
            "alcohol"
        ]
        input_data = pd.DataFrame([raw_json])
        input_data = input_data[required_cols]

        prediction = pipeline.predict(input_data)

        if hasattr(prediction, "ndim") and prediction.ndim > 0:
            predicted_value = prediction[0]
        elif hasattr(prediction, "__len__") and len(prediction) > 0:
            predicted_value = prediction[0]
        else:
            predicted_value = prediction

        return {"predicted_quality": float(predicted_value)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get("/healthcheck")
async def healthcheck() -> dict:
    """Проверка доступности сервиса."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель недоступна")
    return {"status": "ok"}


@app.get("/model-info")
async def model_info() -> dict:
    """Получение метаданных о текущей модели."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    try:
        if hasattr(pipeline, "named_steps"):
            regressor = pipeline.named_steps.get("regressor")
            steps = list(pipeline.named_steps.keys())
        else:
            regressor = pipeline
            steps = ["direct_model"]

        reg_params = regressor.get_params() if regressor else {}
        stringified = {str(k): str(v) for k, v in reg_params.items()}

        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "pipeline_steps": steps,
            "regressor_type": type(regressor).__name__ if regressor else "?",
            "hyperparameters": stringified
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and "--sync" in sys.argv:
        export_mlflow_to_dvc()
    else:
        uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
