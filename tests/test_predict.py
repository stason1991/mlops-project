from fastapi.testclient import TestClient
from src.api.api import app

client = TestClient(app)


def test_predict_valid_data():
    """Тестируем корректный POST-запрос с признаками вина."""
    valid_payload = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    with TestClient(app) as client_ctx:
        response = client_ctx.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert "predicted_quality" in response.json()


def test_predict_invalid_negative_values():
    """Тестируем валидацию Pydantic на отрицательные значения."""
    invalid_payload = {
        "fixed acidity": -7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    with TestClient(app) as client_ctx:
        response = client_ctx.post("/predict", json=invalid_payload)
        assert response.status_code == 422


def test_model_info_endpoint():
    """Тест ручки информации о модели (/model-info)."""
    with TestClient(app) as client_ctx:
        response = client_ctx.get("/model-info")
        assert response.status_code == 200

        json_data = response.json()
        assert "model_name" in json_data
        assert "model_version" in json_data
        assert "regressor_type" in json_data
        assert "hyperparameters" in json_data
        assert json_data["regressor_type"] == "RandomForestRegressor"