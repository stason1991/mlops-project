from fastapi.testclient import TestClient
from src.api.api import app

client = TestClient(app)

def test_healthcheck_endpoint():
    # Тест ручки здоровья API
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}