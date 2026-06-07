from fastapi.testclient import TestClient
from src.api.api import app

client = TestClient(app)


def test_healthcheck_endpoint():
    """Тест ручки здоровья API согласно ТЗ (/healthcheck)."""
    with TestClient(app) as client_ctx:
        response = client_ctx.get("/healthcheck")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
