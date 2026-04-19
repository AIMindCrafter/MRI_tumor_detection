"""tests/test_api.py — Basic API smoke tests (no real model required)."""
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient

# Patch model loading so tests run without a .keras file
with patch("backend.main.load_model"):
    from backend.main import app, CLASS_NAMES, predict_fn

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_missing_file():
    resp = client.post("/predict")
    assert resp.status_code == 422    # Unprocessable — no file


def test_class_names_count():
    assert len(CLASS_NAMES) == 4
