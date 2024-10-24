import pytest
from fastapi.testclient import TestClient
from .main import app  # Import relatif car le fichier est dans le même dossier

client = TestClient(app)

def test_predict_positive():
    response = client.post("/predict", json={"text": "This is a great day!"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}

def test_predict_negative():
    response = client.post("/predict", json={"text": "This is a terrible day!"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "negative"}

def test_feedback_positive():
    response = client.post("/feedback", json={
        "text": "This is a great day!",
        "prediction": "positive",
        "validation": True
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback reçu, merci !"}

def test_feedback_negative():
    response = client.post("/feedback", json={
        "text": "This is a terrible day!",
        "prediction": "positive",
        "validation": False
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback reçu, merci !"}

