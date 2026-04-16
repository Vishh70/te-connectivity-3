import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.api import app
import pytest

client = TestClient(app)

@pytest.fixture
def auth_headers():
    response = client.post(
        "/api/login",
        data={"username": "admin", "password": "ai-cup"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_api_status_machine_id(monkeypatch, auth_headers):
    # Mock the `run` method to avoid executing pandas/ML logic in the basic API test
    def mock_run(machine_id):
        return {
            "machine_id": machine_id,
            "timestamp": "2026-04-13 12:00:00",
            "ml_risk_probability": 0.15,
            "alert_level": "LOW",
            "decision_reason": "AI WARNING",
            "violations": []
        }
    
    # We patch inside `backend.api` where it was imported
    monkeypatch.setattr("backend.api.run", mock_run)
    
    response = client.get("/api/status/M231", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["machine_id"] == "M231"
    assert data["alert_level"] == "LOW"

def test_api_machines_list(auth_headers):
    response = client.get("/api/machines", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # In CI there are no local data files so the list may be empty — expected.
    if len(data) > 0:
        assert "id" in data[0]

def test_websocket_status_failure_handling():
    # Since testing the websocket directly connects to EventLoop, 
    # we verify we can import and connect without crashing.
    login_response = client.post(
        "/api/login",
        data={"username": "admin", "password": "ai-cup"},
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    with client.websocket_connect(f"/ws/machines/status?token={token}") as websocket:
        data = websocket.receive_json()
        assert isinstance(data, list)
        
        # Check that we get basic status shapes back
        if len(data) > 0:
             assert "id" in data[0]
             assert "status" in data[0]
