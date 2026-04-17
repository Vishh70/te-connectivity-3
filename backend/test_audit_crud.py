from fastapi.testclient import TestClient
from api import app
import json
import os

client = TestClient(app)

def test_audit_crud_wiring():
    print("=== Starting API CRUD Verification ===")
    
    # Bypass auth for local test by setting LOCAL_DEV=true
    os.environ["LOCAL_DEV"] = "true"
    
    # 1. Fetch current status
    print("1. Testing GET /api/audit/validation...")
    response = client.get("/api/audit/validation")
    assert response.status_code == 200
    initial_count = len(response.json()["results"])
    print(f"   Success. Initial cases: {initial_count}")

    # 2. Add a new case
    print("2. Testing POST /api/audit/case (Add)...")
    new_case = {
        "id": "Verification Case",
        "machine": "VERIFY-99",
        "date": "2026-04-17",
        "start": "10:00",
        "end": "10:05",
        "comment": "Test CRUD Wiring"
    }
    response = client.post("/api/audit/case", json=new_case)
    assert response.status_code == 200
    print("   Success. Case added.")

    # 3. Verify it was saved to disk
    with open("audit_cases.json", "r") as f:
        disk_data = json.load(f)
        added_case = next((c for c in disk_data if c["machine"] == "VERIFY-99"), None)
        assert added_case is not None
        print("   Verified: Case persists in audit_cases.json on disk.")

    # 4. Delete the case
    print("3. Testing DELETE /api/audit/case/{index} (Delete)...")
    # Finding the index of our new case
    response = client.get("/api/audit/validation")
    results = response.json()["results"]
    verify_index = next((i for i, c in enumerate(results) if c["machine"] == "VERIFY-99"), -1)
    
    assert verify_index != -1
    response = client.delete(f"/api/audit/case/{verify_index}")
    assert response.status_code == 200
    print(f"   Success. Case at index {verify_index} deleted.")

    # 5. Final check
    response = client.get("/api/audit/validation")
    final_count = len(response.json()["results"])
    assert final_count == initial_count
    print(f"   Success. Count restored to {final_count}.")
    
    print("\n=== FINAL CONCLUSION ===")
    print("AUDIT API WIRING IS FULLY FUNCTIONAL AND PERSISTENT.")

if __name__ == "__main__":
    try:
        test_audit_crud_wiring()
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
