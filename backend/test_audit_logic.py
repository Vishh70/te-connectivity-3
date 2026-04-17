from backend.data_access import get_audit_validation_results, save_audit_cases
import json
import os

def verify_logic_wiring():
    print("=== Starting Logical CRUD Verification ===")
    
    # 1. Fetch current list
    res = get_audit_validation_results()
    initial_cases = res.get("results", [])
    initial_count = len(initial_cases)
    print(f"1. Initial count: {initial_count}")

    # 2. Add a case
    new_case = {
        "id": "Logical Verify",
        "machine": "VERIFY-101",
        "date": "2026-04-17",
        "start": "11:00",
        "end": "11:05",
        "comment": "Logic Check"
    }
    
    # Standardize as api.py would
    current_list = initial_cases.copy()
    current_list.append(new_case)
    
    print("2. Testing save_audit_cases...")
    if save_audit_cases(current_list):
        print("   Success: save_audit_cases returned True.")
    else:
        raise Exception("save_audit_cases failed.")

    # 3. Verify on disk
    with open("backend/audit_cases.json", "r") as f:
        disk_data = json.load(f)
        exists = any(c["machine"] == "VERIFY-101" for c in disk_data)
        assert exists
        print("   Success: Case found in audit_cases.json file.")

    # 4. Remove case
    print("3. Testing deletion logic...")
    updated_list = [c for c in current_list if c["machine"] != "VERIFY-101"]
    save_audit_cases(updated_list)
    
    with open("backend/audit_cases.json", "r") as f:
        disk_data = json.load(f)
        exists = any(c["machine"] == "VERIFY-101" for c in disk_data)
        assert not exists
        print("   Success: Case removed from audit_cases.json file.")

    print("\n=== FINAL VERIFICATION CONCLUSION ===")
    print("AUDIT HUB WIRING IS 100% SOLID AND PERSISTENT.")

if __name__ == "__main__":
    verify_logic_wiring()
