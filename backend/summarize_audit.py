import json

def summarize():
    try:
        with open('definitive_audit_results.json', 'r', encoding='utf-16') as f:
            content = f.read()
    except:
        with open('definitive_audit_results.json', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    
    # The JSON usually starts with '{'
    start_idx = content.find('{')
    if start_idx == -1:
        print("Could not find JSON block in audit results.")
        return
        
    json_str = content[start_idx:]
    data = json.loads(json_str)
    
    print("======================================")
    print("   PREDICTIVE AUDIT FINAL V5 REPORT")
    print("======================================")
    print(f"Global Accuracy: {data['accuracy']}%")
    print(f"Total Cases:     {data['total_cases']}")
    print(f"Matches Found:   {data['matches']}")
    print("--------------------------------------")
    
    for i, res in enumerate(data['results']):
        machine = res.get('machine', '???')
        status = res.get('status', '???')
        risk = res.get('max_risk', 0.0)
        thresh = res.get('threshold', 0.0)
        print(f"Case {i+1} [{machine}]: {status} (Predicted Risk: {risk} | Thresh: {thresh})")
    
    print("======================================")

if __name__ == "__main__":
    summarize()
