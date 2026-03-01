import re
content = open('app.py', encoding='utf-8', errors='ignore').read()
routes = re.findall(r"@app\.route\([\'\"]([^'\"]+)", content)
print("Total routes:", len(routes))
for r in routes:
    if any(k in r.lower() for k in ['mock', 'resume', 'interview', 'upload']):
        print("  RELEVANT:", r)
print("\nAll routes:")
for r in routes:
    print(" ", r)
