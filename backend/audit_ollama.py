import os

def find_ollama(directory):
    for root, dirs, files in os.walk(directory):
        if 'node_modules' in dirs:
            dirs.remove('node_modules')
        if 'myenv' in dirs:
            dirs.remove('myenv')
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'ollama' in content.lower():
                        print(f"Residual found in: {path}")

if __name__ == "__main__":
    find_ollama("c:/Users/Admin/Documents/BE Project/be/backend")
