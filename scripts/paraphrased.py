import json
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

KB_PATH = "data/processed/kb_clean.json"
OUTPUT_PATH = "data/processed/kb_eval_paraphrased.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def paraphrase_question(question):
    prompt = f"""
Rewrite the following database interview question in a different way.
Keep the meaning exactly the same.
Do NOT change the intent.
Return only the rewritten question.

Original question:
"{question}"

Paraphrased question:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return None
    except:
        return None


def main():
    kb = load_json(KB_PATH)
    paraphrased_data = []

    for i, item in enumerate(kb):
        print(f"Processing {i+1}/{len(kb)}")
        new_q = paraphrase_question(item["question"])

        if new_q:
            paraphrased_data.append({
                "id": item["id"],
                "paraphrased_question": new_q
            })

        time.sleep(0.5)  # avoid overloading model

    save_json(paraphrased_data, OUTPUT_PATH)
    print("Paraphrased dataset saved.")

if __name__ == "__main__":
    main()