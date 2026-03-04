import time
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral
import os

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma3:1b"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-large-latest"

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

DATASET_PATH = "D:/skin disease/BE_PROJECT/data/raw/evaluation.csv"

# =========================
# MODEL CALLS
# =========================

def generate_with_gemma(prompt):

    start = time.time()

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": GEMMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        text = r.json()["response"]

    except Exception as e:
        print("Gemma error:", e)
        text = ""

    latency = time.time() - start

    return text, latency


def generate_with_mistral(prompt):

    start = time.time()

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content

    except Exception as e:
        print("Mistral error:", e)
        text = ""

    latency = time.time() - start

    return text, latency


# =========================
# PROMPT TEMPLATE
# =========================

def build_prompt(row):

    return f"""
Generate ONE technical interview question.

Topic: {row.topic}
Subtopic: {row.subtopic}

The question must include:
{row.concept1} and {row.concept2}

Difficulty: {row.difficulty}

Return ONLY the question.
"""


# =========================
# METRICS
# =========================

def concept_coverage(question, c1, c2):

    q = question.lower()

    score = 0

    if c1.lower() in q:
        score += 1

    if c2.lower() in q:
        score += 1

    return score / 2


def difficulty_alignment(question, difficulty):

    words = len(question.split())

    if difficulty == "easy":
        return 1 if words < 18 else 0

    if difficulty == "medium":
        return 1 if 18 <= words <= 30 else 0

    if difficulty == "hard":
        return 1 if words > 30 else 0


def semantic_diversity(questions):

    embeddings = EMBEDDER.encode(questions, batch_size=32)

    sims = []

    for i in range(len(questions)):
        for j in range(i+1, len(questions)):

            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]

            sims.append(sim)

    return 1 - np.mean(sims)


def duplicate_rate(questions):

    embeddings = EMBEDDER.encode(questions, batch_size=32)

    duplicates = 0
    total = 0

    for i in range(len(questions)):
        for j in range(i+1, len(questions)):

            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]

            total += 1

            if sim > 0.9:
                duplicates += 1

    return duplicates / total if total else 0


# =========================
# MAIN EVALUATION
# =========================

def evaluate_model(df, generator, model_name):

    questions = []
    latencies = []

    coverage = []
    difficulty = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):

        print(f"{model_name} progress: {i}/{total}")

        prompt = build_prompt(row)

        q, latency = generator(prompt)

        questions.append(q)
        latencies.append(latency)

        coverage.append(
            concept_coverage(q, row.concept1, row.concept2)
        )

        difficulty.append(
            difficulty_alignment(q, row.difficulty)
        )

    return {
        "concept_coverage": np.mean(coverage),
        "difficulty_alignment": np.mean(difficulty),
        "duplicate_rate": duplicate_rate(questions),
        "diversity": semantic_diversity(questions),
        "latency": np.mean(latencies)
    }


# =========================
# RUN EXPERIMENT
# =========================

def main():

    df = pd.read_csv(DATASET_PATH)

    print("Dataset size:", len(df))

    print("\nRunning GEMMA evaluation...\n")

    gemma_results = evaluate_model(df, generate_with_gemma, "Gemma")

    print("\nRunning MISTRAL evaluation...\n")

    mistral_results = evaluate_model(df, generate_with_mistral, "Mistral")

    print("\n========== RESULTS ==========\n")

    print("Gemma 1B")
    for k,v in gemma_results.items():
        print(k, round(v,3))

    print("\nMistral Large")
    for k,v in mistral_results.items():
        print(k, round(v,3))


if __name__ == "__main__":
    main()