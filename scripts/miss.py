import time
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral

# =========================
# ENV SETUP (same as rag.py)
# =========================

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

MISTRAL_MODEL = "mistral-large-latest"

# =========================
# CONFIG
# =========================

DATASET_PATH = "D:/skin disease/BE_PROJECT/data/raw/evaluation.csv"

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# MODEL CALL
# =========================

def generate_with_mistral(prompt):

    start = time.time()

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        text = response.choices[0].message.content.strip()

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

The question must include both concepts:
{row.concept1}
{row.concept2}

Difficulty: {row.difficulty}

Return ONLY the question text.
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
        for j in range(i + 1, len(questions)):

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
        for j in range(i + 1, len(questions)):

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

def evaluate_model(df):

    questions = []
    latencies = []

    coverage = []
    difficulty = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):

        print(f"Mistral progress: {i}/{total}")

        prompt = build_prompt(row)

        q, latency = generate_with_mistral(prompt)

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
    print("\nRunning MISTRAL evaluation...\n")

    results = evaluate_model(df)

    print("\n========== RESULTS ==========\n")

    print("Mistral Large")

    for k, v in results.items():
        print(k, round(v, 3))


if __name__ == "__main__":
    main()