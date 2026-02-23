import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# CONFIG
# =========================
TOP_K = 5

INDEX_PATH = "data/processed/faiss_gemini/index.faiss"
METAS_PATH = "data/processed/faiss_gemini/metas.json"
KB_PATH = "data/processed/kb_clean.json"
PARAPHRASE_PATH = "data/processed/kb_eval_paraphrased.json"

# =========================
# LOAD DATA
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

index = faiss.read_index(INDEX_PATH)
metas = load_json(METAS_PATH)
kb = load_json(KB_PATH)
paraphrases = load_json(PARAPHRASE_PATH)

kb_lookup = {item["id"]: item for item in kb}
paraphrase_lookup = {item["id"]: item["paraphrased_question"] for item in paraphrases}

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================
# RETRIEVAL FUNCTIONS
# =========================

def faiss_search(query, k=TOP_K * 5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    _, I = index.search(q_emb, k)
    return [metas[i] for i in I[0]]

def apply_topic_filter(results, topic):
    return [r for r in results if r["topic"] == topic]

def apply_reranking(query, candidates, k=TOP_K):
    pairs = []
    valid_metas = []

    for meta in candidates:
        item = kb_lookup.get(meta["id"])
        if not item:
            continue

        text = f"Q: {item['question']} A: {item['answer']}"
        pairs.append((query, text))
        valid_metas.append(meta)

    if not pairs:
        return candidates[:k]

    scores = reranker.predict(pairs)
    scored = list(zip(valid_metas, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in scored[:k]]

# =========================
# EVALUATION CORE
# =========================

def evaluate_system(use_topic=False, use_rerank=False):

    recall = 0
    mrr = 0
    precision = 0
    evaluated = 0

    for item in kb:
        true_id = item["id"]
        topic = item["topic"]

        # 🔥 USE PARAPHRASED QUERY
        query = paraphrase_lookup.get(true_id)

        # Skip if paraphrase missing
        if not query:
            continue

        evaluated += 1

        # Stage 1: FAISS
        results = faiss_search(query)

        # Stage 2: Topic filtering
        if use_topic:
            results = apply_topic_filter(results, topic)

        # Stage 3: Reranking
        if use_rerank:
            results = apply_reranking(query, results, TOP_K)
        else:
            results = results[:TOP_K]

        retrieved_ids = [r["id"] for r in results]

        # Recall@K
        if true_id in retrieved_ids:
            recall += 1
            rank = retrieved_ids.index(true_id) + 1
            mrr += 1 / rank

        # Precision@K
        correct = sum(1 for rid in retrieved_ids if rid == true_id)
        precision += correct / TOP_K

    return {
        "Samples Evaluated": evaluated,
        "Recall@5": recall / evaluated,
        "MRR": mrr / evaluated,
        "Precision@5": precision / evaluated
    }

# =========================
# RUN ALL EXPERIMENTS
# =========================

if __name__ == "__main__":

    print("\n===== Experiment 1: FAISS Only =====")
    print(evaluate_system(use_topic=False, use_rerank=False))

    print("\n===== Experiment 2: FAISS + Topic Filter =====")
    print(evaluate_system(use_topic=True, use_rerank=False))

    print("\n===== Experiment 3: FAISS + Rerank =====")
    print(evaluate_system(use_topic=False, use_rerank=True))

    print("\n===== Experiment 4: FAISS + Topic Filter + Rerank =====")
    print(evaluate_system(use_topic=True, use_rerank=True))