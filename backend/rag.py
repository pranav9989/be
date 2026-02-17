import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# ================== ENV SETUP ==================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"


# ================== PATHS ==================
FAISS_DIR = "data/processed/faiss_gemini"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METAS_PATH = os.path.join(FAISS_DIR, "metas.json")

KB_CLEAN_PATH = "data/processed/kb_clean.json"
TOPIC_RULES_PATH = "config/topic_rules.json"


# ================== LOADERS ==================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index_and_metas():
    index = faiss.read_index(INDEX_PATH)
    metas = load_json(METAS_PATH)
    return index, metas


def build_kb_lookup():
    kb = load_json(KB_CLEAN_PATH)
    return {item["id"]: item for item in kb}


# ================== TOPIC DETECTION ==================
def get_topic_and_subtopic_from_query(query, topic_rules):
    q = query.lower()
    for rule in topic_rules:
        for kw in rule["keywords"]:
            if kw in q:
                return rule["topic"], rule["subtopic"]
    return None, None


# ================== FAISS RETRIEVAL ==================
def get_relevant_chunks_filtered(query, index, metas, model, topic=None, k=5):
    """
    Vector search + HARD topic filtering
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Search more than needed, then filter
    _, I = index.search(query_embedding, k * 4)

    results = []
    for idx in I[0]:
        meta = metas[idx]
        if topic is None or meta["topic"] == topic:
            results.append(meta)
        if len(results) == k:
            break

    return results


def generate_rag_response(query, context):
    prompt = f"""
You are an expert Computer Science interviewer providing a MODEL ANSWER.
The user has just answered this question. Provide a CONCISE, HUMAN-LIKE expected answer.

RULES:
- Keep it brief - 2-3 sentences maximum
- Sound like a knowledgeable person, not a textbook
- Include key technical terms naturally
- DO NOT use markdown or formatting
- DO NOT explain concepts in depth - just state the core answer

Context for reference:
{context}

Question: {query}

Expected answer (concise, 2-3 sentences):
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
            return f"❌ Ollama error: {response.text}"

    except Exception as e:
        return f"❌ Ollama connection error: {str(e)}"

# ================== MAIN ==================
def main(user_query):
    # Load resources
    topic_rules = load_json(TOPIC_RULES_PATH)
    index, metas = load_index_and_metas()
    kb_lookup = build_kb_lookup()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Topic detection
    topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)

    if topic:
        print(f"🧠 Detected Topic: {topic} | Subtopic: {subtopic}")
        augmented_query = f"{topic} {subtopic}: {user_query}"
    else:
        print("🧠 Topic not detected → semantic search only")
        augmented_query = user_query

    # Retrieve (topic-filtered)
    retrieved = get_relevant_chunks_filtered(
        augmented_query,
        index,
        metas,
        embedder,
        topic=topic,
        k=5
    )

    # Build context STRICTLY from kb_clean.json
    context_blocks = []
    for meta in retrieved:
        item = kb_lookup.get(meta["id"])
        if not item:
            continue
        context_blocks.append(
            f"Q: {item['question']}\nA: {item['answer']}"
        )

    context_text = "\n\n".join(context_blocks)

    # Generate answer
    answer = generate_rag_response(user_query, context_text)

    print("\n================ RAG ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ===================\n")
    for meta in retrieved:
        print(f"- {meta['id']} | {meta['topic']} > {meta['subtopic']}")

    return answer, retrieved
