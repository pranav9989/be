import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ================== CONFIG ==================
PROCESSED_DIR = Path("data/processed")
KB_CLEAN_PATH = PROCESSED_DIR / "kb_clean.json"

FAISS_DIR = PROCESSED_DIR / "faiss_gemini"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = FAISS_DIR / "faiss_index_gemini.idx"
METAS_PATH = FAISS_DIR / "metas.json"

# ================== LOADERS ==================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ================== CHUNKING ==================
def create_chunks_and_metas(data):
    chunks = []
    metas = []

    for item in data:
        text_chunk = f"Q: {item['question']}\nA: {item['answer']}"
        chunks.append(text_chunk)

        metas.append({
            "id": item["id"],
            "topic": item["topic"],
            "subtopic": item["subtopic"],
            "difficulty": item["difficulty"],
            "source": item.get("source"),
        })

    return chunks, metas

# ================== FAISS ==================
def build_faiss_index(chunks, metas):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔄 Generating embeddings...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True  # IMPORTANT
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(embeddings, dtype="float32"))

    index_path = FAISS_DIR / "index.faiss"
    metas_path = FAISS_DIR / "metas.json"

    faiss.write_index(index, str(index_path))
    with open(metas_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)

    print(f"✅ FAISS index saved → {index_path}")
    print(f"✅ Metadata saved → {metas_path}")
    print(f"📦 Total vectors: {index.ntotal}")

# ================== MAIN ==================
def main():
    if not KB_CLEAN_PATH.exists():
        print("❌ kb_clean.json not found. Run prepare_kb.py first.")
        return

    data = load_json(KB_CLEAN_PATH)

    if not data:
        print("❌ kb_clean.json is empty.")
        return

    chunks, metas = create_chunks_and_metas(data)
    build_faiss_index(chunks, metas)

if __name__ == "__main__":
    main()
