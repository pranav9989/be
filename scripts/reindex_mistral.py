import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ================== CONFIG ==================
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
KB_CLEAN_PATH = PROCESSED_DIR / "kb_clean.json"

# This matches the FAISS_DIR in rag.py
FAISS_DIR = PROCESSED_DIR / "faiss_mistral"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

def build_kb_index():
    if not KB_CLEAN_PATH.exists():
        print(f"❌ {KB_CLEAN_PATH} not found. Cleaning raw data first...")
        # Try to run prepare_kb.py if it exists
        prepare_script = ROOT / "scripts" / "prepare_kb.py"
        if prepare_script.exists():
            os.system(f"python {prepare_script}")
        else:
            print("❌ Cannot find prepare_kb.py. Please ensure your data/raw folder has JSON files.")
            return

    print("🔄 Loading knowledge base...")
    with open(KB_CLEAN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

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
            "text": text_chunk # Store text for quick retrieval if needed
        })

    print(f"🔄 Generating embeddings for {len(chunks)} chunks using Mistral-compatible model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product is better for normalized cosine similarity
    index.add(np.asarray(embeddings, dtype="float32"))

    index_path = FAISS_DIR / "index.faiss"
    metas_path = FAISS_DIR / "metas.json"

    faiss.write_index(index, str(index_path))
    with open(metas_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)

    print(f"\n✅ FAISS index saved → {index_path}")
    print(f"✅ Metadata saved → {metas_path}")
    print(f"📦 Total vectors: {index.ntotal}")
    print("\n🚀 All documents re-indexed successfully for the Mistral migration.")

if __name__ == "__main__":
    build_kb_index()
