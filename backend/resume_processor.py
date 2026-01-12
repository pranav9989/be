import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# ================== PATHS ==================
RESUME_DATA_DIR = "data/processed/resume_faiss"
os.makedirs(RESUME_DATA_DIR, exist_ok=True)

def get_resume_index_path(user_id):
    return os.path.join(RESUME_DATA_DIR, f"resume_index_{user_id}.faiss")

def get_resume_metas_path(user_id):
    return os.path.join(RESUME_DATA_DIR, f"resume_metas_{user_id}.json")

# ================== LOADERS ==================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ================== RESUME PROCESSING ==================
def process_resume_for_faiss(resume_text, user_id):
    """Process resume text with FAISS - create chunks and store embeddings"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(resume_text)

    # Load or create embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings for all chunks
    embeddings = []
    metas = []

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode([chunk], normalize_embeddings=True)[0]
        embeddings.append(embedding)

        meta = {
            "id": f"resume_chunk_{user_id}_{i}",
            "chunk_id": i,
            "user_id": user_id,
            "text": chunk,
            "source": "resume",
            "chunk_size": len(chunk)
        }
        metas.append(meta)

    # Convert to numpy array for FAISS
    embeddings_array = np.array(embeddings).astype('float32')

    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)

    # Add vectors to index
    index.add(embeddings_array)

    # Save index and metadata
    index_path = get_resume_index_path(user_id)
    metas_path = get_resume_metas_path(user_id)

    faiss.write_index(index, index_path)
    save_json(metas, metas_path)

    return len(chunks)

def search_resume_faiss(query, user_id, top_k=5):
    """Search resume content using FAISS"""
    index_path = get_resume_index_path(user_id)
    metas_path = get_resume_metas_path(user_id)

    if not os.path.exists(index_path) or not os.path.exists(metas_path):
        return []

    try:
        # Load index and metadata
        index = faiss.read_index(index_path)
        metas = load_json(metas_path)

        # Load embedder and encode query
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        query_embedding = np.array([query_embedding]).astype('float32')

        # Search
        scores, indices = index.search(query_embedding, min(top_k, len(metas)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metas):  # Valid index
                meta = metas[idx].copy()
                meta["_score"] = float(score)
                results.append(meta)

        return results

    except Exception as e:
        print(f"Error searching resume FAISS: {e}")
        return []

def get_resume_chunks(user_id):
    """Get all resume chunks for a user"""
    metas_path = get_resume_metas_path(user_id)
    if os.path.exists(metas_path):
        return load_json(metas_path)
    return []

def delete_resume_data(user_id):
    """Delete resume FAISS data for a user"""
    index_path = get_resume_index_path(user_id)
    metas_path = get_resume_metas_path(user_id)

    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metas_path):
            os.remove(metas_path)
        return True
    except Exception as e:
        print(f"Error deleting resume data: {e}")
        return False
