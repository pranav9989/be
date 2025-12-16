"""
rag.py - RAG (Retrieval-Augmented Generation) pipeline and AI functions
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import Config

# Global RAG components
rag_index = None
rag_metas = None
rag_embedder = None
topic_rules = None

def get_gemini_api_key():
    """Get Gemini API key from environment."""
    for name in ("GEMINI_API_KEY", "GEMIN_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(name)
        if v:
            return v
    return None

def find_rag_paths():
    """Find FAISS index and metadata file paths."""
    # return tuple(index_path, metas_path) if found else (None, None)
    for base in Config.RAG_DIRS:
        for idx_name in Config.INDEX_CANDIDATES:
            idx_path = base / idx_name
            if idx_path.exists():
                # find a metas file
                for mname in Config.METAS_CANDIDATES:
                    mpath = base / mname
                    if mpath.exists():
                        return str(idx_path), str(mpath)
                # fallback: any json in dir
                for mpath in base.glob("*.json"):
                    return str(idx_path), str(mpath)
    return None, None

def load_metas_to_map(metas_raw):
    """
    Normalize metas into dict int_id -> meta.
    Support list-of-dicts, dict keyed by int strings, or mapping.
    """
    meta_map = {}
    if metas_raw is None:
        return meta_map
    if isinstance(metas_raw, list):
        # if items include 'int_id', prefer that; else use list index
        has_int_id = any(isinstance(m, dict) and "int_id" in m for m in metas_raw)
        if has_int_id:
            for m in metas_raw:
                try:
                    ik = int(m.get("int_id"))
                    meta_map[ik] = m
                except Exception:
                    pass
        else:
            for i, m in enumerate(metas_raw):
                meta_map[int(i)] = m
    elif isinstance(metas_raw, dict):
        # keys may be numeric strings
        for k, v in metas_raw.items():
            try:
                ik = int(k)
                meta_map[ik] = v
            except Exception:
                # if value contains int_id, use that
                if isinstance(v, dict) and "int_id" in v:
                    try:
                        meta_map[int(v["int_id"])] = v
                    except Exception:
                        meta_map[len(meta_map)] = v
                else:
                    meta_map[len(meta_map)] = v
    return meta_map

def initialize_rag_system():
    """
    Attempt to configure Gemini (if API key present) and load FAISS index + metas + embedder.
    Robust: doesn't crash app when files/missing.
    """
    global rag_index, rag_metas, rag_embedder, topic_rules
    print("🚀 Initializing RAG system...")

    # Load topic rules (optional)
    try:
        if Config.TOPIC_RULES_FILE.exists():
            topic_rules = json.loads(Config.TOPIC_RULES_FILE.read_text(encoding="utf-8"))
            print(f"✅ Loaded topic rules: {len(topic_rules)} rules")
        else:
            topic_rules = None
            print("ℹ️ topic_rules.json not found; continuing without topic routing")
    except Exception as e:
        print("⚠️ Error loading topic rules:", e)
        import traceback
        traceback.print_exc()
        topic_rules = None

    # Configure Gemini client if API key present (do not call generate/test here)
    api_key = get_gemini_api_key()
    if api_key:
        try:
            genai.configure(api_key=api_key)
            print("✅ Gemini configured (API key present). Model set:", Config.GEMINI_MODEL)
        except Exception as e:
            print("⚠️ Error configuring Gemini:", e)
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ No GEMINI_API_KEY found in environment; Gemini features disabled.")

    # Load FAISS index + metas if present
    idx_path, metas_path = find_rag_paths()
    if not idx_path or not metas_path:
        print("ℹ️ No FAISS index or metas found; RAG features disabled.")
        rag_index = None
        rag_metas = None
        rag_embedder = None
        return

    try:
        print(f"🔍 Loading FAISS index from: {idx_path}")
        rag_index = faiss.read_index(idx_path)
        print(f"✅ Loaded FAISS index with {rag_index.ntotal} vectors")
    except Exception as e:
        print("⚠️ Failed to load FAISS index:", e)
        import traceback
        traceback.print_exc()
        rag_index = None

    try:
        print(f"📄 Loading metas from: {metas_path}")
        with open(metas_path, "r", encoding="utf-8") as f:
            metas_raw = json.load(f)
        rag_metas = load_metas_to_map(metas_raw)
        print(f"✅ Loaded {len(rag_metas)} meta entries")
    except Exception as e:
        print("⚠️ Failed to load metas:", e)
        import traceback
        traceback.print_exc()
        rag_metas = None

    if rag_index is not None and rag_metas is not None:
        try:
            print("🤖 Loading embedder model (this may take a moment)...")
            rag_embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedder ready")
        except Exception as e:
            print("⚠️ Failed to load embedder:", e)
            import traceback
            traceback.print_exc()
            rag_embedder = None

def embed_text(embedder, texts):
    """Embed text using SentenceTransformer."""
    emb = embedder.encode(texts, convert_to_numpy=True)
    emb = np.array(emb).astype("float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb

def search_faiss_topk(index, q_emb, k=5):
    """Search FAISS index for top-k similar vectors."""
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return D[0].tolist(), I[0].tolist()

def build_context_from_hits(ids, scores, rag_metas, top_k=5):
    """Build context from search hits."""
    parts = []
    hits = []
    for iid, score in zip(ids[:top_k], scores[:top_k]):
        if iid == -1:
            continue
        meta = rag_metas.get(int(iid))
        if not meta:
            continue
        txt = meta.get("text") or meta.get("chunk") or (f"Item id={meta.get('id')}")
        header = f"[id:{meta.get('id')} int:{meta.get('int_id', iid)} score:{float(score):.4f}]"
        parts.append(header + "\n" + txt)
        mcopy = dict(meta)
        mcopy["_score"] = float(score)
        hits.append(mcopy)
    context = "\n\n".join(parts)
    return context, hits

def generate_rag_response_gemini(api_key, prompt, model_name=Config.GEMINI_MODEL):
    """
    Use Gemini to generate an answer. Try GenerativeModel.generate_content then fallback to responses.create.
    Returns string.
    """
    if not api_key:
        return "Gemini API key not configured."

    genai.configure(api_key=api_key)

    # Try GenerativeModel (some SDK versions support this)
    try:
        gm = genai.GenerativeModel(model_name=model_name)
        resp = gm.generate_content(prompt)
        text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
        if text:
            return text
    except Exception as e:
        # fallback
        gen_err = e

    # Fallback to Responses API
    try:
        resp2 = genai.responses.create(model=model_name, input=prompt)
        text = getattr(resp2, "output_text", None)
        if text:
            return text
        # try nested shapes
        try:
            return resp2["output"][0]["content"][0]["text"]
        except Exception:
            return str(resp2)
    except Exception as e2:
        return f"Error generating response: {e2} (first attempt error: {locals().get('gen_err')})"

def get_topic_and_subtopic_from_query(query, topic_rules):
    """Extract topic and subtopic from user query using keyword matching."""
    if not topic_rules:
        return None, None
    query_lower = query.lower()
    for rule in topic_rules:
        for keyword in rule['keywords']:
            if keyword.lower() in query_lower:
                return rule['topic'], rule['subtopic']
    return None, None

def get_relevant_chunks(query, index, metas, embedder, k=5):
    """Get relevant chunks using FAISS search."""
    if not all([index, metas, embedder]):
        return []
    query_emb = embed_text(embedder, [query])
    scores, ids = search_faiss_topk(index, query_emb, k)
    context, hits = build_context_from_hits(ids, scores, metas, k)
    return hits

def generate_rag_response(query, context):
    """Generate RAG response using Gemini - returns only clean response text."""
    api_key = get_gemini_api_key()
    if not api_key:
        return "Error: No API key available"

    # Use the same clean prompt format as the script
    system_prompt = (
        "You are an expert in computer science, specifically in the domains of DBMS, OOPs, and Operating Systems (OS). "
        "Answer the user's question based on the provided context. "
        "If the context does not contain the answer, state that you cannot answer from the given information. "
        "The context is from a knowledge base of questions and answers. Be concise and helpful."
    )

    full_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=Config.GEMINI_MODEL)
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def build_strict_prompt(context, user_query):
    """Build structured prompt for RAG."""
    system = (
        "You are an expert in computer science (DBMS, OOPs, Operating Systems). "
        "Answer using ONLY the provided CONTEXT. "
        "If the context does not contain the answer, respond exactly: "
        "\"I don't know based on the provided KB.\" "
        "Provide a concise answer and list the source chunk IDs used in square brackets at the end."
    )
    prompt = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_query}\n\nANSWER:"
    return prompt
