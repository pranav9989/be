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


# ================== STRICT TOPIC FILTERING ==================
def get_relevant_chunks_strict(query, index, metas, model, topic=None, k=5):
    """
    Vector search + STRICT topic filtering - ONLY return chunks from specified topic
    Used for technical interview chatbot to ensure topic relevance
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Search more than needed (k * 8) to ensure we find enough of the right topic
    _, I = index.search(query_embedding, k * 8)

    results = []
    seen_ids = set()  # Avoid duplicates
    
    for idx in I[0]:
        meta = metas[idx]
        
        # 🔥 STRICT FILTER: If topic specified, ONLY return chunks with that topic
        if topic is not None:
            if meta["topic"] != topic:
                continue
        
        # Avoid duplicate chunks
        if meta["id"] in seen_ids:
            continue
            
        seen_ids.add(meta["id"])
        results.append(meta)
        
        if len(results) == k:
            break
    
    # 🔥 If we didn't find enough chunks, log warning
    if len(results) < k and topic:
        print(f"⚠️ WARNING: Only found {len(results)}/{k} chunks for topic '{topic}'")
        
    return results


# ================== RELAXED FILTERING (for agentic) ==================
def get_relevant_chunks_relaxed(query, index, metas, model, topic=None, k=5):
    """
    Vector search + RELAXED topic filtering - prefer same topic but accept others
    Used for agentic interview expected answers
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Search more than needed
    _, I = index.search(query_embedding, k * 3)

    results = []
    seen_ids = set()
    
    # First pass: collect same-topic chunks
    for idx in I[0]:
        meta = metas[idx]
        
        if meta["id"] in seen_ids:
            continue
            
        # If topic matches, add it
        if topic and meta["topic"] == topic:
            seen_ids.add(meta["id"])
            results.append(meta)
            if len(results) == k:
                return results
    
    # Second pass: if we need more chunks, add any relevant ones
    if len(results) < k:
        for idx in I[0]:
            meta = metas[idx]
            
            if meta["id"] in seen_ids:
                continue
                
            seen_ids.add(meta["id"])
            results.append(meta)
            if len(results) == k:
                break
    
    return results


# ================== GENERATE TECHNICAL EXPLANATION (DETAILED) ==================
def generate_technical_explanation(query, context, topic=None):
    """
    Generate an IN-DEPTH technical explanation for learning purposes.
    Used by the Technical Interview Chatbot.
    """
    topic_instruction = ""
    if topic:
        topic_instruction = f"The question is about {topic}. ONLY discuss {topic} concepts."
    
    prompt = f"""
You are an expert Computer Science educator providing a DETAILED explanation to help a student learn.

Topic: {topic if topic else 'Computer Science'}
{topic_instruction}

RULES:
- Provide a THOROUGH, EDUCATIONAL explanation (4-6 sentences)
- Include examples where helpful
- Explain core concepts clearly
- Connect ideas to help understanding
- DO NOT mention unrelated topics
- Focus ONLY on {topic if topic else 'the relevant topic'}

Context from knowledge base:
{context}

Student Question: {query}

Detailed Educational Answer:
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


# ================== GENERATE EXPECTED ANSWER (CONCISE) ==================
def generate_expected_answer(query, context, topic=None):
    """
    Generate a CONCISE expected answer (what a human would say).
    Used by the Agentic Interview for comparison.
    """
    topic_instruction = ""
    if topic:
        topic_instruction = f"The question is about {topic}."
    
    prompt = f"""
You are an expert in {topic if topic else 'Computer Science'} providing a MODEL ANSWER.
This is what a knowledgeable person would say in an interview.

RULES:
- Keep it VERY BRIEF - 2-3 sentences maximum
- Sound like a knowledgeable person, not a textbook
- Include key technical terms naturally
- DO NOT use markdown or formatting
- DO NOT explain concepts in depth - just state the core answer
- Be conversational, not academic

Context for reference:
{context}

Interview Question: {query}

Expected answer (concise, 2-3 sentences, human-like):
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


# ================== MAIN ENTRY POINT FOR TECHNICAL INTERVIEW ==================
def technical_interview_query(user_query):
    """
    Main function for Technical Interview Chatbot.
    If topic detected → Return DETAILED RAG explanation
    If no topic detected → Let LLM answer naturally
    """
    # Load resources
    topic_rules = load_json(TOPIC_RULES_PATH)
    index, metas = load_index_and_metas()
    kb_lookup = build_kb_lookup()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Topic detection
    topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)

    # 🔥 NO TOPIC DETECTED - Let LLM answer naturally
    if not topic:
        print("🧠 No topic detected - letting LLM respond naturally")
        
        prompt = f"""
You are a helpful Computer Science educator. The user said: "{user_query}"

Respond naturally and conversationally. If it's a question, answer it. If it's a greeting, greet back.
Be friendly and helpful.
"""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                answer = response.json()["response"].strip()
                return answer, []  # No sources
        except:
            return "Hi! I'm here to help with Computer Science topics. Ask me about DBMS, OS, or OOPs!", []

    # 🔥 TOPIC DETECTED - Use RAG for detailed explanation
    print(f"🧠 Detected Topic: {topic} | Subtopic: {subtopic}")
    augmented_query = f"{topic} {subtopic}: {user_query}"
    
    # Retrieve relevant chunks
    retrieved = get_relevant_chunks_strict(
        augmented_query,
        index,
        metas,
        embedder,
        topic=topic,
        k=5
    )

    # Build context
    context_blocks = []
    for meta in retrieved:
        item = kb_lookup.get(meta["id"])
        if item:
            context_blocks.append(f"Q: {item['question']}\nA: {item['answer']}")
    
    context_text = "\n\n".join(context_blocks)

    # Generate detailed explanation
    answer = generate_technical_explanation(user_query, context_text, topic)

    print("\n================ TECHNICAL EXPLANATION ================\n")
    print(answer)
    print("\n================ SOURCES ===================\n")
    for meta in retrieved:
        print(f"- {meta['id']} | {meta['topic']} > {meta['subtopic']}")

    return answer, retrieved


# ================== MAIN ENTRY POINT FOR AGENTIC INTERVIEW ==================
def agentic_expected_answer(user_query):
    """
    Main function for Agentic Interview.
    Returns CONCISE expected answers for comparison.
    """
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

    # Retrieve with RELAXED filtering (prefer same topic, accept others if needed)
    retrieved = get_relevant_chunks_relaxed(
        augmented_query,
        index,
        metas,
        embedder,
        topic=topic,
        k=3  # Need fewer sources for concise answer
    )

    # Build context
    context_blocks = []
    for meta in retrieved:
        item = kb_lookup.get(meta["id"])
        if not item:
            continue
        context_blocks.append(
            f"Q: {item['question']}\nA: {item['answer']}"
        )

    context_text = "\n\n".join(context_blocks)

    # Generate CONCISE expected answer
    answer = generate_expected_answer(user_query, context_text, topic)

    print("\n================ EXPECTED ANSWER (Concise) ================\n")
    print(answer)

    print("\n================ SOURCES ===================\n")
    for meta in retrieved:
        print(f"- {meta['id']} | {meta['topic']} > {meta['subtopic']}")

    return answer, retrieved


# ================== BACKWARD COMPATIBILITY ==================
# Keep original main for existing code that might call it
def main(user_query):
    """Legacy function - defaults to technical explanation"""
    return technical_interview_query(user_query)