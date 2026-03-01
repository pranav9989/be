import os
import json
import faiss
import time
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# ================== ENV SETUP ==================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from mistralai import Mistral

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-large-latest"

# ================== PATHS ==================
FAISS_DIR = "data/processed/faiss_mistral"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METAS_PATH = os.path.join(FAISS_DIR, "metas.json")

KB_CLEAN_PATH = "data/processed/kb_clean.json"
TOPIC_RULES_PATH = "config/topic_rules.json"

# ================== DOMAIN RESTRICTION ==================
# Only allow these domains
ALLOWED_TOPICS = {"Operating Systems", "DBMS", "OOP"}

# ================== TOPIC ALIASES (CRITICAL FIX) ==================
TOPIC_ALIASES = {
    "OS": "Operating Systems",
    "Operating System": "Operating Systems",
    "Operating Systems": "Operating Systems",
    
    "DBMS": "DBMS",
    "Database": "DBMS",
    "Databases": "DBMS",
    
    "OOP": "OOP",
    "OOPS": "OOP",
    "Object Oriented Programming": "OOP",
}

OUT_OF_DOMAIN_MESSAGE = (
    "I can only answer questions related to Operating Systems, DBMS, and Object-Oriented Programming. "
    "Please ask a question from one of these domains."
)


# ================== GLOBAL CACHE ==================
# Cache for models and data to avoid repeated loading
_INDEX_CACHE = None
_METAS_CACHE = None
_KB_LOOKUP_CACHE = None
_EMBEDDER_CACHE = None
_TOPIC_RULES_CACHE = None
_CACHE_LOADED = False  # Flag to track if cache has been loaded


def get_embedder():
    """Get cached sentence transformer embedder"""
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is None:
        print("🔄 Loading sentence transformer model (cached)...")
        _EMBEDDER_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER_CACHE


def get_topic_rules():
    """Get cached topic rules"""
    global _TOPIC_RULES_CACHE
    if _TOPIC_RULES_CACHE is None:
        _TOPIC_RULES_CACHE = load_json(TOPIC_RULES_PATH)
        print(f"📚 Loaded topic rules from {TOPIC_RULES_PATH}")
    return _TOPIC_RULES_CACHE


# ================== LOADERS ==================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index_and_metas():
    """Load FAISS index and metas with caching"""
    global _INDEX_CACHE, _METAS_CACHE, _CACHE_LOADED
    
    if _INDEX_CACHE is None or _METAS_CACHE is None:
        print(f"📚 Loading FAISS index from {INDEX_PATH}")
        _INDEX_CACHE = faiss.read_index(INDEX_PATH)
        _METAS_CACHE = load_json(METAS_PATH)
        
        # Count topics for debugging (only once)
        topic_counts = {}
        for meta in _METAS_CACHE:
            topic = meta.get('topic', 'Unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        print(f"   Loaded {len(_METAS_CACHE)} total chunks")
        for topic, count in topic_counts.items():
            print(f"      - {topic}: {count} chunks")
    
    return _INDEX_CACHE, _METAS_CACHE


def build_kb_lookup():
    """Build knowledge base lookup with caching"""
    global _KB_LOOKUP_CACHE
    
    if _KB_LOOKUP_CACHE is None:
        kb = load_json(KB_CLEAN_PATH)
        print(f"📚 Loaded knowledge base with {len(kb)} items")
        _KB_LOOKUP_CACHE = {item["id"]: item for item in kb}
    
    return _KB_LOOKUP_CACHE


# ================== TOPIC DETECTION ==================
def get_topic_and_subtopic_from_query(query, topic_rules=None):
    if topic_rules is None:
        topic_rules = get_topic_rules()
    
    q = query.lower()
    for rule in topic_rules:
        for kw in rule["keywords"]:
            if kw in q:
                print(f"   Detected topic: {rule['topic']} -> {rule['subtopic']}")
                return rule["topic"], rule["subtopic"]
    return None, None


# ================== FIX 7: TOPIC DETECTION FUNCTION ==================
def detect_topic(question: str, expected_topic: str = None):
    """
    Detect topic with fallback to expected topic if provided
    Used by adaptive_question_bank to ensure topic consistency
    """
    # Use RAG voting to detect topic
    topic, confidence = detect_topic_via_rag(question)
    
    # If we have an expected topic and RAG detected something different, log it
    if expected_topic and topic and topic != expected_topic:
        print(f"⚠️ RAG topic mismatch: RAG={topic}, expected={expected_topic} → overridden to {expected_topic}")
        return expected_topic
    
    # Return RAG detected topic, or expected if RAG failed
    if topic:
        return topic
    return expected_topic


# ================== STRICT TOPIC FILTERING ==================
def get_relevant_chunks_strict(query, index, metas, model, topic=None, k=5):
    print(f"\n🔍 Searching for relevant chunks (k={k})...")
    
    query_embedding = model.encode([query], normalize_embeddings=True)
    start_time = time.time()
    scores, I = index.search(query_embedding, k * 8)
    search_time = time.time() - start_time
    
    results = []
    seen_ids = set()
    
    for idx, score in zip(I[0], scores[0]):
        if idx < 0 or idx >= len(metas):
            continue
        meta = metas[idx]
        if topic is not None and meta["topic"] != topic:
            continue
        if meta["id"] in seen_ids:
            continue
        seen_ids.add(meta["id"])
        meta_copy = meta.copy()
        meta_copy["_score"] = float(score)
        results.append(meta_copy)
        if len(results) == k:
            break
    
    return results


# ================== GENERATE TECHNICAL EXPLANATION ==================
def generate_technical_explanation(query, context, topic=None):
    """
    Generate an IN-DEPTH technical explanation for learning purposes.
    Used by the Technical Interview Chatbot.
    """
    # Normalize topic if needed
    if topic:
        normalized_topic = TOPIC_ALIASES.get(topic, topic)
    else:
        normalized_topic = None
    
    # STRICT DOMAIN ENFORCEMENT
    if normalized_topic and normalized_topic not in ALLOWED_TOPICS:
        print(f"❌ Explanation blocked for topic: {normalized_topic}")
        return OUT_OF_DOMAIN_MESSAGE
    
    topic_instruction = ""
    if normalized_topic:
        topic_instruction = f"The question is about {normalized_topic}. ONLY discuss {normalized_topic} concepts."
    
    prompt = f"""
You are an expert Computer Science educator providing a DETAILED explanation to help a student learn.

Topic: {normalized_topic if normalized_topic else 'Computer Science'}
{topic_instruction}

RULES:
- Provide a THOROUGH, EDUCATIONAL explanation (4-6 sentences)
- ONLY discuss concepts from the specified topic
- DO NOT answer if topic is outside Operating Systems, DBMS, or OOP
- DO NOT mention unrelated domains like networking, ML, web, etc.
- If question is outside allowed domains, refuse politely

Context from knowledge base:
{context}

Student Question: {query}

Detailed Educational Answer:
"""

    print(f"\n🪄 Generating detailed explanation via Mistral...")
    start_time = time.time()
    
    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        elapsed = time.time() - start_time
        answer = response.choices[0].message.content.strip()
        print(f"✅ Generated answer in {elapsed:.1f}s")
        return answer
            
    except Exception as e:
        print(f"❌ Mistral error: {e}")
        return f"❌ Error triggering Mistral: {str(e)}"

# ================== GENERIC MISTRAL GENERATION ==================
def mistral_generate(prompt, timeout=120):
    print(f"\n🪄 Generating response via Mistral API...")
    start_time = time.time()
    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = time.time() - start_time
        text = response.choices[0].message.content.strip()
        print(f"✅ Generation complete in {elapsed:.1f}s")
        return text
    except Exception as e:
        print(f"❌ Error generating Mistral response: {e}")
        return None

def ollama_generate(prompt, timeout=120):
    return mistral_generate(prompt, timeout)


# ================== GENERATE INTERVIEW QUESTION ==================
def generate_interview_question(prompt, topic=None):
    full_prompt = f"""{prompt}

CRITICAL INSTRUCTION:
- Return ONLY the question text
- NO introductions or commentary
- Maximum 400 characters

Question:"""

    print(f"\n🪄 Generating interview question via Mistral...")
    start_time = time.time()
    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3
        )
        elapsed = time.time() - start_time
        question = response.choices[0].message.content.strip()
        return question
    except Exception as e:
        print(f"❌ Mistral interview error: {e}")
        return None


# ================== RAG VOTING TOPIC DETECTION ==================
def retrieve_similar_qas(query, topic=None, k=3):
    """
    Retrieve similar Q&A pairs from FAISS
    Used for few-shot examples in question generation
    """
    try:
        # Load data (now cached)
        topic_rules = get_topic_rules()
        index, metas = load_index_and_metas()
        kb_lookup = build_kb_lookup()
        embedder = get_embedder()  # Use cached embedder
        
        # Normalize topic if provided
        normalized_topic = None
        if topic:
            normalized_topic = TOPIC_ALIASES.get(topic, topic)
        
        # Detect topic if not provided
        if not normalized_topic:
            detected_topic, _ = get_topic_and_subtopic_from_query(query, topic_rules)
            normalized_topic = TOPIC_ALIASES.get(detected_topic, detected_topic) if detected_topic else None
        
        # Get embedding
        query_emb = embedder.encode([query], normalize_embeddings=True)
        
        # Search
        scores, I = index.search(query_emb, k * 3)
        
        results = []
        seen_ids = set()
        
        for idx, score in zip(I[0], scores[0]):
            if idx < 0 or idx >= len(metas):
                continue
                
            meta = metas[idx]
            
            # Filter by topic if specified
            if normalized_topic and meta.get("topic") != normalized_topic:
                continue
            
            item_id = meta["id"]
            if item_id in seen_ids:
                continue
                
            seen_ids.add(item_id)
            
            # Get full item from KB
            item = kb_lookup.get(item_id, {})
            
            results.append({
                "id": item_id,
                "question": item.get("question", meta.get("text", ""))[:150],
                "answer": item.get("answer", ""),
                "topic": meta.get("topic"),
                "subtopic": meta.get("subtopic"),
                "score": float(score)
            })
            
            if len(results) >= k:
                break
        
        print(f"📤 Retrieved {len(results)} similar Q&A examples")
        return results
        
    except Exception as e:
        print(f"⚠️ Error in retrieve_similar_qas: {e}")
        return []


def retrieve_relevant_chunks(query, k=5, topic=None):
    """
    Retrieve relevant chunks for a query
    Used for expected answer generation and gap analysis
    
    Args:
        query: The search query
        k: Number of chunks to retrieve
        topic: Optional topic filter (if provided, only chunks from this topic are returned)
    """
    try:
        index, metas = load_index_and_metas()
        kb_lookup = build_kb_lookup()
        embedder = get_embedder()  # Use cached embedder
        
        # Normalize topic if provided
        normalized_topic = None
        if topic:
            normalized_topic = TOPIC_ALIASES.get(topic, topic)
            print(f"   Normalized topic: {topic} → {normalized_topic}")
        
        query_emb = embedder.encode([query], normalize_embeddings=True)
        
        # Search with higher k to allow for filtering
        search_k = k * 3 if normalized_topic else k
        scores, I = index.search(query_emb, search_k)
        
        results = []
        seen_ids = set()
        
        for idx, score in zip(I[0], scores[0]):
            if idx < 0 or idx >= len(metas):
                continue
                
            meta = metas[idx]
            
            # Filter by topic if specified
            if normalized_topic and meta.get("topic") != normalized_topic:
                continue
                
            item_id = meta["id"]
            if item_id in seen_ids:
                continue
                
            seen_ids.add(item_id)
            item = kb_lookup.get(item_id, {})
            
            results.append({
                "id": item_id,
                "text": meta.get("text", ""),
                "answer": item.get("answer", meta.get("text", "")),
                "topic": meta.get("topic"),
                "subtopic": meta.get("subtopic"),
                "score": float(score)
            })
            
            if len(results) >= k:
                break
        
        print(f"📤 Retrieved {len(results)} relevant chunks for topic {normalized_topic if normalized_topic else 'all'}")
        return results
        
    except Exception as e:
        print(f"⚠️ Error in retrieve_relevant_chunks: {e}")
        return []


def detect_topic_via_rag(query, k=5):
    """
    RAG VOTING TOPIC DETECTION
    Returns topic based on majority vote from retrieved chunks
    """
    print(f"\n🔍 Detecting topic via RAG voting for: {query[:50]}...")
    
    try:
        # Load data (cached)
        index, metas = load_index_and_metas()
        embedder = get_embedder()  # Use cached embedder
        
        # Get embedding
        query_emb = embedder.encode([query], normalize_embeddings=True)
        
        # Search
        scores, I = index.search(query_emb, k)
        
        # Count votes per topic
        votes = {}
        for idx in I[0]:
            if idx >= 0 and idx < len(metas):
                topic = metas[idx].get("topic", "Unknown")
                votes[topic] = votes.get(topic, 0) + 1
        
        if not votes:
            print("   No votes received")
            return None, 0.0
        
        # Get winner
        topic = max(votes, key=votes.get)
        confidence = votes[topic] / k
        
        print(f"   Votes: {votes}")
        print(f"   Winner: {topic} (confidence: {confidence:.2f})")
        
        return topic, confidence
        
    except Exception as e:
        print(f"⚠️ Error in detect_topic_via_rag: {e}")
        return None, 0.0


# ================== MAIN ENTRY POINT ==================
# ================== MAIN ENTRY POINT FOR CHATBOT ==================
def technical_interview_query(user_query):
    """
    Enhanced chatbot query with topic detection and filtering
    """
    print("\n" + "="*80)
    print(f"📝 CHATBOT QUERY: {user_query}")
    print("="*80)
    
    # Load all required data (cached)
    topic_rules = get_topic_rules()
    index, metas = load_index_and_metas()
    kb_lookup = build_kb_lookup()
    embedder = get_embedder()

    # STEP 1: Detect the primary topic from the query
    detected_topic, detected_subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)
    
    # Normalize detected topic
    if detected_topic:
        detected_topic = TOPIC_ALIASES.get(detected_topic, detected_topic)
    
    # ================== STRICT DOMAIN ENFORCEMENT ==================
    if detected_topic is None:
        print("❌ Query rejected: No valid CS domain detected")
        return OUT_OF_DOMAIN_MESSAGE, []

    if detected_topic not in ALLOWED_TOPICS:
        print(f"❌ Query rejected: {detected_topic} is not allowed")
        return OUT_OF_DOMAIN_MESSAGE, []
    
    if detected_topic:
        print(f"\n🔍 Detected topic: {detected_topic} (subtopic: {detected_subtopic})")
    else:
        print(f"\n⚠️ No specific topic detected, will use pure semantic search")
    
    # STEP 2: Get embeddings and search
    query_embedding = embedder.encode([user_query], normalize_embeddings=True)
    
    # Secondary protection via RAG voting
    rag_topic, confidence = detect_topic_via_rag(user_query)
    
    if rag_topic and rag_topic not in ALLOWED_TOPICS:
        print(f"❌ RAG voting rejected topic: {rag_topic}")
        return OUT_OF_DOMAIN_MESSAGE, []
    
    # Search with higher k to allow for filtering
    search_k = 15  # Search more, then filter down
    scores, I = index.search(query_embedding, search_k)
    
    # STEP 3: Filter results by topic if detected
    retrieved = []
    seen_ids = set()
    
    for idx, score in zip(I[0], scores[0]):
        if idx < 0 or idx >= len(metas):
            continue
            
        meta = metas[idx]
        
        # STRICT domain filter
        if meta.get("topic") not in ALLOWED_TOPICS:
            continue
        
        # Skip if we've seen this ID
        if meta["id"] in seen_ids:
            continue
        
        # If topic was detected, prioritize chunks from that topic
        if detected_topic and meta.get("topic") != detected_topic:
            # Still allow but with lower priority - we'll collect them separately
            continue
            
        seen_ids.add(meta["id"])
        meta_copy = meta.copy()
        meta_copy["_score"] = float(score)
        retrieved.append(meta_copy)
        
        # Stop if we have enough from the detected topic
        if len(retrieved) >= 5:
            break
    
    # STEP 4: If we don't have enough from detected topic, add best matches from other allowed topics
    if len(retrieved) < 5:
        print(f"   Only found {len(retrieved)} chunks from {detected_topic}, adding best from other allowed topics...")
        for idx, score in zip(I[0], scores[0]):
            if idx < 0 or idx >= len(metas):
                continue
                
            meta = metas[idx]
            
            # STRICT domain filter
            if meta.get("topic") not in ALLOWED_TOPICS:
                continue
            
            # Skip if already seen
            if meta["id"] in seen_ids:
                continue
            
            # Add this chunk
            seen_ids.add(meta["id"])
            meta_copy = meta.copy()
            meta_copy["_score"] = float(score)
            retrieved.append(meta_copy)
            
            if len(retrieved) >= 5:
                break
    
    print(f"\n🔍 Retrieved {len(retrieved)} relevant chunks:")
    for i, r in enumerate(retrieved):
        print(f"   {i+1}. [{r.get('topic')}/{r.get('subtopic')}] score={r['_score']:.3f}")

    # STEP 5: Build context from retrieved chunks
    context_blocks = []
    for meta in retrieved:
        item = kb_lookup.get(meta["id"])
        if item and item.get("answer"):
            # Add a small topic marker to help the LLM understand the source
            topic_marker = f"[{meta.get('topic')} - {meta.get('subtopic')}]"
            answer_text = item.get("answer", "")
            context_blocks.append(f"{topic_marker}\n{answer_text}")
    
    context_text = "\n\n".join(context_blocks)
    print(f"\n📚 Context built from {len(context_blocks)} chunks ({len(context_text)} chars)")
    
    # STEP 6: Generate detailed answer with topic context
    answer = generate_technical_explanation(
        user_query, 
        context_text,
        topic=detected_topic  # Pass the detected topic for better prompting
    )
    
    print("="*80 + "\n")
    
    return answer, retrieved


# ================== AGENTIC INTERVIEW ==================
def agentic_expected_answer(user_query, sampled_concepts=None, expected_topic=None):
    """
    Generate concise expected answer for adaptive interviews
    Uses controller-provided topic with alias normalization
    """
    print(f"\n📝 GENERATING EXPECTED ANSWER for: {user_query[:50]}...")
    if sampled_concepts:
        print(f"   Sampled concepts: {sampled_concepts}")
    
    # 🔥 FIX 2: Handle missing expected_topic gracefully
    if expected_topic is None:
        print("⚠️ expected_topic missing — using RAG detection fallback")
        detected_topic, confidence = detect_topic_via_rag(user_query)
        
        if detected_topic is None:
            print("❌ Could not detect topic — defaulting to Operating Systems")
            expected_topic = "Operating Systems"
        else:
            expected_topic = detected_topic
            print(f"📌 RAG detected topic: {expected_topic} (confidence: {confidence:.2f})")
    
    # 🔥 FIX 1: Normalize topic using aliases
    normalized_topic = TOPIC_ALIASES.get(expected_topic, expected_topic)
    print(f"📌 Using normalized topic: {normalized_topic} (from: {expected_topic})")
    
    # 🔥 FIX 3: Normalize topic BEFORE retrieval
    chunks = retrieve_relevant_chunks(
        user_query,
        k=5,
        topic=normalized_topic  # Pass normalized topic
    )
    
    # Filter chunks by topic (additional safety)
    chunks = [c for c in chunks if c.get("topic") == normalized_topic]
    print(f"   Retrieved {len(chunks)} chunks for topic {normalized_topic}")
    
    # Build context
    context_blocks = []
    for chunk in chunks:
        answer_text = chunk.get("answer", chunk.get("text", ""))
        if answer_text:
            context_blocks.append(answer_text)
    
    # 🔥 FIX 4: Enhanced context with better structure
    concept_list = ", ".join(sampled_concepts) if sampled_concepts else "key concepts"
    
    context_text = f"""
Topic: {normalized_topic}
Required concepts: {concept_list}

Knowledge:
{chr(10).join(context_blocks[:3])}
"""
    
    prompt = f"""
You are an expert technical interviewer.

Generate a concise expected answer.

REQUIREMENTS:
• Length: 2-4 sentences
• Must mention ALL required concepts explicitly: {concept_list}
• Must use provided context only
• Must be technically accurate and precise
• Do NOT add explanations or meta-commentary
• Return ONLY the answer text

Topic: {normalized_topic}
Question: {user_query}

Relevant context:
{context_text}

Expected Answer:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 300}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            answer = response.json()["response"].strip()
            print(f"✅ Generated expected answer ({len(answer)} chars)")
            
            # Verify concepts
            if sampled_concepts:
                answer_lower = answer.lower()
                missing = []
                present = []
                for c in sampled_concepts:
                    c_norm = c.lower()
                    if c_norm in answer_lower:
                        present.append(c)
                    else:
                        missing.append(c)
                
                if missing:
                    print(f"   ⚠️ Missing concepts in generated answer: {missing}")
                else:
                    print(f"   ✓ All concepts present: {present}")
            
            return answer, chunks
        else:
            print(f"❌ Failed to generate: {response.status_code}")
            return "", chunks
            
    except Exception as e:
        print(f"❌ Error generating expected answer: {e}")
        return "", chunks


# ================== RESUME GAP ANALYSIS ==================
def generate_resume_gap_analysis(resume_data, job_description):
    print(f"\n🪄 Generating Resume Gap Analysis via Mistral...")
    start_time = time.time()
    
    prompt = f"""You are an expert tech recruiter and career coach.
Your task is to compare a candidate's resume data against a target Job Description.
Identify the gaps in their skills and provide a 2-week study plan to help them prepare.

Candidate Resume Data:
{json.dumps(resume_data, indent=2)}

Target Job Description:
{job_description}

You MUST return your response as a valid JSON object with the following exact structure (do not include markdown block formatting, just the raw JSON):
{{
  "match_score": <an integer between 0 and 100 representing how well they fit>,
  "missing_skills": ["skill1", "skill2", "skill3"],
  "study_plan": [
    {{"day": 1, "topic": "Name of topic", "description": "What to study"}},
    {{"day": 2, "topic": "Name of topic", "description": "What to study"}},
    ... (up to day 14)
  ]
}}
"""

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        elapsed = time.time() - start_time
        answer = response.choices[0].message.content.strip()
        print(f"✅ Generated gap analysis in {elapsed:.1f}s")
        return answer
    except Exception as e:
        print(f"❌ Mistral gap analysis error: {e}")
        return None

if __name__ == '__main__':
    print("RAG module loaded and ready.")