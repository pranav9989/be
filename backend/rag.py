import os
import json
import faiss
import time
from sentence_transformers import SentenceTransformer

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


# ================== LOADERS ==================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index_and_metas():
    print(f"📚 Loading FAISS index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    metas = load_json(METAS_PATH)
    
    topic_counts = {}
    for meta in metas:
        topic = meta.get('topic', 'Unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"   Loaded {len(metas)} total chunks")
    for topic, count in topic_counts.items():
        print(f"      - {topic}: {count} chunks")
    return index, metas


def build_kb_lookup():
    kb = load_json(KB_CLEAN_PATH)
    print(f"📚 Loaded knowledge base with {len(kb)} items")
    return {item["id"]: item for item in kb}


# ================== TOPIC DETECTION ==================
def get_topic_and_subtopic_from_query(query, topic_rules):
    q = query.lower()
    for rule in topic_rules:
        for kw in rule["keywords"]:
            if kw in q:
                print(f"   Detected topic: {rule['topic']} -> {rule['subtopic']}")
                return rule["topic"], rule["subtopic"]
    return None, None


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
- Focus ONLY on {topic if topic else 'the relevant topic'}

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


# ================== MAIN ENTRY POINT ==================
def technical_interview_query(user_query):
    topic_rules = load_json(TOPIC_RULES_PATH)
    index, metas = load_index_and_metas()
    kb_lookup = build_kb_lookup()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)

    if not topic:
        return "I can help with technical topics like DBMS, OOPS, and OS.", []

    retrieved = get_relevant_chunks_strict(user_query, index, metas, embedder, topic=topic, k=5)

    context_blocks = []
    for meta in retrieved:
        item = kb_lookup.get(meta["id"])
        if item and item.get("topic") == topic:
            answer_text = item.get("answer", "")
            if answer_text:
                context_blocks.append(answer_text)
    
    context_text = "\n\n".join(context_blocks)
    answer = generate_technical_explanation(user_query, context_text, topic)

    return answer, retrieved


# ================== AGENTIC INTERVIEW ==================
def agentic_expected_answer(user_query, sampled_concepts=None):
    print(f"\n🪄 GENERATING EXPECTED ANSWER via Mistral...")
    start_time = time.time()
    
    topic_rules = load_json(TOPIC_RULES_PATH)
    index, metas = load_index_and_metas()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)
    retrieved = get_relevant_chunks_strict(user_query, index, metas, embedder, topic=topic, k=3)

    context_blocks = []
    for meta in retrieved:
        item = None
        try:
            kb_lookup = build_kb_lookup()
            item = kb_lookup.get(meta["id"])
        except:
            pass
        if item and item.get("answer"):
            context_blocks.append(item["answer"])
        else:
            context_blocks.append(meta.get("text", ""))
    
    context_text = "\n".join(context_blocks)
    
    concept_hint = ""
    if sampled_concepts:
        concept_hint = f"\nCRITICAL: Your answer MUST explain: {', '.join(sampled_concepts)}"
    
    prompt = f"""Question: {user_query}{concept_hint}
Context: {context_text}
Expected answer (2-3 sentences):"""

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        elapsed = time.time() - start_time
        answer = response.choices[0].message.content.strip()
        return answer, retrieved
    except Exception as e:
        print(f"❌ Mistral expected answer error: {e}")
        return "", retrieved

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