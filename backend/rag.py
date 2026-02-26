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
    print(f"📚 Loading FAISS index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    metas = load_json(METAS_PATH)
    
    # Count topics for better debugging
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
    print(f"   No topic detected")
    return None, None


# ================== STRICT TOPIC FILTERING ==================
def get_relevant_chunks_strict(query, index, metas, model, topic=None, k=5):
    print(f"\n🔍 Searching for relevant chunks (k={k})...")
    
    query_embedding = model.encode([query], normalize_embeddings=True)
    start_time = time.time()
    scores, I = index.search(query_embedding, k * 8)  # Search more to filter
    search_time = time.time() - start_time
    
    print(f"   Search completed in {search_time:.3f}s, found {len(I[0])} candidates")
    
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
        
        # Add similarity score to meta
        meta_copy = meta.copy()
        meta_copy["_score"] = float(score)
        results.append(meta_copy)
        
        if len(results) == k:
            break
    
    print(f"   Retrieved {len(results)} relevant chunks:")
    for i, r in enumerate(results):
        # Truncate text for display
        text_preview = r.get("text", "")[:80] + "..." if len(r.get("text", "")) > 80 else r.get("text", "")
        print(f"      {i+1}. [{r.get('topic')}/{r.get('subtopic')}] score={r['_score']:.3f}")
        print(f"         {text_preview}")
    
    return results


# ================== GENERATE TECHNICAL EXPLANATION ==================
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

    print(f"\n🤖 Generating detailed explanation for: {query[:50]}...")
    start_time = time.time()
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024
                }
            },
            timeout=120
        )

        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            answer = response.json()["response"].strip()
            print(f"✅ Generated answer in {elapsed:.1f}s ({len(answer)} chars)")
            
            # Log the answer (first few lines)
            print("\n📝 ANSWER PREVIEW:")
            preview_lines = answer.split('\n')[:5]
            for line in preview_lines:
                if line.strip():
                    print(f"   {line[:100]}")
            if len(answer.split('\n')) > 5:
                print("   ...")
            
            return answer
        else:
            print(f"❌ Ollama error: {response.status_code} - {response.text}")
            return f"❌ Ollama error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        print(f"❌ Ollama timeout after {time.time() - start_time:.1f}s")
        return "❌ Ollama timeout - please try again"
    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        return f"❌ Error: {str(e)}"


# ================== GENERATE INTERVIEW QUESTION ==================
def generate_interview_question(prompt, topic=None):
    """
    Generate a SINGLE interview question based on the prompt.
    STRICT: Returns ONLY the question text, no explanations, no introductions.
    Used by the adaptive question bank.
    """
    # Add strict instruction to the prompt
    full_prompt = f"""{prompt}

CRITICAL INSTRUCTION - FOLLOW EXACTLY:
- Return ONLY the question text - nothing else
- NO introductions like "Here's a question:" or "Interviewer:"
- NO explanations or commentary
- NO markdown formatting
- NO bullet points or numbering
- Just the question itself, ending with a question mark
- Maximum 400 characters

Question:"""

    print(f"\n🤖 Generating interview question...")
    start_time = time.time()
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            question = response.json()["response"].strip()
            print(f"✅ Generated question in {elapsed:.1f}s ({len(question)} chars)")
            print(f"   Preview: {question[:100]}...")
            return question
        else:
            print(f"❌ Ollama error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error generating question: {e}")
        return None

# ================== MAIN ENTRY POINT ==================
def technical_interview_query(user_query):
    """
    Main function for technical interview chatbot
    Returns DETAILED educational explanation
    """
    print("\n" + "="*80)
    print(f"📝 TECHNICAL QUERY: {user_query}")
    print("="*80)
    
    # Load all required data
    topic_rules = load_json(TOPIC_RULES_PATH)
    index, metas = load_index_and_metas()
    kb_lookup = build_kb_lookup()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Detect topic
    topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)

    if not topic:
        print("⚠️ No specific topic detected")
        return "I can help with technical topics like DBMS, OOPS, and OS. Please ask a specific question.", []

    # Retrieve relevant chunks
    retrieved = get_relevant_chunks_strict(
        user_query,
        index,
        metas,
        embedder,
        topic=topic,
        k=5  # Get more chunks for better context
    )

    # Build context from retrieved chunks
    context_blocks = []
    for meta in retrieved:
        item = kb_lookup.get(meta["id"])
        if item and item.get("topic") == topic:
            # Add the answer text to context
            answer_text = item.get("answer", "")
            if answer_text:
                context_blocks.append(answer_text)
    
    context_text = "\n\n".join(context_blocks)
    
    print(f"\n📚 Context built from {len(context_blocks)} chunks ({len(context_text)} chars)")
    
    # Generate detailed answer
    answer = generate_technical_explanation(
        user_query, 
        context_text, 
        topic
    )

    if not answer:
        # Fallback to simpler response
        answer = f"I can help you learn about {topic}. Please try rephrasing your question or ask about a specific concept."
        print(f"⚠️ Using fallback response")

    print("="*80 + "\n")
    
    return answer, retrieved


# ================== AGENTIC INTERVIEW ==================
# ================== AGENTIC INTERVIEW ==================
def agentic_expected_answer(user_query, sampled_concepts=None):
    """
    Generate concise expected answer for adaptive interviews
    This is used for scoring, so it should be focused and concise
    STRICT: Must explicitly mention the sampled concepts
    """
    print(f"\n📝 GENERATING EXPECTED ANSWER for: {user_query[:50]}...")
    if sampled_concepts:
        print(f"   Sampled concepts: {sampled_concepts}")
    
    topic_rules = load_json(TOPIC_RULES_PATH)
    index, metas = load_index_and_metas()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)

    retrieved = get_relevant_chunks_strict(
        user_query,
        index,
        metas,
        embedder,
        topic=topic,
        k=3
    )

    # Build context
    context_blocks = []
    for meta in retrieved:
        item = None
        # Try to get from kb_lookup if available
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
    
    # ========== STRICT PROMPT WITH CONCEPT ENFORCEMENT ==========
    concept_hint = ""
    concept_list_str = ""
    if sampled_concepts and len(sampled_concepts) > 0:
        concept_list = ", ".join(sampled_concepts)
        concept_hint = f"\nCRITICAL REQUIREMENT: Your answer MUST explicitly address these concepts: {concept_list}"
        concept_list_str = concept_list
    else:
        concept_list_str = "the key concepts"
    
    prompt = f"""Question: {user_query}{concept_hint}

Context information:
{context_text}

STRICT REQUIREMENTS - MUST FOLLOW EXACTLY:

1. Your answer MUST explicitly mention and explain: {concept_list_str}

2. Be concise (2-3 sentences)

3. Be technically accurate

4. Focus ONLY on the required concepts

Expected answer:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 200}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            answer = response.json()["response"].strip()
            print(f"✅ Generated expected answer ({len(answer)} chars)")
            print(f"   Preview: {answer[:100]}...")
            
            # Verify the answer contains the required concepts
            if sampled_concepts and len(sampled_concepts) > 0:
                answer_lower = answer.lower()
                missing_concepts = []
                for concept in sampled_concepts:
                    if concept.lower() not in answer_lower:
                        missing_concepts.append(concept)
                
                if missing_concepts:
                    print(f"   ⚠️ Warning: Expected answer missing concepts: {missing_concepts}")
                else:
                    print(f"   ✓ All required concepts present")
            
            return answer, retrieved
        else:
            print(f"❌ Failed to generate expected answer: {response.status_code}")
            return "", retrieved
            
    except Exception as e:
        print(f"❌ Error generating expected answer: {e}")
        return "", retrieved

if __name__ == '__main__':
    print("RAG module loaded and ready.")