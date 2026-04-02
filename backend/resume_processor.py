import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import PyPDF2
import docx
import re
from datetime import datetime
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

# ================== TEXT EXTRACTION ==================
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return ""

# ================== MISTRAL LLM EXTRACTION ==================

@lru_cache(maxsize=100)
def call_mistral_extraction(text_hash, text):
    """Call Mistral API for resume extraction with caching"""
    
    # Truncate text to 8000 chars (safe for Mistral context)
    truncated_text = text[:8000]
    
    system_prompt = """You are an expert Resume Parser. Extract information accurately from resumes.
    Return ONLY valid JSON. Do not invent information. If a field is missing, use null or empty array.
    
    IMPORTANT RULES:
    1. experiences: ONLY include professional jobs and internships. DO NOT include projects here.
    2. projects: Include personal, academic, or work projects.
    3. skills: Extract technical skills only (programming languages, frameworks, tools, databases).
    4. certifications: Include professional certifications and completed courses.
    5. dates: Use format "MM/YYYY" or "Present" for current roles.
    6. total_experience_years: Calculate from all experiences combined."""
    
    user_prompt = f"""Extract the following information from this resume:

RESUME TEXT:
{truncated_text}

Return JSON with exactly this structure:
{{
    "name": "Full name from resume",
    "email": "email address",
    "phone": "phone number",
    "skills": ["skill1", "skill2", "skill3"],
    "total_experience_years": 0.0,
    "education": {{
        "degree": "Degree name",
        "institution": "University/College name",
        "year": "Year of graduation"
    }},
    "experiences": [
        {{
            "title": "Job title",
            "company": "Company name",
            "start_date": "MM/YYYY",
            "end_date": "MM/YYYY or Present",
            "description": "Brief description of responsibilities"
        }}
    ],
    "projects": [
        {{
            "name": "Project name",
            "tech_stack": ["tech1", "tech2"],
            "description": "Brief description"
        }}
    ],
    "certifications": ["certification1", "certification2"],
    "courses": ["course1", "course2"]
}}

Extract only what is present. Return valid JSON only, no other text."""
    
    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        # Clean the response
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        return json.loads(content)
        
    except Exception as e:
        print(f"Mistral API error: {e}")
        return None

def extract_with_mistral(text):
    """Extract resume data using Mistral LLM"""
    # Create a hash for caching
    text_hash = hash(text[:1000])
    result = call_mistral_extraction(text_hash, text)
    
    if result:
        print("✅ Mistral extraction successful")
        return result
    else:
        print("⚠️ Mistral extraction failed, using fallback")
        return fallback_extraction(text)

def fallback_extraction(text):
    """Fallback extraction using regex (if Mistral fails)"""
    result = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "total_experience_years": 0,
        "education": {"degree": None, "institution": None, "year": None},
        "experiences": [],
        "projects": [],
        "certifications": [],
        "courses": []
    }
    
    # Extract email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        result["email"] = email_match.group()
    
    # Extract phone
    phone_match = re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if not phone_match:
        phone_match = re.search(r'\d{10}', text)
    if phone_match:
        result["phone"] = phone_match.group()
    
    # Extract name (first 5 lines, capitalized words)
    lines = text.split('\n')[:10]
    for line in lines:
        line = line.strip()
        if line and len(line.split()) <= 4:
            words = line.split()
            if all(w[0].isupper() for w in words if w):
                result["name"] = line
                break
    
    # Extract skills (common tech keywords)
    tech_skills = [
        "python", "java", "javascript", "react", "node.js", "sql", "mongodb",
        "express", "django", "flask", "aws", "docker", "git", "github",
        "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
        "html", "css", "typescript", "angular", "vue", "c++", "c#",
        "php", "ruby", "go", "rust", "kotlin", "swift", "langchain"
    ]
    
    text_lower = text.lower()
    for skill in tech_skills:
        if skill in text_lower:
            result["skills"].append(skill.title())
    
    result["skills"] = list(set(result["skills"]))[:30]
    
    return result

# ================== MAIN PARSING FUNCTION ==================

def parse_resume_file(file_path, file_type):
    """
    Parse resume file using Mistral LLM for intelligent extraction.
    This is the main function called by the API.
    """
    
    # Step 1: Extract raw text from file
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'docx':
        text = extract_text_from_docx(file_path)
    else:
        text = ""
    
    if not text:
        print("ERROR: No text extracted from resume")
        return None
    
    print(f"📄 Extracted {len(text)} characters from resume")
    
    # Step 2: Use Mistral LLM to extract structured data
    extracted_data = extract_with_mistral(text)
    
    if not extracted_data:
        print("ERROR: Failed to extract data")
        return None
    
    # Step 3: Format the output for frontend compatibility
    result = {
        "skills": extracted_data.get("skills", []),
        "projects": extracted_data.get("projects", []),
        "experience": extracted_data.get("experiences", []),
        "experience_years": extracted_data.get("total_experience_years", 0),
        "certifications": extracted_data.get("certifications", []),
        "courses": extracted_data.get("courses", []),
        "name": extracted_data.get("name"),
        "email": extracted_data.get("email"),
        "phone": extracted_data.get("phone"),
        "education": extracted_data.get("education", {}),
        "full_text": text[:3000]
    }
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 MISTRAL PARSING SUMMARY")
    print(f"{'='*50}")
    print(f"👤 Name: {result['name']}")
    print(f"📧 Email: {result['email']}")
    print(f"📞 Phone: {result['phone']}")
    print(f"💪 Skills: {len(result['skills'])} found")
    print(f"💼 Experience: {len(result['experience'])} entries")
    print(f"📁 Projects: {len(result['projects'])} found")
    print(f"🏅 Certifications: {len(result['certifications'])} found")
    print(f"📚 Courses: {len(result['courses'])} found")
    print(f"📅 Total Experience: {result['experience_years']} years")
    print(f"{'='*50}\n")
    
    return result

# ================== FAISS PROCESSING ==================

def process_resume_for_faiss(resume_text, user_id):
    """Process resume text with FAISS - create chunks and store embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(resume_text)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)

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
        index = faiss.read_index(index_path)
        metas = load_json(metas_path)

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        query_embedding = np.array([query_embedding]).astype('float32')

        scores, indices = index.search(query_embedding, min(top_k, len(metas)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metas):
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

def store_jd_embedding(job_description, user_id):
    """Store job description embedding for later use in interviews"""
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = embedder.encode([job_description], normalize_embeddings=True)[0]
        
        jd_path = os.path.join(RESUME_DATA_DIR, f"jd_embedding_{user_id}.npy")
        np.save(jd_path, embedding)
        
        jd_text_path = os.path.join(RESUME_DATA_DIR, f"jd_text_{user_id}.txt")
        with open(jd_text_path, "w", encoding="utf-8") as f:
            f.write(job_description)
        
        print(f"✅ JD embedding stored for user {user_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to store JD embedding: {e}")
        return False

def get_jd_embedding(user_id):
    """Retrieve job description embedding for a user"""
    jd_path = os.path.join(RESUME_DATA_DIR, f"jd_embedding_{user_id}.npy")
    jd_text_path = os.path.join(RESUME_DATA_DIR, f"jd_text_{user_id}.txt")
    
    if not os.path.exists(jd_path) or not os.path.exists(jd_text_path):
        return None, None
    
    try:
        embedding = np.load(jd_path)
        with open(jd_text_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
        return embedding, jd_text
    except Exception as e:
        print(f"❌ Failed to load JD embedding: {e}")
        return None, None

def delete_jd_data(user_id):
    """Delete JD embedding data for a user"""
    jd_path = os.path.join(RESUME_DATA_DIR, f"jd_embedding_{user_id}.npy")
    jd_text_path = os.path.join(RESUME_DATA_DIR, f"jd_text_{user_id}.txt")
    
    try:
        if os.path.exists(jd_path):
            os.remove(jd_path)
        if os.path.exists(jd_text_path):
            os.remove(jd_text_path)
        return True
    except Exception as e:
        print(f"❌ Error deleting JD data: {e}")
        return False

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