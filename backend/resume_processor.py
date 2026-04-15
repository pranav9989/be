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

# ================== ADD THIS TO resume_processor.py job fit analysis ==================

@lru_cache(maxsize=50)
def call_mistral_job_fit_analysis(resume_text_hash, resume_text, job_description):
    """
    Directly use Mistral to analyze job fit - NO intermediate JSON needed.
    This is the PRIMARY job fit analyzer.
    """
    
    # Truncate for context limits
    truncated_resume = resume_text[:6000]
    truncated_jd = job_description[:4000]
    
    system_prompt = """You are an expert Technical Recruiter and Career Coach at a FAANG company.
    
Your task: Analyze how well this candidate fits the job description.
Be HONEST, SPECIFIC, and ACTIONABLE.

Return ONLY valid JSON with the following structure:
{
    "match_score": <0-100, BE REALISTIC>,
    "fit_breakdown": {
        "technical_skills": <0-100>,
        "experience_level": <0-100>,
        "project_relevance": <0-100>,
        "communication_leadership": <0-100>
    },
    "strengths": [
        "<Specific strength #1 with explanation>",
        "<Specific strength #2>"
    ],
    "gaps": [
        "<Specific gap #1 - what's missing>",
        "<Specific gap #2>"
    ],
    "recommendations": {
        "immediate_actions": ["<Actionable step 1>", "<Actionable step 2>"],
        "learning_resources": [
            {"skill": "Topic name", "resource": "Coursera/Udemy/YouTube link"}
        ],
        "project_suggestions": ["<Project idea 1>", "<Project idea 2>"]
    },
    "interview_prep_focus": ["<Topic 1>", "<Topic 2>", "<Topic 3>"],
    "resume_improvements": ["<Specific change 1>", "<Specific change 2>"],
    "verdict": "<One sentence: ready or not?>",
    "preparation_time": "<e.g., '2 weeks', 'Ready now'>"
}

CRITICAL RULES:
1. Be HONEST - don't inflate scores
2. Reference SPECIFIC things from the resume
3. Give REAL resources (Coursera, freeCodeCamp, YouTube channels)
4. Consider transferable skills
5. If completely wrong for role, say so clearly"""

    user_prompt = f"""Analyze this candidate's fit for the job:

========================================
CANDIDATE RESUME:
========================================
{truncated_resume}

========================================
JOB DESCRIPTION:
========================================
{truncated_jd}

Return ONLY valid JSON with the analysis. No markdown, no extra text."""

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2500,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        # Clean response
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        analysis = json.loads(content)
        print(f"✅ LLM Job Fit Analysis complete - Score: {analysis.get('match_score', 'N/A')}")
        
        # Add semantic score as sanity check (optional)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            resume_emb = model.encode([truncated_resume[:2000]], normalize_embeddings=True)
            jd_emb = model.encode([truncated_jd], normalize_embeddings=True)
            from sklearn.metrics.pairwise import cosine_similarity
            semantic_score = float(cosine_similarity(resume_emb, jd_emb)[0][0]) * 100
            analysis["semantic_similarity_score"] = round(semantic_score, 1)
        except:
            analysis["semantic_similarity_score"] = analysis.get("match_score", 50)
        
        analysis["analysis_method"] = "mistral_llm_direct"
        
        return analysis
        
    except Exception as e:
        print(f"❌ Mistral job fit analysis failed: {e}")
        return None

def get_job_fit_analysis(resume_text, job_description):
    """
    Primary function to call - returns LLM analysis or falls back to keyword matching
    """
    if not job_description or not resume_text:
        return None
    
    # Create hash for caching
    text_hash = hash(resume_text[:500] + job_description[:500])
    
    # Try LLM analysis first
    llm_analysis = call_mistral_job_fit_analysis(text_hash, resume_text, job_description)
    
    if llm_analysis:
        return llm_analysis
    
    # Fallback to the existing keyword-based analysis
    print("⚠️ LLM analysis failed, falling back to keyword matching")
    
    # First parse resume with Mistral to get structured data
    parsed_data = call_mistral_extraction(hash(resume_text[:500]), resume_text)
    
    if parsed_data:
        # Call the existing analyze_resume_job_fit (you'll need to import or copy it)
        return analyze_resume_job_fit_fallback(parsed_data, job_description)
    
    return None

# ================== ENHANCED JOB FIT ANALYSIS FUNCTION ==================

def get_enhanced_job_fit_analysis(resume_data, job_description, resume_text=None):
    """
    Enhanced job fit analysis using Mistral LLM.
    Falls back to keyword matching if LLM fails.
    
    Args:
        resume_data: Structured JSON from parse_resume_file
        job_description: The target job description
        resume_text: Raw resume text (optional, will use from resume_data if not provided)
    
    Returns:
        dict: Detailed job fit analysis with recommendations
    """
    
    # Get raw resume text if not provided
    if not resume_text:
        resume_text = resume_data.get('full_text', '')
    
    if not resume_text or not job_description:
        return None
    
    # Create hash for caching
    text_hash = hash(resume_text[:500] + job_description[:500])
    
    # Try LLM analysis first
    llm_analysis = call_mistral_job_fit_analysis(text_hash, resume_text, job_description)
    
    if llm_analysis:
        # Enhance with extracted skills data
        llm_analysis['extracted_skills'] = resume_data.get('skills', [])[:15]
        llm_analysis['extracted_projects'] = len(resume_data.get('projects', []))
        llm_analysis['extracted_experience_years'] = resume_data.get('experience_years', 0)
        print(f"✅ Enhanced job fit analysis complete - Score: {llm_analysis.get('match_score', 'N/A')}")
        return llm_analysis
    
    print("⚠️ Enhanced job fit analysis failed - returning None")
    return None

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