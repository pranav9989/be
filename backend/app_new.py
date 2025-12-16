#!/usr/bin/env python3
"""
app.py — Flask app (improved RAG + Gemini handling)
Drop-in replacement for your reference app.py. Keeps templates/routes intact,
but improves RAG initialization, FAISS usage, and Gemini calls.
"""

import os
import json
import traceback
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
import jwt
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template, session, redirect, url_for, flash,
    send_from_directory
)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

import PyPDF2
import docx
import random

# Load environment variables
load_dotenv()

# -------------------- App config --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interview_prep.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key-here')
app.config['JWT_ACCESS_TOKEN_EXPIRE'] = timedelta(hours=24)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('backend/instance', exist_ok=True)

# DB + auth
db = SQLAlchemy(app)
# Allow credentials so React (on different origin) can use cookie sessions
CORS(app, supports_credentials=True)
login_manager = LoginManager()
login_manager.init_app(app)

# For API endpoints, return JSON 401 instead of redirecting to login HTML
@login_manager.unauthorized_handler
def unauthorized_callback():
    return jsonify({'error': 'Authentication required'}), 401

# keep your previous login view for template redirects
login_manager.login_view = 'login'

# -------------------- Template helpers --------------------
@app.template_filter('from_json')
def from_json_filter(json_str):
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []

@app.template_filter('datetime_diff')
def datetime_diff_filter(dt):
    if not dt:
        return datetime.now() - datetime.now()
    return datetime.now() - dt

# -------------------- Models --------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    experience_years = db.Column(db.Integer, default=0)
    skills = db.Column(db.Text, nullable=True)
    resume_filename = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    interviews = db.relationship('InterviewSession', backref='user', lazy=True)

class InterviewSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_type = db.Column(db.String(50), nullable=False)
    questions = db.Column(db.Text, nullable=True)
    score = db.Column(db.Float, nullable=True)
    feedback = db.Column(db.Text, nullable=True)
    duration = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# JWT helper functions
def create_access_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRE'],
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def jwt_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid token'}), 401

        token = auth_header.split(' ')[1]
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401

        # Set current_user for Flask-Login compatibility
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 401

        # Create a mock current_user object for Flask-Login decorators
        global current_user
        current_user = user

        return f(*args, **kwargs)
    return decorated_function

# -------------------- RAG & Gemini state --------------------
# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "faiss_gemini",
    Path("data") / "processed" / "faiss_gemini"
]
# possible index filenames used across versions
INDEX_CANDIDATES = ["faiss_index_gemini.idx", "index.faiss", "faiss_index_gemini.faiss", "faiss_index.idx"]
METAS_CANDIDATES = ["metas.json", "metas.jsonl", "metas_full.json"]

CONFIG_DIR = PROJECT_ROOT / "config"
TOPIC_RULES_FILE = CONFIG_DIR / "topic_rules.json"
TAXONOMY_FILE = CONFIG_DIR / "taxonomy.json"

# In-memory objects
rag_index = None
rag_metas = None   # dict: int_id -> meta
rag_embedder = None
topic_rules = None

# Gemini model name (allow override via env)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

def get_gemini_api_key():
    for name in ("GEMINI_API_KEY", "GEMIN_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(name)
        if v:
            return v
    return None

# -------------------- RAG helpers --------------------
def find_rag_paths():
    # return tuple(index_path, metas_path) if found else (None, None)
    for base in RAG_DIRS:
        for idx_name in INDEX_CANDIDATES:
            idx_path = base / idx_name
            if idx_path.exists():
                # find a metas file
                for mname in METAS_CANDIDATES:
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
        if TOPIC_RULES_FILE.exists():
            topic_rules = json.loads(TOPIC_RULES_FILE.read_text(encoding="utf-8"))
            print(f"✅ Loaded topic rules: {len(topic_rules)} rules")
        else:
            topic_rules = None
            print("ℹ️ topic_rules.json not found; continuing without topic routing")
    except Exception as e:
        print("⚠️ Error loading topic rules:", e)
        traceback.print_exc()
        topic_rules = None

    # Configure Gemini client if API key present (do not call generate/test here)
    api_key = get_gemini_api_key()
    if api_key:
        try:
            genai.configure(api_key=api_key)
            print("✅ Gemini configured (API key present). Model set:", GEMINI_MODEL)
        except Exception as e:
            print("⚠️ Error configuring Gemini:", e)
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
        traceback.print_exc()
        rag_metas = None

    if rag_index is not None and rag_metas is not None:
        try:
            print("🤖 Loading embedder model (this may take a moment)...")
            rag_embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedder ready")
        except Exception as e:
            print("⚠️ Failed to load embedder:", e)
            traceback.print_exc()
            rag_embedder = None

# Global current_user for JWT compatibility
current_user = None

# -------------------- Search / generate helpers --------------------
def embed_text(embedder, texts):
    # texts: list[str]; returns numpy array (n, dim) float32 normalized
    emb = embedder.encode(texts, convert_to_numpy=True)
    emb = np.array(emb).astype("float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb

def search_faiss_topk(index, q_emb, k=5):
    # q_emb shape: (1, dim)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return D[0].tolist(), I[0].tolist()

def build_context_from_hits(ids, scores, rag_metas, top_k=5):
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

def generate_rag_response_gemini(api_key, prompt, model_name=GEMINI_MODEL):
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
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def build_strict_prompt(context, user_query):
    system = (
        "You are an expert in computer science (DBMS, OOPs, Operating Systems). "
        "Answer using ONLY the provided CONTEXT. "
        "If the context does not contain the answer, respond exactly: "
        "\"I don't know based on the provided KB.\" "
        "Provide a concise answer and list the source chunk IDs used in square brackets at the end."
    )
    prompt = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_query}\n\nANSWER:"
    return prompt

# -------------------- File text extraction & resume parsing --------------------
def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print("PDF extraction error:", e)
        traceback.print_exc()
        return None

def extract_text_from_docx(file_stream):
    try:
        doc = docx.Document(file_stream)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text.strip()
    except Exception as e:
        print("DOCX extraction error:", e)
        traceback.print_exc()
        return None

def parse_resume_text(text):
    lines = text.strip().splitlines()
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'html', 'css',
        'machine learning', 'data science', 'flask', 'django', 'mongodb', 'mysql'
    ]
    skills = []
    for line in lines:
        for kw in tech_keywords:
            if kw in line.lower() and kw.title() not in skills:
                skills.append(kw.title())
    experience_years = 0
    for line in lines:
        if 'year' in line.lower() and 'experience' in line.lower():
            tokens = line.split()
            for t in tokens:
                if t.isdigit():
                    experience_years = int(t)
                    break
    return {'skills': skills, 'experience_years': experience_years, 'raw_text': text[:1000]}

# ========== API ROUTES (Updated for React) ==========

@app.route('/')
def home():
    return jsonify({'message': 'Interview Prep API', 'status': 'running'})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        token = create_access_token(user.id)
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name
            }
        })
    else:
        return jsonify({'success': False, 'message': 'Invalid username or password'})


@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name')

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists'})
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered'})

    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
        full_name=full_name
    )
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({
        'success': True,
        'message': 'Registration successful',
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name
        }
    })


@app.route('/api/logout', methods=['POST'])
@jwt_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/profile', methods=['GET'])
@jwt_required
def get_profile():
    skills = json.loads(current_user.skills) if current_user.skills else []
    return jsonify({
        'user': {
            'id': current_user.id,
            'username': current_user.username,
            'email': current_user.email,
            'full_name': current_user.full_name,
            'phone': current_user.phone,
            'experience_years': current_user.experience_years,
            'skills': skills,
            'resume_filename': current_user.resume_filename,
            'created_at': current_user.created_at.isoformat()
        }
    })


@app.route('/api/update_profile', methods=['POST'])
@jwt_required
def update_profile():
    data = request.get_json()
    current_user.full_name = data.get('full_name', current_user.full_name)
    current_user.phone = data.get('phone', current_user.phone)
    current_user.experience_years = data.get('experience_years', current_user.experience_years)
    current_user.skills = json.dumps(data.get('skills', []))
    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile updated successfully'})


@app.route('/api/upload_resume', methods=['POST'])
@jwt_required
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})

    if file:
        filename = secure_filename(f"{current_user.id}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from resume
        file_stream = BytesIO()
        file.stream.seek(0)
        file_stream.write(file.stream.read())
        file_stream.seek(0)

        text = None
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_stream)
        elif filename.lower().endswith('.docx'):
            text = extract_text_from_docx(file_stream)

        if text:
            resume_data = parse_resume_text(text)
            current_user.resume_filename = filename
            current_user.skills = json.dumps(resume_data['skills'])
            current_user.experience_years = resume_data['experience_years']
            db.session.commit()
            return jsonify({
                'success': True,
                'message': 'Resume uploaded and parsed successfully',
                'data': resume_data
            })
        else:
            return jsonify({'success': False, 'message': 'Could not extract text from resume'})


@app.route('/api/query', methods=['POST'])
@jwt_required
def rag_query():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Check if RAG system is available
        if not all([rag_index, rag_metas, rag_embedder, topic_rules]):
            return jsonify({'error': 'RAG system not initialized. Please contact administrator.'}), 500

        print(f"🔍 Processing query: {user_query}")

        topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)
        print(f"📋 Detected topic: {topic}, subtopic: {subtopic}")

        if topic and subtopic:
            augmented_query = f"Question about {subtopic} in {topic}: {user_query}"
        else:
            augmented_query = user_query

        print(f"🔎 Augmented query: {augmented_query}")
        relevant_chunks = get_relevant_chunks(augmented_query, rag_index, rag_metas, rag_embedder)
        print(f"📚 Found {len(relevant_chunks)} relevant chunks")

        context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        print(f"📝 Context length: {len(context_text)} characters")

        # Generate clean response for frontend
        response_text = generate_rag_response(user_query, context_text)

        # Log debugging info to console only
        print(f"✅ Generated response, length: {len(response_text)} characters")
        print("\n--- Source Chunks (for debugging) ---")
        for chunk in relevant_chunks[:3]:  # Show only first 3 for debugging
            print(f"Source ID: {chunk.get('id', 'N/A')}")
            text_preview = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
            print(f"Text: {text_preview}\n")

        response_data = {
            'success': True,
            'query': user_query,
            'answer': response_text,
            'detected_topic': topic,
            'detected_subtopic': subtopic,
            'source_count': len(relevant_chunks)
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"❌ RAG Query Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/hr_questions', methods=['POST'])
@jwt_required
def generate_hr_questions():
    try:
        skills = json.loads(current_user.skills) if current_user.skills else []
        experience = current_user.experience_years

        gemini_model = genai.GenerativeModel("models/gemini-flash-latest")

        prompt = f"""
        Generate 5 HR interview questions for a candidate with the following profile:
        - Skills: {', '.join(skills)}
        - Experience: {experience} years

        Include a mix of:
        1. General HR questions
        2. Behavioral questions
        3. Technical leadership questions (if experienced)
        4. Questions about their specific skills

        Return as JSON array with questions and expected answer guidelines.
        """
        response = gemini_model.generate_content(prompt)
        try:
            questions = json.loads(response.text)
        except:
            questions = [
                {"question": "Tell me about yourself", "type": "general"},
                {"question": "Why do you want to work here?", "type": "general"},
                {"question": "Describe a challenging project you worked on", "type": "behavioral"},
                {"question": "How do you handle tight deadlines?", "type": "behavioral"},
                {"question": "Where do you see yourself in 5 years?", "type": "general"}
            ]
        return jsonify({'success': True, 'questions': questions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/user_stats', methods=['GET'])
@jwt_required
def get_user_stats():
    try:
        sessions_count = InterviewSession.query.filter_by(user_id=current_user.id).count()
        completed_sessions = InterviewSession.query.filter(
            InterviewSession.user_id == current_user.id,
            InterviewSession.score.isnot(None)
        ).all()
        avg_score = None
        if completed_sessions:
            avg_score = sum(session.score for session in completed_sessions) / len(completed_sessions)
        questions_answered = sessions_count * 5
        return jsonify({
            'sessions': sessions_count,
            'questions': questions_answered,
            'avg_score': avg_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/topics', methods=['GET'])
def get_topics():
    try:
        with open('config/taxonomy.json', 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        return jsonify(taxonomy)
    except Exception as e:
        return jsonify({'error': f'Error loading topics: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'rag_initialized': all([rag_embedder is not None, rag_index is not None, rag_metas is not None, topic_rules is not None])
    })


@app.route('/api/save_interview_session', methods=['POST'])
@jwt_required
def save_interview_session():
    """Save interview practice session with questions and answers"""
    try:
        data = request.get_json()
        session_type = data.get('session_type', 'hr')
        questions = data.get('questions', [])
        answers = data.get('answers', {})

        # Create new interview session
        session = InterviewSession(
            user_id=current_user.id,
            session_type=session_type,
            questions=json.dumps({
                'questions': questions,
                'answers': answers
            }),
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )

        db.session.add(session)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Session saved successfully',
            'session_id': session.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/interview_history', methods=['GET'])
@jwt_required
def get_interview_history():
    """Get user's interview practice session history"""
    try:
        session_type = request.args.get('type', None)

        query = InterviewSession.query.filter_by(user_id=current_user.id)

        if session_type:
            query = query.filter_by(session_type=session_type)

        sessions = query.order_by(InterviewSession.created_at.desc()).limit(20).all()

        history = []
        for session in sessions:
            questions_data = json.loads(session.questions) if session.questions else {}
            history.append({
                'id': session.id,
                'session_type': session.session_type,
                'created_at': session.created_at.isoformat(),
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'questions_count': len(questions_data.get('questions', [])),
                'score': session.score,
                'feedback': session.feedback
            })

        return jsonify({
            'success': True,
            'sessions': history
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Serve React build in production
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    else:
        return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    # Initialize system - app will start even if some components fail
    initialize_rag_system()
    print("🚀 Starting Flask API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
