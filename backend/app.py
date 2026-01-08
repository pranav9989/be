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

from dotenv import load_dotenv
import google.generativeai as genai

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



# Global current_user for JWT compatibility
current_user = None


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

        # Use the exact same RAG system as rag_query.py
        from rag import main as rag_main
        answer, retrieved = rag_main(user_query)

        # Extract topic info from first retrieved item
        topic = retrieved[0]["topic"] if retrieved else None
        subtopic = retrieved[0]["subtopic"] if retrieved else None

        response_data = {
            'success': True,
            'query': user_query,
            'answer': answer,
            'detected_topic': topic,
            'detected_subtopic': subtopic,
            'source_count': len(retrieved)
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
    try:
        # Check if required files exist
        import os
        files_exist = all([
            os.path.exists("data/processed/faiss_gemini/index.faiss"),
            os.path.exists("data/processed/faiss_gemini/metas.json"),
            os.path.exists("data/processed/kb_clean.json"),
            os.path.exists("config/topic_rules.json")
        ])
        return jsonify({
            'status': 'healthy',
            'rag_initialized': files_exist
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'rag_initialized': False,
            'error': str(e)
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

    print("🚀 Starting Flask API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
