#!/usr/bin/env python3
"""
app.py — Flask app (improved RAG + Gemini handling)
Drop-in replacement for your reference app.py. Keeps templates/routes intact,
but improves RAG initialization, FAISS usage, and Gemini calls.
"""

import os
import json
import re
import traceback
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

from flask import (
    Flask, request, jsonify, render_template, session, redirect, url_for, flash,
    send_from_directory
)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
import google.generativeai as genai
from resume_processor import process_resume_for_faiss, search_resume_faiss, get_resume_chunks

import PyPDF2
import docx
import random
from interview_analyzer import speech_to_text, analyze_interview_response

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

# JWT imports
import jwt
from functools import wraps

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
    """
    Extract skills, experience, and projects from resume text.
    Only extracts information actually present in the resume.
    """
    import re

    lines = text.strip().splitlines()
    text_lower = text.lower()

    skills = []
    projects = []
    experience_years = 0

    # Common technical skills patterns - only add if actually found in resume
    common_skills = {
        'Programming Languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala'],
        'Web Technologies': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring'],
        'Databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server', 'sqlite'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'github actions', 'terraform'],
        'Tools & Technologies': ['git', 'linux', 'windows', 'api', 'rest', 'graphql', 'json', 'xml'],
        'Frameworks & Libraries': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'opencv'],
        'Methodologies': ['agile', 'scrum', 'kanban', 'ci/cd', 'tdd', 'bdd']
    }

    # Extract skills by checking if they appear in the resume
    for category, skill_list in common_skills.items():
        for skill in skill_list:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                # Capitalize properly
                if skill == 'node.js':
                    skills.append('Node.js')
                elif skill == 'c++':
                    skills.append('C++')
                elif skill == 'c#':
                    skills.append('C#')
                elif skill == 'ci/cd':
                    skills.append('CI/CD')
                elif skill == 'tdd':
                    skills.append('TDD')
                elif skill == 'bdd':
                    skills.append('BDD')
                else:
                    skills.append(skill.title())

    # Remove duplicates while preserving order
    skills = list(dict.fromkeys(skills))

    # Extract experience years - multiple patterns
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?experience',
        r'total\s*(?:of\s*)?(\d+)\+?\s*years?',
        r'over\s*(\d+)\s*years?',
        r'more\s*than\s*(\d+)\s*years?'
    ]

    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            years = int(match)
            if years > experience_years:  # Take the highest mentioned experience
                experience_years = years

    # If no years found, look for months and convert to years
    if experience_years == 0:
        month_patterns = [
            r'(\d+)\+?\s*months?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*months?'
        ]
        for pattern in month_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                months = int(match)
                years = months / 12
                if years > experience_years:
                    experience_years = int(years)

    # Extract projects - look for project-related sections
    project_keywords = ['project', 'projects', 'developed', 'built', 'created', 'implemented', 'worked on', 'led', 'managed']
    in_projects_section = False
    current_projects = []

    for line in lines:
        line_lower = line.lower().strip()

        # Check if we're entering a projects section
        if any(keyword in line_lower for keyword in ['projects', 'project experience', 'key projects', 'personal projects']):
            in_projects_section = True
            continue

        # If we're in projects section, collect project descriptions
        if in_projects_section and len(line.strip()) > 20:  # Meaningful project description
            # Stop if we hit another section
            if any(section in line_lower for section in ['education', 'skills', 'experience', 'certifications', 'achievements']):
                in_projects_section = False
                continue

            # Clean and add project
            project_text = line.strip()
            if project_text and not any(word in project_text.lower() for word in ['•', '-', '*']):  # Avoid bullet points alone
                current_projects.append(project_text)

        # Also look for project mentions in regular text
        elif any(keyword in line_lower for keyword in project_keywords) and len(line.strip()) > 30:
            current_projects.append(line.strip())

        # Limit to 5 projects max
        if len(current_projects) >= 5:
            break

    # Clean up projects - remove duplicates and empty entries
    projects = []
    for proj in current_projects:
        proj_clean = proj.strip()
        if len(proj_clean) > 10 and proj_clean not in projects:
            projects.append(proj_clean)

    return {
        'skills': skills[:10],  # Limit to top 10 most relevant skills
        'experience_years': experience_years,
        'projects': projects[:3],  # Limit to top 3 projects
        'raw_text': text[:1000]
    }

def analyze_resume_job_fit(resume_data, job_description):
    """Analyze how well the resume fits the job description"""
    if not job_description:
        return None

    resume_skills = set(resume_data.get('skills', []))
    jd_text = job_description.lower()

    # Extract skills from job description
    jd_skills = []
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'html', 'css',
        'machine learning', 'data science', 'flask', 'django', 'mongodb', 'mysql',
        'aws', 'docker', 'kubernetes', 'git', 'linux', 'windows', 'api', 'rest',
        'graphql', 'agile', 'scrum', 'ci/cd', 'jenkins', 'testing', 'unit test',
        'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'typescript', 'vue', 'angular'
    ]

    for skill in tech_keywords:
        if skill in jd_text and skill.title() not in jd_skills:
            jd_skills.append(skill.title())

    jd_skills_set = set(jd_skills)

    # Calculate match scores
    matching_skills = resume_skills.intersection(jd_skills_set)
    missing_skills = jd_skills_set - resume_skills

    match_percentage = (len(matching_skills) / len(jd_skills_set) * 100) if jd_skills_set else 0

    # Experience analysis
    experience_required = 0
    if 'year' in jd_text and 'experience' in jd_text:
        import re
        exp_match = re.search(r'(\d+)\+?\s*year', jd_text)
        if exp_match:
            experience_required = int(exp_match.group(1))

    experience_fit = "Good fit" if resume_data.get('experience_years', 0) >= experience_required else "May need more experience"

    return {
        'matching_skills': list(matching_skills),
        'missing_skills': list(missing_skills),
        'match_percentage': round(match_percentage, 1),
        'experience_required': experience_required,
        'experience_fit': experience_fit,
        'jd_skills_found': jd_skills
    }

# ========== API ROUTES (Updated for React) ==========

@app.route('/')
def home():
    return jsonify({'message': 'Interview Prep API', 'status': 'running'})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Input validation
    if not username or not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return jsonify({'success': False, 'message': 'Invalid username format.'})
    if not password:
        return jsonify({'success': False, 'message': 'Password is required.'})

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

    # Input validation
    if not username or not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return jsonify({'success': False, 'message': 'Invalid username format. Username must be 3-20 characters long and contain only letters, numbers, and underscores.'})
    if not email or not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({'success': False, 'message': 'Invalid email format.'})
    if not password or len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters long.'})

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

            # Get job description from form data
            job_description = request.form.get('job_description', '').strip()

            # Analyze resume-job fit if JD provided
            job_fit_analysis = None
            if job_description:
                job_fit_analysis = analyze_resume_job_fit(resume_data, job_description)
                resume_data['job_fit_analysis'] = job_fit_analysis

            # Process resume with FAISS for interview questions
            try:
                chunk_count = process_resume_for_faiss(text, current_user.id)
                resume_data['chunks_processed'] = chunk_count
                resume_data['rag_ready'] = True
            except Exception as e:
                print(f"FAISS processing failed: {e}")
                resume_data['rag_ready'] = False

            db.session.commit()
            return jsonify({
                'success': True,
                'message': 'Resume uploaded and analyzed successfully',
                'data': resume_data,
                'job_description_provided': bool(job_description)
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
        user = g.current_user
        skills = json.loads(user.skills) if user.skills else []
        experience = user.experience_years

        api_key = get_gemini_api_key()
        if not api_key:
            return jsonify({'success': False, 'error': 'Gemini API key not configured'})

        genai.configure(api_key=api_key)
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


@app.route('/api/resume_based_questions', methods=['POST'])
@jwt_required
def generate_resume_based_questions():
    """Generate interview questions based on user's resume content using FAISS"""
    try:
        data = request.get_json()
        job_description = data.get('job_description', '')
        question_count = data.get('question_count', 5)
        variation_seed = data.get('variation_seed', '')  # For generating different questions each time

        # Check if user has a processed resume
        if not current_user.resume_filename:
            return jsonify({'success': False, 'error': 'No resume uploaded'})

        # Search resume content for relevant information
        if job_description:
            search_query = f"Generate interview questions for this job: {job_description}"
        else:
            search_query = "Generate technical interview questions based on my experience and skills"

        search_results = search_resume_faiss(search_query, current_user.id, top_k=5)

        # Build context from resume chunks
        resume_context = "\n".join([result['text'] for result in search_results])

        # Generate questions using Gemini with resume context
        api_key = get_gemini_api_key()
        if not api_key:
            return jsonify({'success': False, 'error': 'Gemini API key not configured'})

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("models/gemini-flash-latest")

        skills = json.loads(user.skills) if user.skills else []
        experience = user.experience_years

        # Add variation seed to make questions different each time
        variation_text = f" (Variation: {variation_seed})" if variation_seed else ""

        prompt = f"""
        Based on the following resume content and job requirements, generate {question_count} targeted interview questions{variation_text}.

        Resume Content:
        {resume_context}

        Candidate Profile:
        - Skills: {', '.join(skills)}
        - Experience: {experience} years
        - Job Description: {job_description}

        Generate {question_count} specific, relevant interview questions that:
        1. Test technical skills mentioned in the resume
        2. Probe deeper into projects and experiences described
        3. Assess problem-solving abilities demonstrated in the resume
        4. Evaluate fit for the job description provided

        IMPORTANT: Create COMPLETELY DIFFERENT questions each time. Focus on different aspects of the resume, different skills, and different scenarios. Avoid repeating similar question patterns or themes.

        Return as JSON array with each question object containing 'question' and 'type' fields.
        Types should be: 'technical', 'behavioral', 'project-based', or 'situational'.
        """

        response = gemini_model.generate_content(prompt)
        try:
            questions = json.loads(response.text)
        except:
            # Fallback questions if parsing fails
            questions = [
                {"question": "Can you walk me through your most challenging project?", "type": "project-based"},
                {"question": "How do your technical skills align with this role?", "type": "technical"},
                {"question": "Describe a time when you solved a difficult problem", "type": "behavioral"},
                {"question": "What are your career goals and how does this position fit?", "type": "situational"},
                {"question": "How do you stay updated with industry trends?", "type": "situational"}
            ]

        return jsonify({
            'success': True,
            'questions': questions,
            'resume_chunks_found': len(search_results)
        })

    except Exception as e:
        print(f"Resume-based questions error: {e}")
        import traceback
        traceback.print_exc()
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


@app.route('/api/process_audio', methods=['POST'])
@jwt_required
def process_audio():
    try:
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'message': 'No audio file provided'}), 400

        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        # Check file size (max 10MB)
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)

        if file_size == 0:
            return jsonify({'success': False, 'message': 'Audio file is empty'}), 400

        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'success': False, 'message': 'Audio file too large (max 10MB)'}), 400

        # Save the audio file temporarily with proper extension
        filename = secure_filename(audio_file.filename)
        if not filename.endswith(('.webm', '.wav', '.mp3', '.ogg', '.m4a')):
            filename = filename.rsplit('.', 1)[0] + '.webm'

        temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        audio_file.save(temp_audio_path)

        print(f"Saved audio file: {temp_audio_path}, size: {os.path.getsize(temp_audio_path)} bytes")

        # Convert WebM to WAV if needed (Whisper needs WAV or formats supported by ffmpeg)
        # Since ffmpeg might not be installed, use librosa + soundfile as fallback
        converted_audio_path = temp_audio_path
        if filename.lower().endswith('.webm'):
            converted_audio_path = temp_audio_path.rsplit('.', 1)[0] + '.wav'
            try:
                # Try pydub first (requires ffmpeg)
                from pydub import AudioSegment
                audio = AudioSegment.from_file(temp_audio_path, format="webm")
                audio.export(converted_audio_path, format="wav")
                print(f"Converted WebM to WAV using pydub: {converted_audio_path}")
                # Remove original WebM file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            except (ImportError, Exception) as e:
                print(f"pydub conversion failed ({e}), trying librosa + soundfile...")
                try:
                    # Fallback: Use librosa + soundfile (doesn't require ffmpeg)
                    import librosa
                    import soundfile as sf

                    # Load audio with librosa (handles many formats)
                    y, sr = librosa.load(temp_audio_path, sr=16000)  # 16kHz is good for speech
                    # Save as WAV using soundfile
                    sf.write(converted_audio_path, y, sr)
                    print(f"Converted WebM to WAV using librosa: {converted_audio_path}")
                    # Remove original WebM file
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                except ImportError:
                    print("soundfile not available. Installing soundfile is recommended: pip install soundfile")
                    # Last resort: try with original file (might fail if ffmpeg not available)
                    converted_audio_path = temp_audio_path
                    print("Will attempt transcription with original WebM file (requires ffmpeg)")
                except Exception as e2:
                    print(f"librosa conversion also failed: {e2}")
                    # Last resort: try with original file
                    converted_audio_path = temp_audio_path
                    print("Will attempt transcription with original WebM file (requires ffmpeg)")

        # Transcribe audio
        print("Starting transcription...")
        print(f"Using audio file: {converted_audio_path}")

        # Verify file exists and has content
        if not os.path.exists(converted_audio_path):
            return jsonify({
                'success': False,
                'message': f'Audio file not found: {converted_audio_path}'
            }), 400

        file_size = os.path.getsize(converted_audio_path)
        print(f"Audio file exists, size: {file_size} bytes")

        if file_size == 0:
            return jsonify({
                'success': False,
                'message': 'Audio file is empty after conversion'
            }), 400

        transcribed_text = None
        try:
            # Import faster-whisper model
            import librosa
            from interview_analyzer import whisper_model

            print("Faster-Whisper model loaded, starting transcription...")

            # Load audio with librosa first (bypasses Whisper's ffmpeg requirement)
            # This way we can handle the audio loading ourselves
            try:
                print(f"Loading audio with librosa from: {converted_audio_path}")
                # Load audio at 16kHz (Whisper's native sample rate)
                audio_array, sample_rate = librosa.load(converted_audio_path, sr=16000)
                print(f"Audio loaded successfully: {len(audio_array)} samples at {sample_rate}Hz")

                # Pass numpy array directly to Faster-Whisper (bypasses file loading)
                # faster-whisper can accept numpy arrays directly
                print("Transcribing audio array...")
                segments, info = whisper_model.transcribe(audio_array, beam_size=5)

                # Extract text from segments (segments is a generator, so convert to list)
                segments_list = list(segments)
                transcribed_text = " ".join([segment.text for segment in segments_list]).strip()

                # Log transcription info
                print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
                print(f"Transcription result: '{transcribed_text}' (length: {len(transcribed_text)})")

                # Log segment details
                if len(segments_list) > 0:
                    print(f"Found {len(segments_list)} audio segments")
                    for i, segment in enumerate(segments_list[:3]):  # Print first 3 segments
                        print(f"Segment {i}: {segment.text} (start: {segment.start:.2f}s, end: {segment.end:.2f}s)")
                else:
                    print("Warning: No audio segments found in transcription result")

            except Exception as librosa_error:
                print(f"librosa failed to load audio: {librosa_error}")
                error_msg = str(librosa_error)
                if "ffmpeg" in error_msg.lower() or "winerror 2" in error_msg.lower():
                    # Clear error message about ffmpeg requirement
                    raise Exception(
                        "ffmpeg is required but not found. Please install ffmpeg:\n"
                        "Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH\n"
                        "Or use: choco install ffmpeg (if you have Chocolatey)\n"
                        "After installation, restart your Flask server."
                    )
                else:
                    raise

        except ImportError as e:
            print(f"Import error: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Faster-Whisper library not available: {str(e)}'
            }), 500
        except Exception as e:
            print(f"Exception during transcription: {e}")
            traceback.print_exc()
            error_details = str(e)
            # Check for common Whisper errors
            if "CUDA" in error_details or "cuda" in error_details.lower():
                error_details += " (GPU/CUDA issue - trying CPU mode)"
            return jsonify({
                'success': False,
                'message': f'Transcription failed: {error_details}'
            }), 500

        if not transcribed_text or len(transcribed_text.strip()) == 0:
            error_msg = 'Could not transcribe audio. Please ensure you are speaking clearly and try again.'
            # Check if file exists and has content
            if os.path.exists(converted_audio_path):
                file_size = os.path.getsize(converted_audio_path)
                error_msg += f' (Audio file size: {file_size} bytes)'
            return jsonify({
                'success': False,
                'message': error_msg
            }), 400

        # Perform comprehensive speech analysis
        print("Starting comprehensive speech analysis...")
        try:
            # For conversational interview, we don't have ideal answers, so we'll use generic analysis
            # In a real scenario, you'd have question-specific ideal answers
            ideal_answer = "This is a conversational interview response that should demonstrate good communication skills, clear articulation, and professional speaking patterns."
            ideal_keywords = ["communication", "professional", "clear", "articulate", "confident"]

            analysis_results = analyze_interview_response(
                converted_audio_path,
                ideal_answer,
                ideal_keywords
            )

            print("Speech analysis completed successfully")
            print(f"Analysis results keys: {list(analysis_results.keys()) if analysis_results else 'None'}")
            print(f"Overall score: {analysis_results.get('overall_score', 'N/A') if analysis_results else 'N/A'}")

            # Generate interviewer response based on analysis
            overall_score = analysis_results.get('overall_score', 0)
            fluency_score = analysis_results.get('fluency_score', 0)

            if overall_score >= 80:
                interviewer_response = f"Excellent delivery! I heard: \"{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}\". Your speech patterns are very professional."
            elif overall_score >= 60:
                interviewer_response = f"Good effort! I heard: \"{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}\". Keep practicing to improve your delivery."
            else:
                interviewer_response = f"I heard: \"{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}\". Let's work on improving your speech clarity and fluency."

            # Add specific feedback
            feedback_tips = []
            if fluency_score < 60:
                feedback_tips.append("Try speaking more slowly and clearly")
            if analysis_results.get('pause_ratio', 0) > 0.6:
                feedback_tips.append("Reduce long pauses between words")

            if feedback_tips:
                interviewer_response += f" Tips: {'; '.join(feedback_tips[:2])}."

            # Return comprehensive analysis results
            return jsonify({
                'success': True,
                'transcribed_text': transcribed_text,
                'interviewer_response': interviewer_response,
                'speech_analysis': {
                    'overall_score': analysis_results.get('overall_score', 0),
                    'performance_level': analysis_results.get('performance_level', 'Unknown'),
                    'fluency_score': analysis_results.get('fluency_score', 0),
                    'pitch_score': analysis_results.get('pitch_score', 0),
                    'voice_quality_score': analysis_results.get('voice_quality_score', 0),
                    'wpm': analysis_results.get('wpm', 0),
                    'pause_ratio': analysis_results.get('pause_ratio', 0),
                    'pitch_feedback': analysis_results.get('pitch_feedback', ''),
                    'fluency_feedback': analysis_results.get('fluency_feedback', ''),
                    'voice_quality_feedback': analysis_results.get('voice_quality_feedback', ''),
                    'improvement_suggestions': analysis_results.get('improvement_suggestions', [])[:3]  # Top 3 suggestions
                }
            })

        except Exception as analysis_error:
            print(f"Speech analysis failed: {analysis_error}")
            traceback.print_exc()

            # Fallback response without analysis
            interviewer_response = f"I heard you say: \"{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}\". Let's continue the conversation."

            return jsonify({
                'success': True,
                'transcribed_text': transcribed_text,
                'interviewer_response': interviewer_response,
                'speech_analysis': None,
                'analysis_error': 'Speech analysis unavailable'
            })

    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        error_message = str(e)
        if "No module named" in error_message:
            error_message = "Audio processing module not available. Please install required dependencies."
        return jsonify({'success': False, 'error': error_message}), 500
    finally:
        # Clean up the temporary files
        files_to_remove = []
        if temp_audio_path and os.path.exists(temp_audio_path):
            files_to_remove.append(temp_audio_path)
        # Also remove converted file if different
        if 'converted_audio_path' in locals() and converted_audio_path != temp_audio_path:
            if converted_audio_path and os.path.exists(converted_audio_path):
                files_to_remove.append(converted_audio_path)
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                print(f"Error removing temp file {file_path}: {e}")




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
