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
import wave
import time
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

def save_pcm_as_wav(pcm_bytes, path):
    """Save raw PCM bytes as proper WAV file (16-bit, 16kHz, mono)"""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)      # Mono
        wf.setsampwidth(2)      # 16-bit PCM
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(pcm_bytes)

from flask import (
    Flask, request, jsonify, render_template, session, redirect, url_for, flash,
    send_from_directory, g
)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
import google.generativeai as genai
from resume_processor import process_resume_for_faiss, search_resume_faiss, get_resume_chunks
from assemblyai_websocket_stream import AssemblyAIWebSocketStreamer, warmup_assemblyai
from interview_analyzer import (
    speech_to_text,
    RunningStatistics,
    analyze_audio_chunk_fast,   # 🔥 ADD THIS
    calculate_semantic_similarity,
    compute_research_metrics,
    finalize_interview
)

import PyPDF2
import docx
import random

from agent.analyzer import analyze_answer
from agent.controller import InterviewAgentController

import requests
from flask import request, Response, jsonify
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

def ollama_generate(prompt, timeout=60):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            return None
    except Exception as e:
        print("Ollama error:", e)
        return None


def finalize_user_answer(session_key):
    """
    The Single Source of Truth for ending a turn.
    Triggered ONLY by silence (Clock B).
    """
    session = streaming_sessions.get(session_key)
    if not session:
        return

    # 🔒 ATOMIC LOCK: Prevent double-firing
    if session.get("finalized"):
        return
    
    # 🛑 IMMEDIATE STATE CHANGE (Clock C)
    session["finalized"] = True
    session["turn"] = "INTERVIEWER" 

    # 1. Merge all buffered text (Clock A results)
    final_answer = " ".join(session.get("final_text", [])).strip()
    
    # 2. If user said nothing but silence timed out, handle gracefully
    if not final_answer:
        final_answer = "[User remained silent]"

    print(f"✅ FINAL USER ANSWER (Triggered by Silence): {final_answer}")

    # 3. Call the AI Agent
    try:
        room = session["room"]
        question = session.get("current_question")

        analysis = analyze_answer(
            question=question,
            answer=final_answer
        )

        agent_response = InterviewAgentController.handle_answer(
            session_id=str(session["user_id"]),
            answer=final_answer,
            analysis=analysis
        )

        next_question = agent_response.get("next_question")

        # 4. Update Frontend
        socketio.emit(
            "user_answer_complete",
            {
                "answer": final_answer,
                "question": question,
                "next_question": next_question
            },
            room=room
        )

        # 🔥 NEW: Calculate and store Q&A scores
        if "stats" in session:
            from interview_analyzer import calculate_semantic_similarity, calculate_keyword_coverage
            
            # Calculate scores
            semantic_score = calculate_semantic_similarity(final_answer, question)
            keyword_score = calculate_keyword_coverage(final_answer, question)
            
            # Record in stats
            session["stats"].record_qa_pair(question, final_answer, semantic_score, keyword_score)
            
            print(f"📊 Q&A Scores - Semantic: {semantic_score:.3f}, Keyword: {keyword_score:.3f}")

        if next_question:
            if session.get("terminated"):
                print("🚫 Session terminated — skipping agent_next_question emit")
                return

            session["current_question"] = next_question
            socketio.emit(
                "agent_next_question",
                {"question": next_question},
                room=room
            )
    except Exception as e:
        print(f"❌ Error in agent loop: {e}")
        traceback.print_exc()

    # 5. Clean up thread flags
    session["silence_thread_started"] = False


class MockAssemblyAIStreamer:
    """
    Mock streamer for testing without AssemblyAI API key.
    Accumulates audio chunks and provides basic transcription on stop.
    """
    def __init__(self, on_partial, on_final, on_error=None):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.audio_chunks = []
        self.is_active = False

    def start(self):
        """Mock start - just set active flag"""
        self.is_active = True
        print("🎭 Mock AssemblyAI streamer started (no API key required)")

    def send_audio(self, audio_bytes):
        """Accumulate audio chunks for later transcription"""
        if self.is_active:
            self.audio_chunks.append(audio_bytes)

    def stop(self):
        """Mock stop - transcribe accumulated audio"""
        if not self.is_active:
            return

        try:
            self.is_active = False
            print(f"🎭 Mock streamer stopping with {len(self.audio_chunks)} audio chunks")

            if self.audio_chunks:
                # Combine all audio chunks
                full_audio_bytes = b"".join(self.audio_chunks)

                # Save as temporary WAV file for transcription
                temp_path = f"temp_mock_audio_{int(time.time())}.wav"
                save_pcm_as_wav(full_audio_bytes, temp_path)

                # Transcribe using local Whisper
                transcript = speech_to_text(temp_path)

                if transcript:
                    # Simulate final transcript (we don't have partials in mock mode)
                    self.on_final(transcript)
                    print(f"🎭 Mock transcription complete: {len(transcript)} characters")
                else:
                    print("🎭 Mock transcription failed - no text returned")

                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass

        except Exception as e:
            print(f"🎭 Mock streamer error: {e}")
            if self.on_error:
                self.on_error(str(e))


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

# WebSocket setup for real-time audio streaming
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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

# Global streaming transcription state
streaming_sessions = {}  # user_id -> session data

def notify_backend_ready(user_id, sid):
    """Notify frontend that backend is ready to receive audio"""
    session_key = (user_id, sid)
    
    if session_key not in streaming_sessions:
        print(f"⚠️ notify_backend_ready: session not found for {session_key}")
        return
    
    print(f"🔥 Backend ready notification for user {user_id}")
    
    streaming_sessions[session_key]['ready'] = True
    streaming_sessions[session_key]['primed'] = True
    
    # 🔥 NEW: FLUSH BUFFERED AUDIO
    flush_early_buffer(session_key)
    
    # Send backend_ready event
    try:
        socketio.emit(
            'backend_ready',
            {'status': 'ready', 'timestamp': time.time()},
            room=streaming_sessions[session_key]['room']
        )
        print(f"📡 backend_ready emitted to room {streaming_sessions[session_key]['room']}")
    except Exception as e:
        print(f"❌ Failed to emit backend_ready: {e}")

def silence_watcher(session_key, timeout=15):
    """
    Clock B: The Logic Engine.
    Monitors time since last voice activity. Owns the 'finalize' decision.
    LONG PAUSE detection happens here - when user stops speaking for 5s,
    it will be counted in RunningStatistics via the audio_chunk handler.
    """
    print(f"👂 Silence watcher started for {session_key}")

    last_log_time = time.time()
    
    while True:
        time.sleep(1)  # Tick every second

        session = streaming_sessions.get(session_key)
        
        # 1. Safety Checks
        if not session: 
            return  # Session deleted
        
        if session.get("destroyed"):
            print("💀 Silence watcher exiting (Session Destroyed)")
            return  # User left/stopped

        # 2. Turn Check (Clock C)
        # If it's not the user's turn, we pause watching (or exit)
        if session.get("turn") != "USER":
            # If we already finalized, this thread is done.
            if session.get("finalized"):
                return 
            continue

        # 3. Time Check (Clock B)
        last_voice = session.get("last_voice_time")
        if not last_voice:
            continue

        elapsed = time.time() - last_voice
        
        # Log every 2 seconds instead of 5 for better visibility
        if time.time() - last_log_time >= 2:
            print(f"⏰ Silence elapsed: {elapsed:.1f}s")
            last_log_time = time.time()
            
            # 🔥 LONG PAUSE DETECTION (5s) - just for logging
            # Actual counting happens in analyze_audio_chunk_fast
            if elapsed > 5.0 and elapsed < 5.5:
                print(f"⚠️ LONG PAUSE DETECTED: {elapsed:.1f}s (will be counted in final stats)")

        # 4. THE DECISION POINT
        if elapsed >= timeout:
            print(f"🛑 Silence limit ({timeout}s) reached. Finalizing.")
            finalize_user_answer(session_key)
            return  # Thread ends here

def flush_early_buffer(session_key):
    """Flush early buffered audio chunks to AssemblyAI"""
    if session_key not in streaming_sessions:
        return
    
    session = streaming_sessions[session_key]
    
    if 'early_buffer' not in session or not session['early_buffer']:
        return
    
    print(f"📤 Flushing {len(session['early_buffer'])} buffered audio chunks to AssemblyAI")
    
    # Send all buffered audio chunks to AssemblyAI
    for i, audio_bytes in enumerate(session['early_buffer']):
        try:
            session['interviewer_audio_chunks'].append(audio_bytes)
            session['chunk_count'] = session.get('chunk_count', 0) + 1
            if i < 5:  # Log first few chunks
                print(f"📤 Sent buffered chunk {i+1} ({len(audio_bytes)} bytes)")
        except Exception as e:
            print(f"❌ Failed to send buffered chunk {i+1}: {e}")
    
    # Clear the buffer
    session['early_buffer'] = []
    
    # Update stats
    session['buffer_flushed'] = True
    print(f"✅ Early buffer flushed to AssemblyAI")


@app.route('/api/debug_sessions', methods=['GET'])
def debug_sessions():
    """Debug endpoint to check all active sessions"""
    sessions_info = []
    
    for key, session in streaming_sessions.items():
        sessions_info.append({
            'user_id': key[0],
            'socket_id': key[1],
            'primed': session.get('primed', False),
            'ready': session.get('ready', False),
            'chunk_count': session.get('chunk_count', 0),
            'buffer_count': session.get('buffer_count', 0),
            'early_buffer_len': len(session.get('early_buffer', [])),
            'audio_chunks_len': len(session.get('audio_chunks', []))
        })
    
    return jsonify({
        'total_sessions': len(sessions_info),
        'sessions': sessions_info
    })

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
        g.current_user = user

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

def warmup_models():
    """Warm up all ML models on startup to prevent cold-start delays"""
    print("🔥 Warming up all models...")
    
    # Warm up Whisper models - ONLY MEDIUM.EN for streaming
    try:
        from interview_analyzer import model_manager
        
        # ONLY warm up medium.en (fast for live processing)
        print("🔄 Warming up Whisper medium.en model (fast for live)...")
        medium_model = model_manager.get_model("medium.en")
        
        # Create a short dummy audio for transcription warmup
        sample_rate = 16000
        duration = 1  # seconds (reduced from 2)
        dummy_audio = np.random.randint(-1000, 1000, duration * sample_rate, dtype=np.int16)
        dummy_audio_path = "temp_warmup_audio.wav"
        
        # Save as WAV
        save_pcm_as_wav(dummy_audio.tobytes(), dummy_audio_path)
        
        # Transcribe with medium model only
        print("🔄 Running quick test transcription...")
        try:
            segments, info = medium_model.transcribe(
                dummy_audio_path,
                language="en",
                beam_size=2,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            # Consume the generator
            segments_list = list(segments)
            print(f"✅ Medium model warmed up (transcribed {len(segments_list)} segments)")
        except Exception as e:
            print(f"⚠️ Medium model warmup failed: {e}")
        
        # Clean up
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
            
    except Exception as e:
        print(f"⚠️ Model warmup failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ All models warmed up (medium.en only for streaming)")

# ========== WEBSOCKET EVENT HANDLERS (Real-time Streaming) ==========

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'success'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection and clean up streaming sessions"""
    print(f"Client disconnected: {request.sid}")
    
    # Clean up any streaming sessions for this socket ID
    sessions_to_remove = []
    for session_key, session_data in streaming_sessions.items():
        if session_key[1] == request.sid:  # session_key is (user_id, socket_id)
            try:
                if 'session' in session_data and session_data['session']:
                    session_data['session'].stop()
                    print(f"Cleaned up streaming session for user {session_key[0]}")
            except Exception as e:
                print(f"Error stopping session for user {session_key[0]}: {e}")
            sessions_to_remove.append(session_key)
    
    # Remove the sessions
    for session_key in sessions_to_remove:
        if session_key in streaming_sessions:
            del streaming_sessions[session_key]
            print(f"Removed session {session_key} from tracking")
    
    print(f"Active sessions remaining: {len(streaming_sessions)}")
    
@socketio.on('trigger_backend_ready')
def handle_trigger_backend_ready(data):
    """Handle the trigger to send backend_ready from main context"""
    user_id = data.get('user_id')
    sid = data.get('sid')
    session_key = (user_id, sid)
    
    if session_key in streaming_sessions:
        room = streaming_sessions[session_key].get('room')
        if room:
            print(f"🔄 Sending backend_ready from main context for user {user_id}")
            socketio.emit('backend_ready', {'status': 'ready'}, room=room)
            print(f"✅ backend_ready sent successfully")
        else:
            print(f"⚠️ No room found for session {session_key}")

@socketio.on("stop_interview")
def stop_interview(data):
    """Stop the live interview and perform final analysis using pre-aggregated stats"""
    user_id = data.get('user_id')
    sid = request.sid
    session_key = (user_id, sid)

    if session_key not in streaming_sessions:
        print(f"⚠️ Session {session_key} not found in streaming sessions")
        return

    session_data = streaming_sessions[session_key]
    room = session_data['room']

    try:
        # 1️⃣ IMMEDIATELY set turn to DONE to stop any further processing
        session_data["turn"] = "DONE"
        session_data["destroyed"] = True
        
        # 2️⃣ Stop AssemblyAI session
        if 'session' in session_data and session_data['session']:
            session_data['session'].stop()

        # 3️⃣ Combine final transcripts - FIXED: Ensure all text is captured
        full_transcript = " ".join(session_data.get('final_text', []))
        
        # If transcript is empty, try to get from AssemblyAI session
        if not full_transcript and 'session' in session_data:
            # Try to get any pending transcription
            try:
                if hasattr(session_data['session'], 'get_final_transcript'):
                    pending_text = session_data['session'].get_final_transcript()
                    if pending_text:
                        full_transcript = pending_text
            except:
                pass

        # 4️⃣ Get incremental statistics
        stats = session_data.get('stats', RunningStatistics())
        final_stats = stats.get_current_stats()

        print(
            f"📊 Final aggregated stats | "
            f"WPM={final_stats.get('wpm', 0):.1f}, "
            f"PauseRatio={final_stats.get('pause_ratio', 0):.3f}, "
            f"TotalWords={final_stats.get('total_words', 0)}"
        )

        # 5️⃣ FINAL METRICS
        expected_answer = session_data.get('current_question', '')
        results = finalize_interview(
            stats=stats,
            user_answer=full_transcript,
            expected_answer=expected_answer
        )

        # 6️⃣ Emit FINAL result BEFORE cleanup
        analysis_results = {
            'success': True,
            'processing_method': 'incremental_fast',
            'transcript': full_transcript,
            'metrics': results.get('metrics', {}),
            'semantic_similarity': results.get('semantic_similarity', 0),
            'analysis_valid': results.get('analysis_valid', False),
            'total_duration': final_stats.get('total_duration', 0),
            'speaking_time': final_stats.get('speaking_time', 0),
            'total_words': final_stats.get('total_words', 0)
        }

        emit('interview_complete', analysis_results, room=room)
        print(f"🛑 Interview stopped for user {user_id} — FINAL metrics delivered")

        # 7️⃣ Cleanup session AFTER emitting results
        # Remove any pending audio chunks
        session_data["user_audio_chunks"] = []
        session_data["interviewer_audio_chunks"] = []
        session_data["early_buffer"] = []
        
        # Mark as destroyed
        session_data["destroyed"] = True
        
        # Small delay before removing to ensure all events are processed
        import threading
        def delayed_cleanup():
            time.sleep(2)  # Wait 2 seconds
            if session_key in streaming_sessions:
                del streaming_sessions[session_key]
                print(f"🧹 Cleaned up session {session_key}")
        
        threading.Thread(target=delayed_cleanup).start()

    except Exception as e:
        print(f"❌ Error stopping interview for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        emit('interview_error', {'error': str(e)}, room=room)

@socketio.on('start_interview')
def start_interview(data):
    """Start live interview with AssemblyAI streaming AND audio recording"""
    user_id = data.get('user_id')
    sid = request.sid
    session_key = (user_id, sid)
    
    # 🔥 CRITICAL: Check if session already exists
    if session_key in streaming_sessions:
        print(f"⚠️ Interview already in progress for user {user_id}, cleaning up old session")
        try:
            old_session = streaming_sessions[session_key]
            if 'session' in old_session and old_session['session']:
                old_session['session'].stop()
        except:
            pass
        del streaming_sessions[session_key]
    
    # Initialize session BEFORE any other operations
    room = f"interview_{user_id}"
    join_room(room)
    
    # Initialize fresh session
    streaming_sessions[session_key] = {
        "turn": "INTERVIEWER",  # Start with interviewer turn
        "current_question": None,
        "session": None,
        "room": room,
        "early_buffer": [],
        "final_text": [],
        "user_audio_chunks": [],
        "interviewer_audio_chunks": [],
        "stats": RunningStatistics(),
        "user_id": user_id,
        "chunk_count": 0,
        "ready": False,
        "primed": False,
        "buffer_count": 0
    }

    # Get timestamp for audio file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"interview_{user_id}_{timestamp}.wav"
    audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    
    # Store audio file info
    streaming_sessions[session_key]["audio_filename"] = audio_filename
    streaming_sessions[session_key]["audio_filepath"] = audio_filepath

    # --------------------------------
    # PARTIAL TRANSCRIPTS
    # --------------------------------
    def on_partial(text):
        session = streaming_sessions.get(session_key)
        
        # 1. Update the frontend
        try:
            socketio.emit('live_transcript', {'text': text}, room=room)
        except Exception as e:
            print(f"Error sending partial transcript: {e}")

        # 2. ✅ THE FIX: Update Silence Timer HERE
        # If text is appearing, the human is definitely speaking.
        if session and text.strip():
            session["last_voice_time"] = time.time()

    # --------------------------------
    # FINAL TRANSCRIPTS (AGENT LOOP) - FIXED VERSION
    # --------------------------------
    def on_final(text):
        session = streaming_sessions.get(session_key)
        
        if not session or session.get("destroyed"): 
            print(f"⚠️ Session destroyed, ignoring transcription: {text}")
            return
        
        # Only accept transcription during USER turn or if we're finalizing
        if session.get("turn") != "USER" and not session.get("finalizing"):
            print(f"⚠️ Not USER turn ({session.get('turn')}), ignoring: {text}")
            return

        text = text.strip()
        if not text: 
            return

        print(f"📝 Buffered Final: {text}")
        
        # Append to final_text list
        if "final_text" not in session:
            session["final_text"] = []
        
        session["final_text"].append(text)
        
        # Also update stats
        if "stats" in session:
            session["stats"].update_transcript(text)
        
        # ✅ Reset timer on final sentence completion
        session["last_voice_time"] = time.time()

    streaming_sessions[session_key]["on_final"] = on_final

    def on_error(error):
        """Handle streaming errors"""
        print(f"❌ Streaming error for user {user_id}: {error}")

    def on_ready():
        socketio.start_background_task(
            notify_backend_ready,
            user_id,
            sid
        )
                
    try:
        # Try to create AssemblyAI WebSocket streaming session
        use_mock = False
        try:
            # 🔥 FIXED: Use simple config without invalid parameters
            session = AssemblyAIWebSocketStreamer(
                on_partial=on_partial, 
                on_final=on_final, 
                on_error=on_error,
                on_ready=on_ready,
            )
            session.start()
            print(f"✅ AssemblyAI streaming session started for user {user_id}")
        
        except Exception as assemblyai_error:
            print(f"⚠️ AssemblyAI not available ({assemblyai_error}), using mock streamer")
            use_mock = True
            session = MockAssemblyAIStreamer(on_partial, on_final, on_error)
            session.start()
            # Call on_ready immediately for mock
            socketio.start_background_task(notify_backend_ready, user_id, sid)

        # Update session
        streaming_sessions[session_key].update({
            "turn": "INTERVIEWER",
            "session": session,        # ✅ STORE STREAMER
            "final_text": [],
            "finalized": False,
        })

        # --------------------------------
        # START AGENT SESSION (INTRO)
        # --------------------------------
        print("🤖 Starting interview agent session...")
        intro_question = InterviewAgentController.start_session(
            session_id=str(user_id),
            user_id=user_id
        )
        
        if not intro_question:
            intro_question = "Tell me briefly about yourself."

        print(f"🤖 Agent intro question: {intro_question}")
        
        # Store question in session
        streaming_sessions[session_key]["current_question"] = intro_question
        streaming_sessions[session_key]["turn"] = "INTERVIEWER"
        streaming_sessions[session_key]["agent_session_id"] = str(user_id)

        # 🔥 CRITICAL: Emit intro question IMMEDIATELY
        socketio.emit(
            "agent_intro_question",
            {"question": intro_question},
            room=room
        )

        emit('interview_started', {
            'status': 'success', 
            'use_mock': use_mock,
            'audio_filename': audio_filename,
            'priming_duration': 0,  # 🔥 NO DELAY
            'requires_buffering': False,  # 🔥 NO BUFFERING NEEDED
            'intro_question': intro_question
        }, room=room)
        
        print(f"✅ Live interview fully initialized for user {user_id}")

    except Exception as e:
        print(f"❌ Failed to start live interview: {e}")
        import traceback
        traceback.print_exc()
        emit('interview_error', {'error': str(e)}, room=room)


@socketio.on("interviewer_done")
def interviewer_done(data):
    user_id = data["user_id"]
    sid = request.sid
    session_key = (user_id, sid)

    session = streaming_sessions.get(session_key)
    if not session:
        return
    
    # Prevent if already stopped
    if session.get("turn") == "DONE":
        return

    print(f"🎤 USER turn started for user {user_id}")

    # 1. Flip State (Clock C)
    session["turn"] = "USER"
    session["finalized"] = False
    session["final_text"] = []
    
    # 2. Start Clock B (Silence Timer)
    session["last_voice_time"] = time.time() 

    # 3. Start the Watcher Thread
    if not session.get("silence_thread_started"):
        session["silence_thread_started"] = True
        socketio.start_background_task(silence_watcher, session_key)

@socketio.on("audio_chunk")
def receive_audio(data):
    user_id = data.get("user_id")
    sid = request.sid
    session_key = (user_id, sid)
    session = streaming_sessions.get(session_key)

    # ---- HARD SAFETY ----
    if not session or session.get("destroyed"):
        return
    
    if session.get("turn") == "DONE":
        return
    
    if session.get("turn") != "USER":
        # Still send to AssemblyAI but don't analyze
        audio = data.get("audio")
        if audio and session.get("session"):
            try:
                if isinstance(audio, memoryview):
                    audio_bytes = audio.tobytes()
                elif isinstance(audio, bytearray):
                    audio_bytes = bytes(audio)
                elif isinstance(audio, bytes):
                    audio_bytes = audio
                elif isinstance(audio, str):
                    import base64
                    audio_bytes = base64.b64decode(audio)
                else:
                    return
                
                session["session"].send_audio(audio_bytes)
            except:
                pass
        return
    
    if session.get("finalized"):
        return

    audio = data.get("audio")
    if audio is None:
        return

    try:
        # ---- Normalize audio to bytes ----
        if isinstance(audio, memoryview):
            audio_bytes = audio.tobytes()
        elif isinstance(audio, bytearray):
            audio_bytes = bytes(audio)
        elif isinstance(audio, bytes):
            audio_bytes = audio
        elif isinstance(audio, str):
            import base64
            audio_bytes = base64.b64decode(audio)
        else:
            return

        if len(audio_bytes) < 2:
            return

        # ---- Ensure buffers exist ----
        session.setdefault("user_audio_chunks", [])
        session.setdefault("stats", RunningStatistics())

        # ---- 1️⃣ AssemblyAI (safe) ----
        session["user_audio_chunks"].append(audio_bytes)
        if session.get("session"):
            session["session"].send_audio(audio_bytes)

        # ---- 2️⃣ Fast incremental analysis ----
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)

        # 🔥 FIXED: Use the OLD analyze_audio_chunk_fast but with FIXED silence detection
        from interview_analyzer import analyze_audio_chunk_fast as old_analyze
        
        # Temporary fix: modify the function parameters
        import librosa
        
        # Convert for analysis
        audio_float = pcm.astype(np.float32) / 32768.0
        duration = len(audio_float) / 16000
        
        # 🔥 FIX THE BUG: Use proper VAD parameters (changed from 30 to 25 for better sensitivity)
        intervals = librosa.effects.split(
            audio_float, 
            top_db=25,  # 🔥 CHANGED: More sensitive to speech
            frame_length=2048,
            hop_length=512
        )
        
        # Calculate REALISTIC speaking time
        speaking_time = sum((e - s) / 16000 for s, e in intervals)
        
        # 🔥 REMOVED: Artificial 85% cap that was causing undercounting
        # speaking_time = min(speaking_time, duration * 0.85)  # This line is removed
        
        # Update stats manually
        stats = session["stats"]
        stats.update_time_stats(duration, speaking_time)
        
        # Update pauses - changed threshold from 0.1 to 0.3 to match your function
        pauses = []
        if len(intervals) > 1:
            for i in range(len(intervals) - 1):
                pause = (intervals[i+1][0] - intervals[i][1]) / 16000
                if pause > 0.3:  # 🔥 CHANGED: Only count pauses > 300ms (reduces noise)
                    pauses.append(pause)
        stats.update_pause_stats(pauses)
        
        # Still call old function for pitch/voice quality
        old_analyze(pcm_chunk=pcm, sample_rate=16000, stats=stats)

        # Update silence clock
        before = stats.speaking_time - (duration * 0.3)  # Estimate
        after = stats.speaking_time
        if after > before + 0.05:
            session["last_voice_time"] = time.time()

    except Exception as e:
        print(f"⚠️ audio_chunk handler error: {e}")

        
@socketio.on('interviewer_audio_chunk')
def receive_interviewer_audio(data):
    user_id = data.get('user_id')
    sid = request.sid
    session_key = (user_id, sid)

    session = streaming_sessions.get(session_key)
    if not session or session.get("destroyed"):
        return

    audio_data = data.get('audio')
    if not isinstance(audio_data, bytes):
        return

    session["interviewer_audio_chunks"].append(audio_data)

            
@socketio.on('leave_interview')
def handle_leave_interview(data):
    """Clean up streaming session"""
    user_id = data.get('user_id')
    sid = request.sid
    session_key = (user_id, sid)

    if session_key in streaming_sessions:
        room = streaming_sessions[session_key]['room']
        leave_room(room)

        # Clean up session data
        session = streaming_sessions.get(session_key)
        if session:
            session["destroyed"] = True
            print(f"👋 Session marked destroyed by user {user_id}")

        print(f"👋 User {user_id} left interview room and session cleaned up")

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


@app.route("/api/tts/murf", methods=["POST"])
def murf_tts():
    data = request.get_json()
    text = data.get("text")
    username = data.get("username", "anonymous")  # ✅ SAFE DEFAULT

    if not text:
        return jsonify({"error": "No text provided"}), 400

    api_key = os.getenv("MURF_API_KEY")
    if not api_key:
        return jsonify({"error": "MURF_API_KEY not set"}), 500

    # 1️⃣ Call Murf Generate API
    response = requests.post(
        "https://api.murf.ai/v1/speech/generate",
        headers={
            "Content-Type": "application/json",
            "api-key": api_key
        },
        json={
            "text": text,
            "voiceId": "en-US-natalie",
            "format": "MP3"
        }
    )

    if response.status_code != 200:
        print("❌ Murf error:", response.text)
        return jsonify({"error": "Murf TTS failed"}), 500

    # 2️⃣ Download generated audio
    audio_url = response.json().get("audioFile")
    if not audio_url:
        return jsonify({"error": "No audioFile returned by Murf"}), 500

    audio_data = requests.get(audio_url).content

    # 3️⃣ Save interviewer TTS audio
    user_folder = f"uploads/{username}/interviewer"
    os.makedirs(user_folder, exist_ok=True)

    filename = f"{user_folder}/q_{int(time.time())}.mp3"
    with open(filename, "wb") as f:
        f.write(audio_data)

    print(f"🔊 Interviewer audio saved at: {filename}")

    # 4️⃣ Return audio to frontend
    return Response(audio_data, mimetype="audio/mpeg")


@app.route('/api/logout', methods=['POST'])
@jwt_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/profile', methods=['GET'])
@jwt_required
def get_profile():
    skills = json.loads(g.current_user.skills) if g.current_user.skills else []
    return jsonify({
        'user': {
            'id': g.current_user.id,
            'username': g.current_user.username,
            'email': g.current_user.email,
            'full_name': g.current_user.full_name,
            'phone': g.current_user.phone,
            'experience_years': g.current_user.experience_years,
            'skills': skills,
            'resume_filename': g.current_user.resume_filename,
            'created_at': g.current_user.created_at.isoformat()
        }
    })


@app.route('/api/update_profile', methods=['POST'])
@jwt_required
def update_profile():
    data = request.get_json()
    g.current_user.full_name = data.get('full_name', g.current_user.full_name)
    g.current_user.phone = data.get('phone', g.current_user.phone)
    g.current_user.experience_years = data.get('experience_years', g.current_user.experience_years)
    g.current_user.skills = json.dumps(data.get('skills', []))
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
        filename = secure_filename(f"{g.current_user.id}_{file.filename}")
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
            g.current_user.resume_filename = filename
            g.current_user.skills = json.dumps(resume_data['skills'])
            g.current_user.experience_years = resume_data['experience_years']

            # Get job description from form data
            job_description = request.form.get('job_description', '').strip()

            # Analyze resume-job fit if JD provided
            job_fit_analysis = None
            if job_description:
                job_fit_analysis = analyze_resume_job_fit(resume_data, job_description)
                resume_data['job_fit_analysis'] = job_fit_analysis

            # Process resume with FAISS for interview questions
            try:
                chunk_count = process_resume_for_faiss(text, g.current_user.id)
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

        prompt = f"""
Generate 5 HR interview questions for a candidate with:

Skills: {', '.join(skills)}
Experience: {experience} years

Include:
- General HR
- Behavioral
- Leadership (if experienced)
- Skill-specific

Return ONLY valid JSON.
Format:
[
  {{"question": "...", "type": "general"}},
  {{"question": "...", "type": "behavioral"}}
]
"""

        response_text = ollama_generate(prompt)

        if not response_text:
            raise Exception("No response from Ollama")

        try:
            questions = json.loads(response_text)
        except Exception:
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
        variation_seed = data.get('variation_seed', '')

        if not g.current_user.resume_filename:
            return jsonify({'success': False, 'error': 'No resume uploaded'})

        if job_description:
            search_query = f"Generate interview questions for this job: {job_description}"
        else:
            search_query = "Generate technical interview questions based on my experience and skills"

        search_results = search_resume_faiss(
            search_query,
            g.current_user.id,
            top_k=5
        )

        resume_context = "\n".join([result['text'] for result in search_results])

        skills = json.loads(g.current_user.skills) if g.current_user.skills else []
        experience = g.current_user.experience_years

        variation_text = f"(Variation: {variation_seed})" if variation_seed else ""

        prompt = f"""
Based on this resume content and job description,
generate {question_count} targeted interview questions {variation_text}.

Resume Content:
{resume_context}

Candidate Skills: {', '.join(skills)}
Experience: {experience} years
Job Description: {job_description}

Rules:
- Generate {question_count} DIFFERENT questions
- Return ONLY valid JSON array
- Each item format:
  {{
    "question": "...",
    "type": "technical | behavioral | project-based | situational"
  }}
"""

        response_text = ollama_generate(prompt, timeout=90)

        if not response_text:
            raise Exception("No response from Ollama")

        try:
            questions = json.loads(response_text)
        except Exception:
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
        print("Resume-based questions error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})



@app.route('/api/user_stats', methods=['GET'])
@jwt_required
def get_user_stats():
    try:
        sessions_count = InterviewSession.query.filter_by(user_id=g.current_user.id).count()
        completed_sessions = InterviewSession.query.filter(
            InterviewSession.user_id == g.current_user.id,
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
            user_id=g.current_user.id,
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


@app.route("/api/analyze_audio_final", methods=["POST"])
def analyze_audio_final():
    """
    Research-safe final analysis.
    NO post-hoc scoring.
    NO fluency / pitch / feedback.
    """

    try:
        data = request.get_json()
        audio_path = data.get("audio_path")
        expected_answer = data.get("expected_answer", "")
        live_transcript = data.get("transcript", "")

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"success": False, "error": "Audio file not found"}), 404

        # Optional: Whisper transcript (for display only)
        from interview_analyzer import speech_to_text

        try:
            transcript = speech_to_text(
                audio_path,
                model_name="medium.en",
                use_vad=True,
                min_speech_duration=1000
            )
            transcript = transcript.strip() or live_transcript
        except Exception:
            transcript = live_transcript or "[Transcription unavailable]"

        # 🚫 NO METRICS COMPUTED HERE
        # Metrics must ONLY come from streaming analyzer

        return jsonify({
            "success": True,
            "transcript": transcript
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/interview_history', methods=['GET'])
@jwt_required
def get_interview_history():
    """Get user's interview practice session history"""
    try:
        session_type = request.args.get('type', None)

        query = InterviewSession.query.filter_by(user_id=g.current_user.id)

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
    
    # 🔥 IMPROVED WARMUP SEQUENCE
    print("\n" + "="*50)
    print("SERVER INITIALIZATION - PRE-WARMING ALL SERVICES")
    print("="*50)
    
    # Warm up AssemblyAI FIRST (this is critical)
    try:
        print("\n🔥 Stage 1: Warming up AssemblyAI real-time streaming...")
        warmup_success = warmup_assemblyai()
        if warmup_success:
            print("✅ AssemblyAI successfully pre-warmed")
        else:
            print("⚠️ AssemblyAI warmup had issues, but continuing anyway")
        time.sleep(1)  # Give time for warmup to complete
    except Exception as e:
        print(f"⚠️ AssemblyAI warmup failed: {e}")
    
    # Warm up other models
    try:
        print("\n🔥 Stage 2: Warming up ML models...")
        warmup_models()
        print("✅ ML models warmed up")
    except Exception as e:
        print(f"⚠️ Model warmup failed: {e}")
    
    # Preload Sentence Transformer embeddings
    try:
        from sentence_transformers import SentenceTransformer
        print("\n🔥 Stage 3: Preloading sentence transformer...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _ = embedder.encode(["test sentence for warmup"])
        print("✅ Sentence transformer ready")
    except Exception as e:
        print(f"⚠️ Sentence transformer preload failed: {e}")
    
    print("\n" + "="*50)
    print("SERVER READY - All services pre-warmed")
    print("First interview should start with minimal delay")
    print("="*50 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False, allow_unsafe_werkzeug=True)