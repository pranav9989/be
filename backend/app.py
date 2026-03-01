#!/usr/bin/env python3
"""
app.py — Flask app (Mistral-powered Adaptive Interview)
Drop-in replacement for your reference app.py. Fully migrated to Mistral API
for RAG, Mock Interviews, and Coding exercises.
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

from models import User, InterviewSession, StudyActionPlan, UserMastery

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
# import google.generativeai as genai # REMOVED
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

# Add these imports
from agent.adaptive_controller import AdaptiveInterviewController

# Initialize adaptive controller
adaptive_controller = AdaptiveInterviewController()

import requests
from flask import request, Response, jsonify
import os

# Removed Ollama in favor of Mistral (migrated to rag.py)

class SpeechMetrics:
    """
    Tracks true speech and silence metrics for research-grade analysis.
    """
    def __init__(self):
        self.session_start = time.time()
        self.session_end = None
        
        self.speaking_time = 0.0
        self.current_silence = 0.0
        
        self.long_pause_count = 0
        self.last_audio_timestamp = None
        
        self.questions_answered = 0
        self.last_speech_end_time = None  # When last speech segment ended

class MockAssemblyAIStreamer:
    """
    Mock streamer for testing without AssemblyAI API key.
    This version does NOT use Whisper.
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
        print("🎭 Mock AssemblyAI streamer started (no transcription)")

    def send_audio(self, audio_bytes):
        """Accumulate audio chunks (no transcription)"""
        if self.is_active:
            self.audio_chunks.append(audio_bytes)

    def stop(self):
        """Mock stop - notify that audio was received but no transcription"""
        if not self.is_active:
            return

        try:
            self.is_active = False
            print(f"🎭 Mock streamer stopping with {len(self.audio_chunks)} audio chunks")

            if self.audio_chunks:
                # Just notify that audio was received but don't transcribe
                self.on_final("[Audio received but transcription disabled]")
                print(f"🎭 Mock notification sent")

        except Exception as e:
            print(f"🎭 Mock streamer error: {e}")
            if self.on_error:
                self.on_error(str(e))


# Load environment variables
load_dotenv()

# -------------------- App config --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Robust Absolute Paths for Windows/Linux
BASE_DIR = os.path.abspath(os.path.dirname(__file__)) # /backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # /root
INSTANCE_PATH = os.path.join(PROJECT_ROOT, 'instance')
UPLOAD_PATH = os.path.join(PROJECT_ROOT, 'uploads')

os.makedirs(INSTANCE_PATH, exist_ok=True)
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'processed'), exist_ok=True)

# Database in absolute path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(INSTANCE_PATH, 'interview_prep.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key-here')
app.config['JWT_ACCESS_TOKEN_EXPIRE'] = timedelta(hours=24)

# DB + auth
from models import db
db.init_app(app)
# Allow credentials so React (on different origin) can use cookie sessions
# In app.py, after db = SQLAlchemy(app)

# Import models to ensure they're registered
from models import UserMastery, InterviewSession, QuestionHistory, SubtopicMastery

# Create tables + auto-migrate new columns
with app.app_context():
    db.create_all()

    # Idempotent migration: add reset_token columns if they don't exist yet
    # (db.create_all does NOT alter existing tables, so we do it manually)
    from sqlalchemy import text
    with db.engine.connect() as _conn:
        for _col, _type in [("reset_token", "TEXT"), ("reset_token_expiry", "DATETIME")]:
            try:
                _conn.execute(text(f"ALTER TABLE user ADD COLUMN {_col} {_type}"))
                _conn.commit()
                print(f"[Migration] Added column: user.{_col}")
            except Exception:
                pass  # Column already exists — safe to ignore


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

     # ----- ADJUST TOTAL DURATION FOR FINAL SILENCE -----
    if "stats" in session:
        last_voice = session.get("last_voice_time")
        if last_voice:
            final_silence = time.time() - last_voice
            if final_silence > 0:
                old_duration = session["stats"].total_duration
                session["stats"].total_duration = max(0, old_duration - final_silence)
                print(f"⏱️ Removed final wait {final_silence:.1f}s from total duration "
                      f"({old_duration:.1f}s → {session['stats'].total_duration:.1f}s)")
    # --------------------------------------------------------

    # 3. Call the AI Agent (USE ADAPTIVE CONTROLLER)
    try:
        room = session["room"]
        question = session.get("current_question")
        adaptive_session_id = session.get("adaptive_session_id")
        
        if not adaptive_session_id:
            print(f"❌ No adaptive_session_id found for session {session_key}")
            return
        
        # 🔥 Get expected answer for this question from RAG FIRST
        expected_answer = None
        try:
            from rag import agentic_expected_answer
            sampled_concepts = session.get("current_sampled_concepts", []) if session else []
            expected_answer, _ = agentic_expected_answer(question, sampled_concepts)
            print(f"📝 Generated expected answer: {expected_answer[:100]}...")

        except Exception as e:
            print(f"⚠️ Could not get expected answer: {e}")
            import traceback
            traceback.print_exc()
            expected_answer = ""  # Fixed fallback to prevent scoring against the question itself
        
        # 🔥 Call handle_answer with the expected_answer parameter
        with app.app_context():
            agent_response = adaptive_controller.handle_answer(
                session_id=adaptive_session_id,
                answer=final_answer,
                expected_answer=expected_answer,  # ✅ Pass it here!
                stress_test=session.get("stress_test", False) # Pass curveball intention
            )

        # Check if we got an error
        if "error" in agent_response:
            print(f"❌ Agent error: {agent_response['error']}")
            return

        # Get next question from response
        next_question = None
        if agent_response.get("action") in ["FOLLOW_UP", "SIMPLIFY", "DEEPEN", "MOVE_TOPIC"]:
            next_question = agent_response.get("question")
        
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

        # 🔥 Calculate and store Q&A scores WITH expected answer
        if "stats" in session:
            from interview_analyzer import calculate_semantic_similarity, calculate_keyword_coverage
            
            # Check for silent answer
            is_silent = (final_answer == "[User remained silent]" or len(final_answer.strip()) < 5)
            if is_silent:
                semantic_score = 0.0
                keyword_score = 0.0
                print(f"📊 Silent answer detected, setting scores to 0")
            else:
                # Calculate scores (compare with expected answer)
                semantic_score = calculate_semantic_similarity(final_answer, expected_answer)
                keyword_score = calculate_keyword_coverage(final_answer, question)
            
            # Record in stats with expected answer
            session["stats"].record_qa_pair(question, final_answer, expected_answer, semantic_score, keyword_score)
            
            # 🔥 DON'T INCREMENT HERE - IT'S ALREADY INCREMENTED IN SILENCE_WATCHER
            # session["questions_answered"] = session.get("questions_answered", 0) + 1

            print(f"📊 Q&A Scores - Semantic: {semantic_score:.3f}, Keyword: {keyword_score:.3f}")
            print(f"   Expected answer used: {expected_answer[:100]}...")
            
        if next_question:
            if session.get("terminated"):
                print("🚫 Session terminated — skipping agent_next_question emit")
                return

            session["current_question"] = next_question
            session["current_topic"] = agent_response.get("topic", session["current_topic"])
            session["current_subtopic"] = agent_response.get("subtopic", session.get("current_subtopic"))
            session["difficulty"] = agent_response.get("difficulty", session["difficulty"])
            
            # Record the next question before emitting
            if "stats" in session:
                session["stats"].record_question(next_question)
                # 🔥 Record question end time for latency tracking
                session["stats"].record_question_end()
                session["first_voice_recorded"] = False
            
            socketio.emit(
                "agent_next_question",
                {"question": next_question},
                room=room
            )
        elif agent_response.get("action") == "FINALIZE":
            # Interview is complete
            print("🎉 Interview complete - finalizing...")
            # Let the stop_interview handler take over
            stop_interview({"user_id": session_key[0]}, sid=session_key[1])
            
    except Exception as e:
        print(f"❌ Error in agent loop: {e}")
        traceback.print_exc()

    # 5. Clean up thread flags
    session["silence_thread_started"] = False

def silence_watcher(session_key, timeout=15):
    """
    Clock B: The Logic Engine.
    Monitors time since last voice activity. Owns the 'finalize' decision.
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
        
        # Log every 2 seconds for better visibility
        if time.time() - last_log_time >= 2:
            print(f"⏰ Silence elapsed: {elapsed:.1f}s")
            last_log_time = time.time()
        
        # 🔥 Send timer update to frontend every second
        try:
            # Calculate time remaining in the overall session (30 minutes total)
            if "start_time" in session:
                session_start = session.get("start_time", time.time())
                elapsed_total = time.time() - session_start
                time_remaining = max(0, 30 * 60 - int(elapsed_total))  # 30 minutes in seconds
                
                socketio.emit('timer_update', {
                    'time_remaining': time_remaining,
                    'turn_time_elapsed': elapsed  # Time spent on current turn
                }, room=session.get("room"))
        except Exception as e:
            print(f"⚠️ Error sending timer update: {e}")

        # 4. THE DECISION POINT
        if elapsed >= timeout:
            print(f"🛑 Silence limit ({timeout}s) reached. Finalizing.")
            
            # 🔥 DON'T COUNT THIS FINAL 15s AS SILENCE
            # Just finalize and increment question counter
            metrics = session.get("speech_metrics")
            if metrics:
                metrics.questions_answered += 1
                metrics.current_silence = 0  # Reset silence
                print(f"📊 Question #{metrics.questions_answered} completed")
            
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

# -------------------- RAG & Mistral state --------------------
# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "faiss_mistral",
    Path("data") / "processed" / "faiss_mistral"
]
# possible index filenames used across versions
INDEX_CANDIDATES = ["faiss_index_mistral.idx", "index.faiss", "faiss_index_mistral.faiss", "faiss_index.idx"]
METAS_CANDIDATES = ["metas.json", "metas.jsonl", "metas_full.json"]

CONFIG_DIR = PROJECT_ROOT / "config"
TOPIC_RULES_FILE = CONFIG_DIR / "topic_rules.json"
TAXONOMY_FILE = CONFIG_DIR / "taxonomy.json"

# In-memory objects
rag_index = None
rag_metas = None   # dict: int_id -> meta
rag_embedder = None
topic_rules = None

# Mistral model name (allow override via env)
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

def get_mistral_api_key():
    for name in ("MISTRAL_API_KEY",):
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
    import re

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    section_map = {
        "skills": ["skills", "technical skills"],
        "projects": ["projects", "key projects"],
        "internships": ["internship", "internships", "work experience"],
        "certifications": ["certifications", "certificate", "credentials"]
    }

    sections = {k: [] for k in section_map}
    current_section = None

    # ---------- SECTION SPLIT ----------
    for line in lines:
        line_lower = line.lower()

        header_found = False
        for section, keywords in section_map.items():
            if any(line_lower == k for k in keywords):
                current_section = section
                header_found = True
                break

        if header_found:
            continue

        # Stop when new major header detected
        if re.match(r"^[A-Z\s]{3,}$", line) and len(line.split()) <= 4:
            current_section = None
            continue

        if current_section:
            sections[current_section].append(line)

    # ===============================
    # 🔹 SKILLS CLEANING (STRICT)
    # ===============================

    skills = []
    invalid_words = [
        "tools", "technologies", "programming languages",
        "showcasing", "coding", "extra", "curricular"
    ]

    for line in sections["skills"]:
        parts = re.split(r"[•,|:]", line)
        for p in parts:
            clean = p.strip()

            if not clean:
                continue

            clean_lower = clean.lower()

            # remove unwanted labels
            if any(word in clean_lower for word in invalid_words):
                continue

            # remove punctuation
            clean = re.sub(r"[^\w\s+#]", "", clean)

            clean = clean.strip().upper()

            if 2 < len(clean) < 30:
                skills.append(clean)

    skills = sorted(list(dict.fromkeys(skills)))

    # ===============================
    # 🔹 EXPERIENCE FIX
    # ===============================

    experience_years = 0
    match = re.search(r"(\d+)\+?\s*years?", text.lower())
    if match:
        experience_years = int(match.group(1))

    # If internship exists but no years detected → assume 1
    if experience_years == 0 and sections["internships"]:
        experience_years = 1

    # ===============================
    # 🔹 INTERNSHIPS
    # ===============================

    internships = []
    for line in sections["internships"]:
        if "intern" in line.lower():
            internships.append(line.strip())

    internships = list(dict.fromkeys(internships))

    # ===============================
    # 🔹 CERTIFICATIONS
    # ===============================

    certifications = []
    for line in sections["certifications"]:
        if re.search(r"\S+@\S+", line):
            continue
        if re.search(r"\b(b\.?e|bachelor|master|university|college)\b", line, re.I):
            continue
        if 5 < len(line) < 100:
            certifications.append(line.strip())

    certifications = list(dict.fromkeys(certifications))

    # ===============================
    # 🔹 PROJECTS (NAME + TECH ONLY)
    # ===============================

    projects = []

    for line in sections["projects"]:
        if "Tools" in line or "Technologies" in line:
            # Extract project name before month/year
            name_match = re.match(r"•?\s*(.*?)\s+(January|February|March|April|May|June|July|August|September|October|November|December|20\d{2})", line)
            project_name = None
            if name_match:
                project_name = name_match.group(1).strip()
            else:
                project_name = line.split("Tools")[0].strip("• ").strip()

            # Extract tech stack
            tech_match = re.search(r"Technologies:\s*(.*)", line, re.I)
            tech_stack = []
            if tech_match:
                tech_stack = [t.strip().upper() for t in tech_match.group(1).split(",")]

            if project_name:
                projects.append({
                    "name": project_name,
                    "tech_stack": tech_stack
                })

    projects = projects[:5]

    return {
        "skills": skills,
        "experience_years": experience_years,
        "projects": projects,
        "internships": internships,
        "certifications": certifications
    }

def analyze_resume_job_fit(resume_data, job_description):
    """Analyze how well the resume fits the job description using both keyword and semantic matching"""
    if not job_description:
        return None
    
    # Load semantic model (lazy loading to avoid import issues)
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Initialize model if not already in global scope
        if not hasattr(analyze_resume_job_fit, "model"):
            analyze_resume_job_fit.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        model = analyze_resume_job_fit.model
    except Exception as e:
        print(f"⚠️ Semantic model not available: {e}")
        model = None

    resume_skills = set([s.lower() for s in resume_data.get('skills', [])])
    jd_text = job_description.lower()

    # Extract skills from job description
    jd_skills = []
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'html', 'css',
        'machine learning', 'data science', 'flask', 'django', 'mongodb', 'mysql',
        'aws', 'docker', 'kubernetes', 'git', 'linux', 'windows', 'api', 'rest',
        'graphql', 'agile', 'scrum', 'ci/cd', 'jenkins', 'testing', 'unit test',
        'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'typescript', 'vue', 'angular',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'tableau',
        'power bi', 'excel', 'spark', 'hadoop', 'kafka', 'rabbitmq', 'redis',
        'postgresql', 'mongodb', 'dynamodb', 'firebase', 'supabase'
    ]

    for skill in tech_keywords:
        if skill in jd_text and skill.title() not in jd_skills:
            jd_skills.append(skill.title())

    jd_skills_set = set([s.lower() for s in jd_skills])

    # Calculate keyword match scores
    matching_skills = resume_skills.intersection(jd_skills_set)
    missing_skills = jd_skills_set - resume_skills
    
    match_percentage = (len(matching_skills) / len(jd_skills_set) * 100) if jd_skills_set else 0

    # ----- SEMANTIC SIMILARITY SCORE -----
    semantic_score = 0.0
    if model and resume_data.get('projects') or resume_data.get('skills'):
        # Combine resume content for embedding
        resume_text_parts = []
        resume_text_parts.extend(resume_data.get('skills', []))
        resume_text_parts.extend(resume_data.get('projects', []))
        resume_text_parts.extend(resume_data.get('internships', []))
        resume_text_parts.extend(resume_data.get('certifications', []))
        
        resume_text_combined = " ".join(resume_text_parts)
        
        if resume_text_combined.strip():
            try:
                resume_emb = model.encode([resume_text_combined], normalize_embeddings=True)
                jd_emb = model.encode([job_description], normalize_embeddings=True)
                
                semantic_score = float(cosine_similarity(resume_emb, jd_emb)[0][0])
            except Exception as e:
                print(f"⚠️ Semantic similarity calculation failed: {e}")

    # Experience analysis
    experience_required = 0
    exp_match = re.search(r'(\d+)\+?\s*year', jd_text)
    if exp_match:
        try:
            experience_required = int(exp_match.group(1))
        except:
            pass

    experience_fit = "Good fit" if resume_data.get('experience_years', 0) >= experience_required else "May need more experience"

    # Gap severity
    if match_percentage >= 75:
        gap_severity = "Low"
    elif match_percentage >= 50:
        gap_severity = "Medium"
    else:
        gap_severity = "High"

    # Section-level gaps (where missing skills are from)
    section_gaps = {}
    if missing_skills:
        # This is simplified - in production you'd map skills to sections
        section_gaps = {
            'technical': list(missing_skills)[:5],
            'experience': [],
            'certifications': []
        }

    return {
        'matching_skills': list(matching_skills),
        'missing_skills': list(missing_skills),
        'match_percentage': round(match_percentage, 1),
        'semantic_similarity': round(semantic_score, 3),
        'gap_severity': gap_severity,
        'experience_required': experience_required,
        'experience_fit': experience_fit,
        'jd_skills_found': jd_skills,
        'section_gaps': section_gaps
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
def stop_interview(data, sid=None):
    """Stop the live interview and perform final analysis using pre-aggregated stats"""
    user_id = data.get('user_id')
    
    # If sid is provided, use it; otherwise try to get from request
    if sid is None:
        try:
            sid = request.sid
        except RuntimeError:
            sid = None
    
    # Try to find session by user_id and sid
    session_key = None
    if sid:
        session_key = (user_id, sid)
    
    # If not found, try to find any session with this user_id
    if not session_key or session_key not in streaming_sessions:
        for key in streaming_sessions.keys():
            if key[0] == user_id:
                session_key = key
                break
    
    if not session_key or session_key not in streaming_sessions:
        print(f"⚠️ Session for user {user_id} not found in streaming sessions")
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

        # 3️⃣ Combine final transcripts
        full_transcript = " ".join(session_data.get('final_text', []))
        
        # 4️⃣ Get incremental statistics
        stats = session_data.get('stats', RunningStatistics())
        
        # 🔥 FIX #1: Calculate TRUE wall-clock total duration with speech metrics
        start_time = session_data.get("session_start_time")
        end_time = time.time()
        
        if start_time:
            # Total wall-clock session time
            wall_clock_duration = end_time - start_time
            
            # Get number of questions answered from speech_metrics (NEW)
            speech_metrics = session_data.get("speech_metrics")
            if speech_metrics:
                questions_answered = speech_metrics.questions_answered
                print(f"📊 Using speech_metrics.questions_answered: {questions_answered}")
            else:
                questions_answered = session_data.get("questions_answered", 0)
                print(f"⚠️ No speech_metrics found, using session questions_answered: {questions_answered}")
            
            # Subtract forced 15-second waits (one per answered question)
            forced_silence = 15 * questions_answered
            
            # TRUE effective duration (excludes system-imposed waiting)
            true_total_duration = max(0, wall_clock_duration - forced_silence)
            
            # 🔥 GET TRUE SPEAKING TIME FROM SPEECH METRICS (NEW)
            if speech_metrics:
                true_speaking_time = speech_metrics.speaking_time
                long_pause_count = speech_metrics.long_pause_count
                
                # Calculate pause time
                pause_time = max(0, true_total_duration - true_speaking_time)
                pause_ratio = pause_time / true_total_duration if true_total_duration > 0 else 0
                speaking_ratio = true_speaking_time / true_total_duration if true_total_duration > 0 else 0
                
                # Calculate pause frequency (pauses per minute of speaking)
                if true_speaking_time > 0:
                    pause_frequency = long_pause_count / (true_speaking_time / 60)
                else:
                    pause_frequency = 0
                
                print(f"\n⏱️ DURATION CALCULATION (with speech metrics):")
                print(f"   Wall-clock session: {wall_clock_duration:.1f}s")
                print(f"   Questions answered: {questions_answered}")
                print(f"   Forced silence removed: {forced_silence:.1f}s")
                print(f"   Effective duration: {true_total_duration:.1f}s")
                print(f"   True speaking time: {true_speaking_time:.1f}s")
                print(f"   Pause time: {pause_time:.1f}s")
                print(f"   Speaking ratio: {speaking_ratio:.3f}")
                print(f"   Pause ratio: {pause_ratio:.3f}")
                print(f"   Long pauses (>5s blocks): {long_pause_count}")
                print(f"   Pause frequency: {pause_frequency:.2f}/min")
                
                # Override stats with correct values
                stats.total_duration = true_total_duration
                stats.speaking_time = true_speaking_time
                stats.long_pause_count = long_pause_count
            else:
                # Fallback to original calculation
                print(f"⏱️ DURATION CALCULATION (legacy):")
                print(f"   Wall-clock session: {wall_clock_duration:.1f}s")
                print(f"   Questions answered: {questions_answered}")
                print(f"   Forced silence removed: {forced_silence:.1f}s")
                print(f"   Effective duration: {true_total_duration:.1f}s")
                
                # 🔥 CRITICAL: Override the stats.total_duration with wall-clock based value
                stats.total_duration = true_total_duration
        else:
            print(f"⚠️ No session_start_time found, using stats.total_duration = {stats.total_duration:.1f}s")
        
        # Get final stats after duration correction
        final_stats = stats.get_current_stats()

        print(
            f"\n📊 Final aggregated stats | "
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

        # 6️⃣ Emit FINAL result
        analysis_results = {
            'success': True,
            'processing_method': 'incremental_fast',
            'transcript': full_transcript,
            'conversation': final_stats.get('conversation', full_transcript),
            'metrics': results.get('metrics', {}),
            'semantic_similarity': results.get('semantic_similarity', 0),
            'analysis_valid': results.get('analysis_valid', False),
            'total_duration': final_stats.get('total_duration', 0),
            'speaking_time': final_stats.get('speaking_time', 0),
            'total_words': final_stats.get('total_words', 0),
            'qa_pairs': final_stats.get('qa_pairs', [])
        }

        # 💾 SAVE TO DATABASE
        with app.app_context():
            try:
                from models import db, InterviewSession
                from datetime import datetime
                import json
                
                questions = []
                # Restructure answers into the Mock interview format so the frontend can display evaluations
                formatted_answers = {'user_answers': {}, 'evaluations': {}}
                
                for idx, qa in enumerate(final_stats.get('qa_pairs', [])):
                    questions.append({'question': qa.get('question', '')})
                    formatted_answers['user_answers'][str(idx)] = qa.get('answer', '')
                    
                    similarity = qa.get('similarity', 0)
                    kw_coverage = qa.get('keyword_coverage', 0)
                    
                    formatted_answers['evaluations'][str(idx)] = {
                        'ideal_answer': qa.get('expected_answer', ''),
                        'grade': 'A' if similarity > 0.8 else 'B' if similarity > 0.6 else 'C' if similarity > 0.4 else 'D',
                        'score': int(similarity * 100),
                        'strengths': 'Good coverage' if kw_coverage > 0.5 else 'Needs more detail',
                        'improvements': 'Focus on key terms' if kw_coverage <= 0.5 else 'Expand concepts further'
                    }

                session_record = InterviewSession(
                    user_id=user_id,
                    session_type='agentic',
                    questions=json.dumps({'questions': questions, 'answers': formatted_answers}),
                    created_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    score=results.get('semantic_similarity', 0) * 100,
                    duration=final_stats.get('total_duration', 0)
                )
                db.session.add(session_record)
                db.session.commit()
                print(f"✅ Agentic Voice session {session_record.id} saved to DB for user {user_id}")
            except Exception as db_err:
                db.session.rollback()
                print(f"❌ Failed to save Agentic Voice session: {db_err}")

        socketio.emit('interview_complete', analysis_results, room=room)
        print(f"🛑 Interview stopped for user {user_id} — FINAL metrics delivered")

        # 7️⃣ Cleanup
        session_data["user_audio_chunks"] = []
        session_data["interviewer_audio_chunks"] = []
        session_data["early_buffer"] = []
        session_data["destroyed"] = True
        
        import threading
        def delayed_cleanup():
            time.sleep(2)
            if session_key in streaming_sessions:
                del streaming_sessions[session_key]
                print(f"🧹 Cleaned up session {session_key}")
        
        threading.Thread(target=delayed_cleanup).start()

    except Exception as e:
        print(f"❌ Error stopping interview for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('interview_error', {'error': str(e)}, room=room)

@socketio.on('start_interview')
def start_interview(data):
    """Start live interview with AssemblyAI streaming AND audio recording"""
    with app.app_context():  # 🔥 ONE context block for the entire function
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
        
        # Get user info for personalization from database
        from models import User
        user = db.session.get(User, user_id)  # ✅ Now safely inside context
        user_name = user.full_name if user and user.full_name else ""
        
        # Start adaptive session
        import uuid
        session_id = str(uuid.uuid4())
        adaptive_result = adaptive_controller.start_session(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name
        )
        
    
    # Initialize fresh session with adaptive data
    streaming_sessions[session_key] = {
        "turn": "INTERVIEWER",  # Start with interviewer turn
        "current_question": adaptive_result["question"],
        "current_topic": adaptive_result["topic"],
        "difficulty": adaptive_result["difficulty"],
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
        "buffer_count": 0,
        "adaptive_session_id": session_id,  # Store for later use
        "first_voice_recorded": False,
        "session_start_time": time.time(),  # 🔥 CRITICAL: Wall-clock start time
        "questions_answered": 0,
        "speech_metrics": SpeechMetrics(),
        "stress_test": data.get('stress_test', False) # Curveball mode flag
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

        print(f"📝 Final transcript received: {text}")
        
        # Get speech metrics
        metrics = session.get("speech_metrics")
        if not metrics:
            metrics = SpeechMetrics()
            session["speech_metrics"] = metrics
        
        now = time.time()
        
        # 🔥 SPEECH TRACKING LOGIC
        if metrics.last_audio_timestamp is None:
            # First speech in this user turn
            metrics.last_audio_timestamp = now
            metrics.last_speech_end_time = now
            print(f"🎤 First speech detected at {now:.1f}s")
        else:
            # Calculate time since last speech
            delta = now - metrics.last_audio_timestamp
            
            if delta < 2.0:  # Small gap = continuous speech
                # Add the delta to speaking time (this is speech duration)
                metrics.speaking_time += delta
                print(f"🗣️ Speech continued: +{delta:.2f}s (total speaking: {metrics.speaking_time:.1f}s)")
            else:
                # Large gap = silence occurred between speech segments
                silence_segment = delta
                
                # 🔥 COUNT LONG PAUSES IN 5-SECOND BLOCKS
                long_pauses = int(silence_segment // 5)
                if long_pauses > 0:
                    metrics.long_pause_count += long_pauses
                    print(f"⏸️ SILENCE SEGMENT: {silence_segment:.1f}s → {long_pauses} long pause(s) (total: {metrics.long_pause_count})")
                
                # For the speech that just happened, we need to add the duration
                # But we don't have the exact speech duration here - we'll approximate
                # The actual speech duration is captured in audio_chunk handler
                # This is just for cross-segment silence tracking
            
            # Update last audio timestamp
            metrics.last_audio_timestamp = now
            metrics.last_speech_end_time = now
        
        # Append to final_text list
        if "final_text" not in session:
            session["final_text"] = []
        
        session["final_text"].append(text)
        
        # Also update stats (legacy)
        if "stats" in session:
            session["stats"].update_transcript(text)
        
        # Reset silence timer
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

        # 🔥 CRITICAL: Emit intro question from adaptive controller
        socketio.emit(
            "agent_intro_question",
            {"question": adaptive_result["question"]},
            room=room
        )

        emit('interview_started', {
            'status': 'success', 
            'use_mock': use_mock,
            'audio_filename': audio_filename,
            'priming_duration': 0,  # 🔥 NO DELAY
            'requires_buffering': False,  # 🔥 NO BUFFERING NEEDED
            'intro_question': adaptive_result["question"],
            'topic': adaptive_result["topic"],
            'difficulty': adaptive_result["difficulty"],
            'masteries': adaptive_result.get('masteries', {})  # Include mastery data
        }, room=room)
        
        print(f"✅ Live interview fully initialized for user {user_id} with adaptive learning")
        print(f"   Topic: {adaptive_result['topic']}, Difficulty: {adaptive_result['difficulty']}")
        if 'masteries' in adaptive_result:
            print(f"   Current masteries: {adaptive_result['masteries']}")

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
    
    # 🔥 NEW: Record question end for latency tracking
    if "stats" in session:
        session["stats"].record_question_end()
        session["first_voice_recorded"] = False

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
        
        # ---- Ensure speech_metrics exists ----
        if "speech_metrics" not in session:
            from app import SpeechMetrics  # Import if needed
            session["speech_metrics"] = SpeechMetrics()

        # ---- 1️⃣ AssemblyAI (safe) ----
        session["user_audio_chunks"].append(audio_bytes)
        if session.get("session"):
            session["session"].send_audio(audio_bytes)

        # ---- 2️⃣ Fast incremental analysis with SPEECH METRICS ----
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)

        # Import librosa for VAD
        import librosa
        
        # Convert for analysis
        audio_float = pcm.astype(np.float32) / 32768.0
        duration = len(audio_float) / 16000
        
        # Use VAD to detect speech intervals
        intervals = librosa.effects.split(
            audio_float, 
            top_db=20,  # More sensitive to silence
            frame_length=2048,
            hop_length=512
        )
        
        # Calculate speaking time in this chunk
        speaking_time_chunk = sum((e - s) / 16000 for s, e in intervals)
        
        # 🔥 UPDATE SPEECH METRICS WITH CONTINUOUS SPEECH
        metrics = session.get("speech_metrics")
        if metrics:
            # This chunk contains speech
            if speaking_time_chunk > 0.1:
                # Add this chunk's speaking time to total
                metrics.speaking_time += speaking_time_chunk
                print(f"🎤 Chunk speech: +{speaking_time_chunk:.2f}s (total: {metrics.speaking_time:.1f}s)")
                
                # Update last audio timestamp for silence tracking
                now = time.time()
                if metrics.last_audio_timestamp is None:
                    metrics.last_audio_timestamp = now
                    metrics.last_speech_end_time = now
                else:
                    # Check for gap since last speech
                    delta = now - metrics.last_audio_timestamp
                    if delta > 2.0:  # Significant gap detected
                        # This gap will be handled by on_final when next transcript arrives
                        print(f"⏸️ Gap detected: {delta:.1f}s (will be counted as silence when speech resumes)")
                    
                    metrics.last_audio_timestamp = now
                    metrics.last_speech_end_time = now

        # Get stats object
        stats = session["stats"]
        
        # Calculate chunk start time based on session start
        session_start = session.get("session_start_time", time.time())
        chunk_start_time = time.time() - session_start  # Elapsed seconds since session start
        
        # Now update stats with this chunk's duration
        stats.update_time_stats(duration, speaking_time_chunk)
        
        # Update pauses - count pauses within this chunk
        pauses = []
        if len(intervals) > 1:
            for i in range(len(intervals) - 1):
                pause = (intervals[i+1][0] - intervals[i][1]) / 16000
                if pause > 0.3:
                    pauses.append(pause)
        stats.update_pause_stats(pauses)
        
        # Handle chunk boundary for pause tracking
        if hasattr(stats, "handle_chunk_boundary"):
            stats.handle_chunk_boundary(chunk_start_time, chunk_start_time + duration, len(intervals) > 0)
        
        # Still call old function for pitch/voice quality
        from interview_analyzer import analyze_audio_chunk_fast as old_analyze
        old_analyze(pcm_chunk=pcm, sample_rate=16000, stats=stats)
        
        # Update silence clock
        before = stats.speaking_time - (duration * 0.3)
        after = stats.speaking_time
        if after > before + 0.05:
            session["last_voice_time"] = time.time()
            
            # Record first voice for latency tracking
            try:
                if not session.get("first_voice_recorded") and stats.speaking_time > 0.5:
                    if hasattr(stats, "record_first_voice"):
                        stats.record_first_voice()
                        session["first_voice_recorded"] = True
            except Exception as e:
                print(f"⚠️ Error recording first voice: {e}")
        
        # ========== Track chunk timing for debugging ==========
        chunk_index = session.get("chunk_index", 0)
        session["last_chunk_start"] = chunk_start_time
        session["last_chunk_duration"] = duration
        session["chunk_index"] = chunk_index + 1
        # =====================================================

    except Exception as e:
        print(f"⚠️ audio_chunk handler error: {e}")
        import traceback
        traceback.print_exc()
        
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


@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    """
    Step 1 of password reset: Accept email, generate a secure token,
    store it with a 15-minute expiry, and email a reset link.
    Always returns 200 to prevent email enumeration attacks.
    """
    import secrets
    from email_service import send_password_reset_email

    data = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()

    if not email:
        return jsonify({'success': False, 'message': 'Email is required'}), 400

    user = User.query.filter_by(email=email).first()

    if user:
        # Generate a cryptographically secure token (64 hex chars)
        token = secrets.token_urlsafe(48)
        user.reset_token = token
        user.reset_token_expiry = datetime.utcnow() + timedelta(minutes=15)
        db.session.commit()

        email_sent = send_password_reset_email(user.email, token)
        if not email_sent:
            print(f"[WARNING] Reset token generated for {email} but email failed to send.")

    # Always return the same message (security: don't reveal if email exists)
    return jsonify({
        'success': True,
        'message': "If that email is registered, you'll receive a reset link shortly."
    }), 200


@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    """
    Step 2 of password reset: Validate the token, update the password hash,
    and invalidate the token so it cannot be reused.
    """
    data = request.get_json() or {}
    token = (data.get('token') or '').strip()
    new_password = data.get('new_password') or ''

    if not token:
        return jsonify({'success': False, 'message': 'Reset token is missing'}), 400

    if not new_password or len(new_password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400

    user = User.query.filter_by(reset_token=token).first()

    if not user:
        return jsonify({'success': False, 'message': 'Invalid or already-used reset link'}), 400

    if not user.reset_token_expiry or datetime.utcnow() > user.reset_token_expiry:
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        return jsonify({'success': False, 'message': 'This reset link has expired. Please request a new one'}), 400

    # Token is valid — update password and clear the token (one-time use)
    user.password_hash = generate_password_hash(new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    db.session.commit()

    print(f"[Auth] Password successfully reset for user: {user.email}")

    return jsonify({
        'success': True,
        'message': 'Password updated successfully. You can now log in.'
    }), 200


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

            # 🔥 ENFORCE JD REQUIREMENT
            if not job_description:
                return jsonify({
                    'success': False,
                    'message': 'Job Description is required for interview preparation'
                }), 400

            # Analyze resume-job fit (now always runs)
            job_fit_analysis = analyze_resume_job_fit(resume_data, job_description)
            resume_data['job_fit_analysis'] = job_fit_analysis

            # Store JD embedding for later use in interviews
            try:
                from resume_processor import store_jd_embedding
                store_jd_embedding(job_description, g.current_user.id)
                resume_data['jd_embedding_stored'] = True
            except Exception as e:
                print(f"⚠️ Failed to store JD embedding: {e}")
                resume_data['jd_embedding_stored'] = False

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
                'job_description_provided': bool(job_description),
                'job_fit_analysis': job_fit_analysis
            })
        else:
            return jsonify({'success': False, 'message': 'Could not extract text from resume'})



@app.route('/api/user/progress', methods=['GET'])
@jwt_required
def get_user_progress():
    """Get user's learning progress - CLEAR distinction between subtopic and concept status"""
    user_id = g.current_user.id
    
    # Get topic-level masteries
    masteries = UserMastery.query.filter_by(user_id=user_id).all()
    
    # Get subtopic-level masteries
    subtopic_masteries = SubtopicMastery.query.filter_by(user_id=user_id).all()
    
    result = {
        'topics': {},  # Topic-level summary
        'subtopics': {},  # Organized by topic
        'concepts': {},  # Concept-level details (can be large, maybe load on demand)
        'statistics': {
            'total_subtopics': 0,
            'not_started': 0,
            'ongoing': 0,
            'mastered': 0,
            'weak_concepts': 0,
            'strong_concepts': 0,
            'total_questions': 0
        }
    }
    
    # Process topic-level masteries
    for m in masteries:
        data = m.to_dict()
        result['topics'][m.topic] = data
        result['statistics']['total_questions'] += m.questions_attempted
        
        # Get concept counts
        concepts = m.get_concept_masteries()
        for cname, cdata in concepts.items():
            if cdata.get('is_weak'):
                result['statistics']['weak_concepts'] += 1
            if cdata.get('is_strong'):
                result['statistics']['strong_concepts'] += 1
    
    # Process subtopic masteries
    for sm in subtopic_masteries:
        if sm.topic not in result['subtopics']:
            result['subtopics'][sm.topic] = []
        
        subtopic_data = {
            'name': sm.subtopic,
            'mastery': round(sm.mastery_level, 3),
            'attempts': sm.attempts,
            'status': sm.subtopic_status,  # 'not_started', 'ongoing', 'mastered'
            'last_asked': sm.last_asked.isoformat() if sm.last_asked else None
        }
        
        result['subtopics'][sm.topic].append(subtopic_data)
        result['statistics'][sm.subtopic_status] += 1
        result['statistics']['total_subtopics'] += 1
    
    print(f"📊 Progress data loaded:")
    print(f"   Subtopics - Not started: {result['statistics']['not_started']}, "
          f"Ongoing: {result['statistics']['ongoing']}, "
          f"Mastered: {result['statistics']['mastered']}")
    print(f"   Concepts - Weak: {result['statistics']['weak_concepts']}, "
          f"Strong: {result['statistics']['strong_concepts']}")
    
    return jsonify({'success': True, 'data': result})

@app.route('/api/user/sessions', methods=['GET'])
@jwt_required
def get_user_sessions():
    """Get user's interview session history"""
    user_id = g.current_user.id
    
    sessions = InterviewSession.query.filter_by(user_id=user_id)\
                .order_by(InterviewSession.start_time.desc())\
                .limit(20).all()
    
    return jsonify({
        'success': True,
        'sessions': [s.to_dict() for s in sessions]
    })


@app.route('/api/user/topics/<topic>/details', methods=['GET'])
@jwt_required
def get_topic_details(topic):
    """Get detailed information about a specific topic"""
    user_id = g.current_user.id
    
    mastery = UserMastery.query.filter_by(user_id=user_id, topic=topic).first()
    
    if not mastery:
        return jsonify({'success': False, 'error': 'Topic not found'})
    
    # Get recent questions for this topic
    recent = QuestionHistory.query.filter_by(
        user_id=user_id, 
        topic=topic
    ).order_by(QuestionHistory.timestamp.desc()).limit(10).all()
    
    return jsonify({
        'success': True,
        'data': {
            'mastery': mastery.to_dict(),
            'recent_questions': [
                {
                    'question': q.question[:100] + '...' if len(q.question) > 100 else q.question,
                    'semantic_score': q.semantic_score,
                    'keyword_score': q.keyword_score,
                    'timestamp': q.timestamp.isoformat()
                } for q in recent
            ]
        }
    })

@app.route('/api/user/masteries', methods=['GET'])
@jwt_required
def get_user_masteries():
    """Get user's topic masteries for adaptive learning display"""
    user_id = g.current_user.id
    
    masteries = UserMastery.query.filter_by(user_id=user_id).all()
    
    result = {
        'masteries': {},
        'overall': {
            'avg_mastery': 0,
            'weakest_topics': [],
            'strongest_topics': [],
            'total_questions': 0
        }
    }
    
    if not masteries:
        return jsonify({'success': True, 'data': result})
    
    total_mastery = 0
    topics_list = []
    total_questions = 0
    
    for m in masteries:
        data = m.to_dict()
        result['masteries'][m.topic] = data
        total_mastery += m.mastery_level
        topics_list.append((m.topic, m.mastery_level))
        total_questions += m.questions_attempted
    
    if topics_list:
        result['overall']['avg_mastery'] = total_mastery / len(topics_list)
        topics_list.sort(key=lambda x: x[1])
        result['overall']['weakest_topics'] = [t[0] for t in topics_list[:3]]
        result['overall']['strongest_topics'] = [t[0] for t in topics_list[-3:]]
        result['overall']['total_questions'] = total_questions
    
    return jsonify({'success': True, 'data': result})

@app.route('/api/query', methods=['POST'])
@jwt_required
def rag_query():
    """
    Technical Interview Chatbot endpoint
    Returns DETAILED educational explanations for learning
    """
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Use the DETAILED version for learning
        from rag import technical_interview_query
        answer, retrieved = technical_interview_query(user_query)

        # Extract topic info from first retrieved item
        topic = retrieved[0]["topic"] if retrieved else None
        subtopic = retrieved[0]["subtopic"] if retrieved else None

        response_data = {
            'success': True,
            'query': user_query,
            'answer': answer,
            'detected_topic': topic,
            'detected_subtopic': subtopic,
            'source_count': len(retrieved),
            'type': 'detailed_explanation'
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"❌ Technical Query Error: {str(e)}")
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

        from rag import mistral_generate
        response_text = mistral_generate(prompt)

        if not response_text:
            raise Exception("No response from Mistral")

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

@app.route('/api/user/reset_mastery', methods=['POST'])
@jwt_required
def reset_user_mastery():
    """Reset mastery for the user - COMPLETE CLEANUP"""
    try:
        data = request.get_json() or {}
        topic = data.get('topic')  # Optional: reset only specific topic
        
        user_id = g.current_user.id
        
        # Start a transaction
        try:
            if topic:
                # Reset only specific topic
                # Delete subtopic masteries for this topic
                deleted_subtopics = SubtopicMastery.query.filter_by(
                    user_id=user_id, 
                    topic=topic
                ).delete()
                
                # Delete topic mastery
                deleted_topic = UserMastery.query.filter_by(
                    user_id=user_id, 
                    topic=topic
                ).delete()
                
                # Delete question history for this topic
                deleted_questions = QuestionHistory.query.filter_by(
                    user_id=user_id, 
                    topic=topic
                ).delete()
                
                message = f"Reset {topic} mastery successfully"
                print(f"✅ Reset {topic}: {deleted_subtopics} subtopics, {deleted_topic} topic, {deleted_questions} questions")
            else:
                # Reset ALL topics
                # Delete all subtopic masteries
                deleted_subtopics = SubtopicMastery.query.filter_by(user_id=user_id).delete()
                
                # Delete all topic masteries
                deleted_topics = UserMastery.query.filter_by(user_id=user_id).delete()
                
                # Delete all question history
                deleted_questions = QuestionHistory.query.filter_by(user_id=user_id).delete()
                
                message = "Reset all mastery successfully"
                print(f"✅ Reset all: {deleted_subtopics} subtopics, {deleted_topics} topics, {deleted_questions} questions")
            
            db.session.commit()
            
            # Also clear the tracker in memory
            result = adaptive_controller.reset_user_mastery(user_id, topic)
            
            return jsonify({
                'success': True,
                'message': message
            })
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error during reset: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/subtopic_stats', methods=['GET'])
@jwt_required
def get_subtopic_stats():
    """Get subtopic mastery statistics"""
    try:
        stats = adaptive_controller.get_subtopic_stats(g.current_user.id)
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/subtopics/<topic>', methods=['GET'])
@jwt_required
def get_topic_subtopics(topic):
    """Get all subtopics for a topic with their mastery status"""
    try:
        from agent.subtopic_tracker import SubtopicTracker
        tracker = SubtopicTracker(g.current_user.id)
        
        all_subtopics = tracker.SUBTOPICS_BY_TOPIC.get(topic, [])
        attempted = tracker.get_all_attempted_subtopics(topic)
        
        result = []
        for subtopic in all_subtopics:
            data = {
                'name': subtopic,
                'status': 'new',
                'mastery': 0,
                'attempts': 0
            }
            if subtopic in attempted:
                data['status'] = attempted[subtopic].get('status', 'medium') or 'medium'
                data['mastery'] = round(attempted[subtopic].get('mastery_level', 0), 3)
                data['attempts'] = attempted[subtopic].get('attempts', 0)
            result.append(data)
        
        return jsonify({
            'success': True,
            'topic': topic,
            'subtopics': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/subtopic/<topic>/<path:subtopic>/questions', methods=['GET'])
@jwt_required
def get_subtopic_questions(topic, subtopic):
    """Get all questions for a specific subtopic"""
    try:
        user_id = g.current_user.id
        
        questions = QuestionHistory.query.filter_by(
            user_id=user_id,
            topic=topic,
            subtopic=subtopic
        ).order_by(QuestionHistory.timestamp.desc()).limit(10).all()
        
        return jsonify({
            'success': True,
            'questions': [{
                'question': q.question,
                'answer': q.answer,
                'expected_answer': q.expected_answer,
                'semantic_score': q.semantic_score,
                'keyword_score': q.keyword_score,
                # 🔥 REMOVED: coverage_score
                'timestamp': q.timestamp.isoformat()
            } for q in questions]
        })
    except Exception as e:
        print(f"❌ Error fetching subtopic questions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
        
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

        from rag import mistral_generate
        response_text = mistral_generate(prompt, timeout=90)

        if not response_text:
            raise Exception("No response from Mistral")

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


# ============================================================
#  MOCK INTERVIEW — Mistral-powered question generation & evaluation
# ============================================================

@app.route('/api/mock_interview/questions', methods=['POST'])
@jwt_required
def mock_interview_questions():
    """Generate targeted interview questions using Mistral."""
    try:
        from mock_interview_engine import generate_interview_questions
        from resume_processor import search_resume_faiss, get_resume_chunks

        data = request.get_json() or {}
        job_description = data.get('job_description', '').strip()
        question_count  = max(3, min(15, int(data.get('question_count', 8))))
        variation_seed  = data.get('variation_seed', '')

        if not job_description:
            return jsonify({'success': False, 'error': 'Job description is required'}), 400
        if not g.current_user.resume_filename:
            return jsonify({'success': False, 'error': 'Please upload your resume first'}), 400

        search_results = search_resume_faiss(
            f"skills experience projects for: {job_description[:300]}",
            g.current_user.id, top_k=8
        )
        resume_context = "\n".join(r['text'] for r in search_results)

        if not resume_context:
            chunks = get_resume_chunks(g.current_user.id)
            resume_context = "\n".join(c['text'] for c in chunks[:8])

        skills     = json.loads(g.current_user.skills) if g.current_user.skills else []
        experience = g.current_user.experience_years or 0

        questions = generate_interview_questions(
            resume_context=resume_context,
            job_description=job_description,
            skills=skills,
            experience=experience,
            question_count=question_count,
            variation_seed=variation_seed
        )

        return jsonify({'success': True, 'questions': questions, 'question_count': len(questions)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mock_interview/evaluate_answer', methods=['POST'])
@jwt_required
def mock_interview_evaluate_answer():
    """Evaluate a single interview answer using Mistral."""
    try:
        from mock_interview_engine import evaluate_answer
        from resume_processor import search_resume_faiss

        data = request.get_json() or {}
        question        = data.get('question', {})
        user_answer     = data.get('answer', '').strip()
        job_description = data.get('job_description', '')

        if not question or not question.get('question'):
            return jsonify({'success': False, 'error': 'Question data is required'}), 400

        resume_context = ""
        if g.current_user.resume_filename:
            results = search_resume_faiss(question.get('question', ''), g.current_user.id, top_k=3)
            resume_context = " ".join(r['text'] for r in results)[:600]

        evaluation = evaluate_answer(
            question=question,
            user_answer=user_answer,
            resume_context=resume_context,
            job_description=job_description
        )
        return jsonify({'success': True, 'evaluation': evaluation})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mock_interview/session_summary', methods=['POST'])
@jwt_required
def mock_interview_session_summary():
    """Generate overall session performance summary and save to DB."""
    try:
        from mock_interview_engine import generate_session_summary

        data = request.get_json() or {}
        questions       = data.get('questions', [])
        answers         = data.get('answers', {})
        evaluations     = data.get('evaluations', {})
        job_description = data.get('job_description', '')

        answers_int     = {int(k): v for k, v in answers.items()}
        evaluations_int = {int(k): v for k, v in evaluations.items()}

        summary = generate_session_summary(
            questions=questions,
            answers=answers_int,
            evaluations=evaluations_int,
            job_description=job_description
        )

        try:
            session_record = InterviewSession(
                user_id=g.current_user.id,
                session_type='mock_resume',
                score=summary.get('avg_score'),
                questions=json.dumps(
                    [{'q': q.get('question','')[:80]} for q in questions]
                ),
                duration=data.get('duration_minutes', 0)
            )
            db.session.add(session_record)
            db.session.commit()
        except Exception as db_err:
            print(f"[session_summary] DB save non-critical: {db_err}")

        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
#  CODE DEBUGGING INTERVIEW — Adaptive Bug Generation
# ============================================================

@app.route('/api/generate_debugging_scenario', methods=['POST'])
@jwt_required
def generate_debugging_scenario():
    """Generate a buggy code snippet based on the user's weakest concepts."""
    try:
        # Migrated to Mistral in rag.py
        from rag import mistral_generate
        
        # 1. Fetch user's weakest concepts to make it adaptive
        weakest_concept = "Basic Algorithms"
        topic = "General Programming"
        
        masteries = UserMastery.query.filter_by(user_id=g.current_user.id).all()
        weakest_score = 1.0
        
        for m in masteries:
            concept_data = m.get_concept_masteries()
            for concept_name, cd in concept_data.items():
                if cd.get('mastery_level', 1.0) < weakest_score:
                    weakest_score = cd.get('mastery_level', 1.0)
                    weakest_concept = concept_name
                    topic = m.topic

        print(f"🎯 Code Debugging targeting weakest concept: {weakest_concept} in {topic}")

        # 2. Prompt Ollama to generate a buggy scenario
        prompt = f"""
        You are a Senior Staff Engineer conducting a technical interview.
        You need to test the candidate's architectural and debugging skills by presenting them with a broken code block.
        
        Topic: {topic}
        Specific concept to test: {weakest_concept}
        
        Create a 15-line Python code snippet that contains ONE logical or architectural bug related to {weakest_concept}.
        Do NOT include syntax errors. The bug must be logical.
        
        Return ONLY a JSON object with this exact format:
        {{
            "buggy_code": "def example():\\n    ...",
            "topic": "{weakest_concept}",
            "expected_answer": "The bug is on line X where Y happens. To fix it, you need to change Y to Z because..."
        }}
        """
        
        response_text = mistral_generate(prompt, timeout=120)
        
        if not response_text:
            raise Exception("No response from Mistral")
            
        # Extract JSON
        try:
            # Try to find JSON block if Ollama returned markdown
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                    
            scenario = json.loads(response_text)
        except Exception as e:
            print(f"Failed to parse Mistral JSON: {e}\nRaw output: {response_text}")
            # Fallback scenario
            scenario = {
                "buggy_code": "def calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n\n# Note: What happens if the list is empty?",
                "topic": "Edge Cases",
                "expected_answer": "The bug is a potential ZeroDivisionError. If the 'numbers' list is empty, len(numbers) is 0, causing a division by zero. To fix it, check if not numbers: return 0 before calculating the average."
            }

        return jsonify({
            'success': True,
            'scenario': scenario
        })

    except Exception as e:
        print("Debugging Scenario error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# --- New Iterative Code Debugging Routes ---

@app.route('/api/debugging/start', methods=['POST'])
@jwt_required
def start_debugging_session():
    try:
        from models import DebuggingSession
        session = DebuggingSession(user_id=g.current_user.id)
        db.session.add(session)
        db.session.commit()
        return jsonify({'success': True, 'session_id': session.id})
    except Exception as e:
        print("Error starting debugging session:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debugging/next', methods=['POST'])
@jwt_required
def get_next_debugging_challenge():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        from models import DebuggingSession, DebuggingChallenge
        session = DebuggingSession.query.get(session_id)
        if not session or session.user_id != g.current_user.id:
            return jsonify({'success': False, 'error': 'Invalid session'}), 403
            
        # Determine topic targeting based on weakness
        weakest_concept = "General Logic"
        topic = "Programming"
        masteries = UserMastery.query.filter_by(user_id=g.current_user.id).all()
        weakest_score = 1.0
        for m in masteries:
            concept_data = m.get_concept_masteries()
            for name, cd in concept_data.items():
                if cd.get('mastery_level', 1.0) < weakest_score:
                    weakest_score = cd.get('mastery_level', 1.0)
                    weakest_concept = name
                    topic = m.topic

        from debugging_engine import generate_challenge
        prev_challenges = [c.to_dict() for c in session.challenges]
        challenge_data = generate_challenge(weakest_concept, topic, prev_challenges)
        
        if not challenge_data:
            return jsonify({'success': False, 'error': 'AI failed to generate challenge'}), 500
            
        challenge = DebuggingChallenge(
            session_id=session.id,
            language=challenge_data['language'],
            topic=challenge_data['topic'],
            buggy_code=challenge_data['buggy_code'],
            expected_answer=challenge_data['expected_answer'],
            correct_code=challenge_data['correct_code']
        )
        db.session.add(challenge)
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'challenge': challenge.to_dict()
        })
    except Exception as e:
        print("Error fetching next challenge:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debugging/evaluate', methods=['POST'])
@jwt_required
def evaluate_debugging_challenge():
    try:
        data = request.get_json()
        challenge_id = data.get('challenge_id')
        user_explanation = data.get('explanation')
        
        from models import DebuggingChallenge
        challenge = DebuggingChallenge.query.get(challenge_id)
        if not challenge or challenge.session.user_id != g.current_user.id:
            return jsonify({'success': False, 'error': 'Invalid challenge'}), 403
            
        from debugging_engine import evaluate_explanation
        eval_result = evaluate_explanation(challenge.expected_answer, user_explanation, challenge.language)
        
        challenge.user_explanation = user_explanation
        challenge.ai_score = eval_result['score']
        challenge.ai_feedback = eval_result['feedback']
        db.session.commit()
        
        # Update session stats
        session = challenge.session
        challenges_with_scores = [c for c in session.challenges if c.ai_score > 0]
        if challenges_with_scores:
            session.count = len(challenges_with_scores)
            session.avg_score = sum(c.ai_score for c in challenges_with_scores) / len(challenges_with_scores)
            db.session.commit()
            
        return jsonify({
            'success': True, 
            'evaluation': eval_result,
            'correct_code': challenge.correct_code
        })
    except Exception as e:
        print("Error evaluating challenge:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debugging/end', methods=['POST'])
@jwt_required
def end_debugging_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        from models import DebuggingSession
        session = DebuggingSession.query.get(session_id)
        if not session or session.user_id != g.current_user.id:
            return jsonify({'success': False, 'error': 'Invalid session'}), 403
            
        session.end_time = datetime.utcnow()
        
        # Generate final summary
        from rag import mistral_generate
        perf_data = [f"Topic: {c.topic}, Lang: {c.language}, Score: {c.ai_score}" for c in session.challenges]
        prompt = f"Summarize the user's performance across these debugging challenges: {', '.join(perf_data)}. Provide a 2-sentence wrap-up."
        session.summary = mistral_generate(prompt)
        
        db.session.commit()
        return jsonify({'success': True, 'session': session.to_dict()})
    except Exception as e:
        print("Error ending debugging session:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/profile/debugging_history', methods=['GET'])
@jwt_required
def get_debugging_history():
    try:
        from models import DebuggingSession
        sessions = DebuggingSession.query.filter_by(user_id=g.current_user.id).order_by(DebuggingSession.start_time.desc()).all()
        return jsonify({
            'success': True, 
            'history': [s.to_dict() for s in sessions if s.count > 0]
        })
    except Exception as e:
        print("Error fetching debugging history:", e)
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
#  POST-INTERVIEW "ACTION PLAN" GENERATION
# ============================================================

@app.route('/api/generate_action_plan', methods=['POST'])
@jwt_required
def generate_action_plan():
    """Generate a personalized study plan based on user's weak concepts and time available."""
    try:
        from rag import mistral_generate as ollama_generate
        
        data = request.get_json() or {}
        days = int(data.get('days', 7))
        selected_topics = data.get('topics', ['General Programming'])

        # Fetch user's weak concepts for the selected topics
        from rag import mistral_generate as ollama_generate
        weak_concepts_by_topic = {}
        masteries = UserMastery.query.filter_by(user_id=g.current_user.id).all()
        
        for m in masteries:
            if m.topic in selected_topics:
                weak_concepts = []
                concept_data = m.get_concept_masteries()
                for concept_name, cd in concept_data.items():
                    if cd.get('mastery_level', 1.0) < 0.6: # Threshold for 'weak'
                        weak_concepts.append(concept_name)
                
                if weak_concepts:
                    weak_concepts_by_topic[m.topic] = weak_concepts

        # Format weak concepts string for the prompt
        weaknesses_str = ""
        if weak_concepts_by_topic:
            for topic, concepts in weak_concepts_by_topic.items():
                weaknesses_str += f"- {topic}: {', '.join(concepts)}\n"
        else:
            weaknesses_str = "No specific weak concepts identified yet. General review needed for selected topics."

        print(f"📅 Action Plan requested for {days} days on topics: {selected_topics}")

        # Prompt Ollama to generate a markdown action plan
        prompt = f"""
        You are an expert technical interviewer and career coach.
        Create a personalized {days}-day study plan for a software engineering candidate.
        
        The candidate has selected these core topics to study:
        {', '.join(selected_topics)}
        
        Based on their previous interview performance, they have demonstrated weaknesses in these specific technical concepts:
        {weaknesses_str}
        
        Provide a detailed, day-by-day markdown schedule.
        - The schedule should heavily focus on improving their identified weaknesses.
        - Include actionable tasks, resource suggestions (e.g., Leetcode, documentation), and conceptual review goals.
        - Ensure the pace is realistic for {days} days.
        
        CRITICAL FORMATTING RULES:
        1. Return ONLY valid markdown text.
        2. Do NOT wrap the entire response in ```markdown or any backticks.
        3. Do NOT use backticks for the title or headers.
        4. Use standard Markdown headers: # for title, ## for Days, ### for Sessions.
        5. Use standard bullet points (-) for tasks.
        6. No introductory or concluding remarks outside the markdown schedule itself.
        """
        
        from rag import mistral_generate
        response_text = mistral_generate(prompt, timeout=300)
        
        if not response_text:
            raise Exception("No response from Mistral after 300s")

        # Save to database
        try:
            from models import StudyActionPlan
            plan_record = StudyActionPlan(
                user_id=g.current_user.id,
                days=days,
                topics=",".join(selected_topics), # Fixed variable name to selected_topics
                plan_markdown=response_text
            )
            db.session.add(plan_record)
            db.session.commit()
            print(f"✅ Saved Action Plan for user {g.current_user.id}")
        except Exception as db_err:
            print(f"[generate_action_plan] DB save non-critical: {db_err}")

        return jsonify({
            'success': True,
            'plan_markdown': response_text
        })

    except Exception as e:
        print("Action Plan Generation error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/profile/action_plans', methods=['GET'])
@jwt_required
def get_action_plans():
    """Retrieve all saved study action plans for the current user."""
    try:
        # Use the imported StudyActionPlan model correctly
        from models import StudyActionPlan
        plans = StudyActionPlan.query.filter_by(user_id=g.current_user.id).order_by(StudyActionPlan.created_at.desc()).all()
        return jsonify({
            'success': True,
            'action_plans': [plan.to_dict() for plan in plans]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/resume/analysis', methods=['GET'])
@jwt_required
def get_resume_analysis():
    """Return stored resume data for the current user."""
    try:
        user   = g.current_user
        skills = json.loads(user.skills) if user.skills else []
        return jsonify({
            'success': True,
            'has_resume': bool(user.resume_filename),
            'resume_filename': user.resume_filename,
            'skills': skills,
            'experience_years': user.experience_years or 0,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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


# ==============================================================================
# ── DATA SCIENCE CODING PRACTICE ROUTES ───────────────────────────────────────
# ==============================================================================
import coding_engine as ce

@app.route('/api/coding/questions', methods=['POST'])
@jwt_required
def get_coding_questions():
    """Generate Data Science coding questions"""
    data = request.json
    count = int(data.get('question_count', 3))
    difficulty = data.get('difficulty', 'medium')
    
    try:
        questions = ce.generate_coding_questions(question_count=count, difficulty=difficulty)
        return jsonify({'success': True, 'questions': questions})
    except Exception as e:
        print(f"Error generating coding questions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/coding/evaluate', methods=['POST'])
@jwt_required
def evaluate_coding_answer():
    """Evaluate user's SQL or Pandas code"""
    data = request.json
    question = data.get('question', {})
    user_code = data.get('user_code', '')
    language = data.get('language', 'sql')
    
    try:
        evaluation = ce.evaluate_coding_answer(
            question=question,
            user_code=user_code,
            language=language
        )
        return jsonify({'success': True, 'evaluation': evaluation})
    except Exception as e:
        print(f"Error evaluating coding answer: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # Check if required files exist
        import os
        files_exist = all([
            os.path.exists("data/processed/faiss_mistral/index.faiss"),
            os.path.exists("data/processed/faiss_mistral/metas.json"),
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


@app.route('/api/user/history', methods=['GET'])
@jwt_required
def get_user_history():
    """Fetch complete interview history for the user across all session types."""
    try:
        type_filter = request.args.get('type')
        
        # 1. Fetch InterviewSession records
        query = InterviewSession.query.filter_by(user_id=g.current_user.id)
        if type_filter and type_filter != 'debugging':
            query = query.filter_by(session_type=type_filter)
        sessions = query.order_by(InterviewSession.created_at.desc()).all()
        
        history = []
        for s in sessions:
            history.append({
                'id': s.id,
                'session_type': s.session_type,
                'created_at': s.created_at.isoformat(),
                'score': s.score,
                'duration': s.duration,
                'data': json.loads(s.questions) if s.questions else {}
            })
            
        # 2. Fetch DebuggingSession records (if not filtered out)
        if not type_filter or type_filter == 'debugging':
            from models import DebuggingSession
            debug_sessions = DebuggingSession.query.filter_by(user_id=g.current_user.id).filter(DebuggingSession.count > 0).order_by(DebuggingSession.start_time.desc()).all()
            for ds in debug_sessions:
                history.append({
                    'id': f"debug_{ds.id}", # unique ID for frontend
                    'session_type': 'debugging',
                    'created_at': ds.start_time.isoformat(),
                    'score': ds.avg_score * 100, # normalized to 100
                    'duration': None,
                    'data': {
                        'summary': ds.summary,
                        'challenges': [c.to_dict() for c in ds.challenges]
                    }
                })
        
        # Sort combined history by created_at desc
        history.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        print(f"Error fetching history: {e}")
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

        # REMOVE THE WHISPER TRANSCRIPTION ATTEMPT
        # Just use the live transcript from the session
        transcript = live_transcript or "[Transcription only available from AssemblyAI live stream]"

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

@app.route('/api/resume/gap-analysis', methods=['POST'])
@jwt_required
def gap_analysis():
    """Compare user's resume with a job description to find missing skills"""
    try:
        data = request.json
        job_description = data.get('job_description')
        if not job_description:
            return jsonify({'success': False, 'error': 'Job description is required'}), 400

        user = g.current_user
        if not user.skills or user.skills == "[]" or user.skills == "":
            return jsonify({'success': False, 'error': 'No skills found. Please upload your resume first.'}), 400

        try:
            resume_data = json.loads(user.skills)
        except:
            resume_data = user.skills # fallback if it's just raw text
        
        # Add years of experience
        resume_data_payload = {
            "skills": resume_data,
            "experience_years": user.experience_years
        }
        
        from rag import generate_resume_gap_analysis
        result_json_str = generate_resume_gap_analysis(resume_data_payload, job_description)
        
        if not result_json_str:
            return jsonify({'success': False, 'error': 'Failed to generate gap analysis.'}), 500
            
        analysis_data = json.loads(result_json_str)
        return jsonify({
            'success': True,
            'gap_analysis': analysis_data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
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