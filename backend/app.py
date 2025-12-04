from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import PyPDF2
import docx
from io import BytesIO
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interview_prep.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('backend/instance', exist_ok=True)

db = SQLAlchemy(app)
CORS(app)  # Enable CORS for all routes
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    experience_years = db.Column(db.Integer, default=0)
    skills = db.Column(db.Text, nullable=True)  # JSON string of skills
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

# Initialize RAG components
model = None
index = None
metas = None
topic_rules = None

def initialize_rag_system():
    global model, index, metas, topic_rules
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ Warning: No GEMINI_API_KEY found in environment. Gemini features will be disabled.")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test Gemini API connection
        try:
            test_model = genai.GenerativeModel("models/gemini-2.5-flash")
            test_response = test_model.generate_content("Hello")
            print("✅ Gemini API connection successful!")
        except Exception as gemini_error:
            print(f"⚠️ Gemini API error: {gemini_error}")
            return False
        
        # Load RAG components if they exist
        try:
            topic_rules_path = 'config/topic_rules.json'
            with open(topic_rules_path, 'r', encoding='utf-8') as f:
                topic_rules = json.load(f)
            
            faiss_dir = 'data/processed/faiss_gemini'
            index_path = os.path.join(faiss_dir, 'faiss_index_gemini.idx')
            metas_path = os.path.join(faiss_dir, 'metas.json')
            
            index = faiss.read_index(index_path)
            with open(metas_path, 'r', encoding='utf-8') as f:
                metas = json.load(f)
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ RAG system initialized successfully!")
        except Exception as rag_error:
            print(f"⚠️ RAG system files not found: {rag_error}")
            print("📝 App will work without RAG features")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False

# Utility and helper functions (UNCHANGED - keep all your existing functions)
def get_topic_and_subtopic_from_query(query, topic_rules):
    if not topic_rules:
        return None, None
    query_lower = query.lower()
    for rule in topic_rules:
        for keyword in rule['keywords']:
            if keyword in query_lower:
                return rule['topic'], rule['subtopic']
    return None, None

def get_relevant_chunks(query, index, metas, model, k=5):
    if not all([index, metas, model]):
        return []
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, k)
    chunks = [metas[i] for i in I[0]]
    return chunks

def generate_rag_response(query, context):
    try:
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        full_prompt = (
            f"You are an expert computer science interview assistant specializing in DBMS, OOPs, and Operating Systems.\n"
            f"Provide a clear, concise answer based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer in a structured way that would help in interview preparation:"
        )
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def extract_text_from_docx(file_stream):
    try:
        doc = docx.Document(file_stream)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return None

def parse_resume_text(text):
    lines = text.strip().split('\n')
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'html', 'css', 
        'machine learning', 'data science', 'flask', 'django', 'mongodb', 'mysql'
    ]
    skills = []
    for line in lines:
        line_lower = line.lower()
        for keyword in tech_keywords:
            if keyword in line_lower and keyword not in [s.lower() for s in skills]:
                skills.append(keyword.title())
    experience_years = 0
    for line in lines:
        if 'year' in line.lower() and 'experience' in line.lower():
            words = line.split()
            for i, word in enumerate(words):
                if word.isdigit() and i < len(words) - 1 and 'year' in words[i+1].lower():
                    experience_years = int(word)
                    break
    return {
        'skills': skills,
        'experience_years': experience_years,
        'raw_text': text[:1000]
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
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        return jsonify({
            'success': True, 
            'message': 'Login successful',
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
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/profile', methods=['GET'])
@login_required
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
@login_required
def update_profile():
    data = request.get_json()
    current_user.full_name = data.get('full_name', current_user.full_name)
    current_user.phone = data.get('phone', current_user.phone)
    current_user.experience_years = data.get('experience_years', current_user.experience_years)
    current_user.skills = json.dumps(data.get('skills', []))
    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile updated successfully'})

@app.route('/api/upload_resume', methods=['POST'])
@login_required
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
@login_required
def rag_query():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Check if RAG system is available
        if not all([index, metas, model, topic_rules]):
            return jsonify({'error': 'RAG system not initialized. Please contact administrator.'}), 500
        
        topic, subtopic = get_topic_and_subtopic_from_query(user_query, topic_rules)
        if topic and subtopic:
            augmented_query = f"Question about {subtopic} in {topic}: {user_query}"
        else:
            augmented_query = user_query
        relevant_chunks = get_relevant_chunks(augmented_query, index, metas, model)
        context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        response_text = generate_rag_response(user_query, context_text)
        response_data = {
            'success': True,
            'query': user_query,
            'answer': response_text,
            'detected_topic': topic,
            'detected_subtopic': subtopic,
            'source_count': len(relevant_chunks),
            'sources': [{'id': chunk['id'], 'text': chunk['text'][:200] + '...'} for chunk in relevant_chunks[:3]]
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/hr_questions', methods=['POST'])
@login_required
def generate_hr_questions():
    try:
        skills = json.loads(current_user.skills) if current_user.skills else []
        experience = current_user.experience_years
        
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        
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
@login_required
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
        'rag_initialized': all([model is not None, index is not None, metas is not None, topic_rules is not None])
    })

@app.route('/api/save_interview_session', methods=['POST'])
@login_required
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
@login_required
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