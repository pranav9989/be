from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

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

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        try:
            skills = json.loads(self.skills) if self.skills else []
        except:
            skills = []

        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "phone": self.phone,
            "experience_years": self.experience_years,
            "skills": skills,
            "resume_filename": self.resume_filename,
            "created_at": self.created_at.isoformat()
        }

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

# backend/models.py - ADD THESE CLASSES AFTER YOUR EXISTING InterviewSession CLASS

class UserMastery(db.Model):
    """Tracks user's mastery across topics (persistent across sessions)"""
    __tablename__ = 'user_mastery'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(50), nullable=False)
    
    # Mastery scores (0-1)
    mastery_level = db.Column(db.Float, default=0.0)
    semantic_avg = db.Column(db.Float, default=0.0)
    keyword_avg = db.Column(db.Float, default=0.0)
    coverage_avg = db.Column(db.Float, default=0.0)
    
    # Statistics
    questions_attempted = db.Column(db.Integer, default=0)
    correct_count = db.Column(db.Integer, default=0)
    avg_response_time = db.Column(db.Float, default=0.0)
    
    # Learning velocity
    mastery_velocity = db.Column(db.Float, default=0.0)
    last_mastery = db.Column(db.Float, default=0.0)
    
    # Difficulty tracking
    current_difficulty = db.Column(db.String(20), default="medium")
    consecutive_good = db.Column(db.Integer, default=0)
    consecutive_poor = db.Column(db.Integer, default=0)
    
    # Concept gaps (stored as JSON)
    missing_concepts = db.Column(db.Text, default='[]')
    weak_concepts = db.Column(db.Text, default='[]')
    strong_concepts = db.Column(db.Text, default='[]')
    
    # Timestamps
    first_attempt = db.Column(db.DateTime, default=datetime.utcnow)
    last_attempt = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('masteries', lazy='dynamic'))
    
    # Unique constraint to ensure one mastery record per user per topic
    __table_args__ = (
        db.UniqueConstraint('user_id', 'topic', name='unique_user_topic'),
    )
    
    def get_missing_concepts(self):
        return json.loads(self.missing_concepts) if self.missing_concepts else []
    
    def set_missing_concepts(self, concepts):
        self.missing_concepts = json.dumps(concepts[:10])  # Keep top 10
    
    def get_weak_concepts(self):
        return json.loads(self.weak_concepts) if self.weak_concepts else []
    
    def set_weak_concepts(self, concepts):
        self.weak_concepts = json.dumps(concepts[:10])
    
    def get_strong_concepts(self):
        return json.loads(self.strong_concepts) if self.strong_concepts else []
    
    def set_strong_concepts(self, concepts):
        self.strong_concepts = json.dumps(concepts[:10])
    
    def update_mastery(self, semantic_score, keyword_score, coverage_score, response_time, missing=None):
        """Update mastery using exponential moving average"""
        # 🔥 SAFETY FIX: Ensure no None values before math operations
        self.semantic_avg = self.semantic_avg or 0.0
        self.keyword_avg = self.keyword_avg or 0.0
        self.coverage_avg = self.coverage_avg or 0.0
        self.mastery_level = self.mastery_level or 0.0
        self.mastery_velocity = self.mastery_velocity or 0.0
        self.last_mastery = self.last_mastery or 0.0
        self.avg_response_time = self.avg_response_time or 0.0

        self.questions_attempted = self.questions_attempted or 0
        self.correct_count = self.correct_count or 0
        self.consecutive_good = self.consecutive_good or 0
        self.consecutive_poor = self.consecutive_poor or 0

        alpha = 0.3  # EMA factor
        
        # Ensure all values are floats and not None
        semantic_score = float(semantic_score) if semantic_score is not None else 0.0
        keyword_score = float(keyword_score) if keyword_score is not None else 0.0
        coverage_score = float(coverage_score) if coverage_score is not None else 0.0
        response_time = float(response_time) if response_time is not None else 0.0
        
        # Update averages
        self.semantic_avg = (alpha * semantic_score) + ((1 - alpha) * (self.semantic_avg or 0.0))
        self.keyword_avg = (alpha * keyword_score) + ((1 - alpha) * (self.keyword_avg or 0.0))
        self.coverage_avg = (alpha * coverage_score) + ((1 - alpha) * (self.coverage_avg or 0.0))
        
        # Store old mastery for velocity calculation
        old_mastery = self.mastery_level or 0.0
        
        # Calculate new mastery
        self.mastery_level = (
            (self.semantic_avg or 0.0) * 0.4 +
            (self.keyword_avg or 0.0) * 0.3 +
            (self.coverage_avg or 0.0) * 0.3
        )
        
        # Update learning velocity
        self.mastery_velocity = self.mastery_level - old_mastery
        self.last_mastery = old_mastery
        
        # Update response time (EMA)
        self.avg_response_time = (alpha * response_time) + ((1 - alpha) * (self.avg_response_time or 0.0))
        
        # Update counters
        self.questions_attempted += 1
        if coverage_score > 0.6:
            self.correct_count += 1
        
        # Update consecutive patterns
        combined_score = (semantic_score * 0.4 + keyword_score * 0.3 + coverage_score * 0.3)
        if combined_score > 0.7:
            self.consecutive_good += 1
            self.consecutive_poor = 0
        elif combined_score < 0.4:
            self.consecutive_poor += 1
            self.consecutive_good = 0
        else:
            self.consecutive_good = 0
            self.consecutive_poor = 0
        
        # Update difficulty
        if self.consecutive_good >= 3:
            self.current_difficulty = "hard"
        elif self.consecutive_poor >= 2:
            self.current_difficulty = "easy"
        else:
            self.current_difficulty = "medium"
        
        # Update missing concepts
        if missing:
            current_missing = self.get_missing_concepts()
            for concept in missing:
                if concept not in current_missing:
                    current_missing.append(concept)
            self.set_missing_concepts(current_missing)
        
        self.last_attempt = datetime.utcnow()
    
    def to_dict(self):
        return {
            'topic': self.topic,
            'mastery_level': round(self.mastery_level, 3),
            'questions_attempted': self.questions_attempted,
            'current_difficulty': self.current_difficulty,
            'missing_concepts': self.get_missing_concepts(),
            'weak_concepts': self.get_weak_concepts(),
            'strong_concepts': self.get_strong_concepts(),
            'learning_velocity': round(self.mastery_velocity, 3)
        }


class AdaptiveInterviewSession(db.Model):
    """Enhanced interview session tracking with adaptive metrics"""
    __tablename__ = 'adaptive_interview_session'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    
    # Session stats
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    duration = db.Column(db.Integer, default=0)
    
    # Performance
    questions_asked = db.Column(db.Integer, default=0)
    avg_semantic = db.Column(db.Float, default=0.0)
    avg_keyword = db.Column(db.Float, default=0.0)
    avg_coverage = db.Column(db.Float, default=0.0)
    overall_score = db.Column(db.Float, default=0.0)
    
    # Learning metrics
    learning_velocity = db.Column(db.Float, default=0.0)
    attention_score = db.Column(db.Float, default=1.0)
    
    # Topics covered (JSON)
    topics_covered = db.Column(db.Text, default='[]')
    
    # Weakest/strongest topics (JSON)
    weakest_topics = db.Column(db.Text, default='[]')
    strongest_topics = db.Column(db.Text, default='[]')
    
    # Relationship
    user = db.relationship('User', backref=db.backref('adaptive_sessions', lazy='dynamic'))
    
    def set_topics_covered(self, topics):
        self.topics_covered = json.dumps(topics)
    
    def get_topics_covered(self):
        return json.loads(self.topics_covered) if self.topics_covered else []
    
    def set_weakest_topics(self, topics):
        self.weakest_topics = json.dumps(topics[:3])
    
    def get_weakest_topics(self):
        return json.loads(self.weakest_topics) if self.weakest_topics else []
    
    def set_strongest_topics(self, topics):
        self.strongest_topics = json.dumps(topics[:3])
    
    def get_strongest_topics(self):
        return json.loads(self.strongest_topics) if self.strongest_topics else []
    
    def update_session_stats(self):
        """Update session statistics from question history"""
        from sqlalchemy import func
        
        # Get all questions for this session
        questions = QuestionHistory.query.filter_by(session_id=self.session_id).all()
        
        if questions:
            self.questions_asked = len(questions)
            self.avg_semantic = sum(q.semantic_score for q in questions) / len(questions)
            self.avg_keyword = sum(q.keyword_score for q in questions) / len(questions)
            self.avg_coverage = sum(q.coverage_score for q in questions) / len(questions)
            
            self.overall_score = (
                self.avg_semantic * 0.4 +
                self.avg_keyword * 0.3 +
                self.avg_coverage * 0.3
            )
    
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'questions_asked': self.questions_asked,
            'avg_semantic': round(self.avg_semantic, 3),
            'avg_keyword': round(self.avg_keyword, 3),
            'avg_coverage': round(self.avg_coverage, 3),
            'overall_score': round(self.overall_score, 3),
            'topics_covered': self.get_topics_covered(),
            'weakest_topics': self.get_weakest_topics(),
            'strongest_topics': self.get_strongest_topics(),
            'learning_velocity': round(self.learning_velocity, 3)
        }


class QuestionHistory(db.Model):
    """Track each question asked across all sessions"""
    __tablename__ = 'question_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    session_id = db.Column(db.String(100), nullable=False, index=True)
    
    # Question details
    topic = db.Column(db.String(50), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)
    expected_answer = db.Column(db.Text, nullable=True)
    
    # Scores
    semantic_score = db.Column(db.Float, default=0.0)
    keyword_score = db.Column(db.Float, default=0.0)
    coverage_score = db.Column(db.Float, default=0.0)
    
    # Analysis signals
    depth = db.Column(db.String(20), default="medium")  # shallow, medium, deep
    confidence = db.Column(db.String(20), default="medium")  # low, medium, high
    key_terms = db.Column(db.Text, default='[]')  # JSON list
    
    # Metadata
    difficulty = db.Column(db.String(20), default="medium")
    response_time = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Missing concepts (JSON)
    missing_concepts = db.Column(db.Text, default='[]')
    
    # Relationship
    user = db.relationship('User', backref=db.backref('question_history', lazy='dynamic'))
    
    def set_missing_concepts(self, concepts):
        self.missing_concepts = json.dumps(concepts[:10])
    
    def get_missing_concepts(self):
        return json.loads(self.missing_concepts) if self.missing_concepts else []
    
    def set_key_terms(self, terms):
        self.key_terms = json.dumps(terms[:10])
    
    def get_key_terms(self):
        return json.loads(self.key_terms) if self.key_terms else []
    
    def to_dict(self):
        return {
            'id': self.id,
            'topic': self.topic,
            'question': self.question[:100] + '...' if len(self.question) > 100 else self.question,
            'semantic_score': round(self.semantic_score, 3),
            'keyword_score': round(self.keyword_score, 3),
            'coverage_score': round(self.coverage_score, 3),
            'difficulty': self.difficulty,
            'depth': self.depth,
            'confidence': self.confidence,
            'response_time': round(self.response_time, 2),
            'timestamp': self.timestamp.isoformat(),
            'missing_concepts': self.get_missing_concepts()
        }
