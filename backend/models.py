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


class UserMastery(db.Model):
    """Tracks user's mastery across topics with COMPLETE concept-level data"""
    __tablename__ = 'user_mastery'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(50), nullable=False)
    sessions_attempted = db.Column(db.Integer, default=0)
    last_session_date = db.Column(db.DateTime, nullable=True)
    
    # Mastery scores (0-1)
    mastery_level = db.Column(db.Float, default=0.0)
    semantic_avg = db.Column(db.Float, default=0.0)
    keyword_avg = db.Column(db.Float, default=0.0)
    
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
    
    # 🔥 NEW: Complete concept-level data storage
    concept_masteries = db.Column(db.Text, default='{}')  # JSON with ALL concept data
    
    # Legacy fields (keep for backward compatibility, but will be derived)
    missing_concepts = db.Column(db.Text, default='[]')
    weak_concepts = db.Column(db.Text, default='[]')
    strong_concepts = db.Column(db.Text, default='[]')
    concept_stagnation = db.Column(db.Text, default='{}')
    
    # Timestamps
    first_attempt = db.Column(db.DateTime, default=datetime.utcnow)
    last_attempt = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.Float, default=0.0)  # Unix timestamp
    
    # Relationship
    user = db.relationship('User', backref=db.backref('masteries', lazy='dynamic'))
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'topic', name='unique_user_topic'),
    )
    
    def get_concept_masteries(self):
        """Load complete concept masteries from JSON"""
        if not self.concept_masteries:
            return {}
        try:
            return json.loads(self.concept_masteries)
        except:
            return {}
    
    def set_concept_masteries(self, concept_dict):
        """Save complete concept masteries to JSON"""
        self.concept_masteries = json.dumps(concept_dict)
    
    def get_missing_concepts(self):
        """Legacy method - derive from concept_masteries"""
        concepts = self.get_concept_masteries()
        missing = []
        for name, data in concepts.items():
            if data.get('is_weak', False):
                missing.append(name)
        return missing
    
    def get_weak_concepts(self):
        """Get weak concepts from concept_masteries"""
        concepts = self.get_concept_masteries()
        return [name for name, data in concepts.items() if data.get('is_weak', False)]
    
    def get_strong_concepts(self):
        """Get strong concepts from concept_masteries"""
        concepts = self.get_concept_masteries()
        return [name for name, data in concepts.items() if data.get('is_strong', False)]
    
    def get_concept_stagnation(self):
        """Get stagnation counts from concept_masteries"""
        concepts = self.get_concept_masteries()
        return {name: data.get('stagnation_count', 0) for name, data in concepts.items()}
    
    def update_mastery(self, semantic_score, keyword_score, response_time, missing=None):
        """
        Legacy update method - now just updates summary stats
        Full concept tracking happens in adaptive_controller
        """
        # Safety fixes
        self.semantic_avg = self.semantic_avg or 0.0
        self.keyword_avg = self.keyword_avg or 0.0
        self.mastery_level = self.mastery_level or 0.0
        self.mastery_velocity = self.mastery_velocity or 0.0
        self.last_mastery = self.last_mastery or 0.0
        self.avg_response_time = self.avg_response_time or 0.0

        self.questions_attempted = self.questions_attempted or 0
        self.correct_count = self.correct_count or 0
        self.consecutive_good = self.consecutive_good or 0
        self.consecutive_poor = self.consecutive_poor or 0
        self.current_difficulty = self.current_difficulty or "medium"

        alpha = 0.3
        
        old_semantic = self.semantic_avg
        old_keyword = self.keyword_avg
        old_mastery = self.mastery_level or 0.0
        old_good = self.consecutive_good
        old_poor = self.consecutive_poor
        old_difficulty = self.current_difficulty
        
        semantic_score = float(semantic_score) if semantic_score is not None else 0.0
        keyword_score = float(keyword_score) if keyword_score is not None else 0.0
        response_time = float(response_time) if response_time is not None else 0.0
        
        # Update averages
        self.semantic_avg = (alpha * semantic_score) + ((1 - alpha) * self.semantic_avg)
        self.keyword_avg = (alpha * keyword_score) + ((1 - alpha) * self.keyword_avg)
        
        old_mastery = self.mastery_level or 0.0
        self.mastery_level = (
            (self.semantic_avg or 0.0) * 0.7 +
            (self.keyword_avg or 0.0) * 0.3
        )
        
        self.mastery_velocity = self.mastery_level - old_mastery
        self.last_mastery = old_mastery
        
        self.avg_response_time = (alpha * response_time) + ((1 - alpha) * self.avg_response_time)
        
        self.questions_attempted += 1
        if keyword_score > 0.6:
            self.correct_count += 1
        
        combined_score = (semantic_score * 0.7 + keyword_score * 0.3)
        
        if combined_score > 0.7:
            self.consecutive_good += 1
            self.consecutive_poor = 0
        elif combined_score < 0.4:
            self.consecutive_poor += 1
            self.consecutive_good = 0
        else:
            self.consecutive_good = 0
            self.consecutive_poor = 0
        
        if self.consecutive_good >= 3:
            self.current_difficulty = "hard"
        elif self.consecutive_poor >= 2:
            self.current_difficulty = "easy"
        else:
            if self.current_difficulty not in ["easy", "medium", "hard"]:
                self.current_difficulty = "medium"
        
        self.last_attempt = datetime.utcnow()
        self.last_seen = datetime.utcnow().timestamp()
        
        return self
    
    def to_dict(self):
        """Convert to dict for API responses"""
        concepts = self.get_concept_masteries()
        return {
            'topic': self.topic,
            'mastery_level': round(self.mastery_level, 3),
            'questions_attempted': self.questions_attempted,
            'current_difficulty': self.current_difficulty,
            'weak_concepts': self.get_weak_concepts(),
            'strong_concepts': self.get_strong_concepts(),
            'learning_velocity': round(self.mastery_velocity, 3),
            'stagnant_concepts': self.get_concept_stagnation(),
            'concept_count': len(concepts)
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
        
        questions = QuestionHistory.query.filter_by(session_id=self.session_id).all()
        
        if questions:
            self.questions_asked = len(questions)
            self.avg_semantic = sum(q.semantic_score for q in questions) / len(questions)
            self.avg_keyword = sum(q.keyword_score for q in questions) / len(questions)
            
            self.overall_score = (
                self.avg_semantic * 0.7 +
                self.avg_keyword * 0.3
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
    subtopic = db.Column(db.String(100), nullable=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)
    expected_answer = db.Column(db.Text, nullable=True)
    
    # 🔥 NEW: Track which concepts were sampled in this question
    sampled_concepts = db.Column(db.Text, default='[]')  # JSON list
    
    # Scores
    semantic_score = db.Column(db.Float, default=0.0)
    keyword_score = db.Column(db.Float, default=0.0)
    
    # Analysis signals
    depth = db.Column(db.String(20), default="medium")
    confidence = db.Column(db.String(20), default="medium")
    key_terms = db.Column(db.Text, default='[]')
    
    # Metadata
    difficulty = db.Column(db.String(20), default="medium")
    response_time = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Missing concepts (JSON)
    missing_concepts = db.Column(db.Text, default='[]')
    
    # Relationship
    user = db.relationship('User', backref=db.backref('question_history', lazy='dynamic'))
    
    def set_sampled_concepts(self, concepts):
        """Store which concepts were asked in this question"""
        self.sampled_concepts = json.dumps(concepts)
    
    def get_sampled_concepts(self):
        """Get concepts that were asked in this question"""
        return json.loads(self.sampled_concepts) if self.sampled_concepts else []
    
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
            'subtopic': self.subtopic,
            'question': self.question[:100] + '...' if len(self.question) > 100 else self.question,
            'semantic_score': round(self.semantic_score, 3),
            'keyword_score': round(self.keyword_score, 3),
            'difficulty': self.difficulty,
            'depth': self.depth,
            'confidence': self.confidence,
            'response_time': round(self.response_time, 2),
            'timestamp': self.timestamp.isoformat(),
            'sampled_concepts': self.get_sampled_concepts(),
            'missing_concepts': self.get_missing_concepts()
        }


class SubtopicMastery(db.Model):
    """Tracks mastery of individual subtopics (subtopic-level, not concept-level)"""
    __tablename__ = 'subtopic_mastery'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    topic = db.Column(db.String(50), nullable=False)
    subtopic = db.Column(db.String(200), nullable=False)
    
    # Subtopic-level mastery (0-1)
    mastery_level = db.Column(db.Float, default=0.0)
    
    # Statistics
    attempts = db.Column(db.Integer, default=0)
    last_asked = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 🔥 FIXED: Clear status for subtopic itself
    # Values: 'not_started', 'ongoing', 'mastered'
    subtopic_status = db.Column(db.String(20), default='not_started')
    
    # Store concept-level data for this subtopic (optional, for quick access)
    concept_data = db.Column(db.Text, default='{}')
    
    # Relationship
    user = db.relationship('User', backref=db.backref('subtopic_masteries', lazy='dynamic'))
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'topic', 'subtopic', name='unique_user_topic_subtopic'),
    )
    
    def to_dict(self):
        """For API responses"""
        return {
            'topic': self.topic,
            'subtopic': self.subtopic,
            'mastery_level': round(self.mastery_level, 3),
            'attempts': self.attempts,
            'last_asked': self.last_asked.isoformat(),
            'status': self.subtopic_status,  # 'not_started', 'ongoing', 'mastered'
            'concept_count': len(json.loads(self.concept_data)) if self.concept_data else 0
        }