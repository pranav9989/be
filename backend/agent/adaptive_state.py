"""
Adaptive Interview State Management
Tracks user progress, mastery levels, and interview state across sessions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import time


@dataclass
class ConceptMastery:
    """
    Tracks mastery of a single concept EXACTLY as per rule summary
    mastery_level = times_mentioned / attempts
    Includes priority score calculation for adaptive sampling
    """
    name: str
    attempts: int = 0
    times_mentioned: int = 0
    times_missed_when_sampled: int = 0
    mastery_level: float = 0.0
    is_weak: bool = False
    is_strong: bool = False
    stagnation_count: int = 0
    last_seen: float = 0.0
    priority_score: float = 1.0

    def record_attempt(self, mentioned: bool):
        """Record an attempt at this concept - called when concept is sampled"""
        self.attempts += 1
        self.last_seen = time.time()

        if mentioned:
            self.times_mentioned += 1
            # Decrease stagnation when mentioned
            self.stagnation_count = max(0, self.stagnation_count - 1)
        else:
            self.times_missed_when_sampled += 1
            # Increase stagnation when missed
            self.stagnation_count += 1

        # INVARIANT 1: attempts = times_mentioned + times_missed_when_sampled
        # This assertion ensures mathematical correctness
        assert self.times_mentioned + self.times_missed_when_sampled == self.attempts, \
            f"Concept {self.name}: attempts={self.attempts}, mentioned={self.times_mentioned}, missed={self.times_missed_when_sampled}"

        # EXACT mastery formula: times_mentioned / attempts
        if self.attempts > 0:
            self.mastery_level = self.times_mentioned / self.attempts

        # Classify only after 3 attempts (per rules)
        if self.attempts >= 3:
            miss_ratio = self.times_missed_when_sampled / self.attempts
            correct_ratio = self.times_mentioned / self.attempts

            self.is_weak = miss_ratio > 0.7
            self.is_strong = correct_ratio > 0.7
        else:
            self.is_weak = False
            self.is_strong = False

        # Update priority score after each attempt
        self.update_priority_score()

    def update_priority_score(self, velocity: float = 0.0):
        """
        STRICT RULE IMPLEMENTATION WITH VELOCITY FACTOR

        base_priority = 1.0 - mastery_level
        stagnation_boost = (times_missed_when_sampled / attempts) * 0.5
        recency_boost = min(0.3, days_since_last_seen * 0.05)

        velocity adjustment:
            positive velocity → reduce priority
            negative velocity → increase priority
        """
        if self.attempts == 0:
            self.priority_score = 1.5
            return

        # Base priority: lower mastery = higher priority
        base_priority = 1.0 - self.mastery_level

        # Stagnation boost: higher miss ratio = higher priority
        if self.attempts > 0:
            stagnation_boost = (self.times_missed_when_sampled / self.attempts) * 0.5
        else:
            stagnation_boost = 0

        # Recency boost: longer unseen = higher priority
        if self.last_seen > 0:
            days_since = (time.time() - self.last_seen) / (24 * 3600)
            recency_boost = min(0.3, days_since * 0.05)
        else:
            recency_boost = 0.3  # Never seen before = high priority

        raw_priority = base_priority + stagnation_boost + recency_boost

        # 🔥 VELOCITY ADJUSTMENT
        # Positive velocity → reduce priority (they're learning)
        # Negative velocity → increase priority (they're struggling)
        velocity_adjustment = -velocity  # Note the negative sign
        raw_priority += velocity_adjustment

        # Apply classification rules
        if self.is_weak:
            self.priority_score = 2.0  # Maximum priority
        elif self.is_strong:
            self.priority_score = raw_priority * 0.3  # Low priority
        elif self.attempts < 3:
            self.priority_score = raw_priority * 1.2  # Exploration boost
        else:
            self.priority_score = raw_priority  # Normal

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'attempts': self.attempts,
            'times_mentioned': self.times_mentioned,
            'times_missed_when_sampled': self.times_missed_when_sampled,
            'mastery_level': self.mastery_level,
            'is_weak': self.is_weak,
            'is_strong': self.is_strong,
            'stagnation_count': self.stagnation_count,
            'last_seen': self.last_seen,
            'priority_score': self.priority_score
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Create ConceptMastery from dictionary (for DB loading)"""
        return cls(
            name=data["name"],
            attempts=data.get("attempts", 0),
            times_mentioned=data.get("times_mentioned", 0),
            times_missed_when_sampled=data.get("times_missed_when_sampled", 0),
            mastery_level=data.get("mastery_level", 0),
            is_weak=data.get("is_weak", False),
            is_strong=data.get("is_strong", False),
            stagnation_count=data.get("stagnation_count", 0),
            last_seen=data.get("last_seen", 0),
            priority_score=data.get("priority_score", 1.0)
    )


@dataclass
class TopicSessionState:
    """Tracks state for a specific topic within a session"""
    topic: str
    questions_asked: int = 0
    correct_answers: int = 0
    current_difficulty: str = "medium"
    concepts_covered: List[str] = field(default_factory=list)
    last_question_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'topic': self.topic,
            'questions_asked': self.questions_asked,
            'correct_answers': self.correct_answers,
            'current_difficulty': self.current_difficulty,
            'concepts_covered': self.concepts_covered,
            'last_question_time': self.last_question_time.isoformat() if self.last_question_time else None
        }
    
    def add_answer(self, semantic: float, keyword: float, depth: str):
        """Update topic state with new answer"""
        self.questions_asked += 1
        
        # Calculate combined score
        combined = (semantic * 0.7) + (keyword * 0.3)
        
        if combined > 0.6:
            self.correct_answers += 1
        
        self.last_question_time = datetime.utcnow()


@dataclass
class AdaptiveQARecord:
    """Records a single question-answer interaction"""
    question: str
    topic: str
    subtopic: str
    difficulty: str
    answer: str
    analysis: dict
    semantic_score: float
    keyword_score: float
    response_time: float
    missing_concepts: List[str]
    sampled_concepts: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'question': self.question[:100] + '...' if len(self.question) > 100 else self.question,
            'answer': self.answer[:100] + '...' if len(self.answer) > 100 else self.answer,
            'topic': self.topic,
            'subtopic': self.subtopic,
            'difficulty': self.difficulty,
            'semantic_score': round(self.semantic_score, 3),
            'keyword_score': round(self.keyword_score, 3),
            'combined_score': round((self.semantic_score * 0.7 + self.keyword_score * 0.3), 3),
            'response_time': round(self.response_time, 2),
            'missing_concepts': self.missing_concepts[:5],
            'sampled_concepts': self.sampled_concepts,
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_combined_score(self) -> float:
        """Get combined score (70% semantic, 30% keyword)"""
        return (self.semantic_score * 0.7) + (self.keyword_score * 0.3)


@dataclass
class TopicMastery:
    """
    Tracks mastery for a topic, including all concepts within it
    """
    topic: str
    mastery_level: float = 0.0
    semantic_avg: float = 0.0
    keyword_avg: float = 0.0
    total_questions: int = 0
    sessions_attempted: int = 0
    current_difficulty: str = "medium"
    consecutive_good: int = 0
    consecutive_poor: int = 0
    mastery_velocity: float = 0.0
    last_mastery: float = 0.0
    
    # Concept tracking
    concepts: Dict[str, ConceptMastery] = field(default_factory=dict)
    weak_concepts: set = field(default_factory=set)
    strong_concepts: set = field(default_factory=set)
    
    def get_recommended_difficulty(self) -> str:
        """Get recommended difficulty based on mastery"""
        if self.mastery_level < 0.3:
            return "easy"
        elif self.mastery_level > 0.7:
            return "hard"
        else:
            return self.current_difficulty
    
    def update(self, semantic: float, keyword: float, response_time: float,
               sampled_concepts: List[str], mentioned_concepts: List[str],
               missing_from_sampled: List[str]):
        """Update topic mastery with new answer data"""
        # Store old mastery for velocity calculation
        self.last_mastery = self.mastery_level
        
        # Update averages with EMA
        alpha = 0.3
        self.semantic_avg = (alpha * semantic) + ((1 - alpha) * self.semantic_avg)
        self.keyword_avg = (alpha * keyword) + ((1 - alpha) * self.keyword_avg)
        
        # Calculate new mastery level (70% semantic, 30% keyword)
        self.mastery_level = (self.semantic_avg * 0.7) + (self.keyword_avg * 0.3)
        
        # Update counters
        self.total_questions += 1
        
        # Update consecutive performance
        combined = (semantic * 0.7) + (keyword * 0.3)
        if combined > 0.6:
            self.consecutive_good += 1
            self.consecutive_poor = 0
        elif combined < 0.4:
            self.consecutive_poor += 1
            self.consecutive_good = 0
        else:
            self.consecutive_good = 0
            self.consecutive_poor = 0
        
        # Adjust difficulty
        if self.consecutive_good >= 2:
            if self.current_difficulty == "easy":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "hard"
        elif self.consecutive_poor >= 2:
            if self.current_difficulty == "hard":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "easy"


class AdaptiveInterviewState:
    """
    Main state manager for adaptive interviews.
    Tracks user progress across sessions and manages current interview state.
    """
    
    def __init__(self, session_id: str, user_id: int, user_name: str = ""):
        self.session_id = session_id
        self.user_id = user_id
        self.user_name = user_name
        self.start_time = time.time()
        
        # Topic tracking
        self.topic_order: List[str] = []
        self.current_topic_index: int = 0
        self.current_topic: Optional[str] = None
        self.current_subtopic: Optional[str] = None
        
        # Question tracking
        self.current_question: Optional[str] = None
        self.current_difficulty: str = "medium"
        self.current_sampled_concepts: List[str] = []
        self.question_start_time: Optional[float] = None
        
        # History
        self.history: List[AdaptiveQARecord] = []
        self.followup_count: int = 0
        
        # Topic mastery
        self.topic_mastery: Dict[str, TopicMastery] = {}
        
        # Topic sessions (per-session state)
        self.topic_sessions: Dict[str, TopicSessionState] = {}
        
        # Weak topics history
        self.weak_topics_history: Dict[str, List[str]] = {}
        
        # Topics covered in this session
        self.topics_covered_this_session: List[str] = []
        
        # Session limits
        self.max_questions_total = 15
        self.max_duration_minutes = 30
    
    def ensure_topic_mastery(self, topic: str) -> TopicMastery:
        """Get or create topic mastery"""
        if topic not in self.topic_mastery:
            self.topic_mastery[topic] = TopicMastery(topic=topic)
        return self.topic_mastery[topic]
    
    def get_topic_session(self, topic: str) -> TopicSessionState:
        """Get or create topic session state"""
        if topic not in self.topic_sessions:
            self.topic_sessions[topic] = TopicSessionState(topic=topic)
        return self.topic_sessions[topic]
    
    def add_to_history(self, record: AdaptiveQARecord):
        """Add a Q&A record to history"""
        self.history.append(record)
    
    def total_questions_asked(self) -> int:
        """Get total questions asked in this session"""
        return len(self.history)
    
    def time_remaining_sec(self) -> int:
        """Get seconds remaining in session"""
        elapsed = time.time() - self.start_time
        remaining = (self.max_duration_minutes * 60) - elapsed
        return max(0, int(remaining))
    
    def is_time_over(self) -> bool:
        """Check if session time is over"""
        return self.time_remaining_sec() <= 0
    
    def advance_to_next_topic(self) -> bool:
        """Move to next topic in order, returns True if moved, False if completed cycle"""
        self.current_topic_index += 1
        if self.current_topic_index < len(self.topic_order):
            self.current_topic = self.topic_order[self.current_topic_index]
            self.current_subtopic = None
            self.followup_count = 0
            return True
        return False
    
    def get_sorted_topics_by_priority(self) -> List[str]:
        """Get topics sorted by priority (lowest mastery first)"""
        topics_with_mastery = []
        for topic, mastery in self.topic_mastery.items():
            topics_with_mastery.append((topic, mastery.mastery_level))
        
        topics_with_mastery.sort(key=lambda x: x[1])  # Sort by mastery (ascending)
        return [t[0] for t in topics_with_mastery]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'start_time': self.start_time,
            'current_topic': self.current_topic,
            'current_subtopic': self.current_subtopic,
            'current_difficulty': self.current_difficulty,
            'questions_asked': len(self.history),
            'time_remaining': self.time_remaining_sec(),
            'topic_order': self.topic_order,
            'current_topic_index': self.current_topic_index,
            'followup_count': self.followup_count,
            'masteries': {
                topic: {
                    'level': m.mastery_level,
                    'velocity': m.mastery_velocity,
                    'weak_concepts': list(m.weak_concepts),
                    'strong_concepts': list(m.strong_concepts)
                }
                for topic, m in self.topic_mastery.items()
            }
        }


# Export the classes
__all__ = [
    'AdaptiveInterviewState',
    'AdaptiveQARecord',
    'TopicSessionState',
    'TopicMastery',
    'ConceptMastery'
]