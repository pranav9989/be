# backend/agent/adaptive_state.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Set
import time
import numpy as np

@dataclass
class AdaptiveQARecord:
    question: str
    topic: Optional[str]
    difficulty: str
    answer: str
    analysis: Dict
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    coverage_score: float = 0.0
    response_time: float = 0.0
    missing_concepts: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class TopicMastery:
    """Per-topic mastery tracking"""
    topic: str
    mastery_level: float = 0.0
    semantic_avg: float = 0.0
    keyword_avg: float = 0.0
    coverage_avg: float = 0.0
    questions_attempted: int = 0
    correct_count: int = 0
    avg_response_time: float = 0.0
    current_difficulty: str = "medium"
    consecutive_good: int = 0
    consecutive_poor: int = 0
    missing_concepts: Set[str] = field(default_factory=set)
    weak_concepts: Set[str] = field(default_factory=set)
    strong_concepts: Set[str] = field(default_factory=set)
    
    def update(self, semantic: float, keyword: float, coverage: float, 
               response_time: float, missing: List[str] = None):
        """Update mastery with new answer"""
        alpha = 0.3  # EMA factor
        
        # Update averages
        self.semantic_avg = (alpha * semantic) + ((1 - alpha) * self.semantic_avg)
        self.keyword_avg = (alpha * keyword) + ((1 - alpha) * self.keyword_avg)
        self.coverage_avg = (alpha * coverage) + ((1 - alpha) * self.coverage_avg)
        
        # Calculate new mastery
        old_mastery = self.mastery_level
        self.mastery_level = (
            self.semantic_avg * 0.4 +
            self.keyword_avg * 0.3 +
            self.coverage_avg * 0.3
        )
        
        # Update response time
        self.avg_response_time = (alpha * response_time) + ((1 - alpha) * self.avg_response_time)
        
        # Update counters
        self.questions_attempted += 1
        if coverage > 0.6:
            self.correct_count += 1
        
        # Update consecutive patterns
        combined = (semantic * 0.4 + keyword * 0.3 + coverage * 0.3)
        if combined > 0.7:
            self.consecutive_good += 1
            self.consecutive_poor = 0
        elif combined < 0.4:
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
        
        # Update concept sets
        if missing:
            for concept in missing:
                self.missing_concepts.add(concept)
        
        # Update weak/strong based on mastery
        if self.mastery_level > 0.7:
            self.strong_concepts.add(self.topic)
            if self.topic in self.weak_concepts:
                self.weak_concepts.remove(self.topic)
        elif self.mastery_level < 0.4:
            self.weak_concepts.add(self.topic)
            if self.topic in self.strong_concepts:
                self.strong_concepts.remove(self.topic)
    
    def get_recommended_difficulty(self) -> str:
        """Get recommended next question difficulty"""
        return self.current_difficulty
    
    def should_move_on(self, max_followups: int = 3) -> bool:
        """Determine if we should move to next topic"""
        return (self.mastery_level > 0.8 or 
                self.questions_attempted >= max_followups + 2)

@dataclass
class AdaptiveInterviewState:
    # Session info
    session_id: str
    user_id: int
    user_name: str = ""
    start_time: float = field(default_factory=time.time)
    
    # Current context
    current_question: Optional[str] = None
    current_topic: Optional[str] = None
    current_difficulty: str = "medium"
    followup_count: int = 0
    question_start_time: Optional[float] = None
    
    # History
    history: List[AdaptiveQARecord] = field(default_factory=list)
    
    # Topic mastery (in-memory cache, will sync with DB)
    topic_mastery: Dict[str, TopicMastery] = field(default_factory=dict)
    
    # Learning metrics
    learning_velocity: float = 0.0  # Rate of improvement
    attention_score: float = 1.0    # Estimated attention level
    session_quality: float = 0.0     # Overall session quality
    
    # Limits
    max_followups_per_topic: int = 3
    max_questions_total: int = 12
    max_duration_sec: int = 30 * 60
    
    # Adaptive thresholds
    mastery_threshold: float = 0.7
    weak_threshold: float = 0.4
    
    def add_to_history(self, record: AdaptiveQARecord):
        """Add record to history and update learning metrics"""
        self.history.append(record)
        
        # Update learning velocity (exponential moving average)
        if len(self.history) > 1:
            prev = self.history[-2]
            curr = record
            score_delta = (curr.semantic_score - prev.semantic_score)
            self.learning_velocity = (0.3 * score_delta) + (0.7 * self.learning_velocity)
    
    def get_topic_mastery(self, topic: str) -> Optional[TopicMastery]:
        """Get mastery for a topic"""
        return self.topic_mastery.get(topic)
    
    def ensure_topic_mastery(self, topic: str) -> TopicMastery:
        """Get or create topic mastery"""
        if topic not in self.topic_mastery:
            self.topic_mastery[topic] = TopicMastery(topic=topic)
        return self.topic_mastery[topic]
    
    def get_weakest_topics(self, n: int = 3) -> List[str]:
        """Get n weakest topics"""
        topics_with_scores = [
            (topic, mastery.mastery_level) 
            for topic, mastery in self.topic_mastery.items()
        ]
        topics_with_scores.sort(key=lambda x: x[1])
        return [topic for topic, _ in topics_with_scores[:n]]
    
    def get_strongest_topics(self, n: int = 3) -> List[str]:
        """Get n strongest topics"""
        topics_with_scores = [
            (topic, mastery.mastery_level) 
            for topic, mastery in self.topic_mastery.items()
        ]
        topics_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in topics_with_scores[:n]]
    
    def get_next_topic(self, available_topics: List[str]) -> str:
        """Intelligently choose next topic based on weaknesses"""
        if not available_topics:
            return "DBMS"  # Default
        
        # Get weakest topics that are available
        weak_topics = self.get_weakest_topics(2)
        for topic in weak_topics:
            if topic in available_topics:
                return topic
        
        # If no weak topics, choose the one with fewest questions
        topic_counts = {}
        for record in self.history:
            if record.topic:
                topic_counts[record.topic] = topic_counts.get(record.topic, 0) + 1
        
        if topic_counts:
            # Pick least asked topic
            return min(available_topics, key=lambda t: topic_counts.get(t, 0))
        
        # Default to first available
        return available_topics[0]
    
    def get_missing_concepts_for_topic(self, topic: str) -> List[str]:
        """Get missing concepts for a topic"""
        mastery = self.topic_mastery.get(topic)
        if mastery:
            return list(mastery.missing_concepts)[:5]
        return []
    
    def time_remaining_sec(self) -> int:
        elapsed = time.time() - self.start_time
        return max(0, int(self.max_duration_sec - elapsed))
    
    def is_time_over(self) -> bool:
        return self.time_remaining_sec() <= 0
    
    def total_questions_asked(self) -> int:
        return len(self.history)
    
    def reset_for_new_topic(self, topic: str):
        self.current_topic = topic
        self.followup_count = 0
        self.current_difficulty = self.ensure_topic_mastery(topic).get_recommended_difficulty()
    
    def to_dict(self):
        """Convert to dict for API responses"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'questions_asked': self.total_questions_asked(),
            'time_remaining': self.time_remaining_sec(),
            'current_topic': self.current_topic,
            'current_difficulty': self.current_difficulty,
            'learning_velocity': round(self.learning_velocity, 3),
            'weakest_topics': self.get_weakest_topics(3),
            'strongest_topics': self.get_strongest_topics(3),
            'masteries': {
                topic: {
                    'level': round(m.mastery_level, 3),
                    'difficulty': m.current_difficulty,
                    'questions': m.questions_attempted
                }
                for topic, m in self.topic_mastery.items()
            }
        }