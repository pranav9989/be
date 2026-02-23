# backend/agent/adaptive_state.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Set
import time
import numpy as np
import random


@dataclass
class TopicSessionState:
    """Tracks per-topic state within a single session"""
    topic: str
    questions_asked: int = 0
    cumulative_semantic: float = 0.0
    cumulative_keyword: float = 0.0
    avg_score: float = 0.0
    depth_scores: List[str] = field(default_factory=list)
    is_covered: bool = False
    is_weak: bool = False
    last_question_time: float = 0.0
    silent_count: int = 0  # Track silent answers
    
    # Fixed limits - exactly 3 questions per topic per session
    QUESTIONS_PER_TOPIC = 3
    
    def add_answer(self, semantic: float, keyword: float, depth: str):
        """Add an answer to this topic's session state"""
        self.questions_asked += 1
        self.cumulative_semantic += semantic
        self.cumulative_keyword += keyword
        self.depth_scores.append(depth)
        self.avg_score = self.cumulative_semantic / self.questions_asked
        self.last_question_time = time.time()
        
        # Track silent answers (very short answers)
        if len(self.depth_scores) > 0 and depth == "shallow" and semantic < 0.2:
            self.silent_count += 1
    
    def should_move_on(self) -> bool:
        """Determine if we should move to next topic - EXACTLY 3 questions"""
        
        # Case 1: Haven't reached 3 questions yet
        if self.questions_asked < self.QUESTIONS_PER_TOPIC:
            return False
            
        # Case 2: Reached exactly 3 questions - definitely move on
        if self.questions_asked >= self.QUESTIONS_PER_TOPIC:
            print(f"📊 Topic {self.topic}: Reached {self.QUESTIONS_PER_TOPIC} questions - moving on")
            self.is_covered = True
            return True


@dataclass
class AdaptiveQARecord:
    question: str
    topic: Optional[str]
    subtopic: Optional[str]  # Track subtopic
    difficulty: str
    answer: str
    analysis: Dict
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    response_time: float = 0.0
    missing_concepts: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConceptMastery:
    """Track mastery at the concept level"""
    name: str
    topic: str
    mastery_level: float = 0.0
    attempts: int = 0
    last_seen: float = 0.0
    stagnation_count: int = 0
    is_weak: bool = False
    is_strong: bool = False


@dataclass
class TopicMastery:
    """Per-topic mastery tracking with longitudinal data"""
    topic: str
    
    # Core mastery (long-term EMA across sessions)
    mastery_level: float = 0.0
    semantic_avg: float = 0.0
    keyword_avg: float = 0.0
    
    # Statistics
    total_questions: int = 0
    sessions_attempted: int = 0
    last_session_date: float = 0.0
    
    # Learning velocity
    mastery_velocity: float = 0.0
    last_mastery: float = 0.0
    
    # Difficulty tracking
    current_difficulty: str = "medium"
    consecutive_good: int = 0
    consecutive_poor: int = 0
    
    # Concept-level tracking
    concepts: Dict[str, ConceptMastery] = field(default_factory=dict)
    
    # Concept gaps (for cross-session prioritization)
    missing_concepts: Set[str] = field(default_factory=set)
    weak_concepts: Set[str] = field(default_factory=set)
    strong_concepts: Set[str] = field(default_factory=set)
    
    # Stagnation tracking - Keep for backward compatibility but will be derived
    concept_stagnation: Dict[str, int] = field(default_factory=dict)
    
    # Recent scores for stability
    recent_scores: List[float] = field(default_factory=list)
    stability_score: float = 0.0
    
    # Add avg_response_time as a class attribute with default
    avg_response_time: float = 0.0
    
    def update(self, semantic: float, keyword: float,
               response_time: float, missing: List[str] = None, 
               concepts: List[str] = None):
        """
        Update mastery with new answer - EVIDENCE-BASED STAGNATION
        Now uses concept-level tracking with proper decay and weak detection
        """
        alpha = 0.3  # EMA factor
        
        # Store old mastery for velocity
        old_mastery = self.mastery_level
        
        # Update averages
        self.semantic_avg = (alpha * semantic) + ((1 - alpha) * self.semantic_avg)
        self.keyword_avg = (alpha * keyword) + ((1 - alpha) * self.keyword_avg)
        
        # NEW FORMULA: Semantic dominant (70%), keyword (30%)
        self.mastery_level = (
            self.semantic_avg * 0.7 +
            self.keyword_avg * 0.3
        )
        
        # Update learning velocity
        self.mastery_velocity = self.mastery_level - old_mastery
        self.last_mastery = old_mastery
        
        # Update recent scores for stability
        self.recent_scores.append(self.mastery_level)
        if len(self.recent_scores) > 3:
            self.recent_scores.pop(0)
        
        # Calculate stability
        if len(self.recent_scores) >= 2:
            self.stability_score = 1.0 - min(1.0, np.std(self.recent_scores) / 0.3)
        else:
            self.stability_score = 0.0
        
        # Update counters
        self.total_questions += 1
        
        # Update response time average (ensure it exists)
        self.avg_response_time = (alpha * response_time) + ((1 - alpha) * self.avg_response_time)
        
        # Update consecutive patterns (using weighted score)
        combined = (semantic * 0.7 + keyword * 0.3)
        if combined > 0.7:
            self.consecutive_good += 1
            self.consecutive_poor = 0
        elif combined < 0.4:
            self.consecutive_poor += 1
            self.consecutive_good = 0
        else:
            self.consecutive_good = 0
            self.consecutive_poor = 0
        
        # Update difficulty based on sustained performance
        if self.consecutive_good >= 3 and self.stability_score > 0.7:
            self.current_difficulty = "hard"
        elif self.consecutive_poor >= 2:
            self.current_difficulty = "easy"
        else:
            self.current_difficulty = "medium"
        
        # ========== EVIDENCE-BASED CONCEPT STAGNATION ==========
        
        print("\n" + "─"*70)
        print("📈 CONCEPT TRACKING UPDATE")
        print("─"*70)
        print(f"   Topic: {self.topic}")
        
        # Initialize concepts dictionary if needed
        if not hasattr(self, 'concepts'):
            self.concepts = {}
        
        # STEP 1: RESET stagnation for concepts that WERE mentioned
        if concepts and len(concepts) > 0:
            print(f"\n   ✅ Mentioned correctly:")
            for concept in concepts[:5]:
                if concept in self.concepts:
                    old_stag = self.concepts[concept].stagnation_count
                    self.concepts[concept].stagnation_count = 0
                    self.concepts[concept].last_seen = time.time()
                    self.concepts[concept].mastery_level = (alpha * 1.0) + ((1 - alpha) * self.concepts[concept].mastery_level)
                    self.concepts[concept].attempts += 1
                    
                    # Remove from weak concepts if it was there
                    if concept in self.weak_concepts:
                        self.weak_concepts.discard(concept)
                        print(f"      ✓ {concept} - stagnation reset (was weak, now correct)")
                    else:
                        print(f"      ✓ {concept} - stagnation reset (was {old_stag} → 0)")
                else:
                    # Create new concept record for mentioned concept
                    from .adaptive_state import ConceptMastery
                    self.concepts[concept] = ConceptMastery(
                        name=concept, 
                        topic=self.topic,
                        mastery_level=1.0,
                        attempts=1,
                        last_seen=time.time(),
                        stagnation_count=0
                    )
                    print(f"      ✓ {concept} - new concept, mentioned correctly")
        
        # STEP 2: Handle missing concepts (not mentioned)
        if missing and len(missing) > 0:
            print(f"\n   ❌ Missing concepts:")
            for concept in missing[:10]:  # Show up to 10 missing concepts
                # Create concept if it doesn't exist
                if concept not in self.concepts:
                    from .adaptive_state import ConceptMastery
                    self.concepts[concept] = ConceptMastery(
                        name=concept, 
                        topic=self.topic,
                        attempts=1,
                        last_seen=time.time(),
                        stagnation_count=1
                    )
                    print(f"      {concept}: new concept, stagnation=1")
                else:
                    concept_m = self.concepts[concept]
                    old_stag = concept_m.stagnation_count
                    concept_m.attempts += 1
                    concept_m.last_seen = time.time()
                    
                    # Update concept mastery (lower score for missing)
                    concept_m.mastery_level = (alpha * 0.3) + ((1 - alpha) * concept_m.mastery_level)
                    
                    # Increment stagnation count
                    concept_m.stagnation_count += 1
                    new_stag = concept_m.stagnation_count
                    
                    # Calculate miss ratio
                    miss_ratio = new_stag / concept_m.attempts if concept_m.attempts > 0 else 1.0
                    
                    status = ""
                    
                    # Mark as weak after repeated failure (3+ attempts, >70% miss ratio)
                    if concept_m.attempts >= 3 and miss_ratio > 0.7:
                        if not concept_m.is_weak:
                            concept_m.is_weak = True
                            self.weak_concepts.add(concept)
                            status = " → ⚠️ WEAK (threshold crossed)"
                    
                    # Mark as strong if consistently good
                    elif concept_m.attempts >= 3 and (1 - miss_ratio) > 0.7:
                        if not concept_m.is_strong:
                            concept_m.is_strong = True
                            self.strong_concepts.add(concept)
                            status = " → 💪 STRONG"
                    
                    print(f"      {concept}: stagnation {old_stag}→{new_stag} (missed {new_stag}/{concept_m.attempts} times, ratio {miss_ratio:.2f}){status}")
        
        # If neither mentioned nor missing concepts, show appropriate message
        if (not concepts or len(concepts) == 0) and (not missing or len(missing) == 0):
            print(f"\n   No concept updates for this answer")
        
        # Update missing_concepts set (for backward compatibility)
        self.missing_concepts = set(missing) if missing else set()
        
        # Update concept_stagnation dictionary (for backward compatibility)
        self.concept_stagnation = {
            name: cm.stagnation_count 
            for name, cm in self.concepts.items() 
            if cm.stagnation_count > 0
        }
        
        # Show summary of weak/strong concepts
        if self.weak_concepts:
            print(f"\n   ⚠️ Current weak concepts: {', '.join(list(self.weak_concepts)[:5])}")
        if self.strong_concepts:
            print(f"   💪 Current strong concepts: {', '.join(list(self.strong_concepts)[:5])}")
        
        print("─"*70)
    
    def get_priority_score(self) -> float:
        """
        Calculate priority for next session (higher = more priority)
        Evidence-based: only counts concepts with significant stagnation
        """
        # Base priority on low mastery
        priority = 1.0 - self.mastery_level
        
        # REDUCED STAGNATION PENALTY - Only count concepts with >2 stagnation
        # These are concepts that have been missed multiple times
        stagnant_concepts = len([c for c in self.concepts.values() if c.stagnation_count > 2])
        stagnation_penalty = min(stagnant_concepts * 0.05, 0.2)  # Max 0.2
        
        # Add weak concepts penalty
        weak_penalty = len(self.weak_concepts) * 0.1
        
        # Subtract if strongly mastered
        strong_bonus = len(self.strong_concepts) * 0.05
        
        final_priority = priority + stagnation_penalty + weak_penalty - strong_bonus
        return max(0.0, final_priority)  # Ensure non-negative
    
    def get_recommended_difficulty(self) -> str:
        return self.current_difficulty
    
    def is_topic_mastered(self) -> bool:
        """Check if topic is truly mastered (requires stability)"""
        return (self.mastery_level > 0.7 and 
                self.stability_score > 0.6 and 
                self.total_questions >= 5 and
                self.consecutive_good >= 2)


@dataclass
class AdaptiveInterviewState:
    # Session info
    session_id: str
    user_id: int
    user_name: str = ""
    start_time: float = field(default_factory=time.time)

    # Topic order for this session (random permutation)
    topic_order: List[str] = field(default_factory=list)
    current_topic_index: int = 0
    
    # Track weak topics from previous sessions
    weak_topics_history: Dict[str, List[str]] = field(default_factory=dict)  # topic -> list of weak concepts
    
    # Track topics covered in this session
    topics_covered_this_session: List[str] = field(default_factory=list)
    
    # Topic session tracking
    topic_sessions: Dict[str, TopicSessionState] = field(default_factory=dict)
    
    # Current context
    current_question: Optional[str] = None
    current_topic: Optional[str] = None
    current_subtopic: Optional[str] = None
    current_difficulty: str = "medium"
    followup_count: int = 0
    question_start_time: Optional[float] = None
    
    # History
    history: List[AdaptiveQARecord] = field(default_factory=list)
    
    # Topic mastery (long-term, loaded from DB)
    topic_mastery: Dict[str, TopicMastery] = field(default_factory=dict)
    
    # Learning metrics
    learning_velocity: float = 0.0
    attention_score: float = 1.0
    
    # Limits
    max_followups_per_topic: int = 4
    max_questions_total: int = 15
    max_duration_sec: int = 30 * 60  # 30 minutes
    
    # Adaptive thresholds
    mastery_threshold: float = 0.7
    weak_threshold: float = 0.4
    coverage_min_questions: int = 3
    coverage_threshold: float = 0.65

    def get_current_topic_from_order(self) -> str:
        """Get current topic based on order index"""
        if not self.topic_order or self.current_topic_index >= len(self.topic_order):
            return None
        return self.topic_order[self.current_topic_index]

    def get_next_topic_from_order(self) -> str:
        """Get next topic in the order"""
        next_index = self.current_topic_index + 1
        if next_index >= len(self.topic_order):
            return None  # No more topics
        return self.topic_order[next_index]

    def advance_to_next_topic(self) -> bool:
        """
        Advance to next topic in the cycle
        Returns True if moved to next topic successfully
        Returns False if reached end of cycle (signal to start new cycle)
        """
        if self.current_topic_index + 1 < len(self.topic_order):
            # Move to next topic in the order
            self.current_topic_index += 1
            self.current_topic = self.topic_order[self.current_topic_index]
            self.followup_count = 0
            print(f"   ➡️ Advanced to next topic: {self.current_topic}")
            return True
        else:
            # We've reached the end - signal to start a new cycle
            print(f"   🏁 Reached end of cycle, need to start new cycle")
            return False
    
    def add_to_history(self, record: AdaptiveQARecord):
        """Add record to history"""
        self.history.append(record)
        
        # Update learning velocity
        if len(self.history) > 1:
            prev = self.history[-2]
            curr = record
            score_delta = (curr.semantic_score - prev.semantic_score)
            self.learning_velocity = (0.3 * score_delta) + (0.7 * self.learning_velocity)
    
    def get_topic_session(self, topic: str) -> TopicSessionState:
        """Get or create topic session state"""
        if topic not in self.topic_sessions:
            self.topic_sessions[topic] = TopicSessionState(topic=topic)
        return self.topic_sessions[topic]
    
    def get_topic_mastery(self, topic: str) -> Optional[TopicMastery]:
        """Get long-term mastery for a topic"""
        return self.topic_mastery.get(topic)
    
    def ensure_topic_mastery(self, topic: str) -> TopicMastery:
        """Get or create topic mastery"""
        if topic not in self.topic_mastery:
            self.topic_mastery[topic] = TopicMastery(topic=topic)
        return self.topic_mastery[topic]
    
    def get_sorted_topics_by_priority(self) -> List[str]:
        """Get all topics sorted by priority (weakest first)"""
        topics_with_priority = []
        all_topics = ["DBMS", "OS", "OOPS"]
        
        for topic in all_topics:
            mastery = self.topic_mastery.get(topic)
            if mastery:
                priority = mastery.get_priority_score()
            else:
                priority = 1.5  # New topics get high priority
            
            topics_with_priority.append((topic, priority))
        
        # Sort by priority (highest first)
        topics_with_priority.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in topics_with_priority]
    
    def get_next_topic(self) -> str:
        """Get the next topic to cover (weakest first)"""
        sorted_topics = self.get_sorted_topics_by_priority()
        
        # Filter out topics that are already covered well in this session
        available = []
        for topic in sorted_topics:
            session_state = self.topic_sessions.get(topic)
            if not session_state or not session_state.is_covered:
                available.append(topic)
        
        return available[0] if available else sorted_topics[0]
    
    def get_stagnant_concepts(self, topic: str, threshold: int = 3) -> List[str]:
        """Get concepts that have stagnated (evidence-based)"""
        mastery = self.topic_mastery.get(topic)
        if not mastery:
            return []
        
        # Use concept-level stagnation counts
        stagnant = []
        for concept_name, concept in mastery.concepts.items():
            if concept.stagnation_count >= threshold:
                stagnant.append(concept_name)
        
        return stagnant
    
    def time_remaining_sec(self) -> int:
        elapsed = time.time() - self.start_time
        return max(0, int(self.max_duration_sec - elapsed))
    
    def is_time_over(self) -> bool:
        return self.time_remaining_sec() <= 0
    
    def total_questions_asked(self) -> int:
        return len(self.history)
    
    def reset_for_new_topic(self, topic: str):
        """Reset state for moving to a new topic"""
        self.current_topic = topic
        self.followup_count = 0
        self.current_difficulty = self.ensure_topic_mastery(topic).get_recommended_difficulty()
        
        # Initialize topic session if needed
        self.get_topic_session(topic)
    
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
            'weakest_topics': self.get_sorted_topics_by_priority()[:3],
            'topic_progress': {
                topic: {
                    'session_questions': self.topic_sessions.get(topic, TopicSessionState(topic)).questions_asked,
                    'avg_score': round(self.topic_sessions.get(topic, TopicSessionState(topic)).avg_score, 3),
                    'is_covered': self.topic_sessions.get(topic, TopicSessionState(topic)).is_covered
                }
                for topic in ["DBMS", "OS", "OOPS"]
            },
            'masteries': {
                topic: {
                    'level': round(m.mastery_level, 3),
                    'stability': round(m.stability_score, 3),
                    'difficulty': m.current_difficulty,
                    'total_questions': m.total_questions,
                    'weak_concepts': list(m.weak_concepts)[:5],
                    'strong_concepts': list(m.strong_concepts)[:5]
                }
                for topic, m in self.topic_mastery.items()
            }
        }