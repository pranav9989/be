# backend/agent/adaptive_controller.py

import time
import json
import numpy as np
import random
from typing import Dict, Any, Set, List, Optional
from datetime import datetime

from .adaptive_state import AdaptiveInterviewState, AdaptiveQARecord, TopicSessionState
from .adaptive_analyzer import AdaptiveAnalyzer
from .adaptive_decision import AdaptiveDecisionEngine
from .adaptive_question_bank import AdaptiveQuestionBank
from .adaptive_planner import adaptive_planner
from .semantic_dedup import semantic_dedup
from models import db, UserMastery, QuestionHistory, AdaptiveInterviewSession

class AdaptiveInterviewController:
    """
    Longitudinal Adaptive Interview Controller
    No session-level goals - only time/user limits
    Tracks mastery across sessions for reinforcement
    """
    
    # Available topics
    TOPICS = ["DBMS", "OS", "OOPS"]
    
    def __init__(self):
        self.sessions: Dict[str, AdaptiveInterviewState] = {}
        self.question_bank = AdaptiveQuestionBank()
        self.decision_engine = AdaptiveDecisionEngine()
    
    def start_session(self, session_id: str, user_id: int, user_name: str = "") -> dict:
        """Start a new adaptive session"""
        
        # Load user's mastery from database
        masteries = UserMastery.query.filter_by(user_id=user_id).all()
        
        # Create state
        state = AdaptiveInterviewState(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name
        )
        
        # Load masteries into state
        for m in masteries:
            mastery = state.ensure_topic_mastery(m.topic)
            mastery.mastery_level = m.mastery_level
            mastery.semantic_avg = m.semantic_avg
            mastery.keyword_avg = m.keyword_avg
            mastery.coverage_avg = m.coverage_avg
            mastery.total_questions = m.questions_attempted
            mastery.sessions_attempted = getattr(m, 'sessions_attempted', 1)
            mastery.current_difficulty = m.current_difficulty
            mastery.consecutive_good = m.consecutive_good
            mastery.consecutive_poor = m.consecutive_poor
            mastery.missing_concepts = set(m.get_missing_concepts())
            mastery.weak_concepts = set(m.get_weak_concepts())
            mastery.strong_concepts = set(m.get_strong_concepts())
            
            # Load concept stagnation
            if hasattr(m, 'concept_stagnation') and m.concept_stagnation:
                try:
                    mastery.concept_stagnation = json.loads(m.concept_stagnation)
                except:
                    mastery.concept_stagnation = {}
        
        # 🔥 NEW: Select first topic based on priority (weakest first)
        sorted_topics = state.get_sorted_topics_by_priority()
        first_topic = sorted_topics[0] if sorted_topics else random.choice(self.TOPICS)
        
        # Get personalized first question
        mastery_for_topic = state.ensure_topic_mastery(first_topic)
        difficulty = mastery_for_topic.get_recommended_difficulty()
        
        # Generate unique question
        first_question, subtopic = self._generate_question(
            session_id=session_id,
            topic=first_topic,
            difficulty=difficulty,
            user_name=user_name
        )
        
        # Update state
        state.current_topic = first_topic
        state.current_subtopic = subtopic
        state.current_question = first_question
        state.current_difficulty = difficulty
        state.question_start_time = time.time()
        
        # Initialize topic session
        state.get_topic_session(first_topic)
        
        # Store session
        self.sessions[session_id] = state
        
        # Create interview session record in DB
        db_session = AdaptiveInterviewSession(
            user_id=user_id,
            session_id=session_id,
            start_time=datetime.utcnow()
        )
        db_session.set_topics_covered([first_topic])
        db.session.add(db_session)
        db.session.commit()
        
        return {
            "action": "START",
            "question": first_question,
            "topic": first_topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "masteries": {
                t: {
                    'level': round(m.mastery_level, 3),
                    'total_questions': m.total_questions
                }
                for t, m in state.topic_mastery.items()
            },
            "topic_priority": state.get_sorted_topics_by_priority()
        }
    
    def _generate_question(self, session_id: str, topic: str, difficulty: str, 
                           user_name: str = "", max_attempts: int = 5) -> tuple:
        """
        Generate a question and return (question_text, subtopic)
        """
        # Get all previously asked questions
        asked_questions = []
        subtopic = None
        
        if session_id in self.sessions:
            state = self.sessions[session_id]
            asked_questions = [record.question for record in state.history]
        
        for attempt in range(max_attempts):
            # Check for stagnant concepts first
            if session_id in self.sessions:
                state = self.sessions[session_id]
                stagnant = state.get_stagnant_concepts(topic, threshold=2)
                if stagnant:
                    question = self.question_bank.generate_gap_followup(
                        topic=topic,
                        missing_concepts=[stagnant[0]],
                        difficulty=difficulty
                    )
                    subtopic = stagnant[0]
                else:
                    # Get a random subtopic from taxonomy
                    if topic in self.question_bank.subtopics_by_topic:
                        subtopics = self.question_bank.subtopics_by_topic[topic]
                        subtopic = random.choice(subtopics)
                    
                    question = self.question_bank.generate_first_question(
                        topic=topic,
                        difficulty=difficulty,
                        user_name=user_name
                    )
            else:
                # First question - get random subtopic
                if topic in self.question_bank.subtopics_by_topic:
                    subtopics = self.question_bank.subtopics_by_topic[topic]
                    subtopic = random.choice(subtopics)
                
                question = self.question_bank.generate_first_question(
                    topic=topic,
                    difficulty=difficulty,
                    user_name=user_name
                )
            
            # Check semantic uniqueness
            if not semantic_dedup.is_duplicate(session_id, question, asked_questions):
                return question, subtopic
            
            print(f"🔄 Attempt {attempt + 1}: Question was semantically similar, retrying...")
        
        # Fallback
        alt_difficulty = random.choice([d for d in ["easy", "medium", "hard"] if d != difficulty])
        fallback = random.choice(self.question_bank.fallback_questions[topic][alt_difficulty])
        return fallback, None
    
    def handle_answer(self, session_id: str, answer: str) -> dict:
        """Process user answer"""
        
        state = self.sessions.get(session_id)
        if not state:
            return {"error": "Session not found"}
        
        # Calculate response time
        response_time = time.time() - (state.question_start_time or time.time())
        
        # Get current context
        question = state.current_question
        topic = state.current_topic
        subtopic = state.current_subtopic
        
        # Analyze answer
        analysis = AdaptiveAnalyzer.analyze(question, answer, topic)
        
        # Extract scores
        coverage_score = analysis.get("coverage_score", 0.5)
        word_count = len(answer.split())
        depth_score = 0.3 if analysis.get("depth") == "shallow" else 0.6 if analysis.get("depth") == "medium" else 0.9
        missing = analysis.get("missing_concepts", [])
        
        # 🔥 NEW: Extract concepts from answer (for concept-level tracking)
        concepts_mentioned = analysis.get("key_terms_used", [])
        
        # Calculate semantic score (reduced influence of coverage)
        semantic_score = analysis.get("semantic_similarity", 0.5)
        
        # Length penalty
        if word_count < 10:
            semantic_score *= 0.7
        elif word_count < 20:
            semantic_score *= 0.85
        
        semantic_score = min(0.95, semantic_score)
        keyword_score = coverage_score  # Still track, but weight reduced in mastery
        
        # 🔥 Update long-term topic mastery
        mastery = state.ensure_topic_mastery(topic)
        mastery.update(
            semantic=semantic_score,
            keyword=keyword_score,
            coverage=coverage_score,
            response_time=response_time,
            missing=missing,
            concepts=concepts_mentioned
        )
        
        # 🔥 Update session-level topic tracking
        topic_session = state.get_topic_session(topic)
        topic_session.add_answer(
            semantic=semantic_score,
            coverage=coverage_score,
            keyword=keyword_score,
            depth=analysis.get("depth", "medium")
        )
        
        # Create QA record
        record = AdaptiveQARecord(
            question=question,
            topic=topic,
            subtopic=subtopic,
            difficulty=state.current_difficulty,
            answer=answer,
            analysis=analysis,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            coverage_score=coverage_score,
            response_time=response_time,
            missing_concepts=missing
        )
        
        # Add to history
        state.add_to_history(record)
        
        # Save to database
        self._save_to_db(state.user_id, session_id, record, mastery)
        
        # Decide next action
        action = self.decision_engine.decide(state, analysis)
        
        # Execute action
        if action == "FINALIZE":
            self._finalize_session(state, session_id)
            return {
                "action": "FINALIZE",
                "next_question": None,
                "time_remaining": 0,
                "feedback": self._generate_feedback(state)
            }
        
        if action == "MOVE_TOPIC":
            return self._move_to_next_topic(state, session_id)
        
        if action == "SIMPLIFY":
            return self._simplify_question(state, session_id)
        
        if action == "DEEPEN":
            return self._deepen_question(state, session_id)
        
        # Default: follow-up
        return self._generate_followup(state, analysis, session_id)
    
    def _generate_followup(self, state, analysis, session_id) -> dict:
        """Generate appropriate follow-up question"""
        state.followup_count += 1
        
        missing = analysis.get("missing_concepts", [])
        topic = state.current_topic
        mastery = state.ensure_topic_mastery(topic)
        difficulty = mastery.get_recommended_difficulty()
        
        # Target specific missing concepts
        target_concept = None
        if missing:
            target_concept = missing[0]
        else:
            stagnant = state.get_stagnant_concepts(topic, threshold=2)
            if stagnant:
                target_concept = stagnant[0]
        
        if target_concept:
            question = self.question_bank.generate_gap_followup(
                topic=topic,
                missing_concepts=[target_concept],
                difficulty=difficulty
            )
            subtopic = target_concept
        else:
            question, subtopic = self._generate_question(
                session_id=session_id,
                topic=topic,
                difficulty=difficulty,
                user_name=state.user_name
            )
        
        state.current_question = question
        state.current_subtopic = subtopic
        state.question_start_time = time.time()
        
        return {
            "action": "FOLLOW_UP",
            "question": question,
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "focus_areas": missing[:3]
        }
    
    def _simplify_question(self, state, session_id) -> dict:
        """Generate simpler question"""
        state.followup_count += 1
        
        mastery = state.ensure_topic_mastery(state.current_topic)
        missing = list(mastery.missing_concepts)[:3]
        
        question = self.question_bank.generate_simplified_question(
            topic=state.current_topic,
            missing_concepts=missing
        )
        
        # Ensure uniqueness
        if semantic_dedup.is_duplicate(session_id, question, 
                                        [r.question for r in state.history]):
            question = f"Let's try a simpler approach: {question}"
        
        state.current_question = question
        state.current_difficulty = "easy"
        state.question_start_time = time.time()
        
        return {
            "action": "SIMPLIFY",
            "question": question,
            "topic": state.current_topic,
            "difficulty": "easy",
            "time_remaining": state.time_remaining_sec(),
            "message": "Let's try a simpler question to build understanding."
        }
    
    def _deepen_question(self, state, session_id) -> dict:
        """Generate deeper question"""
        state.followup_count += 1
        
        question = self.question_bank.generate_deeper_dive(
            topic=state.current_topic,
            difficulty="hard"
        )
        
        # Ensure uniqueness
        if semantic_dedup.is_duplicate(session_id, question,
                                        [r.question for r in state.history]):
            question = random.choice(self.question_bank.fallback_questions[state.current_topic]["hard"])
        
        state.current_question = question
        state.current_difficulty = "hard"
        state.question_start_time = time.time()
        
        return {
            "action": "DEEPEN",
            "question": question,
            "topic": state.current_topic,
            "difficulty": "hard",
            "time_remaining": state.time_remaining_sec(),
            "message": "Great answer! Here's a more challenging question."
        }
    
    def _move_to_next_topic(self, state, session_id) -> dict:
        """Move to next topic based on priority"""
        # Get next topic (weakest first)
        new_topic = state.get_next_topic()
        
        # If same as current, force different topic
        if new_topic == state.current_topic and len(self.TOPICS) > 1:  # 🔥 FIX: Use self.TOPICS
            for topic in state.get_sorted_topics_by_priority():
                if topic != state.current_topic:
                    new_topic = topic
                    break
        
        state.reset_for_new_topic(new_topic)
        
        mastery = state.ensure_topic_mastery(new_topic)
        difficulty = mastery.get_recommended_difficulty()
        
        question, subtopic = self._generate_question(
            session_id=session_id,
            topic=new_topic,
            difficulty=difficulty,
            user_name=state.user_name
        )
        
        state.current_question = question
        state.current_subtopic = subtopic
        state.question_start_time = time.time()
        
        return {
            "action": "MOVE_TOPIC",
            "question": question,
            "topic": new_topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "mastery": round(mastery.mastery_level, 3),
            "topic_progress": {
                t: state.topic_sessions.get(t, TopicSessionState(t)).questions_asked
                for t in self.TOPICS  # 🔥 FIX: Use self.TOPICS
            }
        }
    
    def _finalize_session(self, state, session_id):
        """Finalize interview - only called when limits reached"""
        # Update session in DB
        db_session = AdaptiveInterviewSession.query.filter_by(session_id=session_id).first()
        if db_session:
            db_session.end_time = datetime.utcnow()
            db_session.duration = int(time.time() - state.start_time)
            db_session.questions_asked = len(state.history)
            db.session.commit()
        
        # Clean up
        semantic_dedup.clear_session(session_id)
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _generate_feedback(self, state) -> dict:
        """Generate end-of-session feedback"""
        return {
            "questions_answered": len(state.history),
            "duration_minutes": int((time.time() - state.start_time) / 60),
            "topics_covered": list(state.topic_sessions.keys()),
            "weakest_topics": state.get_sorted_topics_by_priority()[:2],
            "next_session_focus": state.get_sorted_topics_by_priority()[0] if state.get_sorted_topics_by_priority() else "DBMS"
        }
    
    def _save_to_db(self, user_id: int, session_id: str, record: AdaptiveQARecord, mastery):
        """Save QA record and update user mastery in database"""
        try:
            # Update UserMastery
            db_mastery = UserMastery.query.filter_by(
                user_id=user_id, 
                topic=record.topic
            ).first()
            
            if not db_mastery:
                db_mastery = UserMastery(
                    user_id=user_id,
                    topic=record.topic
                )
                db.session.add(db_mastery)
            
            # Update mastery using the already-calculated values
            db_mastery.update_mastery(
                semantic_score=record.semantic_score,
                keyword_score=record.keyword_score,
                coverage_score=record.coverage_score,
                response_time=record.response_time,
                missing=record.missing_concepts
            )
            
            # Save concept stagnation
            db_mastery.concept_stagnation = json.dumps(mastery.concept_stagnation)
            db_mastery.weak_concepts = json.dumps(list(mastery.weak_concepts))
            db_mastery.strong_concepts = json.dumps(list(mastery.strong_concepts))
            
            # Save question history
            history = QuestionHistory(
                user_id=user_id,
                session_id=session_id,
                topic=record.topic,
                question=record.question,
                answer=record.answer,
                semantic_score=record.semantic_score,
                keyword_score=record.keyword_score,
                coverage_score=record.coverage_score,
                difficulty=record.difficulty,
                response_time=record.response_time
            )
            history.set_missing_concepts(record.missing_concepts)
            db.session.add(history)
            
            db.session.commit()
            
        except Exception as e:
            print(f"Error saving to DB: {e}")
            db.session.rollback()
    
    def get_session(self, session_id: str) -> dict:
        """Get current session state"""
        state = self.sessions.get(session_id)
        if state:
            return state.to_dict()
        return {"error": "Session not found"}