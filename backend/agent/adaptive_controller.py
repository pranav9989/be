# backend/agent/adaptive_controller.py

import time
import json
import numpy as np
from typing import Dict, Any, Set
from datetime import datetime

from .adaptive_state import AdaptiveInterviewState, AdaptiveQARecord
from .adaptive_analyzer import AdaptiveAnalyzer
from .adaptive_decision import AdaptiveDecisionEngine
from .adaptive_question_bank import AdaptiveQuestionBank
from models import db, UserMastery, InterviewSession, QuestionHistory, AdaptiveInterviewSession

class AdaptiveInterviewController:
    """Main controller for adaptive interviews"""
    
    # Available topics
    TOPICS = ["DBMS", "OS", "OOPS"]
    
    def __init__(self):
        self.sessions: Dict[str, AdaptiveInterviewState] = {}
        self.question_bank = AdaptiveQuestionBank()
        self.decision_engine = AdaptiveDecisionEngine()
        # Track asked questions per session to prevent repeats
        self.asked_questions: Dict[str, Set[str]] = {}
    
    def start_session(self, session_id: str, user_id: int, user_name: str = "") -> dict:
        """Start a new adaptive session, loading user history from DB"""
        
        # Load user's mastery from database
        masteries = UserMastery.query.filter_by(user_id=user_id).all()
        
        # Create state
        state = AdaptiveInterviewState(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name
        )
        
        # Initialize asked questions set for this session
        self.asked_questions[session_id] = set()
        
        # Load masteries into state
        for m in masteries:
            mastery = state.ensure_topic_mastery(m.topic)
            mastery.mastery_level = m.mastery_level
            mastery.semantic_avg = m.semantic_avg
            mastery.keyword_avg = m.keyword_avg
            mastery.coverage_avg = m.coverage_avg
            mastery.questions_attempted = m.questions_attempted
            mastery.correct_count = m.correct_count
            mastery.avg_response_time = m.avg_response_time
            mastery.current_difficulty = m.current_difficulty
            mastery.consecutive_good = m.consecutive_good
            mastery.consecutive_poor = m.consecutive_poor
            mastery.missing_concepts = set(m.get_missing_concepts())
            mastery.weak_concepts = set(m.get_weak_concepts())
            mastery.strong_concepts = set(m.get_strong_concepts())
        
        # Determine first topic based on weaknesses
        if state.get_weakest_topics(1):
            first_topic = state.get_weakest_topics(1)[0]
        else:
            first_topic = "DBMS"  # Default
        
        # Get personalized first question
        mastery_for_topic = state.ensure_topic_mastery(first_topic)
        difficulty = mastery_for_topic.get_recommended_difficulty()
        
        # Generate question and ensure it's not a repeat (first question won't be)
        first_question = self._generate_unique_question(
            session_id=session_id,
            topic=first_topic,
            difficulty=difficulty,
            user_name=user_name
        )
        
        # Update state
        state.current_topic = first_topic
        state.current_question = first_question
        state.current_difficulty = difficulty
        state.question_start_time = time.time()
        
        # Add to asked questions
        self.asked_questions[session_id].add(first_question)
        
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
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "masteries": {
                t: round(m.mastery_level, 3) 
                for t, m in state.topic_mastery.items()
            }
        }
    
    def _generate_unique_question(self, session_id: str, topic: str, difficulty: str, user_name: str = "", max_attempts: int = 10) -> str:
        """Generate a unique question that hasn't been asked in this session"""
        
        if session_id not in self.asked_questions:
            self.asked_questions[session_id] = set()
        
        # Get all previously asked questions for this topic
        topic_questions = []
        for q in self.asked_questions[session_id]:
            if topic in q or any(keyword in q.lower() for keyword in self.question_bank.fallback_questions[topic]["medium"]):
                topic_questions.append(q)
        
        for attempt in range(max_attempts):
            question = self.question_bank.generate_first_question(
                topic=topic,
                difficulty=difficulty,
                user_name=user_name
            )
            
            # Better duplicate detection
            question_lower = question.lower()
            is_duplicate = False
            
            for asked in topic_questions:
                # Check for 70% similarity
                asked_words = set(asked.lower().split())
                question_words = set(question_lower.split())
                if asked_words and question_words:
                    similarity = len(asked_words.intersection(question_words)) / len(asked_words.union(question_words))
                    if similarity > 0.7:
                        is_duplicate = True
                        break
                
                # Check if asking same concept
                if "debug" in question_lower and "debug" in asked.lower():
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                return question
            
            # If we've tried too many times, use a completely different fallback
            if attempt == max_attempts - 1:
                import random
                # Choose a different difficulty level
                alt_difficulty = "easy" if difficulty == "hard" else "hard" if difficulty == "easy" else "medium"
                return random.choice(self.question_bank.fallback_questions[topic][alt_difficulty])
        
        return f"Can you explain a different concept in {topic}?"
    
    def handle_answer(self, session_id: str, answer: str) -> dict:
        """Process user answer and generate next action"""
    
        state = self.sessions.get(session_id)
        if not state:
            return {"error": "Session not found"}
        
        response_time = time.time() - (state.question_start_time or time.time())
        question = state.current_question
        topic = state.current_topic
        
        # Analyze answer using adaptive analyzer
        analysis = AdaptiveAnalyzer.analyze(question, answer, topic)
        
        # IMPROVED SCORING
        coverage_score = analysis.get("coverage_score", 0.5)
        
        # Calculate semantic score based on answer quality and length
        word_count = len(answer.split())
        depth_score = 0.3 if analysis.get("depth") == "shallow" else 0.6 if analysis.get("depth") == "medium" else 0.9
        
        # Base score from coverage
        semantic_score = 0.4 + (coverage_score * 0.4)
        
        # Add depth bonus
        semantic_score += depth_score * 0.2
        
        # Length penalty for very short answers
        if word_count < 10:
            semantic_score *= 0.7
        elif word_count < 20:
            semantic_score *= 0.85
        
        # Cap at 0.95
        semantic_score = min(0.95, semantic_score)
        
        keyword_score = coverage_score
        
        # Update topic mastery
        mastery = state.ensure_topic_mastery(topic)
        mastery.update(
            semantic=semantic_score,
            keyword=keyword_score,
            coverage=coverage_score,
            response_time=response_time,
            missing=analysis["missing_concepts"]
        )
        
        # Create QA record
        record = AdaptiveQARecord(
            question=question,
            topic=topic,
            difficulty=state.current_difficulty,
            answer=answer,
            analysis=analysis,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            coverage_score=coverage_score,
            response_time=response_time,
            missing_concepts=analysis["missing_concepts"]
        )
        
        # Add to history
        state.add_to_history(record)
        
        # Save to database
        self._save_to_db(state.user_id, session_id, record)
        
        # Decide next action
        action = self.decision_engine.decide(state, analysis)
        
        # Execute action
        if action == "FINALIZE":
            # Clean up asked questions
            if session_id in self.asked_questions:
                del self.asked_questions[session_id]
            return self._finalize_session(state, session_id)
        
        if action == "MOVE_TOPIC":
            return self._move_to_new_topic(state, session_id)
        
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
        
        # Generate unique follow-up question
        if missing:
            # Generate question targeting missing concepts
            question = self._generate_unique_question(
                session_id=session_id,
                topic=topic,
                difficulty=difficulty,
                user_name=state.user_name
            )
        else:
            # Use gap followup
            question = self.question_bank.generate_gap_followup(
                topic=topic,
                missing_concepts=missing,
                difficulty=difficulty
            )
        
        # Ensure question is unique
        if question in self.asked_questions.get(session_id, set()):
            # If duplicate, use a fallback
            import random
            question = random.choice(self.question_bank.fallback_questions[topic][difficulty])
        
        self.asked_questions[session_id].add(question)
        state.current_question = question
        state.question_start_time = time.time()
        
        return {
            "action": "FOLLOW_UP",
            "question": question,
            "topic": topic,
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "focus_areas": missing[:3]
        }
    
    def _simplify_question(self, state, session_id) -> dict:
        """Generate simpler question for struggling user"""
        state.followup_count += 1
        
        mastery = state.ensure_topic_mastery(state.current_topic)
        missing = list(mastery.missing_concepts)[:3]
        
        # Generate simplified question
        question = self.question_bank.generate_simplified_question(
            topic=state.current_topic,
            missing_concepts=missing
        )
        
        # Ensure uniqueness
        if question in self.asked_questions.get(session_id, set()):
            question = f"Let's try a simpler approach: {question}"
        
        self.asked_questions[session_id].add(question)
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
        """Generate deeper question for strong performer"""
        state.followup_count += 1
        
        # Generate deeper question
        question = self.question_bank.generate_deeper_dive(
            topic=state.current_topic,
            difficulty="hard"
        )
        
        # Ensure uniqueness
        if question in self.asked_questions.get(session_id, set()):
            import random
            question = random.choice(self.question_bank.fallback_questions[state.current_topic]["hard"])
        
        self.asked_questions[session_id].add(question)
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
    
    def _move_to_new_topic(self, state, session_id) -> dict:
        """Move to next topic based on weaknesses"""
        # Get topics not yet covered in this session
        covered = {q.topic for q in state.history if q.topic}
        available = [t for t in self.TOPICS if t not in covered] or self.TOPICS
        
        # Choose next topic
        new_topic = state.get_next_topic(available)
        
        state.reset_for_new_topic(new_topic)
        
        mastery = state.ensure_topic_mastery(new_topic)
        difficulty = mastery.get_recommended_difficulty()
        
        # Generate unique first question for new topic
        question = self._generate_unique_question(
            session_id=session_id,
            topic=new_topic,
            difficulty=difficulty,
            user_name=state.user_name
        )
        
        self.asked_questions[session_id].add(question)
        state.current_question = question
        state.question_start_time = time.time()
        
        # Update session topics in DB
        db_session = AdaptiveInterviewSession.query.filter_by(session_id=state.session_id).first()
        if db_session:
            topics = db_session.get_topics_covered()
            if new_topic not in topics:
                topics.append(new_topic)
                db_session.set_topics_covered(topics)
                db.session.commit()
        
        return {
            "action": "MOVE_TOPIC",
            "question": question,
            "topic": new_topic,
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "mastery": round(mastery.mastery_level, 3)
        }
    
    def _finalize_session(self, state, session_id) -> dict:
        """Finalize interview and save final stats"""
        # Calculate session stats
        questions = len(state.history)
        avg_semantic = np.mean([r.semantic_score for r in state.history]) if questions > 0 else 0
        avg_keyword = np.mean([r.keyword_score for r in state.history]) if questions > 0 else 0
        
        overall = (avg_semantic * 0.6 + avg_keyword * 0.4)
        
        # Update session in DB
        db_session = AdaptiveInterviewSession.query.filter_by(session_id=session_id).first()
        if db_session:
            db_session.end_time = datetime.utcnow()
            db_session.duration = int(time.time() - state.start_time)
            db_session.questions_asked = questions
            db_session.avg_semantic = avg_semantic
            db_session.avg_keyword = avg_keyword
            db_session.overall_score = overall
            db.session.commit()
        
        # Clean up asked questions
        if session_id in self.asked_questions:
            del self.asked_questions[session_id]
        
        # Generate feedback
        feedback = {
            "questions_answered": questions,
            "overall_score": round(overall, 3),
            "strengths": state.get_strongest_topics(2),
            "areas_for_improvement": state.get_weakest_topics(2),
            "learning_velocity": round(state.learning_velocity, 3)
        }
        
        if state.get_weakest_topics(1):
            feedback["recommended_focus"] = state.get_weakest_topics(1)[0]
        
        # Remove from active sessions
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        return {
            "action": "FINALIZE",
            "next_question": None,
            "time_remaining": 0,
            "feedback": feedback
        }
    
    def _save_to_db(self, user_id: int, session_id: str, record: AdaptiveQARecord):
        """Save QA record and update user mastery in database"""
        try:
            # Update UserMastery
            mastery = UserMastery.query.filter_by(
                user_id=user_id, 
                topic=record.topic
            ).first()
            
            if not mastery:
                mastery = UserMastery(
                    user_id=user_id,
                    topic=record.topic
                )
                db.session.add(mastery)
            
            mastery.update_mastery(
                semantic_score=record.semantic_score,
                keyword_score=record.keyword_score,
                coverage_score=record.coverage_score,
                response_time=record.response_time,
                missing=record.missing_concepts
            )
            
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