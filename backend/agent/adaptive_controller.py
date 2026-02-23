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
from .subtopic_tracker import SubtopicTracker
from models import db, UserMastery, QuestionHistory, AdaptiveInterviewSession, SubtopicMastery


class AdaptiveInterviewController:
    """
    Longitudinal Adaptive Interview Controller
    Tracks mastery across sessions for reinforcement
    Exactly 3 questions per subtopic, cycle through topics continuously
    """
    
    # Available topics
    TOPICS = ["DBMS", "OS", "OOPS"]
    
    def __init__(self):
        self.sessions: Dict[str, AdaptiveInterviewState] = {}
        self.question_bank = AdaptiveQuestionBank()
        self.decision_engine = AdaptiveDecisionEngine()
        self.subtopic_trackers: Dict[int, SubtopicTracker] = {}
    
    def start_session(self, session_id: str, user_id: int, user_name: str = "") -> dict:
        """Start a new adaptive session with random topic order"""
        
        print("\n" + "="*80)
        print("🎬 NEW ADAPTIVE INTERVIEW SESSION STARTED")
        print("="*80)
        print(f"   User ID:    {user_id}")
        print(f"   User:       {user_name}")
        print(f"   Session ID: {session_id}")
        
        # Load user's mastery from database
        masteries = UserMastery.query.filter_by(user_id=user_id).all()
        
        # Create state
        state = AdaptiveInterviewState(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name
        )
        
        # Load masteries into state and identify weak topics
        weak_topics = {}
        for m in masteries:
            mastery = state.ensure_topic_mastery(m.topic)
            mastery.mastery_level = m.mastery_level
            mastery.semantic_avg = m.semantic_avg
            mastery.keyword_avg = m.keyword_avg
            mastery.total_questions = m.questions_attempted
            mastery.sessions_attempted = getattr(m, 'sessions_attempted', 1)
            mastery.current_difficulty = m.current_difficulty
            mastery.consecutive_good = m.consecutive_good
            mastery.consecutive_poor = m.consecutive_poor
            mastery.missing_concepts = set(m.get_missing_concepts())
            mastery.weak_concepts = set(m.get_weak_concepts())
            mastery.strong_concepts = set(m.get_strong_concepts())
            
            # Track weak topics for this session
            weak_concepts = m.get_weak_concepts()
            if weak_concepts:
                weak_topics[m.topic] = weak_concepts
            
            # Reset concept stagnation for fresh session
            mastery.concept_stagnation = {}
            
            # Load concept-level tracking from DB
            if hasattr(m, 'concept_stagnation') and m.concept_stagnation:
                try:
                    stagnation_data = json.loads(m.concept_stagnation)
                    for concept_name in stagnation_data.keys():
                        if concept_name not in mastery.concepts:
                            from .adaptive_state import ConceptMastery
                            mastery.concepts[concept_name] = ConceptMastery(
                                name=concept_name,
                                topic=m.topic,
                                is_weak=(concept_name in mastery.weak_concepts),
                                is_strong=(concept_name in mastery.strong_concepts)
                            )
                            if concept_name in mastery.weak_concepts:
                                mastery.concepts[concept_name].stagnation_count = stagnation_data.get(concept_name, 1)
                                mastery.concepts[concept_name].attempts = stagnation_data.get(concept_name, 1) * 2
                except Exception as e:
                    print(f"⚠️ Error loading concept stagnation: {e}")
                    mastery.concept_stagnation = {}
        
        # Store weak topics in state
        state.weak_topics_history = weak_topics
        
        print("\n   📋 TOPIC ORDER:")
        topic_order = random.sample(self.TOPICS, len(self.TOPICS))
        for i, topic in enumerate(topic_order, 1):
            print(f"      {i}. {topic}")
        
        # Store order in state
        state.topic_order = topic_order
        state.current_topic_index = 0
        first_topic = topic_order[0]
        
        print(f"\n   📊 LOADED MASTERIES:")
        for topic in self.TOPICS:
            mastery = state.topic_mastery.get(topic)
            if mastery:
                weak = list(mastery.weak_concepts)[:3]
                strong = list(mastery.strong_concepts)[:3]
                print(f"      {topic}:")
                print(f"         Level:     {mastery.mastery_level:.3f}")
                print(f"         Questions: {mastery.total_questions}")
                print(f"         Weak:      {weak if weak else 'None'}")
                print(f"         Strong:    {strong if strong else 'None'}")
            else:
                print(f"      {topic}: NEW (no history)")
        
        # Get mastery for first topic
        mastery_for_topic = state.ensure_topic_mastery(first_topic)
        difficulty = mastery_for_topic.get_recommended_difficulty()
        
        # Initialize subtopic tracker for this user
        if user_id not in self.subtopic_trackers:
            self.subtopic_trackers[user_id] = SubtopicTracker(user_id)
        
        tracker = self.subtopic_trackers[user_id]
        
        # Use tracker to select first subtopic
        print("\n" + "-"*80)
        print("🎯 SUBTOPIC SELECTION")
        print("-"*80)
        
        stats = tracker.get_statistics()
        topic_stats = stats['by_topic'].get(first_topic, {})
        
        print(f"   Topic: {first_topic}")
        print(f"\n   Current subtopic status:")
        print(f"      WEAK:   {topic_stats.get('weak', 0)} subtopics")
        print(f"      NEW:    {topic_stats.get('new', 0)} subtopics")  # 🔥 FIXED: Use 'new' directly
        print(f"      STRONG: {topic_stats.get('strong', 0)} subtopics")
        
        chosen_subtopic = tracker.get_next_subtopic(first_topic)
        print(f"\n   ✅ SELECTED: {chosen_subtopic}")
        print("-"*80)
        
        # 🔥 CRITICAL: Get weak concepts for the topic
        weak_concepts = list(mastery_for_topic.weak_concepts) if mastery_for_topic else []
        print(f"\n   🎯 Weak concepts to target: {weak_concepts[:5]}")
        
        first_question, subtopic = self._generate_question(
            session_id=session_id,
            topic=first_topic,
            difficulty=difficulty,
            user_name=user_name,
            force_subtopic=chosen_subtopic,
            weak_concepts=weak_concepts  # 🔥 PASS WEAK CONCEPTS
        )
        
        # Update state
        state.current_topic = first_topic
        state.current_subtopic = subtopic
        state.current_question = first_question
        state.current_difficulty = difficulty
        state.question_start_time = time.time()
        state.topics_covered_this_session.append(first_topic)
        
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

        print("\n" + "="*80)
        print(f"✅ SESSION INITIALIZED")
        print("="*80)
        
        return {
            "action": "START",
            "question": first_question,
            "topic": first_topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "topic_order": topic_order,
            "current_topic_index": 0,
            "progress": f"Topic 1/{len(topic_order)}",
            "masteries": {
                t: {
                    'level': round(m.mastery_level, 3),
                    'total_questions': m.total_questions,
                    'weak_concepts': weak_topics.get(t, [])
                }
                for t, m in state.topic_mastery.items()
            }
        }
    
    def _generate_question(self, session_id: str, topic: str, difficulty: str, 
                          user_name: str = "", max_attempts: int = 5,
                          force_subtopic: str = None,
                          weak_concepts: list = None) -> tuple:  # 🔥 ADD weak_concepts parameter
        """
        Generate a question and return (question_text, subtopic)
        If force_subtopic is provided, use that specific subtopic
        """
        # Get all previously asked questions
        asked_questions = []
        subtopic = None
        
        # 🔥 DEBUG: Show incoming difficulty
        print(f"🔍 [DEBUG] _generate_question received difficulty = {difficulty}")
        
        if session_id in self.sessions:
            state = self.sessions[session_id]
            asked_questions = [record.question for record in state.history]
            
            # 🔥 If weak_concepts not provided, get them from state
            if weak_concepts is None:
                mastery = state.topic_mastery.get(topic)
                weak_concepts = list(mastery.weak_concepts) if mastery else []
        
        # Determine which subtopic to use
        if force_subtopic:
            subtopic = force_subtopic
            print(f"🎯 Using forced subtopic: {subtopic}")
        else:
            # Get available subtopics for this topic
            available_subtopics = self.question_bank.subtopics_by_topic.get(topic, [])
            
            # If we have a current subtopic in state, try to stick with it
            if session_id in self.sessions and self.sessions[session_id].current_subtopic:
                state = self.sessions[session_id]
                topic_session = state.get_topic_session(topic)
                
                # If we've asked less than 3 questions on this topic, stay on same subtopic
                if topic_session.questions_asked < 3 and state.current_subtopic in available_subtopics:
                    subtopic = state.current_subtopic
                    print(f"🎯 Continuing with current subtopic: {subtopic}")
                else:
                    # Pick a new subtopic using tracker
                    if state.user_id in self.subtopic_trackers:
                        tracker = self.subtopic_trackers[state.user_id]
                        subtopic = tracker.get_next_subtopic(topic)
                        print(f"🎯 Tracker selected: {subtopic}")
                    else:
                        subtopic = random.choice(available_subtopics) if available_subtopics else "core concepts"
                        print(f"🎯 Random selected: {subtopic}")
            else:
                # First question for this topic - pick using tracker
                if session_id in self.sessions:
                    state = self.sessions[session_id]
                    if state.user_id in self.subtopic_trackers:
                        tracker = self.subtopic_trackers[state.user_id]
                        subtopic = tracker.get_next_subtopic(topic)
                        print(f"🎯 Tracker selected first: {subtopic}")
                    else:
                        subtopic = random.choice(available_subtopics) if available_subtopics else "core concepts"
                        print(f"🎯 Random selected first: {subtopic}")
                else:
                    subtopic = random.choice(available_subtopics) if available_subtopics else "core concepts"
                    print(f"🎯 Random selected (no state): {subtopic}")
        
        print(f"\n   🎯 Weak concepts for this question: {weak_concepts[:5] if weak_concepts else 'None'}")
        
        for attempt in range(max_attempts):
            # 🔥 Generate question with weak concepts
            question = self.question_bank.generate_first_question(
                topic=topic,
                subtopic=subtopic,
                difficulty=difficulty,
                user_name=user_name,
                weak_concepts=weak_concepts  # 🔥 PASS WEAK CONCEPTS
            )
            
            print(f"🔍 [DEBUG] Question bank returned difficulty, final question: {question[:50]}...")
            
            # Check semantic uniqueness
            if not semantic_dedup.is_duplicate(session_id, question, asked_questions):
                return question, subtopic
            
            print(f"🔄 Attempt {attempt + 1}: Question was semantically similar, retrying...")
        
        # Fallback - use a simple template
        print(f"⚠️ Using fallback for {topic} - {subtopic}")
        fallback_question = f"Explain the key concepts of {subtopic} in {topic}."
        return fallback_question, subtopic

    def _generate_question_targeting_weakness(self, session_id: str, topic: str, weak_concepts: list, 
                                              difficulty: str, user_name: str = "") -> tuple:
        """Generate question specifically targeting weak concepts"""
        
        state = self.sessions.get(session_id)
        asked_questions = [record.question for record in state.history if record.topic == topic] if state else []
        
        # Try each weak concept (up to 3 attempts)
        for concept in weak_concepts[:3]:
            question = self.question_bank.generate_gap_followup(
                topic=topic,
                missing_concepts=[concept],
                difficulty=difficulty
            )
            
            # Check if it's semantically unique
            if not semantic_dedup.is_duplicate(session_id, question, asked_questions):
                print(f"🎯 Generated question targeting weak concept: {concept}")
                return question, concept
        
        # Fallback to regular generation
        print(f"⚠️ Could not generate unique question for weak concepts, using regular generation")
        return self._generate_question(session_id, topic, difficulty, user_name, weak_concepts=weak_concepts)
    
    def handle_answer(self, session_id: str, answer: str, expected_answer: str = "") -> dict:
        """Process user answer with expected answer from RAG"""
        
        state = self.sessions.get(session_id)
        if not state:
            return {"error": "Session not found"}

        print("\n" + "█"*80)
        print(f"📊 ANSWER ANALYSIS")
        print("█"*80)
        print(f"   Question #{len(state.history) + 1}")
        print(f"   Topic:     {state.current_topic}")
        print(f"   Subtopic:  {state.current_subtopic}")

        # Handle silent answers properly
        if not answer or answer == "[User remained silent]" or len(answer.strip()) < 5:
            print(f"\n   ⚠️ USER WAS SILENT")
            answer = "[User remained silent]"
            # Force scores to 0 for silent answers
            analysis = {
                "keyword_coverage": 0.0,
                "depth": "shallow",
                "missing_concepts": [],
                "covered_concepts": [],
                "confidence": "low",
                "key_terms_used": [],
                "response_length": 0,
                "grammatical_quality": 0.0,
                "has_example": False,
                "estimated_difficulty": "easy",
                "semantic_similarity": 0.0,
                "expected_answer": expected_answer
            }
            semantic_score = 0.0
            keyword_score = 0.0
            missing = []
            concepts_mentioned = []
        else:
            print(f"\n   Answer:    {answer[:100]}..." if len(answer) > 100 else f"   Answer:    {answer}")
            print(f"\n   Expected:  {expected_answer[:100]}..." if expected_answer else "   Expected:  None")
            
            # Normal analysis for non-silent answers
            analysis = AdaptiveAnalyzer.analyze(
                question=state.current_question,
                answer=answer,
                topic=state.current_topic,
                subtopic=state.current_subtopic,
                question_bank=self.question_bank,
                expected_answer=expected_answer
            )
            
            semantic_score = analysis.get("semantic_similarity", 0.0)
            keyword_score = analysis.get("keyword_coverage", 0.0)
            missing = analysis.get("missing_concepts", [])
            concepts_mentioned = analysis.get("key_terms_used", [])
        
        # Calculate response time
        response_time = time.time() - (state.question_start_time or time.time())
        
        # Get current context
        question = state.current_question
        topic = state.current_topic
        subtopic = state.current_subtopic
        
        # Length penalty (but don't apply if we already have 0 from empty expected answer)
        word_count = len(answer.split())
        if word_count < 10 and semantic_score > 0:
            semantic_score *= 0.7
        elif word_count < 20 and semantic_score > 0:
            semantic_score *= 0.85
        
        semantic_score = min(0.95, semantic_score)
        
        print(f"\n   📈 SCORES:")
        print(f"      Semantic similarity: {semantic_score:.3f}")
        print(f"      Keyword coverage:    {keyword_score:.3f}")
        
        if missing:
            print(f"\n   ❌ MISSING CONCEPTS:")
            for i, concept in enumerate(missing[:5], 1):
                print(f"      {i}. {concept}")
        
        if concepts_mentioned:
            print(f"\n   ✅ COVERED CONCEPTS:")
            for i, concept in enumerate(concepts_mentioned[:5], 1):
                print(f"      {i}. {concept}")
        
        # Update subtopic mastery
        if state.user_id not in self.subtopic_trackers:
            self.subtopic_trackers[state.user_id] = SubtopicTracker(state.user_id)
        
        tracker = self.subtopic_trackers[state.user_id]
        tracker.update_subtopic_performance(topic, subtopic, semantic_score)
        
        # Update long-term topic mastery
        mastery = state.ensure_topic_mastery(topic)
        mastery.update(
            semantic=semantic_score,
            keyword=keyword_score,
            response_time=response_time,
            missing=missing,
            concepts=concepts_mentioned
        )
        
        # Update session-level topic tracking
        topic_session = state.get_topic_session(topic)
        topic_session.add_answer(
            semantic=semantic_score,
            keyword=keyword_score,
            depth=analysis.get("depth", "medium")
        )
        
        # Create QA record with expected_answer
        record = AdaptiveQARecord(
            question=question,
            topic=topic,
            subtopic=subtopic,
            difficulty=state.current_difficulty,
            answer=answer,
            analysis=analysis,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            response_time=response_time,
            missing_concepts=missing
        )
        
        # Add to history
        state.add_to_history(record)
        
        # Save to database
        self._save_to_db(state.user_id, session_id, record, mastery, expected_answer)
        
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
    
    def _calculate_next_difficulty(self, current_question_num: int, current_score: float, previous_difficulty: str) -> str:
        """
        Calculate next question difficulty based on performance
        
        Question 1: Always MEDIUM
        Question 2: Based on Q1 score
        Question 3: Based on Q2 score and Q2 difficulty
        """
        
        # Question 1 is always MEDIUM
        if current_question_num == 1:
            return "medium"
        
        # For Question 2 (based on Q1 score)
        if current_question_num == 2:
            if current_score < 0.4:
                return "easy"
            elif current_score > 0.7:
                return "hard"
            else:
                return "medium"
        
        # For Question 3 (based on Q2 score and Q2 difficulty)
        if current_question_num == 3:
            # Case 1: Poor performance on Q2
            if current_score < 0.4:
                return "easy"
            
            # Case 2: Good performance on Q2
            elif current_score > 0.7:
                # If Q2 was hard and they did well, keep it hard
                if previous_difficulty == "hard":
                    return "hard"
                # Otherwise, challenge with hard
                else:
                    return "hard"
            
            # Case 3: Medium performance on Q2 (0.4-0.7)
            else:
                # If Q2 was hard, dial back to medium
                if previous_difficulty == "hard":
                    return "medium"
                # If Q2 was easy and they got medium, go to medium
                elif previous_difficulty == "easy":
                    return "medium"
                # If Q2 was medium, stay medium
                else:
                    return "medium"
        
        # Fallback (should never reach here)
        return "medium"
    
    def _generate_followup(self, state, analysis, session_id) -> dict:
        """Generate appropriate follow-up question with dynamic difficulty"""
        
        state.followup_count += 1
        
        missing = analysis.get("missing_concepts", [])
        topic = state.current_topic
        semantic_score = analysis.get("semantic_similarity", 0)
        
        # 🔥 CRITICAL: Get weak concepts from topic mastery
        mastery = state.topic_mastery.get(topic)
        weak_concepts = list(mastery.weak_concepts) if mastery else []
        
        # Count questions on current subtopic (1, 2, or 3)
        questions_on_subtopic = 0
        for record in state.history:
            if record.topic == topic and record.subtopic == state.current_subtopic:
                questions_on_subtopic += 1
        
        current_question_num = questions_on_subtopic + 1  # Next question number (1-3)
        
        print("\n" + "█"*80)
        print(f"🎯 NEXT QUESTION DECISION")
        print("█"*80)
        print(f"   Topic:           {topic}")
        print(f"   Subtopic:        {state.current_subtopic}")
        print(f"   Question #:      {current_question_num}/3")
        print(f"   Current quality: {semantic_score:.2f}")
        print(f"   Missing concepts:{' ' + str(missing[:3]) if missing else ' None'}")
        print(f"   Weak concepts:   {weak_concepts[:5]}")
        
        # Check if we've already asked 3 questions on this subtopic
        if questions_on_subtopic >= 3:
            print(f"\n   → Already asked 3 questions, moving to next topic")
            return self._move_to_next_topic(state, session_id)
        
        # Calculate next difficulty based on performance
        next_difficulty = self._calculate_next_difficulty(
            current_question_num=current_question_num,
            current_score=semantic_score,
            previous_difficulty=state.current_difficulty
        )
        
        print(f"\n   📊 DIFFICULTY LOGIC:")
        if current_question_num == 1:
            print(f"      Rule: First question → MEDIUM")
        elif current_question_num == 2:
            if semantic_score < 0.4:
                print(f"      Rule: Q1 score {semantic_score:.2f} (<0.4) → EASY")
            elif semantic_score > 0.7:
                print(f"      Rule: Q1 score {semantic_score:.2f} (>0.7) → HARD")
            else:
                print(f"      Rule: Q1 score {semantic_score:.2f} (0.4-0.7) → MEDIUM")
        else:  # Question 3
            if semantic_score < 0.4:
                print(f"      Rule: Q2 score {semantic_score:.2f} (<0.4) → EASY")
            elif semantic_score > 0.7:
                if state.current_difficulty == "hard":
                    print(f"      Rule: Q2 was HARD, scored well → keep HARD")
                else:
                    print(f"      Rule: Q2 score {semantic_score:.2f} (>0.7) → HARD")
            else:
                if state.current_difficulty == "hard":
                    print(f"      Rule: Q2 was HARD, scored medium → dial back to MEDIUM")
                elif state.current_difficulty == "easy":
                    print(f"      Rule: Q2 was EASY, scored medium → increase to MEDIUM")
                else:
                    print(f"      Rule: Q2 was MEDIUM, scored medium → stay MEDIUM")
        
        print(f"\n   ✅ NEXT QUESTION: {next_difficulty.upper()}")
        print("█"*80)
        
        # 🔥 DEBUG: Show what we're passing to question bank
        print(f"🔍 [DEBUG] Calling generate_question_for_subtopic with:")
        print(f"   topic = {topic}")
        print(f"   subtopic = {state.current_subtopic}")
        print(f"   difficulty = {next_difficulty}")
        print(f"   weak_concepts = {weak_concepts[:5]}")
        
        # Generate question for the same subtopic with calculated difficulty
        question = None
        
        try:
            question = self.question_bank.generate_question_for_subtopic(
                topic=topic,
                subtopic=state.current_subtopic,
                difficulty=next_difficulty,
                weak_concepts=weak_concepts  # 🔥 PASS WEAK CONCEPTS
            )
            print(f"🔍 [DEBUG] Question generated successfully")
        except Exception as e:
            print(f"⚠️ generate_question_for_subtopic failed: {e}")
            question = None
        
        # If that fails, try generate_first_question with forced subtopic
        if not question or len(question) < 10:
            try:
                print(f"🔍 [DEBUG] Falling back to generate_first_question")
                question = self.question_bank.generate_first_question(
                    topic=topic,
                    subtopic=state.current_subtopic,
                    difficulty=next_difficulty,
                    user_name=state.user_name,
                    weak_concepts=weak_concepts  # 🔥 PASS WEAK CONCEPTS
                )
            except Exception as e:
                print(f"⚠️ generate_first_question failed: {e}")
                question = None
        
        # Ultimate fallback - use a template question
        if not question or len(question) < 10:
            fallback_templates = {
                "easy": f"Can you explain the basic concepts of {state.current_subtopic} in {topic}?",
                "medium": f"What are the key principles of {state.current_subtopic} in {topic}?",
                "hard": f"Can you provide a detailed explanation of {state.current_subtopic} with examples in {topic}?"
            }
            question = fallback_templates[next_difficulty]
            print(f"⚠️ Using fallback template for {state.current_subtopic}")
        
        # Check for semantic duplicates
        asked_questions = [r.question for r in state.history]
        from .semantic_dedup import semantic_dedup
        if semantic_dedup.is_duplicate(session_id, question, asked_questions, threshold=0.9):
            context_modifiers = [
                f"Let's explore this further: {question}",
                f"Building on that, {question}",
                f"To dive deeper: {question}",
                f"Now consider this: {question}"
            ]
            question = random.choice(context_modifiers)
        
        # Update state
        state.current_question = question
        state.current_difficulty = next_difficulty
        state.question_start_time = time.time()
        
        return {
            "action": "FOLLOW_UP",
            "question": question,
            "topic": topic,
            "subtopic": state.current_subtopic,
            "difficulty": next_difficulty,
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
            question = f"Here's a more challenging question: {question}"
        
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
        """Move to next topic - cycle through topics continuously"""
        
        print("\n" + "="*80)
        print("➡️ MOVING TO NEXT TOPIC")
        print("="*80)
        
        # Try to advance to next topic
        if not state.advance_to_next_topic():
            # No more topics left - START A NEW CYCLE!
            print("\n🔄 Completed one full cycle of all topics - starting new cycle")
            
            # Reset to first topic
            state.current_topic_index = 0
            state.current_topic = state.topic_order[0]
            state.followup_count = 0
            
            # Reset ALL topic sessions for new cycle
            for topic in list(state.topic_sessions.keys()):
                state.topic_sessions[topic] = TopicSessionState(topic=topic)
            
            print("🔄 Reset all topic sessions for new cycle")
        
        new_topic = state.current_topic
        print(f"\n➡️ New topic: {new_topic}")
        
        # Get mastery for this topic
        mastery = state.ensure_topic_mastery(new_topic)
        
        # 🔥 Get weak concepts for targeting
        weak_concepts = list(mastery.weak_concepts) if mastery else []
        print(f"   Weak concepts to target: {weak_concepts[:5]}")
        
        # Use subtopic tracker to select the next subtopic
        if state.user_id not in self.subtopic_trackers:
            self.subtopic_trackers[state.user_id] = SubtopicTracker(state.user_id)
        
        tracker = self.subtopic_trackers[state.user_id]
        
        print("\n" + "-"*80)
        print("🎯 SUBTOPIC SELECTION")
        print("-"*80)
        
        stats = tracker.get_statistics()
        topic_stats = stats['by_topic'].get(new_topic, {})
        
        print(f"   Topic: {new_topic}")
        print(f"\n   Current subtopic status:")
        print(f"      WEAK:   {topic_stats.get('weak', 0)} subtopics")
        print(f"      NEW:    {topic_stats.get('new', 0)} subtopics")  # 🔥 FIXED: Use 'new' directly
        print(f"      STRONG: {topic_stats.get('strong', 0)} subtopics")
        
        chosen_subtopic = tracker.get_next_subtopic(new_topic)
        print(f"\n   ✅ SELECTED: {chosen_subtopic}")
        print("-"*80)
        
        # NEW SUBTOPIC - First question is ALWAYS MEDIUM
        difficulty = "medium"
        state.current_difficulty = difficulty
        state.current_subtopic = chosen_subtopic
        
        print(f"\n   🔍 DEBUG - Setting difficulty to: {difficulty} (MUST BE MEDIUM)")
        print(f"\n   → First question difficulty: MEDIUM")
        
        # Generate question for chosen subtopic WITH WEAK CONCEPTS
        question, subtopic = self._generate_question(
            session_id=session_id,
            topic=new_topic,
            difficulty=difficulty,
            user_name=state.user_name,
            force_subtopic=chosen_subtopic,
            weak_concepts=weak_concepts  # 🔥 PASS WEAK CONCEPTS
        )
        
        state.current_question = question
        state.current_subtopic = subtopic
        state.question_start_time = time.time()
        state.topics_covered_this_session.append(new_topic)
        
        # Initialize fresh topic session
        state.get_topic_session(new_topic)
        
        print("\n" + "="*80)
        print(f"✅ MOVED TO {new_topic}")
        print("="*80)
        
        return {
            "action": "MOVE_TOPIC",
            "question": question,
            "topic": new_topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "time_remaining": state.time_remaining_sec(),
            "topic_order": state.topic_order,
            "current_topic_index": state.current_topic_index,
            "progress": f"Cycle continues - Topic: {new_topic}",
            "weak_concepts_targeted": weak_concepts[:3]
        }
    
    def _finalize_session(self, state, session_id):
        """Finalize interview and update weak topics tracking for next session"""
        
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
        
        print(f"\n✅ Session {session_id} finalized with {len(state.history)} questions")
    
    def _generate_feedback(self, state) -> dict:
        """Generate end-of-session feedback"""
        
        # Get subtopic statistics if available
        subtopic_stats = {}
        if state.user_id in self.subtopic_trackers:
            tracker = self.subtopic_trackers[state.user_id]
            subtopic_stats = tracker.get_statistics()
        
        return {
            "questions_answered": len(state.history),
            "duration_minutes": int((time.time() - state.start_time) / 60),
            "topics_covered": list(state.topic_sessions.keys()),
            "weakest_topics": state.get_sorted_topics_by_priority()[:2],
            "next_session_focus": state.get_sorted_topics_by_priority()[0] if state.get_sorted_topics_by_priority() else "DBMS",
            "subtopic_stats": subtopic_stats
        }
    
    def _save_to_db(self, user_id: int, session_id: str, record: AdaptiveQARecord, mastery, expected_answer: str = ""):
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
                response_time=record.response_time,
                missing=record.missing_concepts
            )
            
            # Save concept stagnation
            db_mastery.concept_stagnation = json.dumps(mastery.concept_stagnation)
            db_mastery.weak_concepts = json.dumps(list(mastery.weak_concepts))
            db_mastery.strong_concepts = json.dumps(list(mastery.strong_concepts))
            
            # Save question history with expected_answer
            history = QuestionHistory(
                user_id=user_id,
                session_id=session_id,
                topic=record.topic,
                subtopic=record.subtopic,
                question=record.question,
                answer=record.answer,
                expected_answer=expected_answer or record.analysis.get("expected_answer", ""),
                semantic_score=record.semantic_score,
                keyword_score=record.keyword_score,
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
    
    def reset_user_mastery(self, user_id: int, topic: str = None) -> dict:
        """Reset mastery for a user - COMPLETE in-memory cleanup"""
        try:
            # Clear in-memory tracker
            if user_id in self.subtopic_trackers:
                # Reset the tracker (creates new empty one)
                self.subtopic_trackers[user_id].reset_all_mastery()
                # Optionally, delete it to force fresh creation
                if topic:
                    # For topic-specific reset, we need to refresh
                    self.subtopic_trackers[user_id].reset_topic_mastery(topic)
                else:
                    # For full reset, create fresh tracker
                    self.subtopic_trackers[user_id] = SubtopicTracker(user_id)
            
            # Also clear any session data for this user
            sessions_to_remove = []
            for session_id, state in self.sessions.items():
                if state.user_id == user_id:
                    if topic:
                        # Only remove if it's the specific topic session
                        if state.current_topic == topic:
                            sessions_to_remove.append(session_id)
                    else:
                        # Remove all sessions for this user
                        sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                semantic_dedup.clear_session(session_id)
                del self.sessions[session_id]
            
            return {"success": True, "message": f"Reset {'all' if not topic else topic} mastery successfully"}
        except Exception as e:
            print(f"Error resetting mastery: {e}")
            return {"error": str(e)}
    
    def get_subtopic_stats(self, user_id: int) -> dict:
        """Get subtopic mastery statistics"""
        if user_id in self.subtopic_trackers:
            return self.subtopic_trackers[user_id].get_statistics()
        else:
            tracker = SubtopicTracker(user_id)
            self.subtopic_trackers[user_id] = tracker
            return tracker.get_statistics()
    
    def get_topic_subtopics(self, user_id: int, topic: str) -> list:
        """Get all subtopics for a topic with their mastery status"""
        from .subtopic_tracker import SubtopicTracker
        
        tracker = SubtopicTracker(user_id)
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
        
        return result