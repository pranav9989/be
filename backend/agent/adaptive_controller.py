# backend/agent/adaptive_controller.py

import time
import json
import numpy as np
import random
from typing import Dict, Any, Set, List, Optional
from datetime import datetime

from .adaptive_state import AdaptiveInterviewState, AdaptiveQARecord, TopicSessionState, ConceptMastery
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
    
    TOPICS = ["DBMS", "OS", "OOPS"]
    
    # STRICT 9-CASE DIFFICULTY MATRIX - ADD THIS
    def _calculate_next_difficulty(self, question_number: int, previous_score: float, previous_difficulty: str) -> str:
        """
        STRICT 9-CASE DIFFICULTY MATRIX
        
        Q1: Always MEDIUM
        
        Q2: Based on Q1 score:
            < 0.4  → EASY
            0.4-0.7 → MEDIUM
            > 0.7  → HARD
        
        Q3: Based on Q2 score:
            < 0.4  → EASY (regardless of previous)
            0.4-0.7 → MEDIUM (regardless of previous)
            > 0.7  → HARD (regardless of previous)
        """
        # Q1 always medium
        if question_number == 1:
            return "medium"
        
        # Q2 logic (based on Q1 score)
        if question_number == 2:
            if previous_score < 0.4:
                return "easy"
            elif previous_score > 0.7:
                return "hard"
            else:
                return "medium"
        
        # Q3 logic (based on Q2 score)
        if question_number == 3:
            if previous_score < 0.4:
                return "easy"
            elif previous_score > 0.7:
                return "hard"
            else:
                return "medium"
        
        # Fallback (should not happen)
        return "medium"
    
    def __init__(self):
        self.sessions: Dict[str, AdaptiveInterviewState] = {}
        self.question_bank = AdaptiveQuestionBank()
        self.decision_engine = AdaptiveDecisionEngine()
        self.subtopic_trackers: Dict[int, SubtopicTracker] = {}
    
    def _concept_in_answer(self, concept: str, answer_lower: str) -> bool:
        """
        INVARIANT 1: Detect concept with synonym support
        Used for concept mastery tracking
        """
        concept_lower = concept.lower()
        
        # Direct match
        if concept_lower in answer_lower:
            return True
        
        # Multi-word concept without spaces
        if ' ' in concept_lower:
            concept_no_space = concept_lower.replace(' ', '')
            answer_no_space = answer_lower.replace(' ', '')
            if concept_no_space in answer_no_space:
                return True
        
        # Synonym mapping
        synonyms = {
            'mutex': ['mutex', 'mutual exclusion', 'lock'],
            'semaphore': ['semaphore', 'counting semaphore', 'binary semaphore'],
            'critical section': ['critical section', 'critical region'],
            'deadlock': ['deadlock', 'deadly embrace'],
            'process': ['process', 'task'],
            'thread': ['thread', 'lightweight process'],
            'primary key': ['primary key', 'primary-key', 'pk'],
            'foreign key': ['foreign key', 'foreign-key', 'fk'],
            'avoidance': ['banker', 'safe state', 'avoidance'],
            'prevention': ['prevention', 'prevent'],
            'detection': ['detection', 'detect', 'wait-for graph']
        }
        
        if concept_lower in synonyms:
            for synonym in synonyms[concept_lower]:
                if synonym in answer_lower:
                    return True
        
        return False
    
    def start_session(self, session_id: str, user_id: int, user_name: str = "") -> dict:
        print("\n" + "="*80)
        print("🎬 NEW ADAPTIVE INTERVIEW SESSION STARTED")
        print("="*80)
        print(f"   User ID:    {user_id}")
        print(f"   User:       {user_name}")
        print(f"   Session ID: {session_id}")
        
        masteries = UserMastery.query.filter_by(user_id=user_id).all()
        
        state = AdaptiveInterviewState(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name
        )
        
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
            mastery.mastery_velocity = m.mastery_velocity
            mastery.last_mastery = m.last_mastery
            
            concept_data = m.get_concept_masteries()
            for concept_name, cd in concept_data.items():
                concept = ConceptMastery.from_dict(cd)
                mastery.concepts[concept_name] = concept
                
                if concept.is_weak:
                    mastery.weak_concepts.add(concept_name)
                    weak_topics.setdefault(m.topic, []).append(concept_name)
                if concept.is_strong:
                    mastery.strong_concepts.add(concept_name)
        
        state.weak_topics_history = weak_topics
        
        print("\n   📋 TOPIC ORDER:")
        topic_order = random.sample(self.TOPICS, len(self.TOPICS))
        for i, topic in enumerate(topic_order, 1):
            print(f"      {i}. {topic}")
        
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
                print(f"         Velocity:  {mastery.mastery_velocity:+.3f}")
                print(f"         Weak:      {weak if weak else 'None'}")
                print(f"         Strong:    {strong if strong else 'None'}")
            else:
                print(f"      {topic}: NEW (no history)")
        
        mastery_for_topic = state.ensure_topic_mastery(first_topic)
        difficulty = mastery_for_topic.get_recommended_difficulty()
        
        if user_id not in self.subtopic_trackers:
            self.subtopic_trackers[user_id] = SubtopicTracker(user_id)
        
        tracker = self.subtopic_trackers[user_id]
        
        print("\n" + "-"*80)
        print("🎯 SUBTOPIC SELECTION")
        print("-"*80)
        
        # Get weak concepts for this topic to pass to subtopic tracker
        weak_concepts_for_topic = list(mastery_for_topic.weak_concepts) if mastery_for_topic else []
        chosen_subtopic = tracker.get_next_subtopic(first_topic, weak_concepts=weak_concepts_for_topic, covered_subtopics=[])
        print(f"\n   ✅ SELECTED: {chosen_subtopic}")
        print("-"*80)
        
        weak_concepts = list(mastery_for_topic.weak_concepts) if mastery_for_topic else []
        print(f"\n   🎯 Weak concepts to target: {weak_concepts[:5]}")
        
        first_question, subtopic, sampled_concepts = self._generate_question(
            session_id=session_id,
            topic=first_topic,
            difficulty=difficulty,
            user_name=user_name,
            force_subtopic=chosen_subtopic,
            weak_concepts=weak_concepts
        )
        
        state.current_sampled_concepts = sampled_concepts
        
        state.current_topic = first_topic
        state.current_subtopic = subtopic
        state.current_question = first_question
        state.current_difficulty = difficulty
        state.question_start_time = time.time()
        state.topics_covered_this_session.append(first_topic)
        
        state.get_topic_session(first_topic)
        
        self.sessions[session_id] = state
        
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
            "sampled_concepts": sampled_concepts,
            "masteries": {
                t: {
                    'level': round(m.mastery_level, 3),
                    'velocity': round(m.mastery_velocity, 3),
                    'total_questions': m.total_questions,
                    'weak_concepts': weak_topics.get(t, [])
                }
                for t, m in state.topic_mastery.items()
            }
        }
    
    def _generate_question(self, session_id: str, topic: str, difficulty: str, 
                          user_name: str = "", max_attempts: int = 5,
                          force_subtopic: str = None,
                          weak_concepts: list = None) -> tuple:
        """
        Generate question using priority-based concept sampling
        INVARIANT 3: Uses priority_score sorting (no random)
        """
        asked_questions = []
        subtopic = None
        
        if session_id in self.sessions:
            state = self.sessions[session_id]
            asked_questions = [record.question for record in state.history]
        
        # Get subtopic
        if force_subtopic:
            subtopic = force_subtopic
            print(f"🎯 Using forced subtopic: {subtopic}")
        else:
            if session_id in self.sessions:
                state = self.sessions[session_id]
                if state.user_id in self.subtopic_trackers:
                    tracker = self.subtopic_trackers[state.user_id]
                    mastery = state.topic_mastery.get(topic)
                    weak_for_topic = list(mastery.weak_concepts) if mastery else []
                    covered = state.cycle_covered_subtopics.get(topic, [])
                    subtopic = tracker.get_next_subtopic(topic, weak_concepts=weak_for_topic, covered_subtopics=covered)
                    print(f"🎯 Tracker selected: {subtopic}")
                else:
                    available_subtopics = self.question_bank.subtopics_by_topic.get(topic, [])
                    subtopic = random.choice(available_subtopics) if available_subtopics else "core concepts"
                    print(f"🎯 Random fallback: {subtopic}")
            else:
                available_subtopics = self.question_bank.subtopics_by_topic.get(topic, [])
                subtopic = random.choice(available_subtopics) if available_subtopics else "core concepts"
                print(f"🎯 Random selection (no session): {subtopic}")
        
        # Get mastery state for concept sampling
        mastery = None
        if session_id in self.sessions:
            state = self.sessions[session_id]
            mastery = state.topic_mastery.get(topic)
        
        if not mastery:
            # Create temporary mastery if none exists
            from .adaptive_state import TopicMastery
            mastery = TopicMastery(topic=topic)
            print(f"   📊 Created temporary mastery for {topic}")
        
        # INVARIANT 3: Sample concepts by priority (not random)
        sampled_concepts = self.question_bank.sample_concepts_by_priority(
            topic, subtopic, mastery
        )
        
        print(f"\n   🎯 Sampled concepts: {sampled_concepts}")
        
        # Generate question with guaranteed concept inclusion
        for attempt in range(max_attempts):
            try:
                question = self.question_bank.generate_question(
                    #session_id=session_id,
                    topic=topic,
                    subtopic=subtopic,
                    concepts=sampled_concepts,
                    difficulty=difficulty,
                    user_name=user_name,
                    history=asked_questions
                )
                
                # Check for duplicates
                if semantic_dedup.is_duplicate(session_id, question, asked_questions):
                    print(f"🔄 Attempt {attempt + 1}: Duplicate detected, retrying...")
                    continue
                
                return question, subtopic, sampled_concepts
                
            except Exception as e:
                print(f"🔄 Attempt {attempt + 1} failed: {e}, retrying...")
                continue
        
        # Fallback - should rarely happen
        fallback = f"Explain the key concepts of {subtopic} in {topic}."
        print(f"⚠️ Using fallback question after {max_attempts} attempts")
        return fallback, subtopic, sampled_concepts
    
    def handle_answer(self, session_id: str, answer: str, expected_answer: str = "", stress_test: bool = False) -> dict:
        state = self.sessions.get(session_id)
        if not state:
            return {"error": "Session not found"}

        print("\n" + "█"*80)
        print(f"📊 ANSWER ANALYSIS")
        print("█"*80)
        print(f"   Question #{len(state.history) + 1}")
        print(f"   Topic:     {state.current_topic}")
        print(f"   Subtopic:  {state.current_subtopic}")
        
        sampled_concepts = state.current_sampled_concepts
        print(f"   🎯 Sampled concepts: {sampled_concepts}")

        if not answer or answer == "[User remained silent]" or len(answer.strip()) < 5:
            print(f"\n   ⚠️ USER WAS SILENT")
            answer = "[User remained silent]"
            
            semantic_score = 0.0
            keyword_score = 0.0
            mentioned = []
            missing = sampled_concepts.copy()
        else:
            print(f"\n   Answer:    {answer[:100]}..." if len(answer) > 100 else f"   Answer:    {answer}")
            
            answer_lower = answer.lower()
            mentioned = []
            missing = []
            
            for concept in sampled_concepts:
                if self._concept_in_answer(concept, answer_lower):
                    mentioned.append(concept)
                else:
                    missing.append(concept)
            
            print(f"\n   ✅ Mentioned: {mentioned}")
            print(f"   ❌ Missing: {missing}")
            
            semantic_score = 0.0
            keyword_score = 0.0
            
            if expected_answer and expected_answer.strip():
                from interview_analyzer import calculate_semantic_similarity, calculate_keyword_coverage
                semantic_score = calculate_semantic_similarity(answer, expected_answer)
                keyword_score = calculate_keyword_coverage(answer, state.current_question)
                print(f"📊 Semantic: {semantic_score:.3f}, Keyword: {keyword_score:.3f}")
            
            word_count = len(answer.split())
            if word_count < 10 and semantic_score > 0:
                semantic_score *= 0.7
            elif word_count < 20 and semantic_score > 0:
                semantic_score *= 0.85
            
            semantic_score = min(0.95, semantic_score)
        
        response_time = time.time() - (state.question_start_time or time.time())
        
        mastery = state.ensure_topic_mastery(state.current_topic)
        
        # Store old mastery for velocity calculation
        old_mastery = mastery.mastery_level
        
        mastery.update(
            semantic=semantic_score,
            keyword=keyword_score,
            response_time=response_time,
            sampled_concepts=sampled_concepts,
            mentioned_concepts=mentioned,
            missing_from_sampled=missing
        )

        # ============================================
        # UPDATE CONCEPT PRIORITIES WITH VELOCITY
        # ============================================
        # Pass the topic velocity to each concept's priority calculation
        for concept_name, concept in mastery.concepts.items():
            concept.update_priority_score(velocity=mastery.mastery_velocity)

        print(f"📊 CONCEPT PRIORITIES UPDATED WITH VELOCITY {mastery.mastery_velocity:+.3f}")
        
        # ============================================
        # CONCEPT TRACKING - COMPLETE FIX
        # ============================================
        answer_lower = answer.lower()
        for concept_name in state.current_sampled_concepts:
            # Ensure concept exists in mastery
            if concept_name not in mastery.concepts:
                mastery.concepts[concept_name] = ConceptMastery(name=concept_name)
            
            concept = mastery.concepts[concept_name]
            mentioned = self._concept_in_answer(concept_name, answer_lower)
            
            # ✅ CRITICAL: Record attempt with mentioned flag
            concept.record_attempt(mentioned)
            
            # Update weak/strong sets
            if concept.is_weak:
                mastery.weak_concepts.add(concept_name)
            else:
                mastery.weak_concepts.discard(concept_name)
                
            if concept.is_strong:
                mastery.strong_concepts.add(concept_name)
            else:
                mastery.strong_concepts.discard(concept_name)
        
        print(f"\n📊 CONCEPT TRACKING UPDATED:")
        for concept_name in state.current_sampled_concepts:
            concept = mastery.concepts[concept_name]
            print(f"   {concept_name}: attempts={concept.attempts}, "
                f"mentioned={concept.times_mentioned}, "
                f"missed={concept.times_missed_when_sampled}, "
                f"mastery={concept.mastery_level:.2f}, "
                f"priority={concept.priority_score:.2f}, "
                f"weak={concept.is_weak}, strong={concept.is_strong}")
        
        # ============================================
        # LEARNING VELOCITY CALCULATION
        # ============================================
        mastery.mastery_velocity = mastery.mastery_level - old_mastery
        mastery.last_mastery = old_mastery
        
        print(f"📈 TOPIC VELOCITY: {mastery.mastery_velocity:+.3f} "
            f"(from {old_mastery:.3f} to {mastery.mastery_level:.3f})")
        
        if state.user_id not in self.subtopic_trackers:
            self.subtopic_trackers[state.user_id] = SubtopicTracker(state.user_id)
        
        tracker = self.subtopic_trackers[state.user_id]
        tracker.update_subtopic_performance(state.current_topic, state.current_subtopic, semantic_score)
        
        topic_session = state.get_topic_session(state.current_topic)
        topic_session.add_answer(
            semantic=semantic_score,
            keyword=keyword_score,
            depth="medium"
        )
        
        record = AdaptiveQARecord(
            question=state.current_question,
            topic=state.current_topic,
            subtopic=state.current_subtopic,
            difficulty=state.current_difficulty,
            answer=answer,
            analysis={},
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            response_time=response_time,
            missing_concepts=missing,
            sampled_concepts=sampled_concepts
        )
        
        state.add_to_history(record)
        
        # 🔥 Get expected answer for this question from RAG FIRST - FIXED VERSION
        expected_answer_for_db = None
        try:
            from rag import agentic_expected_answer
            sampled_concepts_for_rag = state.current_sampled_concepts  # ✅ Use state directly
            
            # PASS THE CONTROLLER'S TOPIC - CRITICAL FIX
            current_topic = state.current_topic
            if not current_topic:
                print(f"⚠️ No current_topic found in session")
                current_topic = "DBMS"  # Fallback
                
            # ✅ FIX: Pass controller topic explicitly with correct parameter names
            expected_answer_for_db, chunks = agentic_expected_answer(
                user_query=state.current_question,
                sampled_concepts=sampled_concepts_for_rag,
                expected_topic=current_topic  # THIS IS THE CRITICAL FIX
            )
            print(f"📝 Generated expected answer using topic: {current_topic} ({len(expected_answer_for_db)} chars)")

        except Exception as e:
            print(f"⚠️ Could not get expected answer: {e}")
            expected_answer_for_db = state.current_question  # Simple fallback
        
        self._save_to_db(state.user_id, session_id, record, mastery, expected_answer_for_db, sampled_concepts)
        
        # Build comprehensive analysis dict for decision engine
        analysis_for_decision = {
            "semantic_similarity": semantic_score,
            "keyword_coverage": keyword_score,
            "missing_concepts": missing,
            "mentioned_concepts": mentioned,
            "mastery_velocity": mastery.mastery_velocity,
            "stagnation": {c: mastery.concepts[c].stagnation_count for c in sampled_concepts if c in mastery.concepts}
        }
        
        action = self.decision_engine.decide(state, analysis_for_decision)
        
        if action == "FINALIZE":
            self._finalize_session(state, session_id)
            return {
                "action": "FINALIZE",
                "next_question": None,
                "time_remaining": 0,
                "feedback": self._generate_feedback(state),
                "learning_velocity": round(mastery.mastery_velocity, 3)
            }
        
        if action == "MOVE_TOPIC":
            return self._move_to_next_topic(state, session_id)
        
        if action == "SIMPLIFY":
            return self._simplify_question(state, session_id)
        
        if action == "DEEPEN":
            return self._deepen_question(state, session_id)
        
        return self._generate_followup(state, analysis_for_decision, session_id, stress_test)

    def _generate_followup(self, state, analysis, session_id, stress_test=False) -> dict:
        state.followup_count += 1
        
        topic = state.current_topic
        mastery = state.topic_mastery.get(topic)
        weak_concepts = list(mastery.weak_concepts) if mastery else []
        
        # Count questions on this subtopic
        questions_on_subtopic = 0
        for record in state.history:
            if record.topic == topic and record.subtopic == state.current_subtopic:
                questions_on_subtopic += 1
        
        current_question_num = questions_on_subtopic + 1
        
        print("\n" + "█"*80)
        print(f"🎯 NEXT QUESTION DECISION")
        print("█"*80)
        print(f"   Topic:           {topic}")
        print(f"   Subtopic:        {state.current_subtopic}")
        print(f"   Question #:      {current_question_num}/3")
        print(f"   Learning velocity: {analysis.get('mastery_velocity', 0):+.3f}")
        
        if questions_on_subtopic >= 3:
            print(f"\n   → Already asked 3 questions, moving to next topic")
            return self._move_to_next_topic(state, session_id)
        
        # Get previous score for this subtopic
        prev_scores = [r.semantic_score for r in state.history 
                      if r.topic == topic and r.subtopic == state.current_subtopic]
        prev_score = prev_scores[-1] if prev_scores else 0.5
        
        # 🔥 USE STRICT DIFFICULTY MATRIX
        next_difficulty = self._calculate_next_difficulty(
            question_number=current_question_num,
            previous_score=prev_score,
            previous_difficulty=state.current_difficulty
        )
        
        print(f"\n   ✅ NEXT DIFFICULTY: {next_difficulty.upper()} (score: {prev_score:.2f})")
        
        # 🔥 STRESS TEST INJECTION 🔥
        # If stress mode is active, tell the AI to act skeptical/challenging for the followup
        user_context_override = ""
        if stress_test:
            print("\n   ⚠️ STRESS TEST MODE ACTIVE: Injecting adversarial prompt")
            user_context_override = "STRESS TEST MODE: Act slightly skeptical. Challenge the user's previous answer and ask them to explicitly defend their technical reasoning or point out a potential flaw in what they just said."

        question, subtopic, sampled_concepts = self._generate_question(
            session_id=session_id,
            topic=topic,
            difficulty=next_difficulty,
            user_name=state.user_name,
            force_subtopic=state.current_subtopic,
            weak_concepts=weak_concepts,
            user_context=user_context_override  # Pass adversarial context down
        )
        
        asked_questions = [r.question for r in state.history]
        
        state.current_question = question
        state.current_difficulty = next_difficulty
        state.current_sampled_concepts = sampled_concepts
        state.question_start_time = time.time()
        
        return {
            "action": "FOLLOW_UP",
            "question": question,
            "topic": topic,
            "subtopic": state.current_subtopic,
            "difficulty": next_difficulty,
            "time_remaining": state.time_remaining_sec(),
            "learning_velocity": round(mastery.mastery_velocity, 3) if mastery else 0
        }
    

    
    def _move_to_next_topic(self, state, session_id) -> dict:
        print("\n" + "="*80)
        print("➡️ MOVING TO NEXT TOPIC")
        print("="*80)
        
        # Mark current subtopic as covered before moving
        if state.current_topic and state.current_subtopic:
            if state.current_topic not in state.cycle_covered_subtopics:
                state.cycle_covered_subtopics[state.current_topic] = []
            if state.current_subtopic not in state.cycle_covered_subtopics[state.current_topic]:
                state.cycle_covered_subtopics[state.current_topic].append(state.current_subtopic)
                print(f"✅ Marked {state.current_subtopic} in {state.current_topic} as covered for this cycle")

        if not state.advance_to_next_topic():
            print("\n🔄 Completed full cycle - starting new cycle")
            state.current_topic_index = 0
            state.current_topic = state.topic_order[0]
            state.followup_count = 0
            state.cycle_covered_subtopics = {} # CLEAR COVERED FLAGS
            
            for topic in list(state.topic_sessions.keys()):
                state.topic_sessions[topic] = TopicSessionState(topic=topic)
        
        new_topic = state.current_topic
        print(f"\n➡️ New topic: {new_topic}")
        
        mastery = state.ensure_topic_mastery(new_topic)
        weak_concepts = list(mastery.weak_concepts) if mastery else []
        
        if state.user_id not in self.subtopic_trackers:
            self.subtopic_trackers[state.user_id] = SubtopicTracker(state.user_id)
        
        tracker = self.subtopic_trackers[state.user_id]
        covered = state.cycle_covered_subtopics.get(new_topic, [])
        chosen_subtopic = tracker.get_next_subtopic(new_topic, weak_concepts=weak_concepts, covered_subtopics=covered)
        
        difficulty = "medium"
        state.current_difficulty = difficulty
        state.current_subtopic = chosen_subtopic
        
        question, subtopic, sampled_concepts = self._generate_question(
            session_id=session_id,
            topic=new_topic,
            difficulty=difficulty,
            user_name=state.user_name,
            force_subtopic=chosen_subtopic,
            weak_concepts=weak_concepts
        )
        
        state.current_question = question
        state.current_subtopic = subtopic
        state.current_sampled_concepts = sampled_concepts
        state.question_start_time = time.time()
        state.topics_covered_this_session.append(new_topic)
        state.get_topic_session(new_topic)
        
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
            "learning_velocity": round(mastery.mastery_velocity, 3) if mastery else 0
        }
    
    def _simplify_question(self, state, session_id) -> dict:
        state.followup_count += 1
        
        mastery = state.ensure_topic_mastery(state.current_topic)
        weak_concepts = list(mastery.weak_concepts)[:3] if mastery.weak_concepts else []
        
        question, subtopic, sampled_concepts = self._generate_question(
            session_id=session_id,
            topic=state.current_topic,
            difficulty="easy",
            user_name=state.user_name,
            force_subtopic=state.current_subtopic,
            weak_concepts=weak_concepts
        )
        
        state.current_question = question
        state.current_difficulty = "easy"
        state.current_sampled_concepts = sampled_concepts
        state.question_start_time = time.time()
        
        return {
            "action": "SIMPLIFY",
            "question": question,
            "topic": state.current_topic,
            "subtopic": state.current_subtopic,
            "difficulty": "easy",
            "time_remaining": state.time_remaining_sec(),
            "message": "Let's try a simpler question.",
            "learning_velocity": round(mastery.mastery_velocity, 3) if mastery else 0
        }
    
    def _deepen_question(self, state, session_id) -> dict:
        state.followup_count += 1
        
        mastery = state.ensure_topic_mastery(state.current_topic)
        weak_concepts = list(mastery.weak_concepts) if mastery else []
        
        question, subtopic, sampled_concepts = self._generate_question(
            session_id=session_id,
            topic=state.current_topic,
            difficulty="hard",
            user_name=state.user_name,
            force_subtopic=state.current_subtopic,
            weak_concepts=weak_concepts
        )
        
        state.current_question = question
        state.current_difficulty = "hard"
        state.current_sampled_concepts = sampled_concepts
        state.question_start_time = time.time()
        
        return {
            "action": "DEEPEN",
            "question": question,
            "topic": state.current_topic,
            "subtopic": state.current_subtopic,
            "difficulty": "hard",
            "time_remaining": state.time_remaining_sec(),
            "message": "Great! Here's a deeper question.",
            "learning_velocity": round(mastery.mastery_velocity, 3) if mastery else 0
        }
    
    def _finalize_session(self, state, session_id):
        db_session = AdaptiveInterviewSession.query.filter_by(session_id=session_id).first()
        if db_session:
            db_session.end_time = datetime.utcnow()
            db_session.duration = int(time.time() - state.start_time)
            db_session.questions_asked = len(state.history)
            db.session.commit()
        
        semantic_dedup.clear_session(session_id)
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        print(f"\n✅ Session {session_id} finalized with {len(state.history)} questions")
    
    def _generate_feedback(self, state) -> dict:
        # Calculate average learning velocity
        velocities = [m.mastery_velocity for m in state.topic_mastery.values() if hasattr(m, 'mastery_velocity')]
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        
        return {
            "questions_answered": len(state.history),
            "duration_minutes": int((time.time() - state.start_time) / 60),
            "topics_covered": list(state.topic_sessions.keys()),
            "weakest_topics": state.get_sorted_topics_by_priority()[:2],
            "next_session_focus": state.get_sorted_topics_by_priority()[0] if state.get_sorted_topics_by_priority() else "DBMS",
            "average_learning_velocity": round(avg_velocity, 3)
        }
    
    def _save_to_db(self, user_id: int, session_id: str, record: AdaptiveQARecord, mastery, expected_answer: str = "", sampled_concepts: list = None):
        try:
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
            
            db_mastery.update_mastery(
                semantic_score=record.semantic_score,
                keyword_score=record.keyword_score,
                response_time=record.response_time,
                missing=record.missing_concepts
            )
            
            # Save learning velocity
            db_mastery.mastery_velocity = mastery.mastery_velocity
            db_mastery.last_mastery = mastery.last_mastery
            
            # ============================================
            # CONCEPT MASTERY PERSISTENCE
            # ============================================
            # Save ALL concept data
            concept_dict = {}
            for name, concept in mastery.concepts.items():
                concept_dict[name] = concept.to_dict()
            
            db_mastery.set_concept_masteries(concept_dict)
            
            # Also update legacy fields for backward compatibility
            db_mastery.weak_concepts = json.dumps(list(mastery.weak_concepts))
            db_mastery.strong_concepts = json.dumps(list(mastery.strong_concepts))
            
            # Calculate stagnation dict
            stagnation_dict = {}
            for name, concept in mastery.concepts.items():
                if concept.stagnation_count > 0:
                    stagnation_dict[name] = concept.stagnation_count
            db_mastery.concept_stagnation = json.dumps(stagnation_dict)
            
            print(f"\n💾 SAVED CONCEPT DATA:")
            print(f"   Concepts tracked: {len(mastery.concepts)}")
            for name, concept in list(mastery.concepts.items())[:3]:
                print(f"   • {name}: attempts={concept.attempts}, "
                      f"mastery={concept.mastery_level:.2f}")
            
            history = QuestionHistory(
                user_id=user_id,
                session_id=session_id,
                topic=record.topic,
                subtopic=record.subtopic,
                question=record.question,
                answer=record.answer,
                expected_answer=expected_answer,
                semantic_score=record.semantic_score,
                keyword_score=record.keyword_score,
                difficulty=record.difficulty,
                response_time=record.response_time
            )
            history.set_sampled_concepts(record.sampled_concepts)
            history.set_missing_concepts(record.missing_concepts)
            db.session.add(history)
            
            db.session.commit()
            
            print(f"\n💾 Saved to DB:")
            print(f"   Topic: {record.topic}")
            print(f"   Concepts tracked: {len(mastery.concepts)}")
            print(f"   Learning velocity: {mastery.mastery_velocity:+.3f}")
            print(f"   Weak: {list(mastery.weak_concepts)[:3]}")
            print(f"   Strong: {list(mastery.strong_concepts)[:3]}")
            
        except Exception as e:
            print(f"❌ Error saving to DB: {e}")
            db.session.rollback()
    
    def get_session(self, session_id: str) -> dict:
        state = self.sessions.get(session_id)
        if state:
            return state.to_dict()
        return {"error": "Session not found"}
    
    def reset_user_mastery(self, user_id: int, topic: str = None) -> dict:
        try:
            if user_id in self.subtopic_trackers:
                if topic:
                    self.subtopic_trackers[user_id].reset_topic_mastery(topic)
                else:
                    self.subtopic_trackers[user_id] = SubtopicTracker(user_id)
            
            sessions_to_remove = []
            for session_id, state in self.sessions.items():
                if state.user_id == user_id:
                    if topic and state.current_topic == topic:
                        sessions_to_remove.append(session_id)
                    elif not topic:
                        sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                semantic_dedup.clear_session(session_id)
                del self.sessions[session_id]
            
            return {"success": True, "message": f"Reset {'all' if not topic else topic} mastery"}
        except Exception as e:
            print(f"Error resetting mastery: {e}")
            return {"error": str(e)}
    
    def get_subtopic_stats(self, user_id: int) -> dict:
        if user_id in self.subtopic_trackers:
            return self.subtopic_trackers[user_id].get_statistics()
        else:
            tracker = SubtopicTracker(user_id)
            self.subtopic_trackers[user_id] = tracker
            return tracker.get_statistics()
    
    def get_topic_subtopics(self, user_id: int, topic: str) -> list:
        from .subtopic_tracker import SubtopicTracker
        
        tracker = SubtopicTracker(user_id)
        all_subtopics = tracker.SUBTOPICS_BY_TOPIC.get(topic, [])
        attempted = tracker.get_all_attempted_subtopics(topic)
        
        result = []
        for subtopic in all_subtopics:
            data = {
                'name': subtopic,
                'status': 'not_started',
                'mastery': 0,
                'attempts': 0
            }
            if subtopic in attempted:
                data['status'] = attempted[subtopic].get('subtopic_status', 'ongoing')
                data['mastery'] = round(attempted[subtopic].get('mastery_level', 0), 3)
                data['attempts'] = attempted[subtopic].get('attempts', 0)
            result.append(data)
        
        return result