# backend/agent/adaptive_decision.py

from .adaptive_state import AdaptiveInterviewState

class AdaptiveDecisionEngine:
    """Intelligent decision making based on user performance"""
    
    # Action types
    FOLLOW_UP = "FOLLOW_UP"
    DEEPEN = "DEEPEN"
    SIMPLIFY = "SIMPLIFY"
    MOVE_TOPIC = "MOVE_TOPIC"
    FINALIZE = "FINALIZE"
    
    def decide(self, state: AdaptiveInterviewState, analysis: dict) -> str:
        """
        Make intelligent decision based on comprehensive signals
        EXACTLY 3 questions per topic, max 2 follow-ups within those 3
        """
        
        # Hard stops (only these cause FINALIZE)
        if state.is_time_over():
            print("⏰ Time limit reached (30 minutes) - finalizing")
            return self.FINALIZE
        
        if state.total_questions_asked() >= state.max_questions_total:
            print("📊 Question limit reached (15 questions) - finalizing")
            return self.FINALIZE
        
        # Get current topic and its session
        topic = state.current_topic
        topic_session = state.get_topic_session(topic)
        mastery = state.get_topic_mastery(topic)
        
        # Extract signals
        semantic_score = analysis.get("semantic_similarity", 0)
        coverage = analysis.get("coverage_score", 0)
        missing = analysis.get("missing_concepts", [])
        response_length = analysis.get("response_length", 0)
        depth = analysis.get("depth", "shallow")
        has_example = analysis.get("has_example", False)
        
        # Track follow-ups for THIS topic
        followups_on_this_topic = state.followup_count
        MAX_FOLLOWUPS_PER_TOPIC = 2
        
        # Log decision factors
        print(f"\n📊 DECISION FOR {topic}:")
        print(f"   - Questions asked on this topic: {topic_session.questions_asked}/3")
        print(f"   - Follow-ups on this topic: {followups_on_this_topic}/{MAX_FOLLOWUPS_PER_TOPIC}")
        print(f"   - Current answer quality: {semantic_score:.2f}")
        print(f"   - Missing concepts: {missing[:3] if missing else 'None'}")
        
        # Check if we've hit the 3-question limit for this topic
        if topic_session.questions_asked >= 3:
            print(f"   → DECISION: MOVE_TOPIC (completed 3 questions for {topic})")
            return self.MOVE_TOPIC
        
        # 1️⃣ VERY POOR ANSWER - Try different subtopic first, then simplify
        if semantic_score < 0.3:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                print(f"   → DECISION: FOLLOW_UP (try different subtopic, {followups_on_this_topic + 1}/{MAX_FOLLOWUPS_PER_TOPIC})")
                return self.FOLLOW_UP
            else:
                print(f"   → DECISION: MOVE_TOPIC (max follow-ups reached, still struggling)")
                return self.MOVE_TOPIC
        
        # 2️⃣ POOR ANSWER with missing concepts - Try different subtopic
        if semantic_score < 0.5 and missing:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                print(f"   → DECISION: FOLLOW_UP (try different subtopic for missing concepts, {followups_on_this_topic + 1}/{MAX_FOLLOWUPS_PER_TOPIC})")
                return self.FOLLOW_UP
            else:
                print(f"   → DECISION: MOVE_TOPIC (max follow-ups reached)")
                return self.MOVE_TOPIC
        
        # 3️⃣ GOOD ANSWER - Try different subtopic
        if semantic_score > 0.6:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                print(f"   → DECISION: FOLLOW_UP (good answer, try different subtopic, {followups_on_this_topic + 1}/{MAX_FOLLOWUPS_PER_TOPIC})")
                return self.FOLLOW_UP
            else:
                # If we've used both follow-ups and still have questions left, move to next topic
                if topic_session.questions_asked < 3:
                    print(f"   → DECISION: FOLLOW_UP (need more questions, but max follow-ups used)")
                    return self.FOLLOW_UP
                else:
                    return self.MOVE_TOPIC
        
        # 4️⃣ DEFAULT: If we still need more questions on this topic
        if topic_session.questions_asked < 3:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                print(f"   → DECISION: FOLLOW_UP (need {3 - topic_session.questions_asked} more questions)")
                return self.FOLLOW_UP
            else:
                # We need another question but have used follow-ups, so it must be a new main question
                print(f"   → DECISION: MOVE_TOPIC (max follow-ups, need new question)")
                return self.MOVE_TOPIC
        
        # 5️⃣ Finally, move to next topic
        print(f"   → DECISION: MOVE_TOPIC (default)")
        return self.MOVE_TOPIC