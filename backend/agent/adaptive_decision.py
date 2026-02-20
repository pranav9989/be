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
        NO GOAL-BASED FINALIZATION - only time or user action ends session
        """
        
        # 🔥 Hard stops (only these cause FINALIZE)
        if state.is_time_over():
            print("⏰ Time limit reached (30 minutes) - finalizing")
            return self.FINALIZE
        
        if state.total_questions_asked() >= state.max_questions_total:
            print("📊 Question limit reached (15 questions) - finalizing")
            return self.FINALIZE
        
        # Get current topic mastery and session state
        mastery = state.get_topic_mastery(state.current_topic)
        topic_session = state.get_topic_session(state.current_topic)
        
        # Extract signals
        coverage = analysis.get("coverage_score", 0)
        depth = analysis.get("depth", "shallow")
        missing = analysis.get("missing_concepts", [])
        response_length = analysis.get("response_length", 0)
        has_example = analysis.get("has_example", False)
        
        # 🔥 NEW: Check if we've covered this topic enough
        if topic_session.should_move_on(
            min_questions=state.coverage_min_questions,
            threshold=state.coverage_threshold
        ):
            print(f"✅ Topic {state.current_topic} sufficiently covered - moving on")
            return self.MOVE_TOPIC
        
        # 📊 IMPROVED DECISION TREE
        
        # 1️⃣ Very poor answer - SIMPLIFY
        if coverage < 0.3:
            print("📉 Very poor answer - simplifying")
            return self.SIMPLIFY
        
        # 2️⃣ Poor answer but with some correct concepts - FOLLOW_UP on missing
        if coverage < 0.5:
            if missing and len(missing) > 0:
                print("🔍 Poor answer - following up on missing concepts")
                return self.FOLLOW_UP
            print("📉 Poor answer with no clear gaps - simplifying")
            return self.SIMPLIFY
        
        # 3️⃣ Missing key concepts - FOLLOW_UP on specific gaps
        if missing and len(missing) > 0:
            print("🔍 Missing concepts detected - following up")
            return self.FOLLOW_UP
        
        # 4️⃣ Good answer but shallow - ask for deeper explanation
        if coverage > 0.6 and depth == "shallow" and response_length < 50:
            print("📝 Good but shallow - asking for depth")
            return self.FOLLOW_UP
        
        # 5️⃣ Excellent answer with depth and examples - DEEPEN
        if coverage > 0.7 and depth == "deep" and has_example:
            if mastery and mastery.consecutive_good >= 1:
                print("🚀 Excellent answer - deepening")
                return self.DEEPEN
        
        # 6️⃣ Default - ask follow-up if we haven't reached min questions
        if topic_session.questions_asked < state.coverage_min_questions:
            print("🔄 Still need more questions on this topic")
            return self.FOLLOW_UP
        
        # 7️⃣ Finally, move to new topic
        print("➡️ Moving to new topic")
        return self.MOVE_TOPIC