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
        """
        
        # Hard stops
        if state.is_time_over():
            return self.FINALIZE
        
        if state.total_questions_asked() >= state.max_questions_total:
            return self.FINALIZE
        
        # Get current topic mastery
        mastery = state.get_topic_mastery(state.current_topic)
        
        # Extract signals
        coverage = analysis.get("coverage_score", 0)
        depth = analysis.get("depth", "shallow")
        confidence = analysis.get("confidence", "medium")
        has_example = analysis.get("has_example", False)
        missing = analysis.get("missing_concepts", [])
        
        # Get word count from analysis
        response_length = analysis.get("response_length", 0)
        
        # 📊 IMPROVED DECISION TREE
        
        # 1️⃣ Very poor answer - SIMPLIFY
        if coverage < 0.3:
            return self.SIMPLIFY
        
        # 2️⃣ Poor answer but with some correct concepts - FOLLOW_UP on missing
        if coverage < 0.5:
            if missing and len(missing) > 0:
                return self.FOLLOW_UP
            return self.SIMPLIFY
        
        # 3️⃣ Missing key concepts - FOLLOW_UP on specific gaps
        if missing and len(missing) > 0:
            return self.FOLLOW_UP
        
        # 4️⃣ Good answer but shallow - ask for deeper explanation
        if coverage > 0.6 and depth == "shallow" and response_length < 50:
            return self.FOLLOW_UP
        
        # 5️⃣ Excellent answer with depth and examples - DEEPEN
        if coverage > 0.7 and depth == "deep" and has_example:
            # Only deepen if they're doing well consistently
            if mastery and mastery.consecutive_good >= 1:
                return self.DEEPEN
        
        # 6️⃣ Good answer - move to next topic if we've asked enough
        if state.followup_count >= 2:
            return self.MOVE_TOPIC
        
        # 7️⃣ Default - ask one more follow-up before moving on
        if state.followup_count == 0:
            return self.FOLLOW_UP
        
        # 8️⃣ Finally, move to new topic
        return self.MOVE_TOPIC
    
    def get_next_topic_priority(self, state: AdaptiveInterviewState) -> list:
        """Get topics in priority order (weakest first, considering coverage)"""
        topics_with_scores = []
        
        # Get topics already covered in this session
        covered_in_session = {q.topic for q in state.history if q.topic}
        
        for topic in ["DBMS", "OS", "OOPS"]:
            mastery = state.get_topic_mastery(topic)
            if mastery:
                score = mastery.mastery_level
                questions_asked = mastery.questions_attempted
            else:
                score = 0.5
                questions_asked = 0
            
            # Penalize topics with too many questions already
            if questions_asked > 5:
                score += 0.3  # They've had enough practice
            
            # Boost topics with fewer questions
            if questions_asked < 2:
                score -= 0.2  # Need more practice
            
            topics_with_scores.append((topic, score))
        
        # Sort by score (lowest first = weakest)
        topics_with_scores.sort(key=lambda x: x[1])
        
        return [t[0] for t in topics_with_scores]