# backend/agent/adaptive_decision.py

from .adaptive_state import AdaptiveInterviewState


class AdaptiveDecisionEngine:
    """Intelligent decision making based on user performance"""
    
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
        
        print("\n" + "█"*80)
        print("🎯 NEXT ACTION DECISION")
        print("█"*80)
        
        # Safety check - if analysis is empty, provide defaults
        if not analysis:
            analysis = {
                "semantic_similarity": 0.0,
                "missing_concepts": [],
                "mastery_velocity": 0.0,
                "stagnation": {}
            }
        
        # Check time limit
        if state.is_time_over():
            print(f"\n⏰ TIME LIMIT REACHED (30 minutes)")
            print(f"\n   → DECISION: FINALIZE")
            print("█"*80)
            return self.FINALIZE
        
        # Check question limit
        if state.total_questions_asked() >= state.max_questions_total:
            print(f"\n📊 QUESTION LIMIT REACHED ({state.max_questions_total} questions)")
            print(f"\n   → DECISION: FINALIZE")
            print("█"*80)
            return self.FINALIZE
        
        topic = state.current_topic
        topic_session = state.get_topic_session(topic)
        
        # Get analysis data with safe defaults
        semantic_score = analysis.get("semantic_similarity", 0.0)
        missing = analysis.get("missing_concepts", [])
        velocity = analysis.get("mastery_velocity", 0.0)
        stagnation = analysis.get("stagnation", {})
        
        followups_on_this_topic = state.followup_count
        MAX_FOLLOWUPS_PER_TOPIC = 2
        
        print(f"\n   Topic:           {topic}")
        print(f"   Subtopic:        {state.current_subtopic}")
        print(f"   Questions asked: {topic_session.questions_asked}/3")
        print(f"   Follow-ups used: {followups_on_this_topic}/{MAX_FOLLOWUPS_PER_TOPIC}")
        print(f"   Answer quality:  {semantic_score:.2f}")
        print(f"   Learning velocity: {velocity:+.3f}")
        print(f"   Missing concepts:{' ' + str(missing[:3]) if missing else ' None'}")
        if stagnation:
            print(f"   Stagnation:      {stagnation}")
        
        print(f"\n   📊 DECISION LOGIC:")
        
        # ========== ENFORCE 3-QUESTION LIMIT (PER RULES) ==========
        if topic_session.questions_asked >= 3:
            print(f"      • Topic has reached 3 questions (hard limit)")
            print(f"\n   ✅ DECISION: MOVE_TOPIC")
            print("█"*80)
            return self.MOVE_TOPIC
        
        questions_remaining = 3 - topic_session.questions_asked
        followups_remaining = MAX_FOLLOWUPS_PER_TOPIC - followups_on_this_topic
        
        print(f"      • Questions remaining: {questions_remaining}")
        print(f"      • Follow-ups remaining: {followups_remaining}")
        
        decision = None
        reason = ""
        
        # ========== DECISION TREE WITH FULL DIFFICULTY MATRIX ==========
        
        # 1️⃣ EXTREMELY POOR ANSWER (near silence) - SIMPLIFY
        if semantic_score < 0.2:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.SIMPLIFY
                reason = f"Extremely poor answer ({semantic_score:.2f} < 0.2) → simplify question"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Extremely poor but max follow-ups ({MAX_FOLLOWUPS_PER_TOPIC}) reached → move on"
        
        # 2️⃣ VERY POOR ANSWER (0.2-0.3)
        elif semantic_score < 0.3:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Very poor answer ({semantic_score:.2f} < 0.3) → ask another question on same subtopic"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Very poor answer but max follow-ups ({MAX_FOLLOWUPS_PER_TOPIC}) reached → move on"
        
        # 3️⃣ POOR ANSWER with missing concepts (0.3-0.5)
        elif semantic_score < 0.5 and missing:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Poor answer ({semantic_score:.2f}) with missing concepts → target gaps"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Poor answer but max follow-ups ({MAX_FOLLOWUPS_PER_TOPIC}) reached → move on"
        
        # 4️⃣ MEDIUM ANSWER (0.5-0.6) - need more questions
        elif semantic_score < 0.6:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Medium answer ({semantic_score:.2f}) → need more practice on this subtopic"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Medium answer but max follow-ups reached → new subtopic needed"
        
        # 5️⃣ GOOD ANSWER (0.6-0.7)
        elif semantic_score < 0.7:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                # Check learning velocity
                if velocity > 0.05:
                    decision = self.DEEPEN
                    reason = f"Good answer ({semantic_score:.2f}) with positive velocity (+{velocity:.2f}) → deepen"
                else:
                    decision = self.FOLLOW_UP
                    reason = f"Good answer ({semantic_score:.2f}) but velocity flat → explore deeper"
            else:
                if topic_session.questions_asked < 3:
                    decision = self.FOLLOW_UP
                    reason = f"Good answer but need {questions_remaining} more question(s) → follow-up anyway"
                else:
                    decision = self.MOVE_TOPIC
                    reason = f"Good answer, all questions completed → move on"
        
        # 6️⃣ EXCELLENT ANSWER (> 0.7)
        elif semantic_score >= 0.7:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.DEEPEN
                reason = f"Excellent answer ({semantic_score:.2f}) → challenge with deeper question"
            else:
                if topic_session.questions_asked < 3:
                    decision = self.FOLLOW_UP
                    reason = f"Excellent answer but need {questions_remaining} more question(s) → follow-up"
                else:
                    decision = self.MOVE_TOPIC
                    reason = f"Excellent answer, all questions completed → move on"
        
        # 7️⃣ DEFAULT FALLBACK - Need more questions
        elif topic_session.questions_asked < 3:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Need {questions_remaining} more question(s) on this subtopic (fallback)"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Need more questions but max follow-ups reached → new subtopic needed (fallback)"
        
        # 8️⃣ FINAL FALLBACK
        else:
            decision = self.MOVE_TOPIC
            reason = "Default fallback (should not reach here)"
        
        print(f"      • {reason}")
        
        print(f"\n   ✅ DECISION: ", end="")
        if decision == self.FOLLOW_UP:
            print(f"FOLLOW_UP (continue on {state.current_subtopic})")
        elif decision == self.MOVE_TOPIC:
            print(f"MOVE_TOPIC (next topic in cycle)")
        elif decision == self.DEEPEN:
            print(f"DEEPEN (more challenging question)")
        elif decision == self.SIMPLIFY:
            print(f"SIMPLIFY (easier question)")
        elif decision == self.FINALIZE:
            print(f"FINALIZE (end interview)")
        
        print("█"*80)
        
        return decision