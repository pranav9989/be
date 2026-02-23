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
        
        print("\n" + "█"*80)
        print("🎯 NEXT ACTION DECISION")
        print("█"*80)
        
        # Hard stops (only these cause FINALIZE)
        if state.is_time_over():
            print(f"\n⏰ TIME LIMIT REACHED (30 minutes)")
            print(f"\n   → DECISION: FINALIZE")
            print("█"*80)
            return self.FINALIZE
        
        if state.total_questions_asked() >= state.max_questions_total:
            print(f"\n📊 QUESTION LIMIT REACHED (15 questions)")
            print(f"\n   → DECISION: FINALIZE")
            print("█"*80)
            return self.FINALIZE
        
        # Get current topic and its session
        topic = state.current_topic
        topic_session = state.get_topic_session(topic)
        
        # Extract signals
        semantic_score = analysis.get("semantic_similarity", 0)
        missing = analysis.get("missing_concepts", [])
        
        # Track follow-ups for THIS topic
        followups_on_this_topic = state.followup_count
        MAX_FOLLOWUPS_PER_TOPIC = 2
        
        # Print decision context
        print(f"\n   Topic:           {topic}")
        print(f"   Subtopic:        {state.current_subtopic}")
        print(f"   Questions asked: {topic_session.questions_asked}/3")
        print(f"   Follow-ups used: {followups_on_this_topic}/{MAX_FOLLOWUPS_PER_TOPIC}")
        print(f"   Answer quality:  {semantic_score:.2f}")
        print(f"   Missing concepts:{' ' + str(missing[:3]) if missing else ' None'}")
        
        print(f"\n   📊 DECISION LOGIC:")
        
        # Check if we've hit the 3-question limit for this topic
        if topic_session.questions_asked >= 3:
            print(f"      • Topic has reached 3 questions")
            print(f"\n   ✅ DECISION: MOVE_TOPIC")
            print("█"*80)
            return self.MOVE_TOPIC
        
        # Calculate questions remaining on this topic
        questions_remaining = 3 - topic_session.questions_asked
        followups_remaining = MAX_FOLLOWUPS_PER_TOPIC - followups_on_this_topic
        
        print(f"      • Questions remaining: {questions_remaining}")
        print(f"      • Follow-ups remaining: {followups_remaining}")
        
        # Decision tree
        decision = None
        reason = ""
        
        # 1️⃣ VERY POOR ANSWER
        if semantic_score < 0.3:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Very poor answer ({semantic_score:.2f} < 0.3) → try different subtopic"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Very poor answer but max follow-ups ({MAX_FOLLOWUPS_PER_TOPIC}) reached → move on"
        
        # 2️⃣ POOR ANSWER with missing concepts
        elif semantic_score < 0.5 and missing:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Poor answer ({semantic_score:.2f}) with missing concepts → target gaps"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Poor answer but max follow-ups ({MAX_FOLLOWUPS_PER_TOPIC}) reached → move on"
        
        # 3️⃣ GOOD ANSWER
        elif semantic_score > 0.6:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Good answer ({semantic_score:.2f} > 0.6) → explore deeper"
            else:
                # If we've used both follow-ups but still need questions, do follow-up anyway
                if topic_session.questions_asked < 3:
                    decision = self.FOLLOW_UP
                    reason = f"Good answer but need {questions_remaining} more question(s) → follow-up anyway"
                else:
                    decision = self.MOVE_TOPIC
                    reason = f"Good answer, all questions completed → move on"
        
        # 4️⃣ DEFAULT - Need more questions
        elif topic_session.questions_asked < 3:
            if followups_on_this_topic < MAX_FOLLOWUPS_PER_TOPIC:
                decision = self.FOLLOW_UP
                reason = f"Need {questions_remaining} more question(s) on this subtopic"
            else:
                decision = self.MOVE_TOPIC
                reason = f"Need more questions but max follow-ups reached → new subtopic needed"
        
        # 5️⃣ FINAL FALLBACK
        else:
            decision = self.MOVE_TOPIC
            reason = "Default fallback"
        
        print(f"      • {reason}")
        
        # Print decision with appropriate emoji
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