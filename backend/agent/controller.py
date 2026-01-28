# backend/agent/controller.py

from .state_manager import InterviewStateManager
from .decision_engine import decide_next_action
from .topic_selector import choose_topic
from .question_bank import (
    generate_first_question,
    generate_gap_followup
)


class InterviewAgentController:

    @classmethod
    def start_session(cls, session_id: str, user_id: int):
        state = InterviewStateManager.create_session(session_id, user_id)

        state.current_question = "Tell me briefly about yourself."
        state.current_topic = None
        state.status = "ACTIVE"

        return state.current_question

    @classmethod
    def handle_answer(cls, session_id: str, answer: str, analysis: dict):
        state = InterviewStateManager.get_state(session_id)

        # 1️⃣ Record answer
        InterviewStateManager.record_answer(
            session_id=session_id,
            question=state.current_question,
            topic=state.current_topic,
            difficulty=state.difficulty,
            answer=answer,
            analysis=analysis
        )

        # 2️⃣ Decide next action
        action = decide_next_action(state, analysis)

        # 3️⃣ Execute action (RAG = content only)

        if action == "FINALIZE":
            state.status = "COMPLETED"
            return {
                "action": "FINALIZE",
                "next_question": None,
                "time_remaining": 0
            }

        if action == "MOVE_TOPIC":
            asked_topics = {q.topic for q in state.history if q.topic}
            new_topic = choose_topic(asked_topics)

            state.reset_for_new_topic(new_topic)

            question = generate_first_question(new_topic)
            state.current_topic = new_topic
            state.current_question = question

            return {
                "action": "MOVE_TOPIC",
                "next_question": question,
                "time_remaining": state.time_remaining_sec()
            }

        if action in ("FOLLOW_UP", "TARGET_GAP"):
            state.followup_count += 1

            question = generate_gap_followup(
                topic=state.current_topic,
                missing_concepts=analysis.get("missing_concepts", [])
            )

            state.current_question = question

            return {
                "action": action,
                "next_question": question,
                "time_remaining": state.time_remaining_sec()
            }
