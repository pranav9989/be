# backend/agent/state_manager.py

from .state import InterviewAgentState, QARecord


class InterviewStateManager:
    sessions: dict = {}

    @classmethod
    def create_session(cls, session_id: str, user_id: int):
        state = InterviewAgentState(
            session_id=session_id,
            user_id=user_id,
            status="INTRO"
        )
        cls.sessions[session_id] = state
        return state

    @classmethod
    def get_state(cls, session_id: str) -> InterviewAgentState:
        return cls.sessions[session_id]

    @classmethod
    def record_answer(
        cls,
        session_id: str,
        question: str,
        topic: str,
        difficulty: str,
        answer: str,
        analysis: dict
    ):
        state = cls.get_state(session_id)

        state.history.append(
            QARecord(
                question=question,
                topic=topic,
                difficulty=difficulty,
                answer=answer,
                analysis=analysis
            )
        )

        if topic:
            if analysis.get("coverage_score", 0) >= 0.7:
                state.strengths.append(topic)
            else:
                state.weaknesses.append(topic)
