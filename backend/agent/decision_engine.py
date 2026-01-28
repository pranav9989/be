# backend/agent/decision_engine.py

from .state import InterviewAgentState

FOLLOW_UP = "FOLLOW_UP"
TARGET_GAP = "TARGET_GAP"
MOVE_TOPIC = "MOVE_TOPIC"
FINALIZE = "FINALIZE"


def decide_next_action(
    state: InterviewAgentState,
    analysis: dict
) -> str:
    """
    Deterministic agent brain.
    NO LLM. NO randomness.
    """

    # ⏱️ Hard stop on time
    if state.is_time_over():
        return FINALIZE

    # Hard stop on question count
    if state.total_questions_asked() >= state.max_questions_total:
        return FINALIZE

    # Too many follow-ups
    if state.followup_count >= state.max_followups_per_topic:
        return MOVE_TOPIC

    coverage = analysis.get("coverage_score", 0)
    depth = analysis.get("depth", "shallow")

    if coverage < 0.5:
        return TARGET_GAP

    if depth == "shallow":
        return FOLLOW_UP

    return MOVE_TOPIC
