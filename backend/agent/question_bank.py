# backend/agent/question_bank.py

from rag import generate_rag_response


def generate_first_question(topic: str) -> str:
    prompt = f"""
You are an expert technical interviewer.

Generate ONE interview question for topic: {topic}

Rules:
- Medium difficulty
- One question only
- No explanation
"""
    return generate_rag_response("first question", prompt).strip()


def generate_gap_followup(topic: str, missing_concepts: list) -> str:
    gaps = ", ".join(missing_concepts) if missing_concepts else "core understanding"

    prompt = f"""
You are an interviewer for a FINAL YEAR undergraduate student.
Ask a CLEAR, FUNDAMENTAL question.
Avoid advanced edge cases.
Focus on definitions, intuition, and basic examples.
Difficulty level: EASY–MEDIUM.

Topic: {topic}
Weak areas: {gaps}

Generate ONE focused follow-up interview question.
"""
    return generate_rag_response("gap followup", prompt).strip()
