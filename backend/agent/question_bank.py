# backend/agent/question_bank.py

from rag import generate_rag_response

def clean_question_text(raw_text):
    """
    Extract only the question part, removing any "Question:" labels or answers.
    """
    # Remove "Question:" prefix if present
    text = raw_text.replace("Question:", "").strip()
    
    # If there's an "Answer:" section, only keep text before it
    if "Answer:" in text:
        text = text.split("Answer:")[0].strip()
    
    # Remove markdown formatting
    text = text.replace("**", "").replace("##", "").replace("```", "")
    
    return text.strip()

def generate_first_question(topic: str) -> str:
    prompt = f"""
You are an expert technical interviewer.

Generate ONE interview question for topic: {topic}

IMPORTANT RULES:
- Return ONLY the question text
- DO NOT include "Question:" or any labels
- DO NOT include the answer
- One question only, no explanations
- Keep it concise (under 200 characters if possible)

Example format:
What is the difference between abstraction and encapsulation?
"""
    raw_question = generate_rag_response("first question", prompt).strip()
    return clean_question_text(raw_question)


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

IMPORTANT RULES:
- Return ONLY the question text
- DO NOT include "Question:" or any labels
- DO NOT include the answer
- One question only
- Keep it concise

Example:
Explain how indexing works in databases.
"""
    raw_question = generate_rag_response("gap followup", prompt).strip()
    return clean_question_text(raw_question)