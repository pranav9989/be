# backend/agent/analyzer.py

def analyze_answer(question: str, answer: str) -> dict:
    """
    PURE FUNCTION.
    Extracts signals only.
    """

    if not answer or len(answer.strip()) < 5:
        return {
            "coverage_score": 0.0,
            "depth": "shallow",
            "missing_concepts": [],
            "covered_concepts": [],
            "confidence": "low"
        }

    # Placeholder heuristic (your RAG logic plugs here)
    return {
        "coverage_score": 0.6,
        "depth": "medium",
        "missing_concepts": [],
        "covered_concepts": [],
        "confidence": "medium"
    }
