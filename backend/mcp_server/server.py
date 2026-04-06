from mcp.server.fastmcp import FastMCP

# ===== IMPORT YOUR EXISTING MODULES =====
from agent.adaptive_analyzer import AdaptiveAnalyzer
from agent.adaptive_question_bank import AdaptiveQuestionBank
from agent.adaptive_decision import AdaptiveDecisionEngine

from rag import agentic_expected_answer
from resume_processor import search_resume_faiss

# ===== INIT MCP SERVER =====
mcp = FastMCP("Interview MCP Server")

# ===== INIT SHARED OBJECTS =====
question_bank = AdaptiveQuestionBank()
decision_engine = AdaptiveDecisionEngine()


# =========================================================
# 🧠 TOOL 1: ANALYZE ANSWER (CORE INTELLIGENCE)
# =========================================================
@mcp.tool()
def analyze_answer(answer: str, concepts: list):
    """
    Analyze user answer for concept coverage + semantic understanding
    """

    mentioned, missing = AdaptiveAnalyzer.detect_concepts_semantically(
        answer, concepts
    )

    return {
        "mentioned_concepts": mentioned,
        "missing_concepts": missing,
        "coverage_score": len(mentioned) / max(1, len(concepts))
    }


# =========================================================
# 🧠 TOOL 2: GENERATE QUESTION
# =========================================================
@mcp.tool()
def generate_question(topic: str, difficulty: str):
    """
    Generate next interview question
    """

    try:
        question_data = question_bank.generate(topic, difficulty)

        return {
            "question": question_data.get("question"),
            "concepts": question_data.get("concepts", []),
            "difficulty": difficulty,
            "topic": topic
        }

    except Exception as e:
        return {
            "error": str(e),
            "question": f"Explain {topic} basics."
        }


# =========================================================
# 🧠 TOOL 3: EXPECTED ANSWER (RAG)
# =========================================================
@mcp.tool()
def get_expected_answer(question: str, concepts: list):
    """
    Generate expected answer using RAG
    """

    try:
        answer, chunks = agentic_expected_answer(question, concepts)

        return {
            "expected_answer": answer,
            "supporting_chunks": chunks[:3] if chunks else []
        }

    except Exception as e:
        return {
            "expected_answer": "",
            "error": str(e)
        }


# =========================================================
# 🧠 TOOL 4: RESUME CONTEXT
# =========================================================
@mcp.tool()
def get_resume_context(user_id: int, query: str):
    """
    Fetch relevant resume context
    """

    try:
        results = search_resume_faiss(user_id, query)

        return {
            "context": results[:5] if results else []
        }

    except Exception as e:
        return {
            "context": [],
            "error": str(e)
        }


# =========================================================
# 🧠 TOOL 5: DECISION ENGINE (HYBRID CONTROL 🔥)
# =========================================================
@mcp.tool()
def decide_next_action(state: dict, analysis: dict):
    """
    Use your AdaptiveDecisionEngine as safety layer
    """

    try:
        decision = decision_engine.decide(state, analysis)

        return {
            "action": decision
        }

    except Exception as e:
        return {
            "action": "FOLLOW_UP",
            "error": str(e)
        }


# =========================================================
# 🧠 TOOL 6: SEMANTIC SCORE (OPTIONAL BUT STRONG FOR PAPER)
# =========================================================
@mcp.tool()
def evaluate_answer_quality(answer: str, expected_answer: str):
    """
    Compute semantic + keyword score
    """

    from interview_analyzer import (
        calculate_semantic_similarity,
        calculate_keyword_coverage
    )

    try:
        semantic = calculate_semantic_similarity(answer, expected_answer)
        keyword = calculate_keyword_coverage(answer, expected_answer)

        return {
            "semantic_score": float(semantic),
            "keyword_score": float(keyword),
            "final_score": float((semantic + keyword) / 2)
        }

    except Exception as e:
        return {
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "final_score": 0.0,
            "error": str(e)
        }


# =========================================================
# 🚀 RUN SERVER
# =========================================================
if __name__ == "__main__":
    print("🚀 Starting MCP Interview Server...")
    mcp.run()