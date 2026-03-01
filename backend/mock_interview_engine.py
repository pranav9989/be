"""
mock_interview_engine.py — AI-powered mock interview answer evaluator
Uses Mistral API
 for:
  - Targeted question generation from resume + JD
  - Per-question evaluation with detailed feedback & model answers
  - Session-level AI narrative insights
"""

import os
import json
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Removed Ollama in favor of Mistral (via rag.py)
from rag import mistral_generate as _generate


def _parse_json(text: str):
    """Strip markdown fences from text and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Find first '[' or '{' in case there's preamble
    start = min(
        (text.find(c) for c in ["[", "{"] if c in text),
        default=0
    )
    return json.loads(text[start:])


# ─── Question Generation ─────────────────────────────────────────────────────

def generate_interview_questions(resume_context: str, job_description: str,
                                  skills: list, experience: int,
                                  question_count: int = 8,
                                  variation_seed: str = "") -> list:
    """
    Generate targeted interview questions using Mistral.
    Returns list of {question, type, difficulty, focus_area, expected_keywords, hint}
    """
    variation_note = (
        f"\n(Session seed: {variation_seed} — ensure questions are unique from previous sets.)"
        if variation_seed else ""
    )

    prompt = f"""You are a senior technical interviewer at a FAANG company conducting a real interview.
Generate exactly {question_count} interview questions for this specific candidate.{variation_note}

CANDIDATE RESUME (key sections):
{resume_context[:2500]}

CANDIDATE'S SKILLS: {', '.join(skills[:25]) if skills else 'Not specified'}
EXPERIENCE: {experience} year(s)

JOB DESCRIPTION:
{job_description[:2500]}

CRITICAL RULES:
1. Questions MUST be specific to THIS candidate's resume and THIS job description
2. Reference actual projects, technologies, or experiences from the resume
3. Mix types: technical, behavioral, project-based, situational, system-design
4. Difficulty: first {max(1, question_count//3)} questions easy, next {max(1, question_count//3)} medium, last {max(1, question_count//3)} hard
5. expected_keywords must be 4-6 specific technical/conceptual terms a great answer must include
6. hint must be a concrete, specific pointer (not generic advice like "be specific")

Return ONLY a valid JSON array. No markdown. No explanation. No preamble.

[
  {{
    "question": "<specific, tailored interview question>",
    "type": "technical|behavioral|project-based|situational|system-design",
    "difficulty": "easy|medium|hard",
    "focus_area": "<short label e.g. React Hooks, System Scalability, Leadership>",
    "expected_keywords": ["kw1", "kw2", "kw3", "kw4"],
    "hint": "<concrete hint: what a strong answer must include, e.g. mention STAR method with quantified impact>"
  }}
]"""

    try:
        text = _generate(prompt)
        questions = _parse_json(text)
        if isinstance(questions, list) and len(questions) > 0:
            # Validate structure
            valid = []
            for q in questions:
                if isinstance(q, dict) and q.get("question"):
                    valid.append(q)
            if valid:
                return valid[:question_count]
    except Exception as e:
        print(f"[MockInterviewEngine] Question generation failed: {e}")

    return _fallback_questions(question_count)


def _fallback_questions(count: int) -> list:
    """Only used when Mistral completely fails."""
    base = [
        {"question": "Tell me about your most challenging project and the technical decisions you made.",
         "type": "project-based", "difficulty": "medium", "focus_area": "Projects",
         "expected_keywords": ["architecture", "challenge", "solution", "impact", "trade-offs"],
         "hint": "Use STAR: describe the specific challenge, your decision process, implementation, and quantified outcome"},
        {"question": "How do you ensure code quality and maintainability in your projects?",
         "type": "technical", "difficulty": "easy", "focus_area": "Code Quality",
         "expected_keywords": ["testing", "code review", "documentation", "refactoring", "SOLID"],
         "hint": "Mention specific tools (linters, CI/CD), testing strategy, and code review practices"},
        {"question": "Describe a time you had to learn a new technology under tight deadline pressure.",
         "type": "behavioral", "difficulty": "medium", "focus_area": "Adaptability",
         "expected_keywords": ["learning strategy", "documentation", "prototyping", "deadline", "outcome"],
         "hint": "Show how you structured the learning, what resources you used, and the result"},
        {"question": "How would you design a URL shortening service that handles 1 billion requests/day?",
         "type": "system-design", "difficulty": "hard", "focus_area": "System Design",
         "expected_keywords": ["load balancer", "caching", "database sharding", "hashing", "CDN", "rate limiting"],
         "hint": "Start with requirements, then cover: data model, API design, scaling strategy, caching, and failure handling"},
        {"question": "Walk me through debugging a critical production issue with no logs available.",
         "type": "behavioral", "difficulty": "hard", "focus_area": "Debugging",
         "expected_keywords": ["reproduce", "monitoring", "metrics", "hypothesis", "root cause", "rollback"],
         "hint": "Show systematic approach: gather info → form hypothesis → isolate → fix → post-mortem"},
    ]
    return (base * ((count // len(base)) + 1))[:count]


# ─── Model Answer Generator ──────────────────────────────────────────────────

def generate_model_answer(question: dict, job_description: str = "",
                           resume_context: str = "") -> str:
    """Generate a detailed, standalone model answer for a question."""
    prompt = f"""You are an expert senior engineer. Write a complete, detailed model answer to this interview question.

QUESTION: {question.get('question', '')}
QUESTION TYPE: {question.get('type', 'general')}
DIFFICULTY: {question.get('difficulty', 'medium')}
KEY CONCEPTS TO COVER: {', '.join(question.get('expected_keywords', []))}
WHAT MAKES A STRONG ANSWER: {question.get('hint', '')}

JOB CONTEXT: {job_description[:400] if job_description else 'General software engineering role'}

Write a 200-250 word model answer that:
1. Directly and fully answers the question
2. Covers ALL key concepts listed above with appropriate technical depth
3. Uses a concrete real-world example or scenario (can be hypothetical but specific)
4. Includes specific numbers, metrics, or outcomes where appropriate
5. Is structured clearly and professionally

IMPORTANT: Write the answer directly. Do NOT start with "Here is a model answer:" or any preamble.
IMPORTANT: Do NOT reference the candidate's actual answer — this is a standalone ideal answer.
Write only the answer text itself."""

    try:
        text = _generate(prompt)
        if len(text) > 80:
            return text
    except Exception as e:
        print(f"[MockInterviewEngine] Model answer gen failed: {e}")

    # Simple fallback if Mistral fails
    kws = question.get("expected_keywords", [])
    focus = question.get("focus_area", "this topic")
    return (
        f"A strong answer to this question would demonstrate deep understanding of {focus}. "
        f"Key concepts to address: {', '.join(kws)}. "
        f"Structure your response using the STAR method for behavioral questions, or a "
        f"requirements → design → trade-offs → scaling approach for system design questions. "
        f"Always include concrete examples with quantifiable outcomes."
    )


# ─── Answer Evaluation ───────────────────────────────────────────────────────

def evaluate_answer(question: dict, user_answer: str, resume_context: str = "",
                    job_description: str = "") -> dict:
    """
    Evaluate a single interview answer using Mistral.
    """
    if not user_answer or not user_answer.strip():
        ideal = generate_model_answer(question, job_description, resume_context)
        return {
            "score": 0, "grade": "F",
            "strengths": [],
            "improvements": [
                "This question was not answered.",
                "Study the model answer below to understand what a complete response looks like.",
                f"Required concepts: {', '.join(question.get('expected_keywords', [])[:4])}"
            ],
            "ideal_answer": ideal,
            "keyword_coverage": [],
            "missing_keywords": question.get("expected_keywords", []),
            "verdict": "skipped",
            "score_breakdown": {"relevance": 0, "technical": 0, "structure": 0, "keywords": 0}
        }

    expected_kw = question.get("expected_keywords", [])
    hint = question.get("hint", "")

    prompt = f"""You are a rigorous FAANG technical interviewer evaluating a candidate's answer.

QUESTION: {question.get('question', '')}
TYPE: {question.get('type', '')} | DIFFICULTY: {question.get('difficulty', 'medium')} | FOCUS: {question.get('focus_area', '')}
KEY CONCEPTS EXPECTED: {', '.join(expected_kw)}
WHAT A STRONG ANSWER COVERS: {hint}

CANDIDATE'S ANSWER:
{user_answer[:2000]}

RESUME CONTEXT (for reference):
{resume_context[:400] if resume_context else 'N/A'}

SCORING DIMENSIONS:
- Relevance & Completeness (0-30): Does it directly answer what was asked?
- Technical Accuracy (0-40): Are concepts correct, specific, and at appropriate depth?
- Structure & Clarity (0-20): Is it well-organized?
- Keyword Coverage (0-10): Which expected concepts were covered?
NOTE: Be highly polarizing with your score. A very poor answer should get a score between 10-30. An average answer 50-70. An excellent answer 85-100. Do not default to 60.

STRICT RULES FOR YOUR RESPONSE:
- "strengths" items must quote or paraphrase something SPECIFIC the candidate actually said. Never write generic statements like "Attempted the question."
- "improvements" items must identify SPECIFICALLY what was missing or incorrect, and suggest exactly what the candidate should have said/included.
- "ideal_answer" must be 200-250 words covering ALL key concepts with concrete examples. Write it as if you're the ideal candidate — do NOT say "the model answer should be". Just write the answer.
- Grade strictly: an answer of 5 words should score ≤ 15/100.

Return ONLY valid JSON. No markdown. No preamble.

{{
  "score": <0-100>,
  "grade": "<A|B|C|D|F>",
  "verdict": "<excellent|good|average|below-average|poor>",
  "strengths": [
    "<specific thing candidate said well, quoting their words if possible>",
    "<another specific strength>"
  ],
  "improvements": [
    "<specific missing concept/detail with explanation of what should have been said>",
    "<another specific improvement>"
  ],
  "ideal_answer": "<200-250 word complete model answer covering all key concepts with specific examples>",
  "keyword_coverage": ["<keywords candidate actually mentioned>"],
  "missing_keywords": ["<expected keywords not covered>"],
  "score_breakdown": {{
    "relevance": <0-40>,
    "technical": <0-30>,
    "structure": <0-20>,
    "keywords": <0-10>
  }}
}}"""

    try:
        text = _generate(prompt)
        result = _parse_json(text)
        result["score"] = max(0, min(100, int(result.get("score", 0))))

        # Enforce grade matches score
        s = result["score"]
        result["grade"] = "A" if s >= 85 else "B" if s >= 70 else "C" if s >= 55 else "D" if s >= 40 else "F"

        # Validate ideal_answer is substantial; regenerate if not
        if not result.get("ideal_answer") or len(result.get("ideal_answer", "")) < 100:
            result["ideal_answer"] = generate_model_answer(question, job_description, resume_context)

        return result

    except Exception as e:
        print(f"[MockInterviewEngine] Evaluation failed: {e}")
        # Better fallback than before — at least tries to score based on content
        word_count = len(user_answer.split())
        covered = [kw for kw in expected_kw if kw.lower() in user_answer.lower()]
        missing = [kw for kw in expected_kw if kw.lower() not in user_answer.lower()]
        score = min(100, max(10, word_count * 2 + len(covered) * 15))
        grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D" if score >= 40 else "F"

        return {
            "score": score, "grade": grade, "verdict": "average",
            "strengths": [
                f"You wrote {word_count} words showing engagement." if word_count > 20 else "You attempted the question.",
                f"Mentioned {len(covered)} of {len(expected_kw)} key concepts." if covered else "Answer was provided."
            ],
            "improvements": [
                f"Missing concepts: {', '.join(missing[:3])}. Include these with specific examples." if missing else "Expand on technical depth.",
                "Use STAR method for behavioral questions; requirements→design→trade-offs for system design.",
                "Include specific numbers, outcomes, or metrics to make answers memorable."
            ],
            "ideal_answer": generate_model_answer(question, job_description, resume_context),
            "keyword_coverage": covered,
            "missing_keywords": missing,
            "score_breakdown": {"relevance": score//2, "technical": score//4,
                                "structure": score//8, "keywords": min(10, len(covered)*2)}
        }


# ─── Session Summary ─────────────────────────────────────────────────────────

def generate_session_summary(questions: list, answers: dict, evaluations: dict,
                              job_description: str = "") -> dict:
    """
    Generate comprehensive session summary with AI-powered narrative insights.
    """
    total = len(questions)
    answered_count = len([a for a in answers.values() if a and str(a).strip()])

    # All scores — skipped = 0
    all_scores = []
    for i in range(total):
        ev = evaluations.get(i, evaluations.get(str(i), {}))
        all_scores.append(ev.get("score", 0))

    avg_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    # Grade from avg of ALL questions
    overall_grade = ("A" if avg_score >= 85 else "B" if avg_score >= 70
                     else "C" if avg_score >= 55 else "D" if avg_score >= 40 else "F")

    # Grade counts — ALL questions including skipped
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for i in range(total):
        ev = evaluations.get(i, evaluations.get(str(i), {}))
        grade = ev.get("grade", "F")
        grade_counts[grade if grade in grade_counts else "F"] += 1

    # Type-level performance
    type_scores = {}
    for i, q in enumerate(questions):
        t = q.get("type", "other")
        ev = evaluations.get(i, evaluations.get(str(i), {}))
        type_scores.setdefault(t, []).append(ev.get("score", 0))
    type_avg = {t: round(sum(s)/len(s)) for t, s in type_scores.items() if s}

    strongest = max(type_avg, key=type_avg.get) if type_avg else "N/A"
    weakest   = min(type_avg, key=type_avg.get) if type_avg else "N/A"

    # Collect strengths / improvements
    all_strengths = []
    all_improvements = []
    for ev in evaluations.values():
        if isinstance(ev, dict):
            all_strengths.extend(ev.get("strengths", []))
            all_improvements.extend(ev.get("improvements", []))

    unique_strengths    = list(dict.fromkeys(s for s in all_strengths    if s and len(s) > 12))[:5]
    unique_improvements = list(dict.fromkeys(s for s in all_improvements if s and len(s) > 12))[:5]

    # AI narrative
    ai = _generate_ai_insights(questions, answers, evaluations,
                                avg_score, overall_grade, type_avg,
                                answered_count, total, job_description)

    return {
        "avg_score": avg_score,
        "overall_grade": overall_grade,
        "questions_answered": answered_count,
        "total_questions": total,
        "completion_rate": round((answered_count / total) * 100) if total else 0,
        "grade_counts": grade_counts,
        "type_performance": type_avg,
        "strongest_area": strongest,
        "weakest_area": weakest,
        "top_strengths": unique_strengths,
        "top_improvements": unique_improvements,
        "recommendation": _get_recommendation(avg_score, answered_count, total),
        "ai_narrative":      ai.get("narrative", ""),
        "skill_gaps":        ai.get("skill_gaps", []),
        "study_plan":        ai.get("study_plan", []),
        "interview_tips":    ai.get("interview_tips", []),
        "readiness_verdict": ai.get("readiness_verdict", ""),
    }


def _generate_ai_insights(questions, answers, evaluations, avg_score, overall_grade,
                           type_performance, answered, total, job_description):
    qa_summary = []
    for i, q in enumerate(questions[:10]):
        ev = evaluations.get(i, evaluations.get(str(i), {}))
        ans = answers.get(i, answers.get(str(i), ""))
        qa_summary.append({
            "q": q.get("question", "")[:80],
            "type": q.get("type", ""),
            "difficulty": q.get("difficulty", ""),
            "score": ev.get("score", 0),
            "grade": ev.get("grade", "F"),
            "answered": bool(ans and str(ans).strip()),
            "issues": ev.get("improvements", [])[:2]
        })

    comp_rate = round((answered/total)*100) if total else 0

    prompt = f"""You are a senior career coach reviewing a mock interview. Provide HONEST, SPECIFIC, PERSONALISED insights.

SESSION DATA:
- Grade: {overall_grade} | Score: {avg_score}/100 | Completion: {comp_rate}%
- Answered: {answered}/{total} questions
- By type: {json.dumps(type_performance)}

Q&A BREAKDOWN:
{json.dumps(qa_summary, indent=2)}

JOB CONTEXT: {job_description[:400] if job_description else 'Software Engineering role'}

Provide deeply personalised feedback. Reference specific question types that went well or poorly.
Be direct and actionable — no generic advice.

Return ONLY valid JSON. No markdown.

{{
  "narrative": "<3-4 sentences of honest, personalised performance analysis referencing specific question types and scores>",
  "skill_gaps": [
    "<specific technical/soft skill gap identified from the answers, with 1-sentence explanation>",
    "<another gap>",
    "<another gap>"
  ],
  "study_plan": [
    "<specific action: what to study, with a suggested resource or approach>",
    "<another action>",
    "<another action>"
  ],
  "interview_tips": [
    "<tip specific to THIS candidate's weaknesses — not generic>",
    "<another specific tip>",
    "<another tip>"
  ],
  "readiness_verdict": "<One honest sentence: ready/not ready, and what the single biggest blocker is>"
}}"""

    try:
        text = _generate(prompt)
        return _parse_json(text)
    except Exception as e:
        print(f"[MockInterviewEngine] AI insights failed: {e}")
        return {
            "narrative": (
                f"You scored {avg_score}/100 (Grade {overall_grade}), answering {answered} of {total} questions "
                f"with a {comp_rate}% completion rate. "
                + ("Strong performance overall — focus on consistency." if avg_score >= 70
                   else "Several areas need significant improvement — use the model answers as study guides.")
            ),
            "skill_gaps": ["Review weak question types", "Improve technical depth", "Practice structured answers"],
            "study_plan": ["Practice STAR method daily", "Review technical concepts in weak areas", "Do mock interviews weekly"],
            "interview_tips": ["Be more specific with examples", "Use structured responses", "Cover all key concepts"],
            "readiness_verdict": (
                f"Grade {overall_grade} — {'Ready with minor polish.' if avg_score >= 70 else 'Not yet ready. Significant practice required.'}"
            )
        }


def _get_recommendation(avg_score: float, answered: int, total: int) -> str:
    if answered < total * 0.5:
        return "You skipped many questions. In a real interview, always attempt every question — even a partial structured answer scores better than silence."
    if avg_score >= 80:
        return "Excellent! You're well-prepared. Focus on consistency and edge-case depth."
    if avg_score >= 60:
        return "Good foundation! Deepen technical answers with specific metrics and use STAR for behavioral questions."
    if avg_score >= 40:
        return "Keep practicing! Cover all key concepts, structure answers clearly, and be specific with examples."
    return "Significant improvement needed. Study the model answers carefully, review fundamentals, and practice daily."
