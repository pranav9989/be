"""
coding_engine.py — AI-powered Data Science Coding Practice
Uses Mistral API
 for:
  - Generating LeetCode/Stratascratch style SQL and Pandas questions
  - Evaluating user code against AI-generated hidden test cases
  - Providing model answers in both SQL and Pandas
"""

import os
import json
import re
import time
import requests

# Removed Ollama in favor of Mistral (via rag.py)
from rag import mistral_generate as _generate

def _parse_json(text: str):
    """Strip markdown fences from text and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    start = min(
        (text.find(c) for c in ["[", "{"] if c in text),
        default=0
    )
    return json.loads(text[start:])


# ─── Question Generation ─────────────────────────────────────────────────────

def generate_coding_questions(question_count: int = 3, difficulty: str = "medium") -> list:
    """
    Generate Data Science coding questions (SQL & Pandas).
    Returns a list of dicts containing problem schema, data, and expected shapes.
    """
    prompt = f"""You are an expert Data Science Interviewer at a top tech company (like Stratascratch or LeetCode).
Generate exactly {question_count} Data Science coding interview questions of '{difficulty}' difficulty.
These questions must be solvable using EITHER SQL or Python (Pandas).

CRITICAL RULES:
1. Return ONLY a valid JSON array of objects. Do NOT wrap in markdown blocks like ```json.
2. Each object MUST have this exact structure:
{{
  "id": "q1",
  "title": "Find top salaries by department",
  "difficulty": "{difficulty}",
  "description": "Write a query to find the top 3 highest paid employees in each department. Return department_name, employee_name, and salary.",
  "tables": [
     {{
       "name": "employees",
       "schema": "id INT, name VARCHAR, salary INT, department_id INT",
       "sample_data": "[{{'id':1, 'name':'Alice', 'salary':100, 'department_id':1}}]"
     }}
  ],
  "expected_output_columns": ["department_name", "employee_name", "salary"]
}}
3. Ensure the schema and sample_data are very realistic and mock an actual enterprise database. Provide at least 4-5 rows of sample_data per table.
4. Make sure questions test things like WINDOW FUNCTIONS, JOINS, AGGREGATIONS, or CTEs in SQL, and equivalent operations (groupby, merge, apply) in Pandas.

Generate {question_count} questions now as a JSON array:"""

    response_text = _generate(prompt)
    try:
        data = _parse_json(response_text)
        if not isinstance(data, list):
            return [data]
        return data
    except Exception as e:
        print(f"[CodingEngine] Failed to parse generated DB questions: {e}\\nResponse: {{response_text[:300]}}")
        # Build a safe fallback
        return [
            {
                "id": "q1-fallback",
                "title": "Most Profitable Companies",
                "difficulty": "medium",
                "description": "Find the 3 most profitable companies in the entire world. Return the company name and profit.",
                "tables": [
                    {
                        "name": "forbes_global",
                        "schema": "company VARCHAR, sector VARCHAR, profits FLOAT",
                        "sample_data": "[{'company': 'ICBC', 'sector': 'Financials', 'profits': 42.7}, {'company': 'China Construction Bank', 'sector': 'Financials', 'profits': 34.2}]"
                    }
                ],
                "expected_output_columns": ["company", "profits"]
            }
        ]


# ─── Code Evaluation ─────────────────────────────────────────────────────────

def evaluate_coding_answer(question: dict, user_code: str, language: str) -> dict:
    """
    Evaluate user's SQL or Pandas code. 
    Returns Pass/Fail status, test case results, hints, and the AI's perfect Model Answers for both languages.
    """
    
    prompt = f"""You are an expert Data Science Grader. Rate the following user's {language.upper()} code for a data science question.

QUESTION TITLE: {question.get('title')}
DESCRIPTION: {question.get('description')}
TABLES AVAILABLE:
{json.dumps(question.get('tables'), indent=2)}

USER'S SUBMITTED {language.upper()} CODE:
```
{user_code}
```

Evaluate the code by simulating execution against standard edge cases. 
Return ONLY a valid JSON object. Do not wrap in ```json.

REQUIRED JSON STRUCTURE:
{{
  "passed": true/false,
  "score": 0_to_100,
  "feedback": "A very specific 2-3 sentence explanation of what they did right, or where the code fails (e.g. 'You forgot to handle ties by using DENSE_RANK() instead of RANK()').",
  "test_cases": [
    {{"description": "Basic exact match", "passed": true/false}},
    {{"description": "Handles null/empty values", "passed": true/false}},
    {{"description": "Handles duplicate/tied records appropriately", "passed": true/false}}
  ],
  "model_answer_sql": "-- The optimal SQL solution\\nSELECT ...",
  "model_answer_pandas": "# The optimal Pandas solution\\nimport pandas as pd\\n..."
}}

Only output the raw JSON object string:"""

    response_text = _generate(prompt)
    try:
        evaluation = _parse_json(response_text)
        
        # Ensure fallback fields
        evaluation.setdefault('passed', False)
        evaluation.setdefault('score', 0)
        evaluation.setdefault('feedback', "Analysis failed.")
        evaluation.setdefault('test_cases', [])
        evaluation.setdefault('model_answer_sql', "Select * from table")
        evaluation.setdefault('model_answer_pandas', "df.head()")
        
        return evaluation
    except Exception as e:
        print(f"[CodingEngine] Failed to parse code evaluation: {e}\nResponse: {response_text[:300]}")
        return {
            "passed": False,
            "score": 0,
            "feedback": f"Could not evaluate code due to parsing error. Ensure your syntax is strictly valid {language}.",
            "test_cases": [{"description": "Syntax check", "passed": False}],
            "model_answer_sql": "Error evaluating code.",
            "model_answer_pandas": "Error evaluating code."
        }
