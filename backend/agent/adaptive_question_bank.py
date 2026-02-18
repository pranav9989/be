# backend/agent/adaptive_question_bank.py

from rag import generate_rag_response
import random

class AdaptiveQuestionBank:
    """Generates personalized questions based on user state"""
    
    def __init__(self):
        # Fallback questions by topic and difficulty
        self.fallback_questions = {
            "DBMS": {
                "easy": [
                    "What is a database?",
                    "What is a primary key?",
                    "What is the difference between SQL and NoSQL?"
                ],
                "medium": [
                    "Explain the ACID properties of a database transaction.",
                    "What is normalization? Why is it used?",
                    "Describe different types of joins in SQL."
                ],
                "hard": [
                    "Explain the differences between BCNF and 3NF with examples.",
                    "How do database indexing structures work internally?",
                    "Describe the two-phase locking protocol for concurrency control."
                ]
            },
            "OS": {
                "easy": [
                    "What is an operating system?",
                    "What is a process?",
                    "What is the difference between a process and a thread?"
                ],
                "medium": [
                    "Explain different CPU scheduling algorithms.",
                    "What is deadlock? What are its necessary conditions?",
                    "How does virtual memory work?"
                ],
                "hard": [
                    "Compare and contrast paging and segmentation.",
                    "Explain the banker's algorithm for deadlock avoidance.",
                    "How do modern operating systems handle memory management?"
                ]
            },
            "OOPS": {
                "easy": [
                    "What is object-oriented programming?",
                    "What is a class?",
                    "What is inheritance?"
                ],
                "medium": [
                    "Explain the four pillars of OOP with examples.",
                    "What is polymorphism? Give examples.",
                    "Difference between abstraction and encapsulation."
                ],
                "hard": [
                    "Explain the diamond problem in multiple inheritance and how languages handle it.",
                    "What are virtual functions and how do they work?",
                    "Compare composition vs inheritance with examples."
                ]
            }
        }
    
    def generate_first_question(self, topic: str, difficulty: str = "medium", user_name: str = "") -> str:
        """Generate first question for a topic with personalization"""
        
        personalization = f" for {user_name}" if user_name else ""
        
        prompt = f"""
You are an expert technical interviewer conducting a personalized interview{personalization}.

Generate ONE interview question for topic: {topic}
Difficulty: {difficulty}

IMPORTANT RULES:
- Return ONLY the question text
- DO NOT include "Question:" or any labels
- DO NOT include the answer
- Make it appropriate for the difficulty level
- One question only, no explanations

Example format:
What is the difference between abstraction and encapsulation?
"""
        try:
            raw_question = generate_rag_response("first question", prompt).strip()
            return self._clean_question(raw_question)
        except:
            # Fallback to random question from bank
            return random.choice(self.fallback_questions[topic][difficulty])
    
    def generate_gap_followup(self, topic: str, missing_concepts: list, difficulty: str = "medium") -> str:
        """Generate question targeting specific gaps"""
        
        gaps = ", ".join(missing_concepts[:3]) if missing_concepts else "core understanding"
        
        prompt = f"""
You are an interviewer helping a student improve.

Topic: {topic}
Student needs help with: {gaps}
Difficulty: {difficulty}

Ask ONE focused question to help them understand these concepts better.
Make it clear and fundamental.

IMPORTANT RULES:
- Return ONLY the question text
- DO NOT include labels
- One question only
"""
        try:
            raw_question = generate_rag_response("gap followup", prompt).strip()
            return self._clean_question(raw_question)
        except:
            # Fallback
            return f"Can you explain more about {missing_concepts[0] if missing_concepts else topic}?"
    
    def generate_simplified_question(self, topic: str, missing_concepts: list) -> str:
        """Generate very simple question for struggling users"""
        
        gaps = ", ".join(missing_concepts[:2]) if missing_concepts else "basic concepts"
        
        prompt = f"""
You are a helpful tutor helping a student who is struggling with {topic}.

The student needs help with: {gaps}

Ask a VERY SIMPLE, BASIC question that:
- Uses plain language
- Asks for a definition or simple example
- Builds confidence
- Is easy to answer

IMPORTANT RULES:
- Return ONLY the question text
- Keep it very simple
- No technical jargon overload
"""
        try:
            raw_question = generate_rag_response("simplified", prompt).strip()
            return self._clean_question(raw_question)
        except:
            return f"What is {topic} in simple terms?"
    
    def generate_deeper_dive(self, topic: str, difficulty: str = "hard") -> str:
        """Generate challenging question for strong performers"""
        
        prompt = f"""
You are an expert interviewer quizzing a strong candidate on {topic}.

The candidate has shown excellent understanding. Ask a DEEPER, CHALLENGING question that:
- Tests practical application
- Explores edge cases
- Requires synthesis of multiple concepts
- Goes beyond textbook definitions

Difficulty: {difficulty}

IMPORTANT RULES:
- Return ONLY the question text
- Make it thought-provoking
- No labels or explanations
"""
        try:
            raw_question = generate_rag_response("deep dive", prompt).strip()
            return self._clean_question(raw_question)
        except:
            return random.choice(self.fallback_questions[topic].get("hard", self.fallback_questions[topic]["medium"]))
    
    def _clean_question(self, text: str) -> str:
        """Clean question text"""
        text = text.replace("Question:", "").strip()
        if "Answer:" in text:
            text = text.split("Answer:")[0].strip()
        text = text.replace("**", "").replace("##", "").replace("```", "")
        return text.strip()