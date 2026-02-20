# backend/agent/adaptive_question_bank.py


from rag import generate_technical_explanation as generate_rag_response
import random
import json
from typing import List, Dict

class AdaptiveQuestionBank:
    """Generates personalized questions based on user state"""
    
    def __init__(self, taxonomy_path="config/taxonomy.json"):
        # Load taxonomy from file
        try:
            with open(taxonomy_path, 'r') as f:
                self.taxonomy = json.load(f)
        except:
            # Fallback taxonomy if file not found
            self.taxonomy = {
                "topics": [
                    {
                        "name": "DBMS",
                        "subtopics": ["Normalization", "ACID", "Transactions", "Indexing", "SQL", "Joins"]
                    },
                    {
                        "name": "OOPs",
                        "subtopics": ["Classes", "Objects", "Inheritance", "Polymorphism", "Encapsulation", "Abstraction"]
                    },
                    {
                        "name": "OS",
                        "subtopics": ["Processes", "Threads", "Memory Management", "Scheduling", "Deadlocks", "File Systems"]
                    }
                ]
            }
        
        # Build subtopic lists by topic
        self.subtopics_by_topic = {}
        for topic in self.taxonomy["topics"]:
            self.subtopics_by_topic[topic["name"]] = topic["subtopics"]
        
        # Fallback questions by topic and difficulty (still keep as backup)
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
    
    def _get_subtopics_str(self, topic: str) -> str:
        """Get formatted string of subtopics for a topic"""
        subtopics = self.subtopics_by_topic.get(topic, [])
        return ", ".join(subtopics)
    
    def generate_first_question(self, topic: str, difficulty: str = "medium", user_name: str = "") -> str:
        """Generate first question for a topic - CONSTRAINED by taxonomy"""
        
        subtopics_str = self._get_subtopics_str(topic)
        
        personalization = f" for {user_name}" if user_name else ""
        
        prompt = f"""
You are an expert technical interviewer conducting a personalized interview{personalization}.

Generate ONE interview question for topic: {topic}
IMPORTANT: The question must be about ONE of these specific subtopics: {subtopics_str}
Difficulty level: {difficulty}

RULES:
- The question MUST be about a concept from the allowed subtopics list above
- Return ONLY the question text, no labels or explanations
- Make it appropriate for the difficulty level
- Be specific and focused on a single concept
- DO NOT ask about topics outside the allowed list

Example format:
What is the difference between process and thread?

Generate your question now:
"""
        try:
            raw_question = generate_rag_response("first question", prompt).strip()
            return self._clean_question(raw_question)
        except:
            # Fallback to random question from bank
            return random.choice(self.fallback_questions[topic][difficulty])
    
    def generate_gap_followup(self, topic: str, missing_concepts: list, difficulty: str = "medium") -> str:
        """Generate question targeting specific gaps - CONSTRAINED by taxonomy"""
        
        subtopics_str = self._get_subtopics_str(topic)
        gaps = ", ".join(missing_concepts[:3]) if missing_concepts else "core concepts"
        
        prompt = f"""
You are an interviewer helping a student improve their understanding.

Topic: {topic}
The student is struggling with: {gaps}

IMPORTANT RULES:
- Your question MUST be about ONE of these allowed subtopics: {subtopics_str}
- Focus specifically on the struggling concepts if they are in the allowed list
- If struggling concepts aren't in the allowed list, choose the closest related allowed subtopic
- Ask ONE focused question to help them understand better
- Make it clear and fundamental
- Difficulty: {difficulty}

Return ONLY the question text, no labels or explanations.

Generate your question:
"""
        try:
            raw_question = generate_rag_response("gap followup", prompt).strip()
            return self._clean_question(raw_question)
        except:
            # Fallback
            return f"Can you explain more about {missing_concepts[0] if missing_concepts else topic}?"
    
    def generate_simplified_question(self, topic: str, missing_concepts: list) -> str:
        """Generate very simple question for struggling users - CONSTRAINED by taxonomy"""
        
        subtopics_str = self._get_subtopics_str(topic)
        gaps = ", ".join(missing_concepts[:2]) if missing_concepts else "basic concepts"
        
        prompt = f"""
You are a helpful tutor helping a student who is struggling with {topic}.

The student needs help with: {gaps}

IMPORTANT RULES:
- Ask a VERY SIMPLE, BASIC question about ONE of these allowed subtopics: {subtopics_str}
- Use plain, simple language
- Ask for a definition or a very simple example
- Build confidence - make it easy to answer
- No technical jargon overload

Return ONLY the question text, no labels or explanations.

Generate your simplified question:
"""
        try:
            raw_question = generate_rag_response("simplified", prompt).strip()
            return self._clean_question(raw_question)
        except:
            return f"What is {topic} in simple terms?"
    
    def generate_deeper_dive(self, topic: str, difficulty: str = "hard") -> str:
        """Generate challenging question for strong performers - CONSTRAINED by taxonomy"""
        
        subtopics_str = self._get_subtopics_str(topic)
        
        prompt = f"""
You are an expert interviewer quizzing a strong candidate on {topic}.

The candidate has shown excellent understanding. Ask a DEEPER, CHALLENGING question that:
- Tests practical application
- Explores edge cases
- Requires synthesis of multiple concepts
- Goes beyond textbook definitions

IMPORTANT RULES:
- The question MUST be about ONE of these allowed subtopics: {subtopics_str}
- Make it thought-provoking
- Difficulty: {difficulty}

Return ONLY the question text, no labels or explanations.

Generate your challenging question:
"""
        try:
            raw_question = generate_rag_response("deep dive", prompt).strip()
            return self._clean_question(raw_question)
        except:
            return random.choice(self.fallback_questions[topic].get("hard", self.fallback_questions[topic]["medium"]))
    
    def generate_question_for_subtopic(self, topic: str, subtopic: str, difficulty: str = "medium") -> str:
        """Generate a question for a specific subtopic"""
        
        prompt = f"""
You are an expert technical interviewer.

Generate ONE interview question about the following topic and subtopic:
Topic: {topic}
Subtopic: {subtopic}
Difficulty: {difficulty}

RULES:
- Focus ONLY on {subtopic} within {topic}
- Return ONLY the question text
- Make it appropriate for the difficulty level
- Be specific and focused

Generate your question:
"""
        try:
            raw_question = generate_rag_response("subtopic question", prompt).strip()
            return self._clean_question(raw_question)
        except:
            return f"Can you explain {subtopic} in {topic}?"
    
    def _clean_question(self, text: str) -> str:
        """Clean question text"""
        text = text.replace("Question:", "").strip()
        if "Answer:" in text:
            text = text.split("Answer:")[0].strip()
        text = text.replace("**", "").replace("##", "").replace("```", "")
        return text.strip()