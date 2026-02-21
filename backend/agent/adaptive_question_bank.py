# backend/agent/adaptive_question_bank.py

from rag import generate_technical_explanation as generate_rag_response
import random
import re
import numpy as np
from typing import Dict, List, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AdaptiveQuestionBank:
    """Advanced adaptive question bank with intent-based rotation and duplicate prevention"""
    
    MAX_CHARS = 400
    MAX_RETRIES = 5
    DUP_THRESHOLD = 0.82  # Slightly lower than semantic_dedup for more variety
    
    # 🔥 Layer 1: Intent-Based Question Rotation (NOT type-based)
    INTENTS = [
        "core_definition",
        "conceptual_difference",
        "mechanism_flow",
        "real_world_scenario",
        "problem_case",
        "edge_case",
        "tradeoff_analysis",
        "debugging_case",
        "optimization_reasoning",
        "misconception_check"
    ]
    
    # 🔥 Layer 4: Subtopic Normalization (concept clusters)
    SUBTOPIC_CLUSTERS = {
        "Polymorphism": [
            "Polymorphism (Compile-time & Runtime)",
            "Method Overloading vs Overriding",
            "Virtual functions",
            "Dynamic dispatch"
        ],
        "Memory Management": [
            "Memory Management (Paging, Segmentation)",
            "Virtual Memory",
            "Demand Paging & Page Replacement (LRU, FIFO)"
        ],
        "Process Management": [
            "Process vs Thread",
            "Process States & PCB",
            "Context Switching"
        ],
        "Synchronization": [
            "Synchronization (Mutex, Semaphore, Monitor)",
            "Deadlock (Conditions & Prevention)"
        ],
        "Database Design": [
            "Normalization (1NF, 2NF, 3NF, BCNF)",
            "Keys (Primary, Foreign, Candidate, Composite)"
        ],
        "Query Optimization": [
            "Indexing (B+ Tree, Hash Index)",
            "Joins (Inner, Left, Right, Full)",
            "SQL Queries (GROUP BY, HAVING, Subqueries)"
        ],
        "Transaction Management": [
            "ACID Properties",
            "Transactions & Concurrency Control",
            "Isolation Levels",
            "Locking (Shared, Exclusive Locks)",
            "Deadlocks in DBMS"
        ]
    }
    
    def __init__(self):
        # Initialize sentence transformer for semantic duplicate detection
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # YOUR EXACT TOPICS AND SUBTOPICS - HARDCODED
        self.taxonomy = {
            "topics": [
                {
                    "name": "DBMS",
                    "subtopics": [
                        "Normalization (1NF, 2NF, 3NF, BCNF)",
                        "Keys (Primary, Foreign, Candidate, Composite)",
                        "ACID Properties",
                        "Transactions & Concurrency Control",
                        "Isolation Levels",
                        "Indexing (B+ Tree, Hash Index)",
                        "Joins (Inner, Left, Right, Full)",
                        "SQL Queries (GROUP BY, HAVING, Subqueries)",
                        "Locking (Shared, Exclusive Locks)",
                        "Deadlocks in DBMS"
                    ]
                },
                {
                    "name": "OOPS",
                    "subtopics": [
                        "Classes & Objects",
                        "Encapsulation",
                        "Abstraction",
                        "Inheritance (Types & Diamond Problem)",
                        "Polymorphism (Compile-time & Runtime)",
                        "Method Overloading vs Overriding",
                        "Interfaces vs Abstract Classes",
                        "Constructors",
                        "Access Modifiers",
                        "SOLID Principles"
                    ]
                },
                {
                    "name": "OS",
                    "subtopics": [
                        "Process vs Thread",
                        "Process States & PCB",
                        "Context Switching",
                        "CPU Scheduling Algorithms (FCFS, SJF, RR, Priority)",
                        "Synchronization (Mutex, Semaphore, Monitor)",
                        "Deadlock (Conditions & Prevention)",
                        "Memory Management (Paging, Segmentation)",
                        "Virtual Memory",
                        "Demand Paging & Page Replacement (LRU, FIFO)",
                        "System Calls"
                    ]
                }
            ]
        }
        
        # Build subtopic lookup dictionaries
        self.subtopics_by_topic = {}
        self.subtopics_by_topic_upper = {}
        self.subtopics_by_topic_lower = {}
        
        for topic in self.taxonomy["topics"]:
            topic_name = topic["name"]
            subtopics = topic["subtopics"]
            
            self.subtopics_by_topic[topic_name] = subtopics
            self.subtopics_by_topic_upper[topic_name.upper()] = subtopics
            self.subtopics_by_topic_lower[topic_name.lower()] = subtopics
        
        # Add common variations
        self.subtopics_by_topic["OOP"] = self.subtopics_by_topic["OOPS"]
        self.subtopics_by_topic["OOPs"] = self.subtopics_by_topic["OOPS"]
        self.subtopics_by_topic["Operating System"] = self.subtopics_by_topic["OS"]
        self.subtopics_by_topic["Operating Systems"] = self.subtopics_by_topic["OS"]
        self.subtopics_by_topic["Database"] = self.subtopics_by_topic["DBMS"]
        self.subtopics_by_topic["Databases"] = self.subtopics_by_topic["DBMS"]
        
        print(f"✅ HARDCODED subtopics loaded successfully:")
        for topic_name, subtopics in self.subtopics_by_topic.items():
            if topic_name in ["DBMS", "OOPS", "OS"]:
                print(f"   - {topic_name}: {len(subtopics)} subtopics")
        
        # 🔥 Layer 2: Hard Duplicate Avoidance Memory
        # Stores last 3 semantic embeddings per subtopic
        self.embedding_history = {}  # {topic: {subtopic: [embeddings]}}
        
        # 🔥 Intent tracking per subtopic
        self.intent_tracker = {}  # {topic: {subtopic: [intents_used]}}
        
        # 🔥 Verbal-safe phrase replacements
        self.verbal_phrases = {
            r"show me": "describe",
            r"demonstrate": "explain",
            r"write code": "explain conceptually",
            r"implement": "describe the implementation approach",
            r"draw": "describe",
            r"diagram": "describe"
        }
    
    def _normalize_subtopic(self, topic: str, subtopic: str) -> str:
        """Map detailed subtopic to its conceptual cluster"""
        for cluster_name, members in self.SUBTOPIC_CLUSTERS.items():
            if subtopic in members:
                return cluster_name
        return subtopic
    
    def _get_next_intent(self, topic: str, subtopic: str) -> str:
        """Get next intent ensuring variety and progression"""
        
        # Initialize tracker
        if topic not in self.intent_tracker:
            self.intent_tracker[topic] = {}
        
        if subtopic not in self.intent_tracker[topic]:
            self.intent_tracker[topic][subtopic] = []
        
        used_intents = self.intent_tracker[topic][subtopic]
        
        # If we've used all intents, reset and start over
        if len(used_intents) >= len(self.INTENTS):
            used_intents = []
            self.intent_tracker[topic][subtopic] = []
        
        # Available intents (not used yet)
        available = [i for i in self.INTENTS if i not in used_intents]
        
        # Progression logic:
        # - Start with core_definition
        # - Then mechanism_flow
        # - Then conceptual_difference
        # - Then real_world_scenario
        # - Then tradeoff_analysis
        # - Then edge_case/debugging/misconception for depth
        
        if len(used_intents) == 0:
            chosen = "core_definition"
        elif len(used_intents) == 1:
            chosen = "mechanism_flow"
        elif len(used_intents) == 2:
            chosen = "conceptual_difference"
        elif len(used_intents) == 3:
            chosen = "real_world_scenario"
        elif len(used_intents) == 4:
            chosen = "tradeoff_analysis"
        else:
            # Pick randomly from remaining deep intents
            deep_intents = [i for i in available if i in ["edge_case", "debugging_case", "misconception_check", "optimization_reasoning"]]
            if deep_intents:
                chosen = random.choice(deep_intents)
            else:
                chosen = random.choice(available)
        
        # Record this intent
        self.intent_tracker[topic][subtopic].append(chosen)
        print(f"🎯 Intent for {topic} - {subtopic}: {chosen}")
        return chosen
    
    def _is_duplicate(self, topic: str, subtopic: str, question: str) -> bool:
        """
        Layer 2: Hard duplicate avoidance using semantic embeddings
        Compare with last 3 questions for this subtopic
        """
        # Get embedding for new question
        new_emb = self.embedder.encode([question])[0]
        
        # Initialize history if needed
        if topic not in self.embedding_history:
            self.embedding_history[topic] = {}
        
        if subtopic not in self.embedding_history[topic]:
            self.embedding_history[topic][subtopic] = []
        
        history = self.embedding_history[topic][subtopic]
        
        # Compare with historical embeddings
        for old_emb in history:
            similarity = cosine_similarity([new_emb], [old_emb])[0][0]
            if similarity > self.DUP_THRESHOLD:
                print(f"🔍 Duplicate detected (similarity: {similarity:.3f} > {self.DUP_THRESHOLD})")
                return True
        
        # Add to history (keep last 3)
        history.append(new_emb)
        if len(history) > 3:
            history.pop(0)
        
        return False
    
    def _make_verbal_safe(self, question: str) -> str:
        """Layer 3: Convert to verbal-friendly phrasing"""
        import re
        
        # Replace visual/action phrases with verbal ones
        for pattern, replacement in self.verbal_phrases.items():
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question
    
    def _build_prompt(self, topic: str, subtopic: str, intent: str, difficulty: str = "medium", user_name: str = "") -> str:
        """Build prompt with intent and constraints"""
        
        personalization = f" for {user_name}" if user_name else ""
        
        intent_descriptions = {
            "core_definition": "Ask for the fundamental definition or core concept",
            "conceptual_difference": "Ask how this concept differs from related concepts",
            "mechanism_flow": "Ask about the internal mechanism or workflow",
            "real_world_scenario": "Ask for a real-world application scenario",
            "problem_case": "Present a problem that requires this concept to solve",
            "edge_case": "Ask about edge cases or unusual situations",
            "tradeoff_analysis": "Ask about tradeoffs, pros and cons",
            "debugging_case": "Present a debugging scenario related to this concept",
            "optimization_reasoning": "Ask about optimization strategies",
            "misconception_check": "Address a common misconception about this topic"
        }
        
        intent_desc = intent_descriptions.get(intent, "Ask a technical question")
        
        prompt = f"""
You are a technical interviewer conducting a job interview{personalization}.

Generate ONE interview question about:
Topic: {topic}
Subtopic: {subtopic}
Intent: {intent}
Difficulty: {difficulty}

INTENT GUIDANCE:
{intent_desc}

STRICT RULES:
- Ask ONLY the question - NO introductions, NO commentary, NO explanations
- NO phrases like "let's talk about", "I'd like to ask", "here's a question"
- NO "show me" - use "describe" or "explain" instead
- Question must be answerable VERBALLY in an interview
- Question must be UNDER {self.MAX_CHARS} characters
- Focus specifically on {subtopic}
- Be direct and conversational, like a real interviewer

Generate ONLY the question text:
"""
        return prompt
    
    def _enforce_length(self, question: str) -> str:
        """Enforce length limit - reject if too long, don't truncate"""
        if len(question) <= self.MAX_CHARS:
            return question
        return None  # Signal to retry
    
    def _clean_question(self, text: str) -> str:
        """Clean question text"""
        # Remove common prefixes
        text = re.sub(r"^(Question:|Interviewer:|AI:|Assistant:)", "", text, flags=re.IGNORECASE)
        text = text.strip()
        
        # Remove markdown
        text = re.sub(r"[`*_#]", "", text)
        
        # Ensure it's a question
        if "?" in text:
            # Take everything up to and including first question mark
            text = text[:text.index("?")+1]
        else:
            # Add question mark if missing
            text += "?"
        
        return text
    
    def generate_question(self, topic: str, subtopic: str, difficulty: str = "medium", user_name: str = "") -> str:
        """
        Main question generation method with full pipeline:
        1. Intent selection with progression
        2. Prompt building
        3. RAG generation
        4. Duplicate check
        5. Length enforcement
        6. Verbal-safe conversion
        """
        
        for attempt in range(self.MAX_RETRIES):
            # Get next intent for variety
            intent = self._get_next_intent(topic, subtopic)
            
            # Build prompt
            prompt = self._build_prompt(topic, subtopic, intent, difficulty, user_name)
            
            try:
                # Generate with RAG
                from rag import generate_technical_explanation as generate_rag_response
                raw = generate_rag_response("question", prompt)
                
                # Clean
                question = self._clean_question(raw)
                
                # Make verbal-safe
                question = self._make_verbal_safe(question)
                
                # Check length (reject if too long)
                question = self._enforce_length(question)
                if question is None:
                    print(f"⚠️ Attempt {attempt + 1}: Question too long, retrying...")
                    continue
                
                # Check for duplicates
                if not self._is_duplicate(topic, subtopic, question):
                    print(f"✅ Generated {intent} question ({len(question)} chars): {question}")
                    return question
                else:
                    print(f"🔄 Attempt {attempt + 1}: Duplicate detected, retrying with different intent...")
                    
            except Exception as e:
                print(f"⚠️ Generation attempt {attempt + 1} failed: {e}")
                continue
        
        # 🔥 Hard fallback - simple template question
        print(f"⚠️ All {self.MAX_RETRIES} attempts failed, using fallback")
        fallback = f"What are the key concepts and practical applications of {subtopic} in {topic}?"
        return fallback
    
    # Convenience wrappers
    def generate_first_question(self, topic: str, subtopic: str = None, difficulty: str = "medium", user_name: str = "") -> str:
        """Wrapper for first question in a topic"""
        if subtopic is None:
            subtopics = self.subtopics_by_topic.get(topic, [])
            subtopic = random.choice(subtopics) if subtopics else "core concepts"
        
        return self.generate_question(topic, subtopic, difficulty, user_name)
    
    def generate_question_for_subtopic(self, topic: str, subtopic: str, difficulty: str = "medium") -> str:
        """Wrapper for subtopic-specific question"""
        return self.generate_question(topic, subtopic, difficulty)
    
    def generate_gap_followup(self, topic: str, missing_concepts: list, difficulty: str = "medium", 
                             current_subtopic: str = None, available_subtopics: list = None) -> str:
        """Generate follow-up targeting missing concepts"""
        
        if current_subtopic:
            # Use a different intent for gap follow-up
            # Force a deeper intent
            forced_intents = ["misconception_check", "debugging_case", "problem_case"]
            intent = random.choice(forced_intents)
            
            prompt = self._build_prompt(topic, current_subtopic, intent, difficulty)
            try:
                from rag import generate_technical_explanation as generate_rag_response
                raw = generate_rag_response("gap followup", prompt)
                question = self._clean_question(raw)
                question = self._make_verbal_safe(question)
                
                if question and len(question) <= self.MAX_CHARS:
                    if not self._is_duplicate(topic, current_subtopic, question):
                        return question
            except:
                pass
            
            # Fallback
            return f"Can you explain how {missing_concepts[0] if missing_concepts else current_subtopic} works in practice?"
        
        return f"Let's focus on {missing_concepts[0] if missing_concepts else topic}. Can you explain that concept?"
    
    def generate_simplified_question(self, topic: str, missing_concepts: list) -> str:
        """Generate a simpler question for struggling users"""
        subtopic = missing_concepts[0] if missing_concepts else topic
        
        prompt = f"""
You are a helpful tutor. Ask a VERY SIMPLE question about {subtopic} in {topic}.
Use plain language. Make it easy to answer.
Question must be under {self.MAX_CHARS} characters.
Return ONLY the question.
"""
        try:
            from rag import generate_technical_explanation as generate_rag_response
            raw = generate_rag_response("simplified", prompt)
            question = self._clean_question(raw)
            question = self._make_verbal_safe(question)
            
            if question and len(question) <= self.MAX_CHARS:
                return question
        except:
            pass
        
        return f"What is {subtopic} in simple terms?"
    
    def generate_deeper_dive(self, topic: str, difficulty: str = "hard") -> str:
        """Generate challenging question for strong performers"""
        # Force advanced intents
        advanced_intents = ["optimization_reasoning", "tradeoff_analysis", "edge_case"]
        intent = random.choice(advanced_intents)
        
        # Pick an advanced subtopic
        subtopics = self._get_subtopics(topic)
        advanced_keywords = ["Deadlock", "Synchronization", "Polymorphism", "Indexing", "Virtual Memory", "Optimization"]
        advanced_subtopics = [s for s in subtopics if any(k.lower() in s.lower() for k in advanced_keywords)]
        
        if advanced_subtopics:
            subtopic = random.choice(advanced_subtopics)
        else:
            subtopic = random.choice(subtopics) if subtopics else topic
        
        return self.generate_question(topic, subtopic, "hard")
    
    def _get_subtopics(self, topic: str) -> list:
        """Get subtopics for a topic"""
        return self.subtopics_by_topic.get(topic, [])
    
    def reset_tracker(self, user_id: int = None):
        """Reset intent and embedding trackers (useful for testing or reset)"""
        self.intent_tracker = {}
        self.embedding_history = {}
        print("🔄 Question bank trackers reset")