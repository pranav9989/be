# backend/agent/adaptive_question_bank.py

from rag import generate_technical_explanation as generate_rag_response
import random
import re
import numpy as np
from typing import Dict, List, Set, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AdaptiveQuestionBank:
    """Advanced adaptive question bank with intent-based rotation and duplicate prevention"""
    
    MAX_CHARS = 400
    MAX_RETRIES = 5
    DUP_THRESHOLD = 0.82  # Slightly lower than semantic_dedup for more variety
    
    # Intent-Based Question Rotation
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
    
    # Question types for dynamic prompting
    QUESTION_TYPES = [
        "definition",
        "comparison",
        "scenario",
        "code",
        "debugging",
        "optimization"
    ]
    
    # Subtopic Normalization (concept clusters)
    SUBTOPIC_CLUSTERS = {
        "Polymorphism": [
            "Polymorphism",
            "method overloading",
            "method overriding",
            "runtime binding",
            "compile-time binding",
            "dynamic dispatch"
        ],
        "Memory Management": [
            "Memory Management",
            "paging",
            "segmentation",
            "Virtual Memory",
            "page replacement"
        ],
        "Process Management": [
            "Processes",
            "Threads",
            "process states",
            "PCB",
            "Context Switching"
        ],
        "Synchronization": [
            "Synchronization",
            "mutex",
            "semaphore",
            "monitor",
            "critical section"
        ],
        "Database Design": [
            "Normalization",
            "Keys",
            "functional dependency",
            "anomalies"
        ],
        "Query Optimization": [
            "Indexing",
            "Joins",
            "SQL Aggregation",
            "query optimization"
        ],
        "Transaction Management": [
            "ACID",
            "Transactions",
            "Concurrency Control",
            "Isolation Levels",
            "Locking",
            "Deadlocks"
        ]
    }
    
    # Intent progression mapping for 3-question sequence
    INTENT_PROGRESSION = {
        1: "core_definition",
        2: "mechanism_flow",
        3: "conceptual_difference"
    }
    
    def __init__(self):
        # Initialize sentence transformer for semantic duplicate detection
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # RESTRUCTURED TAXONOMY - Atomic subtopics with internal concepts
        self.taxonomy = {
            "topics": [
                {
                    "name": "DBMS",
                    "subtopics": [
                        {
                            "name": "Normalization",
                            "concepts": ["1NF", "2NF", "3NF", "BCNF", "functional dependency", "anomalies"]
                        },
                        {
                            "name": "Keys",
                            "concepts": ["Primary Key", "Foreign Key", "Candidate Key", "Composite Key", "Super Key"]
                        },
                        {
                            "name": "ACID",
                            "concepts": ["Atomicity", "Consistency", "Isolation", "Durability"]
                        },
                        {
                            "name": "Transactions",
                            "concepts": ["commit", "rollback", "transaction states", "savepoint"]
                        },
                        {
                            "name": "Concurrency Control",
                            "concepts": ["2PL", "timestamp ordering", "optimistic locking", "pessimistic locking"]
                        },
                        {
                            "name": "Isolation Levels",
                            "concepts": ["Read Uncommitted", "Read Committed", "Repeatable Read", "Serializable", "dirty read", "non-repeatable read", "phantom read"]
                        },
                        {
                            "name": "Indexing",
                            "concepts": ["B+ Tree", "Hash Index", "clustered index", "non-clustered index", "composite index"]
                        },
                        {
                            "name": "Joins",
                            "concepts": ["Inner Join", "Left Join", "Right Join", "Full Join", "Self Join", "Natural Join", "Equi Join", "cross join"]
                        },
                        {
                            "name": "SQL Aggregation",
                            "concepts": ["GROUP BY", "HAVING", "Subqueries", "COUNT", "SUM", "AVG", "MIN", "MAX", "correlated subquery"]
                        },
                        {
                            "name": "Locking",
                            "concepts": ["Shared Lock", "Exclusive Lock", "Lock Granularity", "row lock", "table lock", "deadlock"]
                        },
                        {
                            "name": "Deadlocks",
                            "concepts": ["Wait-for graph", "detection", "prevention", "avoidance", "banker's algorithm"]
                        }
                    ]
                },
                {
                    "name": "OOPS",
                    "subtopics": [
                        {
                            "name": "Classes",
                            "concepts": ["class structure", "attributes", "methods", "static members", "instance members"]
                        },
                        {
                            "name": "Objects",
                            "concepts": ["instantiation", "state", "behavior", "identity", "object lifecycle"]
                        },
                        {
                            "name": "Encapsulation",
                            "concepts": ["data hiding", "getters/setters", "access control", "information hiding"]
                        },
                        {
                            "name": "Abstraction",
                            "concepts": ["abstract classes", "interfaces", "implementation hiding", "contract"]
                        },
                        {
                            "name": "Inheritance",
                            "concepts": ["single inheritance", "multiple inheritance", "multilevel inheritance", "diamond problem", "base class", "derived class"]
                        },
                        {
                            "name": "Polymorphism",
                            "concepts": ["method overloading", "method overriding", "runtime binding", "compile-time binding", "dynamic dispatch", "duck typing", "virtual functions"]
                        },
                        {
                            "name": "Constructors",
                            "concepts": ["default constructor", "parameterized constructor", "copy constructor", "constructor overloading", "destructor"]
                        },
                        {
                            "name": "Access Modifiers",
                            "concepts": ["public", "private", "protected", "default", "package-private"]
                        },
                        {
                            "name": "SOLID Principles",
                            "concepts": ["Single Responsibility", "Open/Closed", "Liskov Substitution", "Interface Segregation", "Dependency Inversion"]
                        }
                    ]
                },
                {
                    "name": "OS",
                    "subtopics": [
                        {
                            "name": "Processes",
                            "concepts": ["process states", "PCB", "process creation", "process termination", "zombie process", "orphan process"]
                        },
                        {
                            "name": "Threads",
                            "concepts": ["user threads", "kernel threads", "multithreading", "thread pool", "green threads"]
                        },
                        {
                            "name": "Context Switching",
                            "concepts": ["CPU state saving", "overhead", "mode switch", "dispatch latency"]
                        },
                        {
                            "name": "CPU Scheduling",
                            "concepts": ["FCFS", "SJF", "Round Robin", "Priority", "preemptive", "non-preemptive", "multilevel queue", "multilevel feedback queue"]
                        },
                        {
                            "name": "Synchronization",
                            "concepts": ["mutex", "semaphore", "monitor", "critical section", "race condition", "spinlock"]
                        },
                        {
                            "name": "Deadlocks",
                            "concepts": ["mutual exclusion", "hold and wait", "no preemption", "circular wait", "prevention", "avoidance", "detection", "recovery"]
                        },
                        {
                            "name": "Memory Management",
                            "concepts": ["paging", "segmentation", "internal fragmentation", "external fragmentation", "memory allocation"]
                        },
                        {
                            "name": "Virtual Memory",
                            "concepts": ["demand paging", "page faults", "page replacement", "thrashing", "working set"]
                        },
                        {
                            "name": "Page Replacement",
                            "concepts": ["LRU", "FIFO", "Optimal", "clock algorithm", "second chance"]
                        },
                        {
                            "name": "System Calls",
                            "concepts": ["fork", "exec", "wait", "open", "read", "write", "close", "pipe", "kill"]
                        }
                    ]
                }
            ]
        }
        
        # Build lookup dictionaries
        self.subtopics_by_topic = {}
        self.subtopics_by_topic_upper = {}
        self.subtopics_by_topic_lower = {}
        self.subtopic_concepts = {}  # Store concepts per subtopic
        
        for topic in self.taxonomy["topics"]:
            topic_name = topic["name"]
            subtopics = [st["name"] for st in topic["subtopics"]]
            
            self.subtopics_by_topic[topic_name] = subtopics
            self.subtopics_by_topic_upper[topic_name.upper()] = subtopics
            self.subtopics_by_topic_lower[topic_name.lower()] = subtopics
            
            # Store concepts for each subtopic
            for subtopic in topic["subtopics"]:
                key = f"{topic_name}:{subtopic['name']}"
                self.subtopic_concepts[key] = subtopic["concepts"]
        
        # Add common variations for backward compatibility
        self.subtopics_by_topic["OOP"] = self.subtopics_by_topic["OOPS"]
        self.subtopics_by_topic["OOPs"] = self.subtopics_by_topic["OOPS"]
        self.subtopics_by_topic["Operating System"] = self.subtopics_by_topic["OS"]
        self.subtopics_by_topic["Operating Systems"] = self.subtopics_by_topic["OS"]
        self.subtopics_by_topic["Database"] = self.subtopics_by_topic["DBMS"]
        self.subtopics_by_topic["Databases"] = self.subtopics_by_topic["DBMS"]
        
        print(f"✅ RESTRUCTURED taxonomy loaded successfully with atomic subtopics:")
        for topic_name in ["DBMS", "OOPS", "OS"]:
            subtopics = self.subtopics_by_topic.get(topic_name, [])
            print(f"   - {topic_name}: {len(subtopics)} atomic subtopics")
            
        # Hard Duplicate Avoidance Memory
        # Stores last 3 semantic embeddings per subtopic
        self.embedding_history = {}  # {topic: {subtopic: [embeddings]}}
        
        # Intent tracking per subtopic
        self.intent_tracker = {}  # {topic: {subtopic: [intents_used]}}
        
        # Question type tracking for variety
        self.question_type_tracker = {}  # {topic: {subtopic: [types_used]}}
        
        # Verbal-safe phrase replacements
        self.verbal_phrases = {
            r"show me": "describe",
            r"demonstrate": "explain",
            r"write code": "explain conceptually",
            r"implement": "describe the implementation approach",
            r"draw": "describe",
            r"diagram": "describe"
        }
    
    def _get_concepts_for_subtopic(self, topic: str, subtopic: str) -> list:
        """Get the internal concepts for a given topic and subtopic"""
        key = f"{topic}:{subtopic}"
        return self.subtopic_concepts.get(key, [])
    
    def _get_question_number(self, topic: str, subtopic: str) -> int:
        """Get the current question number (1-3) for this subtopic based on intent tracker"""
        if topic in self.intent_tracker and subtopic in self.intent_tracker[topic]:
            return len(self.intent_tracker[topic][subtopic]) + 1
        return 1
    
    def _normalize_subtopic(self, topic: str, subtopic: str) -> str:
        """Map detailed subtopic to its conceptual cluster"""
        for cluster_name, members in self.SUBTOPIC_CLUSTERS.items():
            if subtopic in members or any(m in subtopic for m in members):
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
        question_number = len(used_intents) + 1
        
        # Force progression for first 3 questions
        if question_number <= 3:
            return self.INTENT_PROGRESSION[question_number]
        
        # If we've used all intents, reset and start over
        if len(used_intents) >= len(self.INTENTS):
            used_intents = []
            self.intent_tracker[topic][subtopic] = []
        
        # Available intents (not used yet)
        available = [i for i in self.INTENTS if i not in used_intents]
        
        # For beyond 3 questions, use deep intents
        deep_intents = [i for i in available if i in ["edge_case", "debugging_case", "misconception_check", "optimization_reasoning"]]
        if deep_intents:
            chosen = random.choice(deep_intents)
        else:
            chosen = random.choice(available) if available else random.choice(self.INTENTS)
        
        # Record this intent
        self.intent_tracker[topic][subtopic].append(chosen)
        return chosen
    
    def _get_next_question_type(self, topic: str, subtopic: str) -> str:
        """Get next question type for variety"""
        
        # Initialize tracker
        if topic not in self.question_type_tracker:
            self.question_type_tracker[topic] = {}
        
        if subtopic not in self.question_type_tracker[topic]:
            self.question_type_tracker[topic][subtopic] = []
        
        used_types = self.question_type_tracker[topic][subtopic]
        
        # If we've used all types, reset
        if len(used_types) >= len(self.QUESTION_TYPES):
            used_types = []
            self.question_type_tracker[topic][subtopic] = []
        
        # Available types
        available = [t for t in self.QUESTION_TYPES if t not in used_types]
        
        if available:
            chosen = random.choice(available)
        else:
            chosen = random.choice(self.QUESTION_TYPES)
        
        # Record
        self.question_type_tracker[topic][subtopic].append(chosen)
        return chosen
    
    def _is_duplicate(self, topic: str, subtopic: str, question: str) -> bool:
        """
        Hard duplicate avoidance using semantic embeddings
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
                return True
        
        # Add to history (keep last 3)
        history.append(new_emb)
        if len(history) > 3:
            history.pop(0)
        
        return False
    
    def _make_verbal_safe(self, question: str) -> str:
        """Convert to verbal-friendly phrasing"""
        import re
        
        # Replace visual/action phrases with verbal ones
        for pattern, replacement in self.verbal_phrases.items():
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question
    
    def _build_prompt(self, topic: str, subtopic: str, intent: str, 
                     difficulty: str = "medium", user_name: str = "",
                     weak_concepts: list = None, question_type: str = None,
                     user_context: str = "") -> str:
        """
        Build prompt with intent, concepts, and user context
        Uses at most 2 concepts for sharper, more focused questions
        """
        
        personalization = f" for {user_name}" if user_name else ""
        
        # Get internal concepts for this subtopic
        concepts = self._get_concepts_for_subtopic(topic, subtopic)
        
        # CRITICAL: Use at most 2 concepts for sharp focus
        focus_concepts = []
        concepts_str = ""
        
        if concepts:
            # If we have weak concepts, prioritize those
            if weak_concepts and len(weak_concepts) > 0:
                # Find which weak concepts are in our concept list
                relevant_weak = [c for c in weak_concepts if c in concepts]
                if relevant_weak:
                    # Use up to 2 weak concepts
                    focus_concepts = relevant_weak[:2]
                    concepts_str = ", ".join(focus_concepts)
                    print(f"      📍 Targeting weak concepts in prompt: {focus_concepts}")
                else:
                    # No relevant weak concepts, sample randomly
                    sample_size = min(2, len(concepts))
                    focus_concepts = random.sample(concepts, sample_size)
                    concepts_str = ", ".join(focus_concepts)
                    print(f"      🎲 Random sampling (no weak match): {focus_concepts}")
            else:
                # No weak concepts, sample randomly (max 2)
                sample_size = min(2, len(concepts))
                focus_concepts = random.sample(concepts, sample_size)
                concepts_str = ", ".join(focus_concepts)
                print(f"      🎲 Random sampling (no weak concepts): {focus_concepts}")
        else:
            concepts_str = subtopic
            focus_concepts = [subtopic]
            print(f"      ⚠️ No concepts found for {topic}:{subtopic}")
        
        # Determine question type if not provided
        if not question_type:
            question_type = self._get_next_question_type(topic, subtopic)
        
        intent_descriptions = {
            "core_definition": f"Ask for the fundamental definition or core concept of {subtopic}",
            "conceptual_difference": f"Ask how concepts within {subtopic} differ from each other or from related concepts",
            "mechanism_flow": f"Ask about the internal mechanism or workflow of {subtopic}",
            "real_world_scenario": f"Ask for a real-world application scenario using {subtopic}",
            "problem_case": f"Present a problem that requires understanding of {subtopic} to solve",
            "edge_case": f"Ask about edge cases or unusual situations related to {subtopic}",
            "tradeoff_analysis": f"Ask about tradeoffs, pros and cons in {subtopic}",
            "debugging_case": f"Present a debugging scenario related to {subtopic}",
            "optimization_reasoning": f"Ask about optimization strategies for {subtopic}",
            "misconception_check": f"Address a common misconception about {subtopic}"
        }
        
        intent_desc = intent_descriptions.get(intent, f"Ask a technical question about {subtopic}")
        
        # Question type guidance
        question_type_guidance = {
            "definition": "Focus on definitions and core concepts",
            "comparison": "Ask to compare different concepts or approaches",
            "scenario": "Present a real-world scenario and ask how to apply the concept",
            "code": "Ask for code or pseudo-code implementation",
            "debugging": "Present broken code or scenario and ask to debug",
            "optimization": "Focus on performance and optimization"
        }
        
        type_guidance = question_type_guidance.get(question_type, "")
        
        # Add user context if available (weak concepts, previous mistakes)
        context_section = ""
        if weak_concepts and len(weak_concepts) > 0:
            context_section = f"\nThe candidate has shown weakness in: {', '.join(weak_concepts[:3])}"
        elif user_context:
            context_section = f"\n{user_context}"
        
        prompt = f"""
You are a technical interviewer conducting a job interview{personalization}.

Generate ONE interview question about:
Topic: {topic}
Subtopic: {subtopic}
Key concepts within this subtopic: {concepts_str}
Intent: {intent}
Question Type: {question_type}
Difficulty: {difficulty}{context_section}

INTENT GUIDANCE:
{intent_desc}

QUESTION TYPE GUIDANCE:
{type_guidance}

STRICT RULES:
- Ask ONLY the question - NO introductions, NO commentary, NO explanations
- NO phrases like "let's talk about", "I'd like to ask", "here's a question"
- NO "show me" - use "describe" or "explain" instead
- Question must be answerable VERBALLY in an interview
- Question must be UNDER {self.MAX_CHARS} characters
- Focus specifically on {subtopic} and its concepts: {concepts_str}
- Be direct and conversational, like a real interviewer
- Do NOT simply list the concepts - create a meaningful question around them

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
        
        # Remove quotes
        text = text.strip('"\'')
        
        # Ensure it's a question
        if "?" in text:
            # Take everything up to and including first question mark
            text = text[:text.index("?")+1]
        else:
            # Add question mark if missing
            text += "?"
        
        return text
    
    def generate_question(self, topic: str, subtopic: str, difficulty: str = "medium", 
                         user_name: str = "", weak_concepts: list = None,
                         user_context: str = "") -> str:
        """
        Main question generation method with full pipeline:
        1. Intent selection with progression
        2. Question type selection for variety
        3. Prompt building with internal concepts
        4. RAG generation
        5. Duplicate check
        6. Length enforcement
        7. Verbal-safe conversion
        
        Now uses atomic subtopics with internal concepts for better questions
        """
        
        print("\n" + "─"*70)
        print("📝 QUESTION GENERATION")
        print("─"*70)
        print(f"   Topic:     {topic}")
        print(f"   Subtopic:  {subtopic}")
        print(f"   Difficulty:{difficulty.upper()}")
        
        # 🔥 CRITICAL DEBUG: Show if weak concepts are received
        if weak_concepts:
            print(f"   🎯 Weak concepts provided: {weak_concepts[:5]}")
        else:
            print(f"   ⚠️ No weak concepts provided")
        
        question_number = self._get_question_number(topic, subtopic)
        print(f"   Question #:{question_number}/3")
        
        # Show concept pool
        concepts = self._get_concepts_for_subtopic(topic, subtopic)
        print(f"\n   Concept pool:")
        for i, concept in enumerate(concepts, 1):
            print(f"      {i}. {concept}")
        
        # Show concept sampling logic
        focus_concepts = []
        if weak_concepts:
            relevant_weak = [c for c in weak_concepts if c in concepts]
            if relevant_weak:
                focus_concepts = relevant_weak[:2]
                print(f"\n   🎯 Targeting weaknesses:")
                for concept in focus_concepts:
                    print(f"      - {concept} (weak)")
            else:
                sample_size = min(2, len(concepts))
                focus_concepts = random.sample(concepts, sample_size)
                print(f"\n   🎲 Random sampling (no weak concepts in this subtopic):")
                for concept in focus_concepts:
                    print(f"      - {concept}")
        else:
            sample_size = min(2, len(concepts))
            focus_concepts = random.sample(concepts, sample_size)
            print(f"\n   🎲 Random sampling:")
            for concept in focus_concepts:
                print(f"      - {concept}")
        
        # Show intent progression
        intent = self._get_next_intent(topic, subtopic)
        intent_map = {
            "core_definition": "1️⃣ DEFINITION (What is it?)",
            "mechanism_flow": "2️⃣ MECHANISM (How does it work?)",
            "conceptual_difference": "3️⃣ COMPARISON (How are they different?)",
            "real_world_scenario": "🌍 REAL-WORLD SCENARIO",
            "problem_case": "🔧 PROBLEM CASE",
            "edge_case": "⚠️ EDGE CASE",
            "tradeoff_analysis": "⚖️ TRADEOFF ANALYSIS",
            "debugging_case": "🐛 DEBUGGING CASE",
            "optimization_reasoning": "⚡ OPTIMIZATION REASONING",
            "misconception_check": "❓ MISCONCEPTION CHECK"
        }
        
        print(f"\n   Intent:    {intent_map.get(intent, intent)}")
        
        for attempt in range(self.MAX_RETRIES):
            # Get question type for variety
            question_type = self._get_next_question_type(topic, subtopic)
            
            # Build prompt with internal concepts AND WEAK CONCEPTS
            prompt = self._build_prompt(
                topic, subtopic, intent, difficulty, 
                user_name, weak_concepts, question_type,
                user_context
            )
            
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
                    print(f"   ⚠️ Attempt {attempt + 1}: Question too long, retrying...")
                    continue
                
                # Check for duplicates
                if not self._is_duplicate(topic, subtopic, question):
                    print(f"\n   ✅ Generated: {question}")
                    print("─"*70)
                    return question
                else:
                    print(f"   🔄 Attempt {attempt + 1}: Duplicate detected, retrying with different intent...")
                    
            except Exception as e:
                print(f"   ⚠️ Generation attempt {attempt + 1} failed: {e}")
                continue
        
        # Hard fallback - simple template question using concepts
        if concepts:
            concept_sample = random.choice(concepts)
            fallback = f"Can you explain {concept_sample} and how it applies to {subtopic} in {topic}?"
        else:
            fallback = f"What are the key concepts and practical applications of {subtopic} in {topic}?"
        
        print(f"   ⚠️ All {self.MAX_RETRIES} attempts failed, using fallback")
        print(f"\n   ✅ Generated (fallback): {fallback}")
        print("─"*70)
        return fallback
    
    # 🔥 FIXED: All wrappers now accept and pass weak_concepts
    def generate_first_question(self, topic: str, subtopic: str = None, 
                               difficulty: str = "medium", user_name: str = "",
                               weak_concepts: list = None) -> str:
        """Wrapper for first question in a topic with weak concept targeting"""
        if subtopic is None:
            subtopics = self.subtopics_by_topic.get(topic, [])
            subtopic = random.choice(subtopics) if subtopics else "core concepts"
        
        return self.generate_question(
            topic, subtopic, difficulty, user_name, 
            weak_concepts=weak_concepts
        )
    
    def generate_question_for_subtopic(self, topic: str, subtopic: str, 
                                      difficulty: str = "medium",
                                      weak_concepts: list = None) -> str:
        """Wrapper for subtopic-specific question with weak concept targeting"""
        return self.generate_question(
            topic, subtopic, difficulty, 
            weak_concepts=weak_concepts
        )
    
    def generate_gap_followup(self, topic: str, missing_concepts: list, 
                             difficulty: str = "medium", 
                             current_subtopic: str = None, 
                             available_subtopics: list = None) -> str:
        """Generate follow-up targeting missing concepts"""
        
        if current_subtopic:
            # Use a different intent for gap follow-up
            forced_intents = ["misconception_check", "debugging_case", "problem_case"]
            intent = random.choice(forced_intents)
            
            # Pass missing concepts as weak concepts
            prompt = self._build_prompt(
                topic, current_subtopic, intent, difficulty,
                weak_concepts=missing_concepts
            )
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
            
            # Fallback with concept-specific question
            if missing_concepts and len(missing_concepts) > 0:
                return f"Can you explain how {missing_concepts[0]} works in the context of {current_subtopic}?"
            return f"Can you explain how {current_subtopic} works in practice?"
        
        if missing_concepts and len(missing_concepts) > 0:
            return f"Let's focus on {missing_concepts[0]}. Can you explain that concept?"
        return f"Let's focus on {topic}. Can you explain the core concepts?"
    
    def generate_simplified_question(self, topic: str, missing_concepts: list) -> str:
        """Generate a simpler question for struggling users"""
        subtopic = missing_concepts[0] if missing_concepts else topic
        
        # Get concepts for simpler targeting
        concepts = self._get_concepts_for_subtopic(topic, subtopic)
        simple_concept = concepts[0] if concepts else subtopic
        
        prompt = f"""
You are a helpful tutor. Ask a VERY SIMPLE question about {simple_concept} in {topic}.
Use plain language. Make it easy to answer.
Focus on the basic definition or core idea.
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
        
        return f"What is {simple_concept} in simple terms?"
    
    def generate_deeper_dive(self, topic: str, difficulty: str = "hard") -> str:
        """Generate challenging question for strong performers"""
        # Force advanced intents
        advanced_intents = ["optimization_reasoning", "tradeoff_analysis", "edge_case"]
        intent = random.choice(advanced_intents)
        
        # Pick an advanced subtopic
        subtopics = self._get_subtopics(topic)
        advanced_keywords = ["Deadlocks", "Synchronization", "Polymorphism", "Indexing", "Virtual Memory", "Optimization"]
        advanced_subtopics = [s for s in subtopics if any(k.lower() in s.lower() for k in advanced_keywords)]
        
        if advanced_subtopics:
            subtopic = random.choice(advanced_subtopics)
        else:
            subtopic = random.choice(subtopics) if subtopics else topic
        
        return self.generate_question(topic, subtopic, "hard")
    
    def generate_question_with_context(self, topic: str, subtopic: str, 
                                      user_context: str, difficulty: str = "medium") -> str:
        """Generate question with specific user context (previous mistakes, weak areas)"""
        return self.generate_question(topic, subtopic, difficulty, 
                                     user_context=user_context)
    
    def _get_subtopics(self, topic: str) -> list:
        """Get subtopics for a topic"""
        return self.subtopics_by_topic.get(topic, [])
    
    def get_concepts_for_subtopic(self, topic: str, subtopic: str) -> list:
        """Public method to get concepts for a subtopic"""
        return self._get_concepts_for_subtopic(topic, subtopic)
    
    def reset_tracker(self, user_id: int = None):
        """Reset intent and embedding trackers (useful for testing or reset)"""
        self.intent_tracker = {}
        self.embedding_history = {}
        self.question_type_tracker = {}
        print("🔄 Question bank trackers reset")