# backend/agent/adaptive_question_bank.py

import random
import re
from typing import Dict, List, Set, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticDeduplicator:
    """Simple semantic deduplication for questions"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.85
    
    def is_duplicate(self, session_id: str, new_question: str, existing_questions: List[str]) -> bool:
        """Check if question is semantically similar to existing ones"""
        if not existing_questions:
            return False
        
        new_emb = self.embedder.encode([new_question])[0]
        
        for existing in existing_questions:
            existing_emb = self.embedder.encode([existing])[0]
            similarity = cosine_similarity([new_emb], [existing_emb])[0][0]
            if similarity > self.similarity_threshold:
                return True
        
        return False


class AdaptiveQuestionBank:
    """
    Advanced adaptive question bank with priority-based concept sampling
    INVARIANT 2: Questions MUST contain sampled concepts
    INVARIANT 3: Concept sampling uses priority_score sorting (never random)
    """
    
    MAX_RETRIES = 5
    MAX_CHARS = 400
    DUP_THRESHOLD = 0.85  # For semantic duplicate detection
    
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
    
    # Intent progression for 3-question sequence
    INTENT_PROGRESSION = {
        1: "core_definition",      # Q1: Establish foundation
        2: "mechanism_flow",        # Q2: Test understanding of how it works
        3: "conceptual_difference"  # Q3: Deepen with comparisons
    }
    
    # Question types for dynamic prompting
    QUESTION_TYPES = [
        "definition",
        "comparison",
        "scenario",
        "code",
        "debugging",
        "optimization"
    ]
    
    def __init__(self):
        # Initialize deduplicator
        self.dedup = SemanticDeduplicator()
        
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
    
    def get_concepts_for_subtopic(self, topic: str, subtopic: str) -> list:
        """Get the internal concepts for a given topic and subtopic"""
        key = f"{topic}:{subtopic}"
        return self.subtopic_concepts.get(key, [])
    
    def _get_concepts_for_subtopic(self, topic: str, subtopic: str) -> list:
        """Internal method to get concepts (alias for get_concepts_for_subtopic)"""
        return self.get_concepts_for_subtopic(topic, subtopic)
    
    def sample_concepts_by_priority(self, topic: str, subtopic: str, mastery_state) -> list:
        """
        STRICT RULE:
        - ALWAYS sample exactly 2 concepts
        - ALWAYS highest priority first
        - NEVER random
        """
        all_concepts = self.get_concepts_for_subtopic(topic, subtopic)
        
        if len(all_concepts) < 2:
            # Not enough concepts, pad with subtopic
            print(f"   ⚠️ Not enough concepts ({len(all_concepts)}), padding with subtopic")
            return all_concepts + [subtopic] * (2 - len(all_concepts))
        
        # Score each concept
        concept_scores = []
        for concept in all_concepts:
            if concept in mastery_state.concepts:
                # Use existing concept's priority score
                score = mastery_state.concepts[concept].priority_score
                print(f"      📊 {concept}: priority={score:.2f}")
            else:
                # New concept - medium priority
                score = 1.5
                print(f"      📊 {concept}: NEW (priority=1.5)")
            concept_scores.append((concept, score))
        
        # INVARIANT 3: Sort by priority (highest first) - NO RANDOM
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take EXACTLY 2 concepts (highest priority)
        sampled = [concept_scores[0][0], concept_scores[1][0]]
        print(f"      ✅ Selected by priority: {sampled}")
        return sampled
    
    def _get_question_number(self, topic: str, subtopic: str) -> int:
        """Get the current question number (1-3) for this subtopic based on intent tracker"""
        if topic in self.intent_tracker and subtopic in self.intent_tracker[topic]:
            return len(self.intent_tracker[topic][subtopic]) + 1
        return 1
    
    def _get_next_intent(self, topic: str, subtopic: str) -> str:
        """
        Enforce intent progression per rules:
        Q1: core_definition
        Q2: mechanism_flow
        Q3: conceptual_difference
        """
        # Initialize tracker
        if topic not in self.intent_tracker:
            self.intent_tracker[topic] = {}
        
        if subtopic not in self.intent_tracker[topic]:
            self.intent_tracker[topic][subtopic] = []
        
        used_intents = self.intent_tracker[topic][subtopic]
        question_number = len(used_intents) + 1
        
        # INTENT PROGRESSION per rules (EXACTLY)
        if question_number == 1:
            intent = "core_definition"
        elif question_number == 2:
            intent = "mechanism_flow"
        elif question_number == 3:
            intent = "conceptual_difference"
        else:
            # Beyond 3 questions (rare), use advanced intents
            advanced_intents = ["edge_case", "debugging_case", "misconception_check", 
                              "optimization_reasoning", "tradeoff_analysis"]
            # Reset if we've used all
            if len(used_intents) >= len(self.INTENTS):
                self.intent_tracker[topic][subtopic] = []
                used_intents = []
            
            available = [i for i in advanced_intents if i not in used_intents]
            if available:
                intent = random.choice(available)
            else:
                # If all advanced used, pick any not used
                all_available = [i for i in self.INTENTS if i not in used_intents]
                intent = random.choice(all_available) if all_available else random.choice(self.INTENTS)
        
        # Record this intent
        self.intent_tracker[topic][subtopic].append(intent)
        
        print(f"   📍 Intent for Q{question_number}: {intent}")
        return intent
    
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
        # Replace visual/action phrases with verbal ones
        for pattern, replacement in self.verbal_phrases.items():
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question
    
    def _build_prompt_strict(self, topic: str, subtopic: str, intent: str, 
                            difficulty: str = "medium", user_name: str = "",
                            weak_concepts: list = None, question_type: str = None,
                            user_context: str = "", sampled_concepts: list = None) -> str:
        """
        STRICT prompt builder - MUST include both sampled concepts
        """
        personalization = f" for {user_name}" if user_name else ""
        
        # CRITICAL: Must have sampled concepts
        if not sampled_concepts or len(sampled_concepts) < 2:
            # Fallback - should never happen in production
            concepts = self.get_concepts_for_subtopic(topic, subtopic)
            if len(concepts) >= 2:
                sampled_concepts = concepts[:2]
            else:
                sampled_concepts = concepts + [subtopic] * (2 - len(concepts))
        
        concept1 = sampled_concepts[0]
        concept2 = sampled_concepts[1]
        
        print(f"      📍 Building STRICT prompt for: {concept1} and {concept2}")
        
        intent_descriptions = {
            "core_definition": f"Ask for definition of {concept1} and {concept2}",
            "conceptual_difference": f"Ask how {concept1} differs from {concept2}",
            "mechanism_flow": f"Ask how {concept1} and {concept2} work together",
            "real_world_scenario": f"Ask for scenario using {concept1} and {concept2}",
            "problem_case": f"Present problem involving {concept1} and {concept2}",
            "edge_case": f"Ask about edge cases with {concept1} and {concept2}",
            "tradeoff_analysis": f"Ask about tradeoffs between {concept1} and {concept2}",
            "debugging_case": f"Present debugging scenario with {concept1} and {concept2}",
            "optimization_reasoning": f"Ask about optimizing {concept1} and {concept2}",
            "misconception_check": f"Address misconception about {concept1} and {concept2}"
        }
        
        intent_desc = intent_descriptions.get(intent, f"Ask about {concept1} and {concept2}")
        
        # Question type guidance
        question_type_guidance = {
            "definition": f"Focus on definitions of {concept1} and {concept2}",
            "comparison": f"Compare and contrast {concept1} and {concept2}",
            "scenario": f"Present a real-world scenario involving {concept1} and {concept2}",
            "code": f"Ask for code example demonstrating {concept1} and {concept2}",
            "debugging": f"Present a debugging scenario involving {concept1} and {concept2}",
            "optimization": f"Focus on performance and optimization of {concept1} and {concept2}"
        }
        
        type_guidance = question_type_guidance.get(question_type, f"Focus on {concept1} and {concept2}")
        
        # Add user context if available (weak concepts, previous mistakes)
        context_section = ""
        if weak_concepts and len(weak_concepts) > 0:
            context_section = f"\nThe candidate has shown weakness in: {', '.join(weak_concepts[:3])}"
        elif user_context:
            context_section = f"\n{user_context}"
        
        # STRICT RULES - MUST include both concepts
        prompt = f"""
You are a technical interviewer conducting a job interview{personalization}.

STRICT REQUIREMENTS - MUST FOLLOW EXACTLY:

1. The question MUST explicitly mention BOTH of these concepts:
   - "{concept1}"
   - "{concept2}"

2. The question MUST directly test understanding of how these concepts relate

3. Do NOT ask about other concepts - focus ONLY on {concept1} and {concept2}

Topic: {topic}
Subtopic: {subtopic}
Intent: {intent} - {intent_desc}
Question Type: {question_type} - {type_guidance}
Difficulty: {difficulty}{context_section}

Generate ONE technical interview question that:
- Explicitly includes "{concept1}" and "{concept2}" in the text
- Tests understanding of both concepts
- Is answerable verbally in an interview
- Is under {self.MAX_CHARS} characters

Question:
"""
        return prompt
    
    def _build_prompt(self, topic: str, subtopic: str, intent: str, 
                     difficulty: str = "medium", user_name: str = "",
                     weak_concepts: list = None, question_type: str = None,
                     user_context: str = "", sampled_concepts: list = None) -> str:
        """
        Legacy prompt builder - kept for backward compatibility
        Delegates to strict version
        """
        return self._build_prompt_strict(
            topic, subtopic, intent, difficulty, user_name,
            weak_concepts, question_type, user_context, sampled_concepts
        )
    
    def _enforce_length(self, question: str) -> str:
        """Enforce length limit - reject if too long, don't truncate"""
        if len(question) <= self.MAX_CHARS:
            return question
        return None  # Signal to retry
    
    def _clean_question(self, text: str) -> str:
        """Clean question text - aggressively remove any non-question content"""
        # Remove common prefixes
        text = re.sub(r"^(Question:|Interviewer:|AI:|Assistant:|Here'?s (a|the) question:)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(Okay|Alright|Sure|Let me|I'll|Let's).*?[:.]", "", text, flags=re.IGNORECASE)
        text = text.strip()
        
        # Remove markdown
        text = re.sub(r"[`*_#]", "", text)
        
        # Remove quotes
        text = text.strip('"\'')
        
        # Find the first question mark and take everything up to it
        if "?" in text:
            # Find the position of the first '?'
            q_pos = text.index('?')
            # Find the start of that sentence (previous '.', '!', or start of string)
            start = max(text.rfind('.', 0, q_pos), text.rfind('!', 0, q_pos), text.rfind('\n', 0, q_pos)) + 1
            if start > 0:
                text = text[start:q_pos+1].strip()
            else:
                text = text[:q_pos+1].strip()
        else:
            # Add question mark if missing
            text += "?"
        
        return text
    
    def validate_question(self, question: str, concepts: List[str]) -> bool:
        """
        INVARIANT 2: Validate that question contains both concepts
        Returns True if valid, False if invalid
        """
        question_lower = question.lower()
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Direct match
            if concept_lower in question_lower:
                continue
            
            # For multi-word concepts, check without spaces
            if ' ' in concept_lower:
                concept_no_space = concept_lower.replace(' ', '')
                if concept_no_space in question_lower.replace(' ', ''):
                    continue
            
            # Concept missing
            print(f"   ❌ Missing concept: '{concept}'")
            return False
        
        return True
    
    def generate_question_with_sampled_concepts(self, session_id: str = None, topic: str = None, 
                                           subtopic: str = None, difficulty: str = "medium", 
                                           user_name: str = "", weak_concepts: list = None,
                                           user_context: str = "", history: List[str] = None) -> Tuple[str, list]:
        """
        Generate question AND return the sampled concepts
        Returns (question_text, sampled_concepts)
        
        Args:
            session_id: Not used (kept for backward compatibility)
            topic: The topic (DBMS, OS, OOPS)
            subtopic: The subtopic within the topic
            difficulty: easy/medium/hard
            user_name: For personalization
            weak_concepts: Concepts to target (may be None)
            user_context: Additional context
            history: List of questions already asked in this session (for duplicate prevention)
        """
        if history is None:
            history = []
        
        print("\n" + "─"*70)
        print("📝 QUESTION GENERATION")
        print("─"*70)
        print(f"   Topic:     {topic}")
        print(f"   Subtopic:  {subtopic}")
        print(f"   Difficulty:{difficulty.upper()}")
        
        question_number = self._get_question_number(topic, subtopic)
        print(f"   Question #:{question_number}/3")
        
        # Get concept pool
        concepts = self.get_concepts_for_subtopic(topic, subtopic)
        print(f"\n   Concept pool: {concepts}")
        
        # Use provided weak_concepts if available, otherwise generate by priority
        if weak_concepts and len(weak_concepts) >= 2:
            sampled_concepts = weak_concepts[:2]
            print(f"\n   🎯 Using provided weak concepts: {sampled_concepts}")
        else:
            # We need mastery state for priority sampling, but if not available,
            # just use first two concepts as fallback
            if len(concepts) >= 2:
                sampled_concepts = concepts[:2]
                print(f"\n   ⚠️ No mastery state, using first concepts: {sampled_concepts}")
            else:
                sampled_concepts = concepts + [subtopic] * (2 - len(concepts))
                print(f"\n   ⚠️ Fallback with padding: {sampled_concepts}")
        
        # Get intent
        intent = self._get_next_intent(topic, subtopic)
        print(f"   📍 Intent: {intent}")
        
        # Generate question for these concepts
        for attempt in range(self.MAX_RETRIES):
            question_type = self._get_next_question_type(topic, subtopic)
            
            # Use the strict prompt builder
            prompt = self._build_prompt_strict(
                topic, subtopic, intent, difficulty, 
                user_name, weak_concepts, question_type,
                user_context, sampled_concepts
            )
            
            try:
                # Use the dedicated interview question generator
                from rag import generate_interview_question as generate_rag_response
                raw = generate_rag_response(prompt, topic)
                
                if not raw:
                    print(f"   ⚠️ Attempt {attempt + 1}: No response, retrying...")
                    continue
                
                # Clean the question
                question = self._clean_question(raw)
                
                # If question is too long, try to extract just the question part
                if len(question) > 300:
                    # Look for lines that might contain the question
                    lines = question.split('\n')
                    for line in lines:
                        line = line.strip()
                        if '?' in line and len(line) < 200:
                            # This might be the actual question
                            question = line
                            break
                
                question = self._make_verbal_safe(question)
                
                # Check length
                if len(question) > self.MAX_CHARS:
                    print(f"   ⚠️ Attempt {attempt + 1}: Question too long ({len(question)} chars), retrying...")
                    continue
                
                # INVARIANT 2: Validate concept inclusion
                if not self.validate_question(question, sampled_concepts):
                    print(f"   ⚠️ Attempt {attempt + 1}: Missing concepts, retrying...")
                    continue
                
                # Check for duplicates using history (current session)
                if question in history:
                    print(f"   ⚠️ Attempt {attempt + 1}: Exact duplicate in session, retrying...")
                    continue
                
                # Check for semantic duplicates within this subtopic (cross-session)
                if self._is_duplicate(topic, subtopic, question):
                    print(f"   ⚠️ Attempt {attempt + 1}: Semantic duplicate found, retrying...")
                    continue
                
                print(f"\n   ✅ Generated: {question}")
                print(f"   🎯 Sampled concepts: {sampled_concepts}")
                print(f"   ✓ Concept verification: Both present")
                print("─"*70)
                return question, sampled_concepts
                    
            except Exception as e:
                print(f"   ⚠️ Attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback
        if concepts and sampled_concepts:
            fallback = f"Explain how {sampled_concepts[0]} and {sampled_concepts[1]} work together in {subtopic}?"
        else:
            fallback = f"What are the key concepts of {subtopic} in {topic}?"
        
        print(f"   ⚠️ Using fallback after {self.MAX_RETRIES} attempts")
        print(f"\n   ✅ Generated: {fallback}")
        print(f"   🎯 Sampled concepts: {sampled_concepts}")
        print("─"*70)
        return fallback, sampled_concepts
    
    def generate_question(self, topic: str, subtopic: str, concepts: List[str], 
                         difficulty: str = "medium", user_name: str = "", 
                         history: List[str] = None) -> str:
        """
        Generate a question that MUST include both sampled concepts
        This is the main method called by the controller
        
        Args:
            topic: The topic (DBMS, OS, OOPS)
            subtopic: The subtopic within the topic
            concepts: List of 2 concepts that MUST appear in the question
            difficulty: easy/medium/hard
            user_name: For personalization
            history: List of questions already asked in this session
        """
        if history is None:
            history = []
        
        print("\n" + "─"*70)
        print("📝 GENERATE QUESTION (DIRECT)")
        print("─"*70)
        print(f"   Topic:     {topic}")
        print(f"   Subtopic:  {subtopic}")
        print(f"   Concepts:  {concepts[0]}, {concepts[1]}")
        print(f"   Difficulty:{difficulty.upper()}")
        
        concept1, concept2 = concepts[0], concepts[1]
        
        # Get intent based on question number
        intent = self._get_next_intent(topic, subtopic)
        print(f"   📍 Intent: {intent}")
        
        # Generate question using RAG with strict prompt
        for attempt in range(self.MAX_RETRIES):
            try:
                # Build strict prompt
                prompt = self._build_prompt_strict(
                    topic=topic,
                    subtopic=subtopic,
                    intent=intent,
                    difficulty=difficulty,
                    user_name=user_name,
                    weak_concepts=None,
                    question_type=None,
                    user_context="",
                    sampled_concepts=concepts
                )
                
                # Generate using RAG
                from rag import generate_interview_question
                raw = generate_interview_question(prompt, topic)
                
                if not raw:
                    print(f"   ⚠️ Attempt {attempt + 1}: No response, retrying...")
                    continue
                
                # Clean the question
                question = self._clean_question(raw)
                question = self._make_verbal_safe(question)
                
                # Check length
                if len(question) > self.MAX_CHARS:
                    print(f"   ⚠️ Attempt {attempt + 1}: Question too long ({len(question)} chars), retrying...")
                    continue
                
                # INVARIANT 2: Validate concept inclusion
                if not self.validate_question(question, concepts):
                    print(f"   ⚠️ Attempt {attempt + 1}: Missing concepts, retrying...")
                    continue
                
                # Check for duplicates in current session
                if question in history:
                    print(f"   ⚠️ Attempt {attempt + 1}: Duplicate in session, retrying...")
                    continue
                
                # Check for semantic duplicates across sessions
                if self._is_duplicate(topic, subtopic, question):
                    print(f"   ⚠️ Attempt {attempt + 1}: Semantic duplicate found, retrying...")
                    continue
                
                print(f"\n   ✅ Generated: {question}")
                print("─"*70)
                return question
                
            except Exception as e:
                print(f"   ⚠️ Attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback - should rarely happen
        fallback = f"Explain how {concept1} and {concept2} work together in {subtopic}?"
        print(f"   ⚠️ Using fallback after {self.MAX_RETRIES} attempts")
        print(f"   ✅ Generated: {fallback}")
        print("─"*70)
        return fallback
    
    # Legacy method for backward compatibility
    def generate_question_legacy(self, topic: str, subtopic: str, difficulty: str = "medium", 
                                user_name: str = "", weak_concepts: list = None,
                                user_context: str = "") -> str:
        """
        Legacy method - kept for backward compatibility
        """
        # For legacy calls, we need to get concepts somehow
        concepts = self.get_concepts_for_subtopic(topic, subtopic)
        if len(concepts) >= 2:
            sampled = concepts[:2]
        else:
            sampled = concepts + [subtopic] * (2 - len(concepts))
        
        return self.generate_question(
            topic=topic,
            subtopic=subtopic,
            concepts=sampled,
            difficulty=difficulty,
            user_name=user_name,
            history=[]
        )
    
    def generate_first_question(self, topic: str, subtopic: str = None, 
                               difficulty: str = "medium", user_name: str = "",
                               weak_concepts: list = None) -> str:
        """Wrapper for first question in a topic with weak concept targeting"""
        if subtopic is None:
            subtopics = self.subtopics_by_topic.get(topic, [])
            subtopic = random.choice(subtopics) if subtopics else "core concepts"
        
        # For first question, we need concepts
        if weak_concepts and len(weak_concepts) >= 2:
            concepts = weak_concepts[:2]
        else:
            concepts = self.get_concepts_for_subtopic(topic, subtopic)
            if len(concepts) >= 2:
                concepts = concepts[:2]
            else:
                concepts = concepts + [subtopic] * (2 - len(concepts))
        
        return self.generate_question(
            topic=topic,
            subtopic=subtopic,
            concepts=concepts,
            difficulty=difficulty,
            user_name=user_name,
            history=[]
        )
    
    def generate_question_for_subtopic(self, topic: str, subtopic: str, 
                                      difficulty: str = "medium",
                                      weak_concepts: list = None) -> str:
        """Wrapper for subtopic-specific question with weak concept targeting"""
        if weak_concepts and len(weak_concepts) >= 2:
            concepts = weak_concepts[:2]
        else:
            concepts = self.get_concepts_for_subtopic(topic, subtopic)
            if len(concepts) >= 2:
                concepts = concepts[:2]
            else:
                concepts = concepts + [subtopic] * (2 - len(concepts))
        
        return self.generate_question(
            topic=topic,
            subtopic=subtopic,
            concepts=concepts,
            difficulty=difficulty,
            user_name="",
            history=[]
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
            prompt = self._build_prompt_strict(
                topic, current_subtopic, intent, difficulty,
                weak_concepts=missing_concepts, sampled_concepts=missing_concepts[:2]
            )
            try:
                from rag import generate_interview_question as generate_rag_response
                raw = generate_rag_response(prompt, topic)
                if raw:
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
        concepts = self.get_concepts_for_subtopic(topic, subtopic)
        simple_concept = concepts[0] if concepts else subtopic
        
        prompt = f"""
You are a helpful tutor. Ask a VERY SIMPLE question about {simple_concept} in {topic}.
Use plain language. Make it easy to answer.
Focus on the basic definition or core idea.
Question must be under {self.MAX_CHARS} characters.
Return ONLY the question.
"""
        try:
            from rag import generate_interview_question as generate_rag_response
            raw = generate_rag_response(prompt, topic)
            if raw:
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
        
        # Get concepts for this subtopic
        concepts = self.get_concepts_for_subtopic(topic, subtopic)
        if len(concepts) >= 2:
            sampled = concepts[:2]
        else:
            sampled = concepts + [subtopic] * (2 - len(concepts))
        
        return self.generate_question(
            topic=topic,
            subtopic=subtopic,
            concepts=sampled,
            difficulty="hard",
            user_name="",
            history=[]
        )
    
    def generate_question_with_context(self, topic: str, subtopic: str, 
                                      user_context: str, difficulty: str = "medium") -> str:
        """Generate question with specific user context (previous mistakes, weak areas)"""
        concepts = self.get_concepts_for_subtopic(topic, subtopic)
        if len(concepts) >= 2:
            sampled = concepts[:2]
        else:
            sampled = concepts + [subtopic] * (2 - len(concepts))
        
        # For context, we need to build a custom prompt
        intent = self._get_next_intent(topic, subtopic)
        prompt = self._build_prompt_strict(
            topic, subtopic, intent, difficulty,
            user_context=user_context, sampled_concepts=sampled
        )
        
        try:
            from rag import generate_interview_question as generate_rag_response
            raw = generate_rag_response(prompt, topic)
            if raw:
                question = self._clean_question(raw)
                question = self._make_verbal_safe(question)
                if question and len(question) <= self.MAX_CHARS:
                    return question
        except:
            pass
        
        return f"Explain {subtopic} in {topic} focusing on {sampled[0]} and {sampled[1]}."
    
    def _get_subtopics(self, topic: str) -> list:
        """Get subtopics for a topic"""
        return self.subtopics_by_topic.get(topic, [])
    
    def reset_tracker(self, user_id: int = None):
        """Reset intent and embedding trackers (useful for testing or reset)"""
        self.intent_tracker = {}
        self.embedding_history = {}
        self.question_type_tracker = {}
        print("🔄 Question bank trackers reset")