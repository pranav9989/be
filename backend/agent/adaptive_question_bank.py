# backend/agent/adaptive_question_bank.py

import random
import re
from typing import Dict, List, Set, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rag import retrieve_similar_qas


class SemanticDeduplicator:
    """Simple semantic deduplication for questions"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.90
    
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
    
    MAX_RETRIES = 3
    MAX_CHARS = 400
    DUP_THRESHOLD = 0.85
    
    # INTENT POOLS BY DIFFICULTY - ONLY DIFFICULTY-BASED INTENTS
    INTENT_POOLS = {
        # EASY
        "easy": [
            "core_definition",        # What is X?
            "identification",         # Identify component
            "listing",                # List types
        ],
        
        # MEDIUM
        "medium": [
            "core_definition",        # What is X?
            "mechanism_flow",         # How does it work? (simple)
            "conceptual_difference",  # X vs Y (basic comparison)
            "application",            # Simple application
        ],
        
        # HARD 
        "hard": [
            "mechanism_flow",         # How does it work? (detailed)
            "conceptual_difference",  # X vs Y (detailed)
            "application",            # Application with context
            "component_relationship", # How components interact
        ]
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
    
    # Class-level cache for embedder (shared across instances)
    _shared_embedder = None
    
    def __init__(self):
        # Initialize deduplicator
        self.dedup = SemanticDeduplicator()
        
        # Initialize sentence transformer for semantic duplicate detection (cached)
        if AdaptiveQuestionBank._shared_embedder is None:
            print("🔄 Loading sentence transformer for question bank (cached)...")
            AdaptiveQuestionBank._shared_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedder = AdaptiveQuestionBank._shared_embedder
        
        # RESTRUCTURED TAXONOMY - Atomic subtopics with internal concepts
        self.taxonomy = {
            "topics": [
                {
                    "name": "DBMS",
                    "subtopics": [
                        {"name": "Normalization", "concepts": [
                            "1NF", "2NF", "3NF", "BCNF",
                            "functional dependency",
                            "normalization",
                            "denormalization"
                        ]},
                        {"name": "Keys", "concepts": [
                            "primary key", "foreign key",
                            "candidate key", "composite key",
                            "referential integrity"
                        ]},
                        {"name": "Joins", "concepts": [
                            "inner join", "left join", "right join",
                            "self join", "subquery", "join vs subquery"
                        ]},
                        {"name": "SQL", "concepts": [
                            "group by", "having",
                            "order of execution",
                            "subquery",
                            "query optimization basics"
                        ]},
                        {"name": "Indexing", "concepts": [
                            "clustered index", "non-clustered index",
                            "B+ tree", "index scan", "full table scan"
                        ]},
                        {"name": "ACID", "concepts": [
                            "atomicity", "consistency",
                            "isolation", "durability"
                        ]},
                        {"name": "Transactions", "concepts": [
                            "isolation levels",
                            "dirty read",
                            "non-repeatable read",
                            "phantom read"
                        ]},
                        {"name": "Locks", "concepts": [
                            "shared lock", "exclusive lock",
                            "two phase locking",
                            "lock granularity"
                        ]},
                        {"name": "Deadlocks", "concepts": [
                            "deadlock conditions",
                            "deadlock prevention",
                            "deadlock avoidance",
                            "wait for graph"
                        ]},
                        {"name": "Query Optimization", "concepts": [
                            "query plan",
                            "execution plan",
                            "cost based optimization"
                        ]},
                        {"name": "Distributed DB", "concepts": [
                            "CAP theorem",
                            "two phase commit",
                            "replication",
                            "sharding"
                        ]},
                        {"name": "NoSQL", "concepts": [
                            "sql vs nosql",
                            "document store",
                            "key value store",
                            "column family",
                            "graph database"
                        ]}
                    ]
                },
                {
                    "name": "OOPS",
                    "subtopics": [
                        {"name": "Core OOP", "concepts": [
                            "encapsulation",
                            "abstraction",
                            "inheritance",
                            "polymorphism",
                            "method overloading",
                            "method overriding"
                        ]},
                        {"name": "Constructors", "concepts": [
                            "constructor",
                            "destructor",
                            "constructor overloading"
                        ]},
                        {"name": "Interfaces & Abstraction", "concepts": [
                            "abstract class",
                            "interface",
                            "abstract vs interface"
                        ]},
                        {"name": "Object Relationships", "concepts": [
                            "association",
                            "aggregation",
                            "composition"
                        ]},
                        {"name": "Virtual Functions", "concepts": [
                            "virtual function",
                            "runtime polymorphism",
                            "vtable"
                        ]},
                        {"name": "Design Patterns", "concepts": [
                            "singleton pattern",
                            "factory pattern",
                            "observer pattern"
                        ]},
                        {"name": "Memory Management", "concepts": [
                            "garbage collection",
                            "memory leak",
                            "stack memory",
                            "heap memory"
                        ]}
                    ]
                },
                {
                    "name": "OS",
                    "subtopics": [
                        {"name": "Processes & Threads", "concepts": [
                            "process",
                            "thread",
                            "process vs thread",
                            "context switching",
                            "multithreading"
                        ]},
                        {"name": "CPU Scheduling", "concepts": [
                            "FCFS",
                            "SJF",
                            "round robin",
                            "priority scheduling",
                            "preemptive scheduling",
                            "non-preemptive scheduling",
                            "starvation"
                        ]},
                        {"name": "Synchronization", "concepts": [
                            "mutex",
                            "semaphore",
                            "critical section",
                            "race condition"
                        ]},
                        {"name": "Deadlocks", "concepts": [
                            "mutual exclusion",
                            "hold and wait",
                            "no preemption",
                            "circular wait",
                            "banker's algorithm",
                            "resource allocation graph"
                        ]},
                        {"name": "Memory Management", "concepts": [
                            "paging",
                            "segmentation",
                            "fragmentation"
                        ]},
                        {"name": "Virtual Memory", "concepts": [
                            "page fault",
                            "thrashing",
                            "working set"
                        ]},
                        {"name": "Page Replacement", "concepts": [
                            "LRU",
                            "FIFO",
                            "optimal replacement"
                        ]},
                        {"name": "System Calls", "concepts": [
                            "fork",
                            "exec",
                            "wait",
                            "user mode",
                            "kernel mode"
                        ]},
                        {"name": "IPC", "concepts": [
                            "pipe",
                            "shared memory",
                            "message queue"
                        ]},
                        {"name": "I/O", "concepts": [
                            "interrupt",
                            "polling",
                            "DMA"
                        ]}
                    ]
                }
            ]
        }
        
        # Build lookup dictionaries
        self.subtopics_by_topic = {}
        self.subtopics_by_topic_upper = {}
        self.subtopics_by_topic_lower = {}
        self.subtopic_concepts = {}
        
        for topic in self.taxonomy["topics"]:
            topic_name = topic["name"]
            subtopics = [st["name"] for st in topic["subtopics"]]
            
            self.subtopics_by_topic[topic_name] = subtopics
            self.subtopics_by_topic_upper[topic_name.upper()] = subtopics
            self.subtopics_by_topic_lower[topic_name.lower()] = subtopics
            
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
        
        # Intent tracking per subtopic (for used intents only)
        self.intent_tracker = {}  # {topic: {subtopic: [intents_used]}}
        
        # Hard Duplicate Avoidance Memory
        self.embedding_history = {}  # {topic: {subtopic: [embeddings]}}
        
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

    def _normalize_concept(self, concept: str) -> str:
        """Normalize concept names for better matching"""
        normalized = concept.replace("/", " ").replace("-", " ")
        normalized = ' '.join(normalized.split())
        return normalized.strip()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for robust concept matching"""
        text = text.lower()
        text = text.replace("/", " ")
        text = text.replace("-", " ")
        text = text.replace("_", " ")
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _concept_present_semantic(self, question: str, concept: str, threshold: float = 0.65) -> bool:
        """Check if concept is semantically present in question"""
        question_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
        concept_emb = self.embedder.encode([concept], normalize_embeddings=True)[0]
        
        similarity = cosine_similarity([question_emb], [concept_emb])[0][0]
        
        if similarity > threshold:
            print(f"      🔍 Semantic concept match: '{concept}' (similarity: {similarity:.3f})")
            return True
        return False
    
    def _select_intent_by_difficulty(self, difficulty: str, used_intents: List[str]) -> str:
        """
        Select intent based on difficulty level ONLY
        Easy → definitions/terminology pool
        Medium → mechanisms/flows/comparisons pool
        Hard → reasoning/analysis pool
        """
        # INTENT POOLS BY DIFFICULTY
                # INTENT POOLS BY DIFFICULTY - SHIFTED DOWN
        INTENT_POOLS = {
            "easy": [
                "core_definition",
                "identification",
                "listing",
            ],
            "medium": [
                "core_definition",
                "mechanism_flow",
                "conceptual_difference",
                "application",
            ],
            "hard": [
                "mechanism_flow",
                "conceptual_difference",
                "application",
                "component_relationship",
            ]
        }
        
        # Get the pool for the requested difficulty
        pool = INTENT_POOLS.get(difficulty, INTENT_POOLS["medium"])
        
        # Filter out already used intents if possible
        available = [i for i in pool if i not in used_intents]
        
        # If all intents in this pool have been used, allow repeats
        if not available:
            available = pool
            
        chosen = random.choice(available)
        print(f"      🎯 Selected intent '{chosen}' from {difficulty} pool")
        return chosen
    
    def _get_question_number(self, topic: str, subtopic: str) -> int:
        """Get the current question number (1-3) for this subtopic"""
        if topic in self.intent_tracker and subtopic in self.intent_tracker[topic]:
            return len(self.intent_tracker[topic][subtopic]) + 1
        return 1
    
    def _get_intent_guidance(self, intent: str, concept1: str, concept2: str) -> str:
        """Get intent-specific guidance for question generation"""
        guidance = {
            "optimization_reasoning": f"""Style: OPTIMIZATION REASONING
Ask how to optimize performance, efficiency, or resource usage.
Example: "How would you optimize a system that uses {concept1} and {concept2} for better throughput?""",
            "tradeoff_analysis": f"""Style: TRADEOFF ANALYSIS
Ask to compare pros/cons, when to use each, or design choices.
Example: "Compare the tradeoffs between using {concept1} versus {concept2} in a high-concurrency system.""",
            "debugging_case": f"""Style: DEBUGGING CASE
Present a failure scenario and ask for debugging approach.
Example: "A system using {concept1} and {concept2} is experiencing deadlocks. How would you debug this?""",
            "prediction": f"""Style: PREDICTION
Ask to predict behavior under certain conditions.
Example: "What would happen if {concept1} and {concept2} are used together in a distributed system?""",
            "design_decision": f"""Style: DESIGN DECISION
Ask about architectural or implementation choices.
Example: "Design a solution using {concept1} and {concept2}. What key decisions must you make?""",
            "core_definition": f"""Style: CORE DEFINITION
Ask for definitions, but phrase creatively.
Example: "In your own words, what are {concept1} and {concept2} and why are they important?""",
            "conceptual_difference": f"""Style: CONCEPTUAL DIFFERENCE
Ask how concepts differ or relate.
Example: "How does {concept1} differ from {concept2} in practice?""",
            "mechanism_flow": f"""Style: MECHANISM FLOW
Ask how things work together step by step.
Example: "Walk me through how {concept1} and {concept2} interact during a typical operation.""",
            "edge_case": f"""Style: EDGE CASE
Ask about boundary conditions or unusual scenarios.
Example: "What edge cases should you consider when implementing {concept1} with {concept2}?""",
            "misconception_check": f"""Style: MISCONCEPTION CHECK
Address common misunderstandings.
Example: "A common misconception is that {concept1} and {concept2} are interchangeable. Why is this incorrect?"""
        }
        return guidance.get(intent, f"Style: {intent.upper()}\nAsk about {concept1} and {concept2}.")
    
    def _fallback_by_intent(self, intent: str, concept1: str, concept2: str, subtopic: str) -> str:
        """Intent-aware fallback question generation"""
        templates = {
            "optimization_reasoning": f"How would you optimize a system using {concept1} and {concept2} in {subtopic}?",
            "tradeoff_analysis": f"Compare the tradeoffs between using {concept1} versus {concept2} in {subtopic}.",
            "debugging_case": f"A system using {concept1} and {concept2} is failing. How would you debug it?",
            "prediction": f"What would happen if {concept1} interacts with {concept2} under heavy load?",
            "design_decision": f"Design a system using {concept1} and {concept2}. What key design decisions must be made?",
            "core_definition": f"What are {concept1} and {concept2} and how do they relate in {subtopic}?",
            "conceptual_difference": f"What is the difference between {concept1} and {concept2} in {subtopic}?",
            "mechanism_flow": f"Explain how {concept1} and {concept2} work together in {subtopic}.",
            "edge_case": f"What edge cases should you consider when using {concept1} and {concept2} together?",
            "misconception_check": f"A common misconception is that {concept1} and {concept2} are the same. Explain why this is incorrect."
        }
        return templates.get(intent, f"How do {concept1} and {concept2} work together in {subtopic}?")
    
    def fallback_question(self, topic: str, subtopic: str, concepts: List[str], intent: str) -> str:
        """Guaranteed fallback that ALWAYS includes BOTH concepts"""
        c1 = concepts[0] if len(concepts) > 0 else ""
        c2 = concepts[1] if len(concepts) > 1 else subtopic
        
        templates = {
            "core_definition": f"What is {c1} and how does it relate to {c2} in {subtopic}?",
            "mechanism_flow": f"Explain step-by-step how {c1} interacts with {c2} in {subtopic}.",
            "conceptual_difference": f"What is the difference between {c1} and {c2} in {subtopic}?",
            "application": f"How would you apply {c1} and {c2} together in a real {subtopic} scenario?",
            "debugging_case": f"A system fails due to improper {c1}. How would you debug it considering {c2} in {subtopic}?",
            "optimization_reasoning": f"How would you optimize a system using {c1} and {c2} together in {subtopic}?",
            "prediction": f"What would happen if {c1} fails while {c2} is active in {subtopic}?",
            "tradeoff_analysis": f"What are the tradeoffs when using {c1} versus {c2} in {subtopic}?",
            "design_decision": f"Design a solution using {c1} and {c2} in {subtopic}. What are your key decisions?",
            "edge_case": f"What edge cases should you consider when using {c1} with {c2} in {subtopic}?",
            "misconception_check": f"A common misconception is that {c1} and {c2} are the same in {subtopic}. Why is this incorrect?",
            "real_world_scenario": f"Describe a real-world scenario where both {c1} and {c2} are crucial in {subtopic}.",
            "problem_case": f"Solve a problem involving both {c1} and {c2} in the context of {subtopic}.",
            "cause_effect": f"What causes {c1} to affect {c2} in {subtopic} and what are the consequences?",
            "component_relationship": f"How does {c1} relate to {c2} in the overall architecture of {subtopic}?",
            "abstraction": f"Explain the abstract relationship between {c1} and {c2} in {subtopic}.",
            "complex_scenario": f"In a complex {subtopic} scenario involving {c1} and {c2}, what factors would you consider?"
        }
        
        fallback = templates.get(intent, f"Explain how {c1} and {c2} work together in {subtopic}?")
        
        # Ensure both concepts are in the question
        q_lower = fallback.lower()
        if c1.lower() not in q_lower and c2.lower() not in q_lower:
            fallback = f"Explain the relationship between {c1} and {c2} in the context of {subtopic}."
        elif c1.lower() not in q_lower:
            fallback = fallback.replace(c2, f"{c1} and {c2}")
        elif c2.lower() not in q_lower:
            fallback = fallback.replace(c1, f"{c1} and {c2}")
        
        print(f"   🎯 Fallback question generated: '{fallback}'")
        return fallback
    
    def generate_question_with_rag(
        self,
        topic: str,
        subtopic: str,
        concepts: List[str],
        difficulty: str,
        used_intents: List[str],
        history: List[str],
        user_name: str = ""
    ) -> tuple:
        """
        Generate question using RAG few-shot examples
        INVARIANT: Exactly 2 concepts must be present in question
        INVARIANT: Intent must match difficulty level
        """
        # INVARIANT: Exactly 2 concepts
        assert len(concepts) == 2, f"INVARIANT FAILED: Expected 2 concepts, got {len(concepts)}"
        
        # Select intent by difficulty
        intent = self._select_intent_by_difficulty(difficulty, used_intents)
        
        # Normalize concepts for better matching
        def normalize_concept(c):
            return c.replace("/", " ").replace("-", " ").strip()
        
        concepts_norm = [normalize_concept(c) for c in concepts]
        
        # Build query for similar example retrieval
        query = f"{topic} {subtopic} {concepts_norm[0]} {concepts_norm[1]} {intent} {difficulty} interview question"
        
        # Retrieve similar Q&A examples from FAISS
        examples = retrieve_similar_qas(query, topic=topic, k=1)
        
        # Build examples text
        examples_text = ""
        if examples and len(examples) > 0:
            examples_text = "\nReference example:\n"
            for i, ex in enumerate(examples):
                q_text = ex.get('question', '')[:150]
                examples_text += f"Example: {q_text}\n"
            print(f"   📚 Using {len(examples)} reference example(s)")
        
        # Get concept-specific context
        context = ""
        try:
            from rag import retrieve_relevant_chunks
            concept_query = f"{topic} {subtopic} {concepts_norm[0]} {concepts_norm[1]}"
            chunks = retrieve_relevant_chunks(concept_query, k=3, topic=topic)
            if chunks:
                context_parts = []
                for chunk in chunks:
                    chunk_text = chunk.get("answer", chunk.get("text", ""))
                    if chunk_text:
                        context_parts.append(chunk_text[:200])
                context = "\n".join(context_parts[:2])
                print(f"   📚 Added concept-specific context ({len(context)} chars)")
        except Exception as e:
            print(f"   ⚠️ Could not get concept context: {e}")
        
        # Build strict prompt with concept injection and difficulty guidance
        personalization = f" for {user_name}" if user_name else ""
        
        # 🔥 UPDATED PROMPT WITH DIFFICULTY GUIDANCE
        prompt = f"""You are an expert technical interviewer{personalization}.

    Generate exactly ONE interview question.

    TOPIC: {topic}
    SUBTOPIC: {subtopic}

    TARGET CONCEPTS (must appear in question):
    {concepts_norm[0]}, {concepts_norm[1]}

    QUESTION INTENT:
    {intent}

    DIFFICULTY LEVEL:
    {difficulty}

    IMPORTANT DIFFICULTY RULES:

    If difficulty = EASY:
    - Ask simple definition or identification questions
    - Do NOT ask design or scenario questions
    - Examples: "What is X?", "Define X", "List the types of X"
    - Focus on basic recall and recognition

    If difficulty = MEDIUM:
    - Ask explanation or comparison questions
    - Avoid system design and complex scenarios
    - Examples: "Explain how X works", "What is the difference between X and Y?"
    - Focus on understanding relationships

    If difficulty = HARD:
    - Ask mechanism or interaction questions
    - But avoid system architecture questions
    - Examples: "How do X and Y interact?", "Explain the relationship between X and Y"
    - Focus on process and interaction

    STRICT RULES:
    1. The question MUST contain BOTH concepts: "{concepts_norm[0]}" and "{concepts_norm[1]}"
    2. Keep question under 40 words
    3. Avoid system design prompts like "How would you design..."
    4. Avoid multi-step scenarios
    5. Focus on conceptual understanding
    6. Return ONLY the question text, no explanations
    7. End with a question mark '?'

    Knowledge context:
    {context}

    Generate ONE interview question only:"""

        # Try multiple attempts
        for attempt in range(self.MAX_RETRIES):
            try:
                from rag import generate_interview_question
                raw = generate_interview_question(prompt, topic)
                
                if not raw:
                    print(f"   ⚠️ Attempt {attempt + 1}: No response, retrying...")
                    continue
                
                # Clean the question
                question = raw.strip()
                
                # Remove common prefixes
                for prefix in ["Question:", "Q:", "Interview Question:", "Answer:", "A:", "Here's a question:", "Here is a question:", "Question :", "Q :"]:
                    if question.startswith(prefix):
                        question = question[len(prefix):].strip()
                
                # Find first sentence with question mark
                if "?" in question:
                    q_pos = question.index('?')
                    start = max(question.rfind('.', 0, q_pos), question.rfind('!', 0, q_pos), question.rfind('\n', 0, q_pos)) + 1
                    if start > 0:
                        question = question[start:q_pos+1].strip()
                    else:
                        question = question[:q_pos+1].strip()
                
                # Apply standard cleaning
                question = self._clean_question(question)
                question = self._make_verbal_safe(question)
                
                # HARD LENGTH ENFORCEMENT
                if len(question) > 300:
                    print(f"   ⚠️ Attempt {attempt + 1}: Too long ({len(question)} chars > 300) - rejecting")
                    continue
                
                if len(question) < 40:
                    print(f"   ⚠️ Attempt {attempt + 1}: Too short ({len(question)} chars < 40) - rejecting")
                    continue
                
                # Validate concept inclusion
                if not self.validate_question(question, concepts_norm):
                    print(f"   ⚠️ Attempt {attempt + 1}: Missing concepts - retrying")
                    continue
                
                # Check for duplicates
                if question in history:
                    print(f"   ⚠️ Attempt {attempt + 1}: Duplicate in session")
                    continue
                
                if self._is_duplicate(topic, subtopic, question):
                    print(f"   ⚠️ Attempt {attempt + 1}: Semantic duplicate")
                    continue
                
                print(f"\n   ✅ Generated with RAG: {question[:100]}... (length: {len(question)} chars)")
                return question, intent
                
            except Exception as e:
                print(f"   ⚠️ Attempt {attempt + 1} error: {e}")
                continue
        
        # Enhanced intent-aware fallback
        fallback = self.fallback_question(topic, subtopic, concepts_norm, intent)
        print(f"   ⚠️ Using enhanced intent-aware fallback: {fallback}")
        return fallback, intent
    
    def get_concepts_for_subtopic(self, topic: str, subtopic: str) -> list:
        """Get the internal concepts for a given topic and subtopic"""
        key = f"{topic}:{subtopic}"
        return self.subtopic_concepts.get(key, [])
    
    def _get_concepts_for_subtopic(self, topic: str, subtopic: str) -> list:
        """Internal method to get concepts"""
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
            print(f"   ⚠️ Not enough concepts ({len(all_concepts)}), padding with subtopic")
            return all_concepts + [subtopic] * (2 - len(all_concepts))
        
        # Score each concept
        concept_scores = []
        for concept in all_concepts:
            if concept in mastery_state.concepts:
                score = mastery_state.concepts[concept].priority_score
                print(f"      📊 {concept}: priority={score:.2f}")
            else:
                score = 1.5
                print(f"      📊 {concept}: NEW (priority=1.5)")
            concept_scores.append((concept, score))
        
        # Sort by priority (highest first)
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take EXACTLY 2 concepts
        sampled = [concept_scores[0][0], concept_scores[1][0]]
        print(f"      ✅ Selected by priority: {sampled}")
        return sampled
    
    def _get_next_question_type(self, topic: str, subtopic: str) -> str:
        """Get next question type for variety"""
        if topic not in self.question_type_tracker:
            self.question_type_tracker[topic] = {}
        
        if subtopic not in self.question_type_tracker[topic]:
            self.question_type_tracker[topic][subtopic] = []
        
        used_types = self.question_type_tracker[topic][subtopic]
        
        if len(used_types) >= len(self.QUESTION_TYPES):
            used_types = []
            self.question_type_tracker[topic][subtopic] = []
        
        available = [t for t in self.QUESTION_TYPES if t not in used_types]
        
        if available:
            chosen = random.choice(available)
        else:
            chosen = random.choice(self.QUESTION_TYPES)
        
        self.question_type_tracker[topic][subtopic].append(chosen)
        return chosen
    
    def _is_duplicate(self, topic: str, subtopic: str, question: str) -> bool:
        """Hard duplicate avoidance using semantic embeddings"""
        new_emb = self.embedder.encode([question])[0]
        
        if topic not in self.embedding_history:
            self.embedding_history[topic] = {}
        
        if subtopic not in self.embedding_history[topic]:
            self.embedding_history[topic][subtopic] = []
        
        history = self.embedding_history[topic][subtopic]
        
        for old_emb in history:
            similarity = cosine_similarity([new_emb], [old_emb])[0][0]
            if similarity > self.DUP_THRESHOLD:
                print(f"   🔍 Duplicate detected (similarity: {similarity:.3f})")
                return True
        
        history.append(new_emb)
        if len(history) > 3:
            history.pop(0)
        
        return False
    
    def _make_verbal_safe(self, question: str) -> str:
        """Convert to verbal-friendly phrasing"""
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
        
        if not sampled_concepts or len(sampled_concepts) < 2:
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
        
        question_type_guidance = {
            "definition": f"Focus on definitions of {concept1} and {concept2}",
            "comparison": f"Compare and contrast {concept1} and {concept2}",
            "scenario": f"Present a real-world scenario involving {concept1} and {concept2}",
            "code": f"Ask for code example demonstrating {concept1} and {concept2}",
            "debugging": f"Present a debugging scenario involving {concept1} and {concept2}",
            "optimization": f"Focus on performance and optimization of {concept1} and {concept2}"
        }
        
        type_guidance = question_type_guidance.get(question_type, f"Focus on {concept1} and {concept2}")
        
        context_section = ""
        if weak_concepts and len(weak_concepts) > 0:
            context_section = f"\nThe candidate has shown weakness in: {', '.join(weak_concepts[:3])}"
        elif user_context:
            context_section = f"\n{user_context}"
        
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
        """Legacy prompt builder - kept for backward compatibility"""
        return self._build_prompt_strict(
            topic, subtopic, intent, difficulty, user_name,
            weak_concepts, question_type, user_context, sampled_concepts
        )
    
    def _enforce_length(self, question: str) -> str:
        """Enforce length limit"""
        if len(question) <= self.MAX_CHARS:
            return question
        return None
    
    def _clean_question(self, text: str) -> str:
        """Clean question text"""
        text = re.sub(r"^(Question:|Interviewer:|AI:|Assistant:|Here'?s (a|the) question:)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(Okay|Alright|Sure|Let me|I'll|Let's).*?[:.]", "", text, flags=re.IGNORECASE)
        text = text.strip()
        
        text = re.sub(r"[`*_#]", "", text)
        text = text.strip('"\'')
        
        if "?" in text:
            q_pos = text.index('?')
            start = max(text.rfind('.', 0, q_pos), text.rfind('!', 0, q_pos), text.rfind('\n', 0, q_pos)) + 1
            if start > 0:
                text = text[start:q_pos+1].strip()
            else:
                text = text[:q_pos+1].strip()
        else:
            text += "?"
        
        return text
    
    def validate_question(self, question: str, concepts: List[str]) -> bool:
        """Robust concept validation using semantic similarity"""
        print(f"\n   🔍 Validating question for concepts: {concepts}")
        
        def normalize_concept(c):
            return c.replace("/", " ").replace("-", " ").lower().strip()
        
        q_lower = question.lower()
        
        missing = []
        present = []
        
        for concept in concepts:
            c_norm = normalize_concept(concept)
            
            if c_norm in q_lower:
                print(f"      ✓ '{concept}' found directly")
                present.append(concept)
                continue
            
            c_no_space = c_norm.replace(" ", "")
            q_no_space = q_lower.replace(" ", "")
            if c_no_space in q_no_space:
                print(f"      ✓ '{concept}' found without spaces")
                present.append(concept)
                continue
            
            try:
                from sentence_transformers import util
                
                q_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
                c_emb = self.embedder.encode([concept], normalize_embeddings=True)[0]
                
                similarity = float(util.cos_sim(q_emb, c_emb)[0][0])
                
                if similarity >= 0.6:
                    print(f"      ✓ '{concept}' semantically present (sim: {similarity:.3f})")
                    present.append(concept)
                    continue
                else:
                    print(f"      ⚠️ '{concept}' semantic similarity low: {similarity:.3f}")
                    missing.append(concept)
            except Exception as e:
                print(f"      ⚠️ Semantic check failed for '{concept}': {e}")
                missing.append(concept)
        
        if len(present) == 0:
            print(f"   ❌ Validation failed - no concepts found at all")
            return False
        
        if len(present) < len(concepts):
            print(f"   ⚠️ Partial match: found {len(present)}/{len(concepts)} concepts")
            return True
        
        print(f"   ✓ All {len(concepts)} concepts present")
        return True
    
    def generate_question_with_sampled_concepts(self, session_id: str = None, topic: str = None, 
                                           subtopic: str = None, difficulty: str = "medium", 
                                           user_name: str = "", weak_concepts: list = None,
                                           user_context: str = "", history: List[str] = None) -> Tuple[str, list]:
        """
        Generate question AND return the sampled concepts
        Returns (question_text, sampled_concepts)
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
        
        # Use provided weak_concepts if available
        if weak_concepts and len(weak_concepts) >= 2:
            sampled_concepts = weak_concepts[:2]
            print(f"\n   🎯 Using provided weak concepts: {sampled_concepts}")
        else:
            if len(concepts) >= 2:
                sampled_concepts = concepts[:2]
                print(f"\n   ⚠️ No mastery state, using first concepts: {sampled_concepts}")
            else:
                sampled_concepts = concepts + [subtopic] * (2 - len(concepts))
                print(f"\n   ⚠️ Fallback with padding: {sampled_concepts}")
        
        # For backward compatibility, use a default intent based on question number
        # but note: actual intent selection happens in generate_question_with_rag
        if question_number == 1:
            intent_display = "core_definition"
        elif question_number == 2:
            intent_display = "mechanism_flow"
        else:
            intent_display = "conceptual_difference"
        print(f"   📍 Expected intent based on Q{question_number}: {intent_display}")
        
        # Generate question for these concepts
        for attempt in range(self.MAX_RETRIES):
            question_type = self._get_next_question_type(topic, subtopic)
            
            # We'll use a placeholder intent here - actual intent will come from difficulty in generate_question
            prompt = self._build_prompt_strict(
                topic, subtopic, "core_definition", difficulty, 
                user_name, weak_concepts, question_type,
                user_context, sampled_concepts
            )
            
            try:
                from rag import generate_interview_question as generate_rag_response
                raw = generate_rag_response(prompt, topic)
                
                if not raw:
                    print(f"   ⚠️ Attempt {attempt + 1}: No response, retrying...")
                    continue
                
                question = self._clean_question(raw)
                
                if len(question) > 300:
                    lines = question.split('\n')
                    for line in lines:
                        line = line.strip()
                        if '?' in line and len(line) < 200:
                            question = line
                            break
                
                question = self._make_verbal_safe(question)
                
                if len(question) > self.MAX_CHARS:
                    print(f"   ⚠️ Attempt {attempt + 1}: Question too long ({len(question)} chars), retrying...")
                    continue
                
                if not self.validate_question(question, sampled_concepts):
                    print(f"   ⚠️ Attempt {attempt + 1}: Missing concepts - retrying")
                    continue
                
                if question in history:
                    print(f"   ⚠️ Attempt {attempt + 1}: Exact duplicate in session, retrying...")
                    continue
                
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
        Uses RAG few-shot retrieval for better quality
        INVARIANT: Intent selected by difficulty level
        """
        if history is None:
            history = []
        
        print("\n" + "─"*70)
        print("📝 GENERATE QUESTION (RAG FEW-SHOT)")
        print("─"*70)
        print(f"   Topic:     {topic}")
        print(f"   Subtopic:  {subtopic}")
        print(f"   Concepts:  {concepts[0]}, {concepts[1]}")
        print(f"   Difficulty:{difficulty.upper()}")
        
        # Get used intents for this subtopic
        used_intents = []
        if topic in self.intent_tracker and subtopic in self.intent_tracker[topic]:
            used_intents = self.intent_tracker[topic][subtopic]
        
        # Generate using RAG
        question, intent = self.generate_question_with_rag(
            topic=topic,
            subtopic=subtopic,
            concepts=concepts,
            difficulty=difficulty,
            used_intents=used_intents,
            history=history,
            user_name=user_name
        )
        
        # Track the intent
        if topic not in self.intent_tracker:
            self.intent_tracker[topic] = {}
        if subtopic not in self.intent_tracker[topic]:
            self.intent_tracker[topic][subtopic] = []
        self.intent_tracker[topic][subtopic].append(intent)
        
        print("─"*70)
        return question
    
    # Legacy methods for backward compatibility
    def generate_question_legacy(self, topic: str, subtopic: str, difficulty: str = "medium", 
                                user_name: str = "", weak_concepts: list = None,
                                user_context: str = "") -> str:
        """Legacy method - kept for backward compatibility"""
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
            forced_intents = ["misconception_check", "debugging_case", "problem_case"]
            intent = random.choice(forced_intents)
            
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
            
            if missing_concepts and len(missing_concepts) > 0:
                return f"Can you explain how {missing_concepts[0]} works in the context of {current_subtopic}?"
            return f"Can you explain how {current_subtopic} works in practice?"
        
        if missing_concepts and len(missing_concepts) > 0:
            return f"Let's focus on {missing_concepts[0]}. Can you explain that concept?"
        return f"Let's focus on {topic}. Can you explain the core concepts?"
    
    def generate_simplified_question(self, topic: str, missing_concepts: list) -> str:
        """Generate a simpler question for struggling users"""
        subtopic = missing_concepts[0] if missing_concepts else topic
        
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
        advanced_intents = ["optimization_reasoning", "tradeoff_analysis", "edge_case"]
        intent = random.choice(advanced_intents)
        
        subtopics = self._get_subtopics(topic)
        advanced_keywords = ["Deadlocks", "Synchronization", "Polymorphism", "Indexing", "Virtual Memory", "Optimization"]
        advanced_subtopics = [s for s in subtopics if any(k.lower() in s.lower() for k in advanced_keywords)]
        
        if advanced_subtopics:
            subtopic = random.choice(advanced_subtopics)
        else:
            subtopic = random.choice(subtopics) if subtopics else topic
        
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
        """Generate question with specific user context"""
        concepts = self.get_concepts_for_subtopic(topic, subtopic)
        if len(concepts) >= 2:
            sampled = concepts[:2]
        else:
            sampled = concepts + [subtopic] * (2 - len(concepts))
        
        # Use default intent for context-based questions
        intent = "core_definition"
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
        """Reset intent and embedding trackers"""
        self.intent_tracker = {}
        self.embedding_history = {}
        self.question_type_tracker = {}
        print("🔄 Question bank trackers reset")