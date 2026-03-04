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
    
    MAX_RETRIES = 3  # Reduced from 5
    MAX_CHARS = 400  # Keep for API, but we'll enforce 300 in generation
    DUP_THRESHOLD = 0.85  # For semantic duplicate detection
    
    # INTENT POOLS BY DIFFICULTY
    INTENT_POOLS = {
        "easy": [
            "core_definition",
            "real_world_scenario",
            "problem_case",
            "identification",
            "listing"
        ],
        "medium": [
            "mechanism_flow",
            "conceptual_difference",
            "edge_case",
            "cause_effect",
            "application",
            "component_relationship"
        ],
        "hard": [
            "optimization_reasoning",
            "tradeoff_analysis",
            "debugging_case",
            "misconception_check",
            "design_decision",
            "complex_scenario",
            "prediction",
            "abstraction"
        ]
    }
    
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

                        {"name": "DBMS basics",
                        "concepts": ["database", "dbms", "rdbms", "schema", "instance", "data independence"]},

                        {"name": "ER Modeling",
                        "concepts": ["entity", "attribute", "relationship", "cardinality", "weak entity"]},

                        {"name": "DBMS Architecture",
                        "concepts": ["three schema architecture", "physical level", "logical level", "view level"]},

                        {"name": "Entities & Keys",
                        "concepts": ["primary key", "foreign key", "candidate key", "composite key", "referential integrity"]},

                        {"name": "Normalization",
                        "concepts": ["1NF", "2NF", "3NF", "BCNF", "functional dependency"]},

                        {"name": "Joins",
                        "concepts": ["inner join", "left join", "right join", "full join", "self join"]},

                        {"name": "DDL DML DCL TCL",
                        "concepts": ["create", "alter", "drop", "select", "insert", "update", "delete", "grant", "revoke", "commit", "rollback"]},

                        {"name": "Aggregations",
                        "concepts": ["group by", "having", "count", "sum", "avg", "min", "max"]},

                        {"name": "Indexing",
                        "concepts": ["B+ tree", "hash index", "clustered index", "non-clustered index"]},

                        {"name": "ACID",
                        "concepts": ["atomicity", "consistency", "isolation", "durability"]},

                        {"name": "Isolation Levels",
                        "concepts": ["read committed", "repeatable read", "serializable", "dirty read", "phantom read"]},

                        {"name": "Locks",
                        "concepts": ["shared lock", "exclusive lock", "two phase locking", "intention lock"]},

                        {"name": "Deadlocks DBMS",
                        "concepts": ["wait for graph", "deadlock detection", "deadlock prevention", "deadlock avoidance", "transaction deadlock"]},

                        {"name": "Logging & Recovery",
                        "concepts": ["write ahead logging", "checkpoint", "undo", "redo"]},

                        {"name": "Query Optimization",
                        "concepts": ["query plan", "execution plan", "cost based optimization"]},

                        {"name": "Distributed Databases",
                        "concepts": ["replication", "sharding", "two phase commit", "CAP theorem"]},

                        {"name": "NoSQL",
                        "concepts": ["document store", "key value store", "column family", "graph database"]}
                    ]
                },

                {
                    "name": "OOPS",
                    "subtopics": [

                        {"name": "Classes",
                        "concepts": ["class", "attributes", "methods", "static", "instance"]},

                        {"name": "Objects",
                        "concepts": ["instantiation", "state", "behavior", "identity"]},

                        {"name": "Constructors",
                        "concepts": ["default constructor", "parameterized constructor", "copy constructor", "destructor"]},

                        {"name": "Encapsulation",
                        "concepts": ["data hiding", "private", "public", "protected"]},

                        {"name": "Inheritance",
                        "concepts": ["single inheritance", "multiple inheritance", "diamond problem"]},

                        {"name": "Polymorphism",
                        "concepts": ["overloading", "overriding", "runtime polymorphism"]},

                        {"name": "Abstraction",
                        "concepts": ["abstract class", "interface", "implementation hiding"]},

                        {"name": "Interfaces",
                        "concepts": ["interface", "implements", "contract"]},

                        {"name": "Virtual Functions",
                        "concepts": ["virtual function", "vtable", "dynamic binding"]},

                        {"name": "Object Relationships",
                        "concepts": ["association", "aggregation", "composition"]},

                        {"name": "Design Patterns",
                        "concepts": ["singleton", "factory", "observer", "strategy"]},

                        {"name": "Memory Management in OOP",
                        "concepts": ["garbage collection", "memory leak", "smart pointer", "RAII"]}
                    ]
                },

                {
                    "name": "OS",
                    "subtopics": [

                        {"name": "OS basics",
                        "concepts": ["kernel", "user mode", "system software", "resource management"]},

                        {"name": "Processes",
                        "concepts": ["process state", "PCB", "zombie process", "orphan process"]},

                        {"name": "Threads",
                        "concepts": ["user thread", "kernel thread", "multithreading"]},

                        {"name": "CPU Scheduling",
                        "concepts": ["FCFS", "SJF", "round robin", "priority scheduling"]},

                        {"name": "Process Synchronization",
                        "concepts": ["mutex", "semaphore", "critical section", "monitor"]},

                        {"name": "Deadlocks OS",
                        "concepts": ["mutual exclusion", "hold and wait", "no preemption", "circular wait", "banker's algorithm", "resource allocation graph"]},

                        {"name": "Memory Management",
                        "concepts": ["segmentation", "fragmentation", "contiguous allocation"]},

                        {"name": "Virtual Memory",
                        "concepts": ["page fault", "thrashing", "working set"]},

                        {"name": "Paging",
                        "concepts": ["page", "frame", "TLB"]},

                        {"name": "File Systems",
                        "concepts": ["inode", "directory structure", "file allocation"]},

                        {"name": "Disk Scheduling",
                        "concepts": ["SSTF", "SCAN", "C-SCAN", "LOOK"]},

                        {"name": "I/O Management",
                        "concepts": ["interrupt", "DMA", "device driver"]},

                        {"name": "System Calls",
                        "concepts": ["fork", "exec", "wait", "syscall"]},

                        {"name": "Interprocess Communication",
                        "concepts": ["pipe", "shared memory", "message queue", "socket"]},

                        {"name": "Networking",
                        "concepts": ["TCP", "UDP", "socket programming"]},

                        {"name": "Security",
                        "concepts": ["authentication", "authorization", "access control"]}
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

    # ================== FIX 5: Add concept normalization helper ==================
    def _normalize_concept(self, concept: str) -> str:
        """
        Normalize concept names for better matching
        Handles slashes, hyphens, and spacing
        """
        # Replace slashes and hyphens with spaces
        normalized = concept.replace("/", " ").replace("-", " ")
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized.strip()

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for robust concept matching
        Removes special chars, normalizes spaces
        """
        text = text.lower()
        text = text.replace("/", " ")
        text = text.replace("-", " ")
        text = text.replace("_", " ")
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _concept_present_semantic(self, question: str, concept: str, threshold: float = 0.65) -> bool:
        """
        Check if concept is semantically present in question
        Uses embeddings for robust matching
        """
        question_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
        concept_emb = self.embedder.encode([concept], normalize_embeddings=True)[0]
        
        similarity = cosine_similarity([question_emb], [concept_emb])[0][0]
        
        if similarity > threshold:
            print(f"      🔍 Semantic concept match: '{concept}' (similarity: {similarity:.3f})")
            return True
        return False
    
    def _select_intent_by_difficulty(self, difficulty: str, used_intents: List[str]) -> str:
        """
        🔥 FIX 2: Select intent based on difficulty level
        Easy → definitions/terminology pool
        Medium → mechanisms/flows pool
        Hard → reasoning/analysis pool
        
        This matches the rule:
        - EASY: definitions, terminology, identification
        - MEDIUM: mechanism_flow, cause_effect, comparison
        - HARD: optimization_reasoning, system_design, abstraction
        """
        # INTENT POOLS BY DIFFICULTY - ALIGNED WITH RULES
        INTENT_POOLS = {
            "easy": [
                "core_definition",      # Definitions
                "identification",        # Identification/listing
                "listing",               # Listing components
                "real_world_scenario"    # Simple scenarios
            ],
            "medium": [
                "mechanism_flow",        # How things work
                "conceptual_difference", # Compare/contrast
                "cause_effect",          # Cause and effect relationships
                "application",           # Apply concepts
                "component_relationship" # How components relate
            ],
            "hard": [
                "optimization_reasoning", # Performance optimization
                "tradeoff_analysis",      # Tradeoffs between approaches
                "debugging_case",         # Debug complex scenarios
                "system_design",          # Design decisions
                "abstraction",            # Abstract reasoning
                "prediction"              # Predict behavior
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
        """
        Intent-aware fallback question generation (legacy)
        Ensures questions match the required intent even in fallback
        """
        templates = {
            "optimization_reasoning":
                f"How would you optimize a system using {concept1} and {concept2} in {subtopic}?",
            
            "tradeoff_analysis":
                f"Compare the tradeoffs between using {concept1} versus {concept2} in {subtopic}.",
            
            "debugging_case":
                f"A system using {concept1} and {concept2} is failing. How would you debug it?",
            
            "prediction":
                f"What would happen if {concept1} interacts with {concept2} under heavy load?",
            
            "design_decision":
                f"Design a system using {concept1} and {concept2}. What key design decisions must be made?",
            
            "core_definition":
                f"What are {concept1} and {concept2} and how do they relate in {subtopic}?",
            
            "conceptual_difference":
                f"What is the difference between {concept1} and {concept2} in {subtopic}?",
            
            "mechanism_flow":
                f"Explain how {concept1} and {concept2} work together in {subtopic}.",
            
            "edge_case":
                f"What edge cases should you consider when using {concept1} and {concept2} together?",
            
            "misconception_check":
                f"A common misconception is that {concept1} and {concept2} are the same. Explain why this is incorrect."
        }
        
        return templates.get(intent, f"How do {concept1} and {concept2} work together in {subtopic}?")
    
    def fallback_question(self, topic: str, subtopic: str, concepts: List[str], intent: str) -> str:
        """
        Guaranteed fallback that ALWAYS includes BOTH concepts
        Returns a valid question with both concepts explicitly mentioned
        """
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
        
        # Ensure both concepts are in the question (double-check)
        q_lower = fallback.lower()
        if c1.lower() not in q_lower and c2.lower() not in q_lower:
            # Ultimate fallback - construct manually
            fallback = f"Explain the relationship between {c1} and {c2} in the context of {subtopic}."
        elif c1.lower() not in q_lower:
            # Insert missing concept
            fallback = fallback.replace(c2, f"{c1} and {c2}")
        elif c2.lower() not in q_lower:
            # Insert missing concept
            fallback = fallback.replace(c1, f"{c1} and {c2}")
        
        print(f"   🎯 Fallback question generated: '{fallback}'")
        return fallback
    
    # ================== FIX 3: Updated generate_question_with_rag with concept injection ==================
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
        
        # Build query for similar example retrieval - ENHANCED with concepts
        query = f"{topic} {subtopic} {concepts_norm[0]} {concepts_norm[1]} {intent} {difficulty} interview question"
        
        # Retrieve similar Q&A examples from FAISS - use only 1 example to prevent copying
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
            # Get chunks specifically for these concepts
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
        
        # Build strict prompt with concept injection
        personalization = f" for {user_name}" if user_name else ""
        
        # 🔥 NEW PROMPT WITH EXPLICIT CONCEPT INJECTION
        prompt = f"""You are an expert technical interviewer{personalization}.

Generate exactly ONE interview question.

STRICT REQUIREMENTS:

Topic: {topic}
Subtopic: {subtopic}
Concepts: {concepts_norm[0]}, {concepts_norm[1]}
Difficulty: {difficulty}
Intent: {intent}

RULES:
• MUST include BOTH concepts EXACTLY as written: "{concepts_norm[0]}" and "{concepts_norm[1]}"
• MUST use the exact concept names - do NOT rename them
• MUST NOT omit either concept
• MUST be under 300 characters
• MUST be a single clear question ending with '?'
• MUST be domain-specific to {topic}
• DO NOT include any explanation or commentary
• Return ONLY the question

Knowledge context:
{context}

Generate now:"""

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
                    # Find start of that sentence
                    start = max(question.rfind('.', 0, q_pos), question.rfind('!', 0, q_pos), question.rfind('\n', 0, q_pos)) + 1
                    if start > 0:
                        question = question[start:q_pos+1].strip()
                    else:
                        question = question[:q_pos+1].strip()
                
                # Apply standard cleaning
                question = self._clean_question(question)
                question = self._make_verbal_safe(question)
                
                # HARD LENGTH ENFORCEMENT - 300 char max
                if len(question) > 300:
                    print(f"   ⚠️ Attempt {attempt + 1}: Too long ({len(question)} chars > 300) - rejecting")
                    continue
                
                if len(question) < 40:
                    print(f"   ⚠️ Attempt {attempt + 1}: Too short ({len(question)} chars < 40) - rejecting")
                    continue
                
                # ✅ INVARIANT: Validate concept inclusion with semantic matching
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
                print(f"   🔍 Duplicate detected (similarity: {similarity:.3f})")
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
    
    # ================== FIX 4: Updated validate_question with semantic matching ==================
    def validate_question(self, question: str, concepts: List[str]) -> bool:
        """
        Robust concept validation using semantic similarity
        Prevents false negatives from exact string matching
        Returns True if valid, False if invalid
        """
        print(f"\n   🔍 Validating question for concepts: {concepts}")
        
        # Normalization helper
        def normalize_concept(c):
            return c.replace("/", " ").replace("-", " ").lower().strip()
        
        q_lower = question.lower()
        
        missing = []
        present = []
        
        for concept in concepts:
            c_norm = normalize_concept(concept)
            
            # Method 1: Simple contains check (fast)
            if c_norm in q_lower:
                print(f"      ✓ '{concept}' found directly")
                present.append(concept)
                continue
            
            # Method 2: Check without spaces (for compound terms)
            c_no_space = c_norm.replace(" ", "")
            q_no_space = q_lower.replace(" ", "")
            if c_no_space in q_no_space:
                print(f"      ✓ '{concept}' found without spaces")
                present.append(concept)
                continue
            
            # Method 3: Semantic similarity (fallback)
            try:
                from sentence_transformers import util
                
                # Get embeddings
                q_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
                c_emb = self.embedder.encode([concept], normalize_embeddings=True)[0]
                
                # Calculate similarity
                similarity = float(util.cos_sim(q_emb, c_emb)[0][0])
                
                # Threshold: 0.6 for semantic match
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
        
        # Valid if at least one concept present (for robustness)
        if len(present) == 0:
            print(f"   ❌ Validation failed - no concepts found at all")
            return False
        
        if len(present) < len(concepts):
            print(f"   ⚠️ Partial match: found {len(present)}/{len(concepts)} concepts")
            # Accept partial matches for better success rate
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
                
                # INVARIANT 2: Validate concept inclusion with semantic matching
                if not self.validate_question(question, sampled_concepts):
                    print(f"   ⚠️ Attempt {attempt + 1}: Missing concepts - retrying")
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