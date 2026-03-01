# backend/agent/subtopic_tracker.py

import json
import random
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
from models import db, SubtopicMastery

class SubtopicTracker:
    """
    Tracks mastery of individual subtopics across sessions
    CLEAR DISTINCTION: Subtopic status vs Concept status
    
    Subtopic status (for the subtopic itself):
        - not_started: never attempted
        - ongoing: attempted but not mastered (any attempts, mastery < 0.7)
        - mastered: 3+ attempts and mastery >= 0.7
    
    Concept status (inside subtopic) - handled by ConceptMastery class:
        - weak: miss_ratio > 70% (3+ attempts)
        - strong: correct_ratio > 70% (3+ attempts)
        - medium: between thresholds (3+ attempts)
        - new: <3 attempts
    """
    
    # Atomic subtopics
    SUBTOPICS_BY_TOPIC = {
        "DBMS": [
            "Normalization",
            "Keys",
            "ACID",
            "Transactions",
            "Concurrency Control",
            "Isolation Levels",
            "Indexing",
            "Joins",
            "SQL Aggregation",
            "Locking",
            "Deadlocks"
        ],
        "OOPS": [
            "Classes",
            "Objects",
            "Encapsulation",
            "Abstraction",
            "Inheritance",
            "Polymorphism",
            "Constructors",
            "Access Modifiers",
            "SOLID Principles"
        ],
        "OS": [
            "Processes",
            "Threads",
            "Context Switching",
            "CPU Scheduling",
            "Synchronization",
            "Deadlocks",
            "Memory Management",
            "Virtual Memory",
            "Page Replacement",
            "System Calls"
        ]
    }
    
    # Legacy name mapping
    OLD_TO_NEW_MAPPING = {
        "Process vs Thread": "Processes",
        "Process States & PCB": "Processes",
        "Context Switching": "Context Switching",
        "CPU Scheduling Algorithms (FCFS, SJF, RR, Priority)": "CPU Scheduling",
        "Synchronization (Mutex, Semaphore, Monitor)": "Synchronization",
        "Deadlock (Conditions & Prevention)": "Deadlocks",
        "Memory Management (Paging, Segmentation)": "Memory Management",
        "Virtual Memory": "Virtual Memory",
        "Demand Paging & Page Replacement (LRU, FIFO)": "Page Replacement",
        "System Calls": "System Calls",
        "Normalization (1NF, 2NF, 3NF, BCNF)": "Normalization",
        "Keys (Primary, Foreign, Candidate, Composite)": "Keys",
        "ACID Properties": "ACID",
        "Transactions & Concurrency Control": "Transactions",
        "Isolation Levels": "Isolation Levels",
        "Indexing (B+ Tree, Hash Index)": "Indexing",
        "Joins (Inner, Left, Right, Full)": "Joins",
        "SQL Queries (GROUP BY, HAVING, Subqueries)": "SQL Aggregation",
        "Locking (Shared, Exclusive Locks)": "Locking",
        "Deadlocks in DBMS": "Deadlocks",
        "Classes & Objects": "Classes",
        "Encapsulation": "Encapsulation",
        "Abstraction": "Abstraction",
        "Inheritance (Types & Diamond Problem)": "Inheritance",
        "Polymorphism (Compile-time & Runtime)": "Polymorphism",
        "Method Overloading vs Overriding": "Polymorphism",
        "Interfaces vs Abstract Classes": "Abstraction",
        "Constructors": "Constructors",
        "Access Modifiers": "Access Modifiers",
        "SOLID Principles": "SOLID Principles"
    }
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.subtopics = {}  # topic -> {subtopic: data}
        self._load_from_db()
    
    def _normalize_subtopic_name(self, topic: str, subtopic: str) -> str:
        """Convert old subtopic names to atomic ones"""
        if subtopic in self.SUBTOPICS_BY_TOPIC.get(topic, []):
            return subtopic
        
        if subtopic in self.OLD_TO_NEW_MAPPING:
            new_name = self.OLD_TO_NEW_MAPPING[subtopic]
            print(f"🔄 Normalizing '{subtopic}' -> '{new_name}'")
            return new_name
        
        if "(" in subtopic:
            base = subtopic.split("(")[0].strip()
            if base in self.SUBTOPICS_BY_TOPIC.get(topic, []):
                print(f"🔄 Extracted base: '{subtopic}' -> '{base}'")
                return base
        
        return subtopic
    
    def _calculate_subtopic_status(self, attempts: int, mastery: float) -> str:
        """
        🔥 FIXED: Calculate status for the SUBTOPIC itself
        Returns: 'not_started', 'ongoing', 'mastered'
        """
        if attempts == 0:
            return 'not_started'
        
        if attempts >= 3 and mastery >= 0.7:
            return 'mastered'
        
        return 'ongoing'
    
    def _load_from_db(self):
        """Load all subtopic masteries from database"""
        self.subtopics = {}
        db_masteries = SubtopicMastery.query.filter_by(user_id=self.user_id).all()
        
        for m in db_masteries:
            normalized = self._normalize_subtopic_name(m.topic, m.subtopic)
            
            if m.topic not in self.subtopics:
                self.subtopics[m.topic] = {}
            
            # Store with normalized name - INCLUDE ALL FIELDS
            self.subtopics[m.topic][normalized] = {
                'name': normalized,
                'topic': m.topic,
                'mastery_level': m.mastery_level,
                'attempts': m.attempts,
                'total_score': m.mastery_level * m.attempts if m.attempts > 0 else 0.0,
                'scores': [],  # 🔥 CRITICAL: Initialize empty scores list
                'last_asked': m.last_asked,
                'original_name': m.subtopic,
                'subtopic_status': self._calculate_subtopic_status(
                    attempts=m.attempts,
                    mastery=m.mastery_level
                )
            }
    
    def _save_subtopic(self, topic: str, subtopic: str, 
                       mastery_level: float = 0.0, 
                       attempts: int = 0):
        """Save or update a subtopic mastery in database"""
        db_subtopic = subtopic
        
        db_mastery = SubtopicMastery.query.filter_by(
            user_id=self.user_id,
            topic=topic,
            subtopic=db_subtopic
        ).first()
        
        if not db_mastery:
            db_mastery = SubtopicMastery(
                user_id=self.user_id,
                topic=topic,
                subtopic=db_subtopic
            )
            db.session.add(db_mastery)
        
        db_mastery.mastery_level = mastery_level
        db_mastery.attempts = attempts
        db_mastery.last_asked = datetime.utcnow()
        
        # 🔥 FIXED: Set subtopic status correctly
        db_mastery.subtopic_status = self._calculate_subtopic_status(attempts, mastery_level)
        
        db.session.commit()
        
        # Update in-memory cache
        if topic not in self.subtopics:
            self.subtopics[topic] = {}
        
        self.subtopics[topic][subtopic] = {
            'mastery_level': mastery_level,
            'attempts': attempts,
            'last_asked': db_mastery.last_asked,
            'subtopic_status': db_mastery.subtopic_status
        }
    
    # ================== FIXED: update_subtopic_performance with proper initialization ==================
    
    def update_subtopic_performance(self, topic: str, subtopic: str, 
                                semantic_score: float):
        """
        Update subtopic mastery based on answer quality
        CORRECT: Simple average of all semantic scores
        FIXED: Safely handles both new and existing subtopics
        """
        # Normalize name
        normalized = self._normalize_subtopic_name(topic, subtopic)
        
        # Ensure topic exists
        if topic not in self.subtopics:
            self.subtopics[topic] = {}
        
        # Ensure subtopic exists with all required fields
        if normalized not in self.subtopics[topic]:
            self.subtopics[topic][normalized] = {
                'mastery_level': 0.0,
                'attempts': 0,
                'total_score': 0.0,
                'scores': [],
                'last_asked': datetime.utcnow(),
                'subtopic_status': 'not_started'
            }
        
        current = self.subtopics[topic][normalized]
        
        # SAFETY: Ensure 'scores' exists (for data loaded from DB)
        if 'scores' not in current:
            current['scores'] = []
        
        # Add new score
        current['scores'].append(semantic_score)
        current['attempts'] = len(current['scores'])
        current['total_score'] = sum(current['scores'])
        
        # Calculate new mastery - SIMPLE AVERAGE (NOT EMA)
        if current['attempts'] > 0:
            current['mastery_level'] = current['total_score'] / current['attempts']
            print(f"   Mastery calculation: total={current['total_score']:.2f}, "
                  f"count={current['attempts']}, avg={current['mastery_level']:.3f}")
        
        # Compute subtopic status
        current['subtopic_status'] = self._calculate_subtopic_status(
            attempts=current['attempts'],
            mastery=current['mastery_level']
        )
        
        current['last_asked'] = datetime.utcnow()
        
        # Save to database
        self._save_subtopic(
            topic=topic,
            subtopic=normalized,
            mastery_level=current['mastery_level'],
            attempts=current['attempts']
        )
        
        print(f"\n📊 SUBTOPIC UPDATE: {topic} - {normalized}")
        print(f"   Attempts: {current['attempts']}")
        print(f"   Mastery: {current['mastery_level']:.2f}")
        print(f"   Status: {current['subtopic_status']}")
    
    # ================== 🔥 FIXED: get_next_subtopic with weak_concepts parameter ==================
    
    def get_next_subtopic(self, topic: str, weak_concepts: Optional[list] = None, covered_subtopics: Optional[list] = None) -> str:
        """
        STRICT PRIORITY ORDER PER RULES:

        1. Weak subtopics (subtopics containing weak concepts) → ALWAYS FIRST
        2. Explore pool (ongoing + not_started) → 80% probability
        3. Mastered subtopics → 20% probability
        """
        all_subtopics = list(self.SUBTOPICS_BY_TOPIC.get(topic, []))
        
        if covered_subtopics:
            all_subtopics = [s for s in all_subtopics if s not in covered_subtopics]
            
        if not all_subtopics:
            print(f"⚠️ No available subtopics found for {topic} after filtering covered ones")
            return "core concepts"
        
        # Get attempted subtopics
        attempted = self.subtopics.get(topic, {})
        
        # Categorize subtopics with strict priority
        weak_subtopics = []      # Contain weak concepts - PRIORITY 1
        explore_pool = []        # ongoing + not_started (no weak concepts) - PRIORITY 2
        mastered_subtopics = []  # mastered with no weak concepts - PRIORITY 3
        
        print(f"\n🔍 Analyzing subtopics for {topic} with weak concepts: {weak_concepts}")
        
        for subtopic in all_subtopics:
            # Check if this subtopic has any weak concepts
            has_weak = False
            if weak_concepts:
                # Get concepts for this subtopic
                subtopic_concepts = self._get_concepts_for_subtopic(topic, subtopic)
                # Check if any weak concept matches any concept in this subtopic
                subtopic_concepts_lower = [c.lower() for c in subtopic_concepts]
                if any(wc.lower() in subtopic_concepts_lower for wc in weak_concepts):
                    has_weak = True
                    weak_subtopics.append(subtopic)
                    print(f"   ⚠️ Weak subtopic detected: {subtopic}")
                    continue
            
            if subtopic in attempted:
                data = attempted[subtopic]
                status = data.get('subtopic_status', 'ongoing')
                
                if status == 'mastered' and not has_weak:
                    mastered_subtopics.append(subtopic)
                    print(f"   ✅ Mastered: {subtopic}")
                else:  # ongoing
                    if not has_weak:
                        explore_pool.append(subtopic)
                        print(f"   📚 Ongoing: {subtopic}")
            else:
                # not_started
                if not has_weak:
                    explore_pool.append(subtopic)
                    print(f"   🆕 Not started: {subtopic}")
        
        print(f"\n🎯 Subtopic selection for {topic}:")
        print(f"   Priority 1 - Weak subtopics: {weak_subtopics}")
        print(f"   Priority 2 - Explore pool: {explore_pool[:5]}... (total: {len(explore_pool)})")
        print(f"   Priority 3 - Mastered: {mastered_subtopics}")
        
        # PRIORITY 1: Weak subtopics (ALWAYS first)
        if weak_subtopics:
            chosen = random.choice(weak_subtopics)
            print(f"   ✅ SELECTED (WEAK - Priority 1): {chosen}")
            return chosen
        
        # PRIORITY 2 vs 3: 80% explore, 20% mastered
        if explore_pool and mastered_subtopics:
            if random.random() < 0.8:
                chosen = random.choice(explore_pool)
                source = "ONGOING" if chosen in attempted else "NOT_STARTED"
                print(f"   ✅ SELECTED ({source} - Priority 2, 80% explore): {chosen}")
                return chosen
            else:
                chosen = random.choice(mastered_subtopics)
                print(f"   ✅ SELECTED (MASTERED - Priority 3, 20% reinforcement): {chosen}")
                return chosen
        
        if explore_pool:
            chosen = random.choice(explore_pool)
            source = "ONGOING" if chosen in attempted else "NOT_STARTED"
            print(f"   ✅ SELECTED ({source} - Priority 2): {chosen}")
            return chosen
        
        if mastered_subtopics:
            chosen = random.choice(mastered_subtopics)
            print(f"   ✅ SELECTED (MASTERED - Priority 3): {chosen}")
            return chosen
        
        # Fallback
        fallback = random.choice(all_subtopics)
        print(f"   ⚠️ Fallback: {fallback}")
        return fallback
    
    # ================== Helper method to get concepts for a subtopic ==================
    
    def _get_concepts_for_subtopic(self, topic: str, subtopic: str) -> list:
        """
        Get concepts for a subtopic from the question bank
        Falls back to hardcoded mapping if question bank not available
        """
        try:
            # Try to import question bank dynamically to avoid circular imports
            from .adaptive_question_bank import AdaptiveQuestionBank
            qb = AdaptiveQuestionBank()
            return qb.get_concepts_for_subtopic(topic, subtopic)
        except (ImportError, AttributeError) as e:
            # Fallback to hardcoded mapping if question bank not available
            concept_map = {
                "DBMS": {
                    "Normalization": ["1NF", "2NF", "3NF", "BCNF", "functional dependency", "anomalies"],
                    "Keys": ["Primary Key", "Foreign Key", "Candidate Key", "Composite Key", "Super Key"],
                    "ACID": ["Atomicity", "Consistency", "Isolation", "Durability"],
                    "Transactions": ["commit", "rollback", "transaction states", "savepoint"],
                    "Concurrency Control": ["2PL", "timestamp ordering", "optimistic locking", "pessimistic locking"],
                    "Isolation Levels": ["Read Uncommitted", "Read Committed", "Repeatable Read", "Serializable"],
                    "Indexing": ["B+ Tree", "Hash Index", "clustered index", "non-clustered index"],
                    "Joins": ["Inner Join", "Left Join", "Right Join", "Full Join"],
                    "SQL Aggregation": ["GROUP BY", "HAVING", "Subqueries", "COUNT", "SUM"],
                    "Locking": ["Shared Lock", "Exclusive Lock", "Lock Granularity"],
                    "Deadlocks": ["Wait for graph", "detection", "prevention", "avoidance"]
                },
                "OOPS": {
                    "Classes": ["class", "object", "constructor", "attributes", "methods"],
                    "Objects": ["instantiation", "state", "behavior", "identity"],
                    "Encapsulation": ["data hiding", "getters setters", "access control"],
                    "Abstraction": ["abstract classes", "interfaces", "implementation hiding"],
                    "Inheritance": ["single inheritance", "multiple inheritance", "diamond problem"],
                    "Polymorphism": ["method overloading", "method overriding", "dynamic dispatch"],
                    "Constructors": ["default constructor", "parameterized constructor", "copy constructor"],
                    "Access Modifiers": ["public", "private", "protected"],
                    "SOLID Principles": ["Single Responsibility", "Open Closed", "Liskov Substitution"]
                },
                "OS": {
                    "Processes": ["process states", "PCB", "process creation", "zombie process"],
                    "Threads": ["user threads", "kernel threads", "multithreading"],
                    "Context Switching": ["CPU state saving", "overhead", "dispatch latency"],
                    "CPU Scheduling": ["FCFS", "SJF", "Round Robin", "Priority"],
                    "Synchronization": ["mutex", "semaphore", "monitor", "critical section"],
                    "Deadlocks": ["mutual exclusion", "hold and wait", "circular wait"],
                    "Memory Management": ["paging", "segmentation", "fragmentation"],
                    "Virtual Memory": ["demand paging", "page faults", "thrashing"],
                    "Page Replacement": ["LRU", "FIFO", "Optimal", "clock algorithm"],
                    "System Calls": ["fork", "exec", "wait", "open", "read", "write"]
                }
            }
            # Return lowercase concepts for better matching
            concepts = concept_map.get(topic, {}).get(subtopic, [])
            return [c.lower() for c in concepts]
    
    def get_all_attempted_subtopics(self, topic: str = None) -> Dict:
        """Get all attempted subtopics, optionally filtered by topic"""
        if topic:
            return self.subtopics.get(topic, {})
        return self.subtopics
    
    def reset_all_mastery(self):
        """Reset ALL subtopic mastery"""
        SubtopicMastery.query.filter_by(user_id=self.user_id).delete()
        db.session.commit()
        self.subtopics = {}
        print(f"🔄 Reset all subtopic mastery for user {self.user_id}")
    
    def reset_topic_mastery(self, topic: str):
        """Reset mastery for a specific topic"""
        SubtopicMastery.query.filter_by(
            user_id=self.user_id,
            topic=topic
        ).delete()
        db.session.commit()
        if topic in self.subtopics:
            del self.subtopics[topic]
        print(f"🔄 Reset {topic} mastery for user {self.user_id}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about subtopic mastery"""
        stats = {
            'total_subtopics': sum(len(topics) for topics in self.SUBTOPICS_BY_TOPIC.values()),
            'not_started': 0,
            'ongoing': 0,
            'mastered': 0,
            'by_topic': {}
        }
        
        for topic, all_subtopics in self.SUBTOPICS_BY_TOPIC.items():
            topic_stats = {
                'total': len(all_subtopics),
                'not_started': 0,
                'ongoing': 0,
                'mastered': 0,
                'subtopics': {}
            }
            
            attempted = self.subtopics.get(topic, {})
            
            for subtopic in all_subtopics:
                if subtopic in attempted:
                    data = attempted[subtopic]
                    status = data.get('subtopic_status', 'ongoing')
                    topic_stats[status] += 1
                    stats[status] += 1
                    
                    topic_stats['subtopics'][subtopic] = {
                        'mastery': round(data.get('mastery_level', 0), 3),
                        'attempts': data.get('attempts', 0),
                        'status': status
                    }
                else:
                    topic_stats['not_started'] += 1
                    stats['not_started'] += 1
            
            stats['by_topic'][topic] = topic_stats
        
        return stats