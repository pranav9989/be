# backend/agent/subtopic_tracker.py

import json
import random
from datetime import datetime
from typing import List, Dict, Set, Tuple
from models import db, SubtopicMastery

class SubtopicTracker:
    """
    Tracks mastery of individual subtopics across sessions
    Priority order: WEAK → NEW → STRONG
    """
    
    # 🔥 UPDATED: Atomic subtopics (no parentheses, no comparisons)
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
    
    # 🔥 NEW: Mapping from old to new names (for backward compatibility with DB)
    OLD_TO_NEW_MAPPING = {
        # OS mappings
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
        
        # DBMS mappings
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
        
        # OOPS mappings
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
        self._load_from_db()
    
    def _normalize_subtopic_name(self, topic: str, subtopic: str) -> str:
        """Convert old subtopic names to atomic ones"""
        # If it's already atomic, return as-is
        if subtopic in self.SUBTOPICS_BY_TOPIC.get(topic, []):
            return subtopic
        
        # Check mapping
        if subtopic in self.OLD_TO_NEW_MAPPING:
            new_name = self.OLD_TO_NEW_MAPPING[subtopic]
            print(f"🔄 Normalizing '{subtopic}' -> '{new_name}'")
            return new_name
        
        # Fallback: try to extract base name (before any parenthesis)
        if "(" in subtopic:
            base = subtopic.split("(")[0].strip()
            if base in self.SUBTOPICS_BY_TOPIC.get(topic, []):
                print(f"🔄 Extracted base name: '{subtopic}' -> '{base}'")
                return base
        
        return subtopic
    
    def _load_from_db(self):
        """Load all subtopic masteries from database, normalizing old names"""
        self.masteries = {}
        db_masteries = SubtopicMastery.query.filter_by(user_id=self.user_id).all()
        
        for m in db_masteries:
            # Normalize the subtopic name from DB (which might have old format)
            normalized = self._normalize_subtopic_name(m.topic, m.subtopic)
            
            if m.topic not in self.masteries:
                self.masteries[m.topic] = {}
            
            # Store with normalized name
            self.masteries[m.topic][normalized] = {
                'mastery_level': m.mastery_level,
                'attempts': m.attempts,
                'last_asked': m.last_asked,
                'status': m.status,  # 'weak', 'strong', or None
                'original_subtopic': m.subtopic  # Keep original for DB updates
            }
    
    def _save_subtopic(self, topic: str, subtopic: str, 
                       mastery_level: float = 0.0, 
                       attempts: int = 0,
                       status: str = None):
        """Save or update a subtopic mastery in database"""
        # Always store the normalized name in DB from now on
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
        db_mastery.status = status
        
        db.session.commit()
        
        # Update in-memory cache with normalized name
        if topic not in self.masteries:
            self.masteries[topic] = {}
        self.masteries[topic][subtopic] = {
            'mastery_level': mastery_level,
            'attempts': attempts,
            'last_asked': db_mastery.last_asked,
            'status': status
        }
    
    def update_subtopic_performance(self, topic: str, subtopic: str, 
                                   semantic_score: float):
        """
        Update subtopic mastery based on answer quality
        - semantic_score > 0.7 → strong
        - semantic_score < 0.4 → weak
        - otherwise → medium (no status)
        """
        # Normalize the subtopic name first
        normalized_subtopic = self._normalize_subtopic_name(topic, subtopic)
        
        # Get current mastery or create new
        current = self.masteries.get(topic, {}).get(normalized_subtopic, {})
        attempts = current.get('attempts', 0) + 1
        
        # Calculate new mastery level (EMA)
        alpha = 0.3
        old_mastery = current.get('mastery_level', 0.0)
        new_mastery = (alpha * semantic_score) + ((1 - alpha) * old_mastery)
        
        # Determine status
        if semantic_score > 0.7:
            status = 'strong'
        elif semantic_score < 0.4:
            status = 'weak'
        else:
            status = None  # Medium - no special status
        
        # Save to database with normalized name
        self._save_subtopic(
            topic=topic,
            subtopic=normalized_subtopic,
            mastery_level=new_mastery,
            attempts=attempts,
            status=status
        )
        
        print(f"📊 Updated subtopic {topic} - {normalized_subtopic}:")
        print(f"   Score: {semantic_score:.2f} → Mastery: {new_mastery:.2f}")
        print(f"   Status: {status if status else 'medium'}")
        print(f"   Attempts: {attempts}")
    
    def get_next_subtopic(self, topic: str) -> str:
        """
        Select next subtopic with priority:
        1. WEAK subtopics (highest priority)
        2. NEW subtopics (never attempted)
        3. STRONG subtopics (lowest priority)
        
        Uses probability: 80% weak/new, 20% strong when weak/new exist
        """
        # Get all subtopics for this topic (atomic names)
        all_subtopics = set(self.SUBTOPICS_BY_TOPIC.get(topic, []))
        
        # Get attempted subtopics with their status (already normalized)
        attempted = self.masteries.get(topic, {})
        
        # Categorize subtopics
        weak_subtopics = []
        strong_subtopics = []
        new_subtopics = []
        
        for subtopic in all_subtopics:
            if subtopic in attempted:
                status = attempted[subtopic].get('status')
                if status == 'weak':
                    weak_subtopics.append(subtopic)
                elif status == 'strong':
                    strong_subtopics.append(subtopic)
                else:
                    # Attempted but medium - treat as new for selection purposes
                    new_subtopics.append(subtopic)
            else:
                # Never attempted
                new_subtopics.append(subtopic)
        
        print(f"\n🎯 Subtopic selection for {topic}:")
        print(f"   Weak: {weak_subtopics}")
        print(f"   New: {new_subtopics[:5]}... (total: {len(new_subtopics)})")
        print(f"   Strong: {strong_subtopics}")
        
        # PRIORITY 1: Weak subtopics (always ask first)
        if weak_subtopics:
            chosen = random.choice(weak_subtopics)
            print(f"   ✅ Selected WEAK subtopic: {chosen}")
            return chosen
        
        # If we have both new and strong subtopics, use probability
        if new_subtopics and strong_subtopics:
            # 80% chance to pick new, 20% chance to pick strong
            if random.random() < 0.8:  # 80% for new
                chosen = random.choice(new_subtopics)
                print(f"   🆕 Selected NEW subtopic (80% probability): {chosen}")
                return chosen
            else:  # 20% for strong
                chosen = random.choice(strong_subtopics)
                print(f"   💪 Selected STRONG subtopic (20% probability): {chosen}")
                return chosen
        
        # Only new subtopics left
        if new_subtopics:
            chosen = random.choice(new_subtopics)
            print(f"   🆕 Selected NEW subtopic: {chosen}")
            return chosen
        
        # Only strong subtopics left
        if strong_subtopics:
            chosen = random.choice(strong_subtopics)
            print(f"   💪 Selected STRONG subtopic (only option): {chosen}")
            return chosen
        
        # Fallback (should never happen with your hardcoded topics)
        fallback = random.choice(list(all_subtopics))
        print(f"   ⚠️ Fallback selected: {fallback}")
        return fallback
    
    def get_all_attempted_subtopics(self, topic: str = None) -> Dict:
        """Get all attempted subtopics, optionally filtered by topic"""
        if topic:
            return self.masteries.get(topic, {})
        return self.masteries
    
    def reset_all_mastery(self):
        """Reset ALL subtopic mastery (for starting over)"""
        SubtopicMastery.query.filter_by(user_id=self.user_id).delete()
        db.session.commit()
        self.masteries = {}
        print(f"🔄 Reset all subtopic mastery for user {self.user_id}")
    
    def reset_topic_mastery(self, topic: str):
        """Reset mastery for a specific topic"""
        SubtopicMastery.query.filter_by(
            user_id=self.user_id,
            topic=topic
        ).delete()
        db.session.commit()
        if topic in self.masteries:
            del self.masteries[topic]
        print(f"🔄 Reset {topic} mastery for user {self.user_id}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about subtopic mastery"""
        stats = {
            'total_subtopics': sum(len(topics) for topics in self.SUBTOPICS_BY_TOPIC.values()),
            'attempted': 0,
            'weak': 0,
            'strong': 0,
            'by_topic': {}
        }
        
        for topic, subtopics in self.SUBTOPICS_BY_TOPIC.items():
            topic_stats = {
                'total': len(subtopics),
                'attempted': 0,
                'weak': 0,
                'strong': 0,
                'subtopics': {}
            }
            
            attempted = self.masteries.get(topic, {})
            topic_stats['attempted'] = len(attempted)
            
            for subtopic, data in attempted.items():
                status = data.get('status')
                if status == 'weak':
                    topic_stats['weak'] += 1
                    stats['weak'] += 1
                elif status == 'strong':
                    topic_stats['strong'] += 1
                    stats['strong'] += 1
                
                topic_stats['subtopics'][subtopic] = {
                    'mastery': round(data.get('mastery_level', 0), 3),
                    'attempts': data.get('attempts', 0),
                    'status': status
                }
            
            stats['attempted'] += topic_stats['attempted']
            stats['by_topic'][topic] = topic_stats
        
        return stats