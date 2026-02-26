# backend/agent/adaptive_analyzer.py

import re
from typing import List, Set, Dict
import numpy as np
# 🔥 ADD THESE IMPORTS
from interview_analyzer import calculate_semantic_similarity
from rag import agentic_expected_answer

class AdaptiveAnalyzer:
    """Enhanced analyzer with adaptive learning signals"""
    
    # Technical keywords by topic for concept extraction
    TECH_KEYWORDS = {
        'DBMS': [
            'database', 'sql', 'query', 'index', 'transaction', 'acid', 
            'normalization', 'join', 'primary key', 'foreign key', 'schema', 
            'table', 'bcnf', '3nf', 'redundancy', 'anomaly', 'lock',
            'deadlock', 'concurrency', 'rollback', 'commit', 'logging'
        ],
        'OS': [
            'process', 'thread', 'memory', 'deadlock', 'scheduling', 
            'virtual memory', 'kernel', 'system call', 'context switch', 
            'semaphore', 'mutex', 'paging', 'segmentation', 'fifo', 'lru',
            'race condition', 'critical section', 'monitor', 'dining philosophers'
        ],
        'OOPS': [
            'class', 'object', 'inheritance', 'polymorphism', 'encapsulation', 
            'abstraction', 'interface', 'method', 'constructor', 'destructor',
            'overloading', 'overriding', 'virtual function', 'abstract class',
            'multiple inheritance', 'diamond problem', 'composition', 'aggregation'
        ]
    }
    
    # Difficulty indicators
    DEPTH_INDICATORS = [
        'because', 'therefore', 'thus', 'hence', 'consequently',
        'for example', 'for instance', 'specifically', 'in particular',
        'first', 'second', 'third', 'finally', 'additionally',
        'furthermore', 'moreover', 'consequently', 'as a result'
    ]
    
    # Confidence indicators
    CONFIDENT_INDICATORS = [
        'definitely', 'certainly', 'absolutely', 'clearly',
        'without doubt', 'undoubtedly', 'i know', 'i understand',
        'obviously', 'of course', 'certainly'
    ]
    
    HESITANT_INDICATORS = [
        'i think', 'maybe', 'perhaps', 'probably', 'i guess',
        'not sure', 'could be', 'might be', 'possibly',
        'sort of', 'kind of', 'approximately'
    ]
    
    @classmethod
    def _concept_in_answer(cls, concept: str, answer_lower: str) -> bool:
        """
        INVARIANT 1: Detect concept with synonym support
        Used for concept mastery tracking
        """
        concept_lower = concept.lower()
        
        # Direct match
        if concept_lower in answer_lower:
            return True
        
        # Multi-word concept without spaces
        if ' ' in concept_lower:
            concept_no_space = concept_lower.replace(' ', '')
            answer_no_space = answer_lower.replace(' ', '')
            if concept_no_space in answer_no_space:
                return True
        
        # Synonym mapping
        synonyms = {
            'mutex': ['mutex', 'mutual exclusion', 'lock'],
            'semaphore': ['semaphore', 'counting semaphore', 'binary semaphore'],
            'critical section': ['critical section', 'critical region'],
            'deadlock': ['deadlock', 'deadly embrace'],
            'process': ['process', 'task'],
            'thread': ['thread', 'lightweight process'],
            'primary key': ['primary key', 'primary-key', 'pk'],
            'foreign key': ['foreign key', 'foreign-key', 'fk'],
            'avoidance': ['banker', 'safe state', 'avoidance'],
            'prevention': ['prevention', 'prevent'],
            'detection': ['detection', 'detect', 'wait-for graph']
        }
        
        if concept_lower in synonyms:
            for synonym in synonyms[concept_lower]:
                if synonym in answer_lower:
                    return True
        
        return False
    
    @classmethod
    def extract_keywords(cls, text: str, topic: str = None) -> Set[str]:
        """Extract important keywords, optionally filtered by topic"""
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        
        # Multi-word terms
        multi_word_terms = []
        for term in ['primary key', 'foreign key', 'virtual memory', 
                     'system call', 'context switch', 'race condition',
                     'critical section', 'dining philosophers', 'deadlock']:
            if term in text.lower():
                multi_word_terms.append(term)
        
        if topic and topic in cls.TECH_KEYWORDS:
            # Filter by topic-specific keywords
            topic_keywords = set(cls.TECH_KEYWORDS[topic])
            return words.intersection(topic_keywords).union(multi_word_terms)
        
        # Return all technical keywords from any topic
        all_keywords = set()
        for kw_list in cls.TECH_KEYWORDS.values():
            all_keywords.update(kw_list)
        
        return words.intersection(all_keywords).union(multi_word_terms)
    
    
    @classmethod
    def assess_depth(cls, answer: str) -> str:
        """Assess depth of answer: shallow, medium, deep"""
        word_count = len(answer.split())
        
        # Count depth indicators
        indicator_count = sum(1 for ind in cls.DEPTH_INDICATORS if ind in answer.lower())
        
        # Check for examples
        has_example = 'example' in answer.lower() or 'instance' in answer.lower()
        
        if word_count > 100 and indicator_count >= 3 and has_example:
            return "deep"
        elif word_count > 50 or indicator_count >= 2:
            return "medium"
        else:
            return "shallow"

    @classmethod
    def get_subtopic_concepts(cls, topic: str, subtopic: str, question_bank) -> Set[str]:
        """
        Get the exact concepts for a specific subtopic from the question bank taxonomy
        This ensures we only check concepts that actually belong to this subtopic
        """
        if not topic or not subtopic or not question_bank:
            return set()
        
        # Look through the taxonomy in question_bank
        for t in question_bank.taxonomy["topics"]:
            if t["name"] == topic:
                for s in t["subtopics"]:
                    if s["name"] == subtopic:
                        # Return all concepts for this subtopic as lowercase set
                        return set([c.lower() for c in s["concepts"]])
        
        return set()
    
    @classmethod
    def assess_confidence(cls, answer: str) -> str:
        """Assess confidence: low, medium, high"""
        answer_lower = answer.lower()
        
        confident_count = sum(1 for ind in cls.CONFIDENT_INDICATORS if ind in answer_lower)
        hesitant_count = sum(1 for ind in cls.HESITANT_INDICATORS if ind in answer_lower)
        
        if confident_count > hesitant_count + 1:
            return "high"
        elif hesitant_count > confident_count + 1:
            return "low"
        else:
            return "medium"
    
    @classmethod
    def identify_missing_concepts(cls, answer: str, expected_concepts: Set[str]) -> List[str]:
        """Identify which expected concepts are missing"""
        if not expected_concepts:
            return []
        
        answer_lower = answer.lower()
        missing = []
        
        for concept in expected_concepts:
            if concept not in answer_lower:
                missing.append(concept)
        
        return missing[:5]  # Return top 5 missing concepts
    

    @classmethod
    def analyze(cls, question: str, answer: str, topic: str = None, 
                subtopic: str = None, question_bank = None,
                expected_answer: str = None) -> dict:
        """
        Comprehensive analysis with adaptive learning signals
        ALL metrics are RAW values (0.0 to 1.0) - NO SCALING
        INVARIANT 1: Concept detection uses synonym support
        """
        if not answer or not answer.strip() or len(answer.strip()) < 5:
            print(f"⚠️ Empty or very short answer detected, returning zeros")
            return {
                "keyword_coverage": 0.0,
                "depth": "shallow",
                "missing_concepts": [],
                "covered_concepts": [],
                "confidence": "low",
                "key_terms_used": [],
                "response_length": 0,
                "grammatical_quality": 0.0,
                "has_example": False,
                "estimated_difficulty": "easy",
                "semantic_similarity": 0.0,
                "expected_answer": expected_answer or ""
            }
        
        # Extract keywords
        key_terms = list(cls.extract_keywords(answer, topic))
        expected_keywords = cls.extract_keywords(question, topic)
        
        # Calculate RAW coverage
        from interview_analyzer import calculate_keyword_coverage
        keyword_coverage = calculate_keyword_coverage(answer, question)
        
        depth = cls.assess_depth(answer)
        confidence = cls.assess_confidence(answer)
        
        # 🔥 FIX: Use subtopic-specific concepts for missing concepts with synonym support
        missing = []
        covered = []
        
        if topic and subtopic and question_bank:
            # Get the EXACT concepts for this subtopic from the taxonomy
            subtopic_concepts = cls.get_subtopic_concepts(topic, subtopic, question_bank)
            
            if subtopic_concepts:
                answer_lower = answer.lower()
                for concept in subtopic_concepts:
                    # Use the enhanced concept detection with synonyms
                    if cls._concept_in_answer(concept, answer_lower):
                        covered.append(concept)
                    else:
                        missing.append(concept)
                
                print(f"🎯 Subtopic concepts for {topic} - {subtopic}: {len(subtopic_concepts)} concepts")
                print(f"   Covered: {len(covered)}, Missing: {len(missing)}")
                
                # Show which specific concepts were detected
                if covered:
                    print(f"   ✓ Detected: {covered[:5]}")
                if missing:
                    print(f"   ✗ Missing: {missing[:5]}")
            else:
                # Fallback to old method if subtopic concepts not found
                missing = cls.identify_missing_concepts(answer, expected_keywords)
                covered = [c for c in expected_keywords if c not in missing]
        else:
            # Fallback to old method
            missing = cls.identify_missing_concepts(answer, expected_keywords)
            covered = [c for c in expected_keywords if c not in missing]
        
        # Check for examples
        has_example = 'example' in answer.lower() or 'instance' in answer.lower()
        
        # Simple grammar check
        sentences = re.split(r'[.!?]+', answer)
        good_sentences = sum(1 for s in sentences if s and s[0].isupper())
        grammatical_quality = good_sentences / len(sentences) if sentences else 0
        
        # Estimate answer difficulty
        if depth == "deep" and confidence == "high" and has_example:
            est_difficulty = "hard"
        elif depth == "shallow" and confidence == "low":
            est_difficulty = "easy"
        else:
            est_difficulty = "medium"
        
        # Calculate semantic similarity using provided expected_answer
        semantic_similarity = 0.0
        if expected_answer and expected_answer.strip():
            try:
                from interview_analyzer import calculate_semantic_similarity
                semantic_similarity = calculate_semantic_similarity(answer, expected_answer)
                print(f"📊 Semantic similarity calculated: {semantic_similarity:.3f}")
            except Exception as e:
                print(f"⚠️ Could not calculate semantic similarity: {e}")
                semantic_similarity = 0.0
        else:
            print(f"⚠️ No expected answer provided for question: {question[:50]}...")
            semantic_similarity = 0.0

        return {
            "keyword_coverage": round(keyword_coverage, 3),
            "depth": depth,
            "missing_concepts": missing[:5],  # Top 5 missing
            "covered_concepts": covered[:10],  # Top 10 covered
            "confidence": confidence,
            "key_terms_used": key_terms[:10],
            "response_length": len(answer.split()),
            "grammatical_quality": round(grammatical_quality, 3),
            "has_example": has_example,
            "estimated_difficulty": est_difficulty,
            "expected_keywords": list(expected_keywords),
            "semantic_similarity": round(semantic_similarity, 3),
            "expected_answer": expected_answer
        }