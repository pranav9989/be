# backend/agent/semantic_dedup.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Dict, Tuple  # 🔥 ADD Tuple here
import hashlib

class SemanticDeduplicator:
    """
    Advanced duplicate detection using embeddings and semantic similarity
    """
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.90  # Questions above this are duplicates
        self.question_cache: Dict[str, Dict] = {}  # session_id -> {questions, embeddings}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return self.embedder.encode([text], normalize_embeddings=True)[0]
    
    def is_duplicate(self, session_id: str, new_question: str, 
                 existing_questions: List[str], threshold: float = None) -> bool:
        """
        Check if a question is semantically similar to existing ones
        threshold: optional override of default similarity_threshold
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        if session_id not in self.question_cache:
            self.question_cache[session_id] = {
                'questions': [],
                'embeddings': []
            }
        
        cache = self.question_cache[session_id]
        
        # If no existing questions, definitely not duplicate
        if not existing_questions:
            return False
        
        # Get embedding for new question
        new_embedding = self.get_embedding(new_question)
        
        # Compare with all existing questions in this session
        for i, existing_question in enumerate(existing_questions):
            # Quick hash-based check for exact matches
            if existing_question == new_question:
                return True
            
            # Check if we have cached embedding
            if i < len(cache['embeddings']):
                existing_embedding = cache['embeddings'][i]
            else:
                existing_embedding = self.get_embedding(existing_question)
                cache['embeddings'].append(existing_embedding)
            
            # Calculate semantic similarity
            similarity = cosine_similarity(
                [new_embedding], 
                [existing_embedding]
            )[0][0]
            
            # For questions from different subtopics, be more lenient
            if similarity > threshold:
                print(f"🔍 Duplicate detected (similarity: {similarity:.3f} > {threshold}):")
                print(f"  New: {new_question[:50]}...")
                print(f"  Old: {existing_question[:50]}...")
                return True
        
        # Not a duplicate - cache it
        cache['questions'].append(new_question)
        cache['embeddings'].append(new_embedding)
        
        return False
    
    def get_similarity_score(self, q1: str, q2: str) -> float:
        """Get semantic similarity score between two questions"""
        emb1 = self.get_embedding(q1)
        emb2 = self.get_embedding(q2)
        return float(cosine_similarity([emb1], [emb2])[0][0])
    
    def get_most_similar(self, session_id: str, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get most similar questions from session history"""
        if session_id not in self.question_cache:
            return []
        
        cache = self.question_cache[session_id]
        if not cache['questions']:
            return []
        
        new_emb = self.get_embedding(question)
        similarities = []
        
        for i, q in enumerate(cache['questions']):
            if i < len(cache['embeddings']):
                emb = cache['embeddings'][i]
            else:
                emb = self.get_embedding(q)
                cache['embeddings'].append(emb)
            
            sim = cosine_similarity([new_emb], [emb])[0][0]
            similarities.append((q, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def clear_session(self, session_id: str):
        """Clear cache for a session"""
        if session_id in self.question_cache:
            del self.question_cache[session_id]

# Global deduplicator instance
semantic_dedup = SemanticDeduplicator()