# backend/agent/adaptive_analyzer.py

import re
from typing import List, Set, Dict, Tuple
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from interview_analyzer import calculate_semantic_similarity
from rag import agentic_expected_answer, retrieve_relevant_chunks

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

# Global embedder for semantic concept detection (loaded once)
_concept_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def normalize_text(text: str) -> str:
    """Normalize text for concept matching"""
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def semantic_concept_match(answer: str, concept: str, threshold: float = 0.65):
    """Match concept using both exact and semantic similarity"""
    answer_norm = normalize_text(answer)
    concept_norm = normalize_text(concept)
    
    # Exact normalized match
    if concept_norm in answer_norm:
        print(f"   ✓ Exact match detected: {concept}")
        return True, 1.0
    
    # Semantic similarity
    answer_emb = _concept_embedder.encode([answer_norm], normalize_embeddings=True)[0]
    concept_emb = _concept_embedder.encode([concept_norm], normalize_embeddings=True)[0]
    
    similarity = cosine_similarity([answer_emb], [concept_emb])[0][0]
    
    print(f"   🔍 Concept similarity: {concept} → {similarity:.3f}")
    
    if similarity >= threshold:
        print(f"   ✓ Semantic match accepted: {concept}")
        return True, similarity
    
    return False, similarity

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
    
    # Add class-level variables for semantic detection
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    concept_similarity_threshold = 0.65
    
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
        Used for concept mastery tracking (legacy method, kept for compatibility)
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
    def detect_concepts_semantically(cls, answer: str, sampled_concepts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Semantic concept detection robust to speech recognition errors
        Uses semantic similarity with text normalization
        
        Returns:
            Tuple of (mentioned_concepts, missing_concepts)
        """
        mentioned = []
        missing = []
        
        if not answer or not sampled_concepts:
            print("   ⚠️ Empty answer or concepts in semantic detection")
            return [], sampled_concepts
        
        print(f"\n🔬 SEMANTIC CONCEPT DETECTION")
        print(f"   Answer: {answer[:100]}..." if len(answer) > 100 else f"   Answer: {answer}")
        print(f"   Concepts to check: {sampled_concepts}")
        
        try:
            for concept in sampled_concepts:
                # Use semantic concept matching
                is_match, sim = semantic_concept_match(answer, concept)
                
                if is_match:
                    mentioned.append(concept)
                    print(f"   ✓ '{concept}' (match with similarity: {sim:.3f})")
                else:
                    missing.append(concept)
                    print(f"   ✗ '{concept}' (similarity: {sim:.3f} < {cls.concept_similarity_threshold})")
                    
        except Exception as e:
            print(f"   ⚠️ Semantic detection failed, falling back: {e}")
            # Fallback to semantic matching (still better than exact)
            for concept in sampled_concepts:
                is_match, sim = semantic_concept_match(answer, concept)
                if is_match:
                    mentioned.append(concept)
                    print(f"   ✓ '{concept}' (fallback match with sim: {sim:.3f})")
                else:
                    missing.append(concept)
                    print(f"   ✗ '{concept}' (fallback missing, sim: {sim:.3f})")
        
        print(f"\n   📊 Final result - Mentioned: {mentioned}, Missing: {missing}")
        return mentioned, missing
    
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
            result = words.intersection(topic_keywords).union(multi_word_terms)
            print(f"🔑 Extracted {len(result)} keywords for topic {topic}")
            return result
        
        # Return all technical keywords from any topic
        all_keywords = set()
        for kw_list in cls.TECH_KEYWORDS.values():
            all_keywords.update(kw_list)
        
        result = words.intersection(all_keywords).union(multi_word_terms)
        print(f"🔑 Extracted {len(result)} keywords (no topic filter)")
        return result
    
    @classmethod
    def assess_depth(cls, answer: str) -> str:
        """Assess depth of answer: shallow, medium, deep"""
        word_count = len(answer.split())
        
        # Count depth indicators
        indicator_count = sum(1 for ind in cls.DEPTH_INDICATORS if ind in answer.lower())
        
        # Check for examples
        has_example = 'example' in answer.lower() or 'instance' in answer.lower()
        
        if word_count > 100 and indicator_count >= 3 and has_example:
            depth = "deep"
        elif word_count > 50 or indicator_count >= 2:
            depth = "medium"
        else:
            depth = "shallow"
        
        print(f"📏 Depth assessment: {depth} (words: {word_count}, indicators: {indicator_count}, examples: {has_example})")
        return depth

    @classmethod
    def get_subtopic_concepts(cls, topic: str, subtopic: str, question_bank) -> Set[str]:
        """
        Get the exact concepts for a specific subtopic from the question bank taxonomy
        This ensures we only check concepts that actually belong to this subtopic
        """
        if not topic or not subtopic or not question_bank:
            print(f"⚠️ Missing parameters for get_subtopic_concepts: topic={topic}, subtopic={subtopic}, question_bank={bool(question_bank)}")
            return set()
        
        # Look through the taxonomy in question_bank
        for t in question_bank.taxonomy["topics"]:
            if t["name"] == topic:
                for s in t["subtopics"]:
                    if s["name"] == subtopic:
                        concepts = set([c.lower() for c in s["concepts"]])
                        print(f"📚 Found {len(concepts)} concepts for {topic} - {subtopic}")
                        return concepts
        
        print(f"⚠️ No concepts found for {topic} - {subtopic}")
        return set()
    
    @classmethod
    def assess_confidence(cls, answer: str) -> str:
        """Assess confidence: low, medium, high"""
        answer_lower = answer.lower()
        
        confident_count = sum(1 for ind in cls.CONFIDENT_INDICATORS if ind in answer_lower)
        hesitant_count = sum(1 for ind in cls.HESITANT_INDICATORS if ind in answer_lower)
        
        if confident_count > hesitant_count + 1:
            confidence = "high"
        elif hesitant_count > confident_count + 1:
            confidence = "low"
        else:
            confidence = "medium"
        
        print(f"🎯 Confidence assessment: {confidence} (confident: {confident_count}, hesitant: {hesitant_count})")
        return confidence
    
    @classmethod
    def identify_missing_concepts(cls, answer: str, expected_concepts: Set[str]) -> List[str]:
        """Identify which expected concepts are missing (legacy method)"""
        if not expected_concepts:
            return []
        
        answer_lower = answer.lower()
        missing = []
        
        for concept in expected_concepts:
            if concept not in answer_lower:
                missing.append(concept)
        
        print(f"🔍 Legacy missing concepts: {missing[:5]}")
        return missing[:5]  # Return top 5 missing concepts
    
    @classmethod
    def generate_rag_expected_answer(cls, question: str, concepts: List[str]) -> str:
        """
        Generate expected answer using RAG retrieval
        Retrieves relevant chunks and synthesizes a concise answer
        """
        if not question or not concepts:
            print("   ⚠️ No question or concepts for RAG expected answer")
            return ""
        
        print(f"\n📚 Generating RAG expected answer for: {question[:50]}...")
        print(f"   Concepts: {concepts}")
        
        # Retrieve relevant chunks
        try:
            chunks = retrieve_relevant_chunks(question, k=5)
            print(f"   Retrieved {len(chunks)} relevant chunks")
        except Exception as e:
            print(f"   ⚠️ RAG retrieval failed: {e}")
            chunks = []
        
        # Extract concept-specific content
        concept_text = ""
        if chunks:
            for concept in concepts:
                concept_lower = concept.lower()
                for chunk in chunks:
                    chunk_text = chunk.get("answer", chunk.get("text", ""))
                    if concept_lower in chunk_text.lower():
                        excerpt = chunk_text[:200]
                        concept_text += f"\n{excerpt}"
                        print(f"   Found content for '{concept}': {excerpt[:50]}...")
        
        # Build prompt
        prompt = f"""Question: {question}

Required concepts to cover: {', '.join(concepts)}

Relevant knowledge:
{concept_text if concept_text else "Use your technical knowledge."}

Generate a concise expected answer (2-4 sentences) that:
1. Explicitly mentions ALL required concepts
2. Is technically accurate
3. Is concise and well-structured

Expected answer:"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 300
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                answer = response.json()["response"].strip()
                print(f"   ✅ Generated ({len(answer)} chars)")
                print(f"   Preview: {answer[:100]}...")
                return answer
            else:
                print(f"   ⚠️ Ollama error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"   ⚠️ Error generating: {e}")
            return ""
    
    @classmethod
    def perform_gap_analysis(cls, user_answer: str, question: str, concepts: List[str]) -> List[str]:
        """
        Perform gap analysis by comparing user answer with retrieved knowledge
        Returns list of missing key points (up to 3)
        """
        if not user_answer or len(user_answer.strip()) < 10:
            print(f"   ⚠️ Answer too brief for gap analysis")
            return ["Answer too brief - needs more detail"]
        
        print(f"\n🔍 Performing gap analysis...")
        print(f"   Question: {question[:50]}...")
        print(f"   Concepts: {concepts}")
        
        # Retrieve relevant chunks
        try:
            chunks = retrieve_relevant_chunks(question, k=5)
            print(f"   Retrieved {len(chunks)} chunks for analysis")
        except Exception as e:
            print(f"   ⚠️ RAG retrieval failed: {e}")
            return []
        
        user_lower = user_answer.lower()
        missing_points = []
        
        for i, chunk in enumerate(chunks[:3]):  # Check top 3 chunks
            chunk_text = chunk.get("answer", chunk.get("text", ""))
            if not chunk_text:
                continue
            
            print(f"\n   Analyzing chunk {i+1}:")
            
            # Extract key sentences from chunk
            sentences = chunk_text.split('.')
            for j, sentence in enumerate(sentences[:2]):  # Check first 2 sentences
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # If key point not in answer, add to missing
                if sentence.lower() not in user_lower:
                    # Extract short version (first 60 chars)
                    short_point = sentence[:60] + "..." if len(sentence) > 60 else sentence
                    missing_points.append(short_point)
                    print(f"      Missing point {len(missing_points)}: {short_point}")
                    break  # One missing point per chunk
            
            if len(missing_points) >= 3:
                break
        
        print(f"\n   📊 Gap analysis complete - found {len(missing_points)} missing points")
        for i, point in enumerate(missing_points):
            print(f"      Missing {i+1}: {point}")
        
        return missing_points[:3]

    @classmethod
    def analyze(cls, question: str, answer: str, topic: str = None, 
                subtopic: str = None, question_bank = None,
                expected_answer: str = None) -> dict:
        """
        Comprehensive analysis with adaptive learning signals
        ALL metrics are RAW values (0.0 to 1.0) - NO SCALING
        INVARIANT 1: Concept detection uses synonym support and semantic matching
        """
        print("\n" + "█"*80)
        print("📊 ADAPTIVE ANALYZER")
        print("█"*80)
        print(f"   Question: {question[:100]}..." if len(question) > 100 else f"   Question: {question}")
        print(f"   Topic: {topic}, Subtopic: {subtopic}")
        
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
                "expected_answer": expected_answer or "",
                "gap_analysis": []
            }
        
        # Extract keywords
        key_terms = list(cls.extract_keywords(answer, topic))
        expected_keywords = cls.extract_keywords(question, topic)
        
        # Calculate RAW coverage
        from interview_analyzer import calculate_keyword_coverage
        keyword_coverage = calculate_keyword_coverage(answer, question)
        print(f"📊 Keyword coverage: {keyword_coverage:.3f}")
        
        depth = cls.assess_depth(answer)
        confidence = cls.assess_confidence(answer)
        
        # Use semantic concept detection
        missing = []
        covered = []
        sampled_concepts = []
        
        if topic and subtopic and question_bank:
            # Get the EXACT concepts for this subtopic from the taxonomy
            subtopic_concepts = cls.get_subtopic_concepts(topic, subtopic, question_bank)
            sampled_concepts = list(subtopic_concepts) if subtopic_concepts else []
            
            if subtopic_concepts:
                # 🔥 USE SEMANTIC CONCEPT DETECTION
                covered, missing = cls.detect_concepts_semantically(
                    answer=answer,
                    sampled_concepts=list(subtopic_concepts)
                )
                
                print(f"\n🎯 Concept detection results:")
                print(f"   Covered: {len(covered)} concepts")
                print(f"   Missing: {len(missing)} concepts")
                if covered:
                    print(f"   ✓ Detected: {covered[:5]}")
                if missing:
                    print(f"   ✗ Missing: {missing[:5]}")
            else:
                # Fallback to old method if subtopic concepts not found
                print(f"⚠️ No subtopic concepts found, using keyword-based fallback")
                missing = cls.identify_missing_concepts(answer, expected_keywords)
                covered = [c for c in expected_keywords if c not in missing]
        else:
            # Fallback to old method
            print(f"⚠️ Missing topic/subtopic/question_bank, using keyword-based fallback")
            missing = cls.identify_missing_concepts(answer, expected_keywords)
            covered = [c for c in expected_keywords if c not in missing]
        
        # Check for examples
        has_example = 'example' in answer.lower() or 'instance' in answer.lower()
        print(f"📝 Has example: {has_example}")
        
        # Simple grammar check
        sentences = re.split(r'[.!?]+', answer)
        good_sentences = sum(1 for s in sentences if s and s[0].isupper())
        grammatical_quality = good_sentences / len(sentences) if sentences else 0
        print(f"📝 Grammatical quality: {grammatical_quality:.3f}")
        
        # Estimate answer difficulty
        if depth == "deep" and confidence == "high" and has_example:
            est_difficulty = "hard"
        elif depth == "shallow" and confidence == "low":
            est_difficulty = "easy"
        else:
            est_difficulty = "medium"
        print(f"📊 Estimated difficulty: {est_difficulty}")
        
        # Generate expected answer if not provided
        final_expected_answer = expected_answer
        if not final_expected_answer or not final_expected_answer.strip():
            # Generate using RAG
            concepts_to_use = covered[:2] if covered else [subtopic] if subtopic else []
            if concepts_to_use:
                final_expected_answer = cls.generate_rag_expected_answer(question, concepts_to_use)
                print(f"📚 Generated RAG expected answer")
            else:
                print(f"⚠️ No concepts available for expected answer generation")
                final_expected_answer = ""
        
        # Calculate semantic similarity
        semantic_similarity = 0.0
        if final_expected_answer and final_expected_answer.strip():
            try:
                from interview_analyzer import calculate_semantic_similarity
                semantic_similarity = calculate_semantic_similarity(answer, final_expected_answer)
                print(f"📊 Semantic similarity calculated: {semantic_similarity:.3f}")
            except Exception as e:
                print(f"⚠️ Could not calculate semantic similarity: {e}")
                semantic_similarity = 0.0
        
        # Perform gap analysis
        gap_analysis_results = []
        if answer and len(answer.strip()) > 10 and question and covered:
            gap_analysis_results = cls.perform_gap_analysis(answer, question, covered[:2])
        
        result = {
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
            "expected_answer": final_expected_answer,
            "gap_analysis": gap_analysis_results
        }
        
        print("\n" + "█"*80)
        print("✅ ANALYSIS COMPLETE")
        print("█"*80)
        print(f"   Keyword coverage: {result['keyword_coverage']:.3f}")
        print(f"   Semantic similarity: {result['semantic_similarity']:.3f}")
        print(f"   Depth: {result['depth']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Missing concepts: {len(result['missing_concepts'])}")
        print(f"   Covered concepts: {len(result['covered_concepts'])}")
        print(f"   Gap analysis points: {len(result['gap_analysis'])}")
        print("█"*80)
        
        return result