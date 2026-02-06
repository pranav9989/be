"""
backend/agent/model_manager.py
Multi-model optimization for different tasks
"""
from typing import Dict, Any, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from enum import Enum
import os

class ModelType(Enum):
    FAST = "fast"      # Small, fast model for simple tasks
    BALANCED = "balanced"  # Medium model for general tasks
    EMBEDDING = "embedding"  # For semantic similarity

class ModelManager:
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize different models for different tasks"""
        # Initialize embedding model (always available)
        self.models[ModelType.EMBEDDING] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Gemini models if API key is available
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                
                # Fast model for simple tasks
                self.models[ModelType.FAST] = genai.GenerativeModel('gemini-1.5-flash')
                
                # Balanced model for general reasoning
                self.models[ModelType.BALANCED] = genai.GenerativeModel('gemini-1.5-flash')
                
                print("✅ Gemini models initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini models: {e}")
        else:
            print("⚠️ No Gemini API key - using fallback models")
    
    def get_model(self, task_type: str) -> Any:
        """Get appropriate model based on task"""
        
        task_model_mapping = {
            # Task: model_type
            "classify_intent": ModelType.FAST,
            "extract_keywords": ModelType.FAST,
            "generate_question": ModelType.BALANCED,
            "analyze_answer": ModelType.BALANCED,
            "semantic_similarity": ModelType.EMBEDDING,
            "summarize": ModelType.BALANCED,
        }
        
        model_type = task_model_mapping.get(task_type, ModelType.BALANCED)
        
        # Fallback to embedding model if Gemini not available
        if model_type in [ModelType.FAST, ModelType.BALANCED] and model_type not in self.models:
            print(f"⚠️ {model_type.value} model not available, using embedding model as fallback")
            return self.models[ModelType.EMBEDDING]
        
        return self.models.get(model_type)
    
    def generate_with_model(self, prompt: str, task_type: str, **kwargs) -> str:
        """Generate content with appropriate model"""
        model = self.get_model(task_type)
        
        if isinstance(model, SentenceTransformer):
            # For embedding tasks, return empty string
            return ""
        
        try:
            response = model.generate_content(prompt, **kwargs)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Model generation failed for {task_type}: {e}")
            return ""
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embedding model"""
        embedding_model = self.models[ModelType.EMBEDDING]
        
        embeddings1 = embedding_model.encode([text1], normalize_embeddings=True)[0]
        embeddings2 = embedding_model.encode([text2], normalize_embeddings=True)[0]
        
        similarity = np.dot(embeddings1, embeddings2) / (
            np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
        )
        
        return float(similarity)

# Global model manager instance
model_manager = ModelManager()