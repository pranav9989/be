"""
config.py - Application configuration settings
"""

import os
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()

class Config:
    """Application configuration."""
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///interview_prep.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # JWT settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key-here')
    JWT_ACCESS_TOKEN_EXPIRE = timedelta(hours=24)

    # Gemini model settings
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")

    # RAG system paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAG_DIRS = [
        PROJECT_ROOT / "data" / "processed" / "faiss_gemini",
        Path("data") / "processed" / "faiss_gemini"
    ]
    INDEX_CANDIDATES = ["faiss_index_gemini.idx", "index.faiss", "faiss_index_gemini.faiss", "faiss_index.idx"]
    METAS_CANDIDATES = ["metas.json", "metas.jsonl", "metas_full.json"]
    CONFIG_DIR = PROJECT_ROOT / "config"
    TOPIC_RULES_FILE = CONFIG_DIR / "topic_rules.json"
    TAXONOMY_FILE = CONFIG_DIR / "taxonomy.json"

    @staticmethod
    def init_app(app):
        """Initialize application with configuration."""
        # Create upload folders
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('backend/instance', exist_ok=True)
