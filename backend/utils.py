"""
utils.py - Utility functions and helpers
"""

import json
import os
from pathlib import Path
from flask import Flask
from datetime import datetime
from io import BytesIO

# -------------------- Template filters --------------------
def from_json_filter(json_str):
    """Convert JSON string to Python object (Jinja2 filter)."""
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []

def datetime_diff_filter(dt):
    """Calculate difference between datetime and now (Jinja2 filter)."""
    if not dt:
        return datetime.now() - datetime.now()
    return datetime.now() - dt

# -------------------- File processing --------------------
def extract_text_from_pdf(file_stream):
    """Extract text from PDF file."""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print("PDF extraction error:", e)
        import traceback
        traceback.print_exc()
        return None

def extract_text_from_docx(file_stream):
    """Extract text from DOCX file."""
    try:
        import docx
        doc = docx.Document(file_stream)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text.strip()
    except Exception as e:
        print("DOCX extraction error:", e)
        import traceback
        traceback.print_exc()
        return None

# -------------------- Resume parsing --------------------
def parse_resume_text(text):
    """Parse resume text to extract skills and experience."""
    lines = text.strip().splitlines()
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'html', 'css',
        'machine learning', 'data science', 'flask', 'django', 'mongodb', 'mysql'
    ]
    skills = []
    for line in lines:
        for kw in tech_keywords:
            if kw in line.lower() and kw.title() not in skills:
                skills.append(kw.title())

    experience_years = 0
    for line in lines:
        if 'year' in line.lower() and 'experience' in line.lower():
            tokens = line.split()
            for t in tokens:
                if t.isdigit():
                    experience_years = int(t)
                    break

    return {
        'skills': skills,
        'experience_years': experience_years,
        'raw_text': text[:1000]
    }
