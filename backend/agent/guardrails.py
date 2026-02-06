"""
backend/agent/guardrails.py
Safety guardrails for the agent system
"""
import re
import json
from typing import Dict, Any, List, Optional
from enum import Enum

class GuardrailType(Enum):
    RELEVANCE = "relevance"
    SAFETY = "safety"
    PII = "pii"
    MODERATION = "moderation"

class Guardrail:
    """Base guardrail class"""
    def __init__(self, name: str, guardrail_type: GuardrailType, description: str):
        self.name = name
        self.type = guardrail_type
        self.description = description
    
    def check(self, input_text: str) -> Dict[str, Any]:
        """Check if input passes this guardrail"""
        return {"passed": True, "message": "", "risk_level": "low"}
    
    def __call__(self, input_text: str) -> bool:
        result = self.check(input_text)
        return result["passed"]

class RelevanceGuardrail(Guardrail):
    """Ensures content stays on technical interview topics"""
    def __init__(self):
        super().__init__(
            name="relevance_filter",
            guardrail_type=GuardrailType.RELEVANCE,
            description="Filters non-technical or off-topic content"
        )
        self.allowed_topics = ["DBMS", "OS", "OOPS", "Data Structures", "Algorithms", 
                              "Networking", "System Design", "Programming"]
        self.blocked_patterns = [
            r"(?i)personal.*(life|family|hobbies|politics|religion)",
            r"(?i)salary|compensation|benefits",
            r"(?i)confidential|proprietary",
            r"(?i)hack|exploit|bypass"
        ]
    
    def check(self, input_text: str) -> Dict[str, Any]:
        input_lower = input_text.lower()
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_lower):
                return {
                    "passed": False,
                    "message": f"Content contains blocked pattern: {pattern}",
                    "risk_level": "high"
                }
        
        return {"passed": True, "message": "", "risk_level": "low"}

class PIIGuardrail(Guardrail):
    """Prevents exposure of personally identifiable information"""
    def __init__(self):
        super().__init__(
            name="pii_filter",
            guardrail_type=GuardrailType.PII,
            description="Detects and filters PII from outputs"
        )
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        }
    
    def check(self, input_text: str) -> Dict[str, Any]:
        detected_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, input_text)
            if matches:
                detected_pii.append({"type": pii_type, "count": len(matches)})
        
        if detected_pii:
            return {
                "passed": False,
                "message": f"Detected PII: {detected_pii}",
                "risk_level": "high",
                "detected_pii": detected_pii
            }
        
        return {"passed": True, "message": "", "risk_level": "low"}

class GuardrailManager:
    """Manages multiple guardrails"""
    def __init__(self):
        self.guardrails = {
            "relevance": RelevanceGuardrail(),
            "pii": PIIGuardrail()
        }
    
    def check_input(self, input_text: str, guardrail_types: List[str] = None) -> Dict[str, Any]:
        """Run multiple guardrails on input"""
        if guardrail_types is None:
            guardrail_types = ["relevance", "pii"]
        
        results = {}
        for gr_type in guardrail_types:
            guardrail = self.guardrails.get(gr_type)
            if guardrail:
                results[gr_type] = guardrail.check(input_text)
        
        # Overall assessment
        all_passed = all(r["passed"] for r in results.values() if r)
        highest_risk = max(
            [r.get("risk_level", "low") for r in results.values() if r],
            key=lambda x: ["low", "medium", "high"].index(x)
        )
        
        return {
            "passed": all_passed,
            "results": results,
            "overall_risk": highest_risk,
            "should_block": highest_risk == "high" or not all_passed
        }
    
    def check_output(self, output_text: str) -> Dict[str, Any]:
        """Check agent output for safety"""
        return self.check_input(output_text, ["relevance", "pii"])

# Global guardrail manager instance
guardrail_manager = GuardrailManager()