"""
backend/agent/tools.py
Well-defined tools for the agent system
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class Tool:
    """Formal tool definition"""
    name: str
    description: str
    parameters: Dict[str, str]
    risk_level: str = "low"  # low, medium, high
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "risk_level": self.risk_level
        }

class ToolRegistry:
    """Registry for managing and executing tools"""
    def __init__(self):
        self.tools = {}
        self._init_default_tools()
    
    def _init_default_tools(self):
        """Initialize default interview agent tools"""
        from rag import generate_rag_response
        
        # Tool 1: Question Generation
        self.register(
            Tool(
                name="generate_technical_question",
                description="Generate technical interview questions",
                parameters={
                    "topic": "str",
                    "difficulty": "str (easy/medium/hard)",
                    "context": "str"
                },
                risk_level="low"
            ),
            self._generate_question_tool
        )
        
        # Tool 2: Answer Analysis
        self.register(
            Tool(
                name="analyze_answer_content",
                description="Analyze technical answers for quality",
                parameters={
                    "question": "str",
                    "answer": "str",
                    "topic": "str"
                },
                risk_level="low"
            ),
            self._analyze_answer_tool
        )
        
        # Tool 3: Topic Selection
        self.register(
            Tool(
                name="select_next_topic",
                description="Select next interview topic based on performance",
                parameters={
                    "covered_topics": "List[str]",
                    "strengths": "List[str]",
                    "weaknesses": "List[str]"
                },
                risk_level="low"
            ),
            self._select_topic_tool
        )
    
    def register(self, tool: Tool, func):
        """Register a tool with its implementation"""
        self.tools[tool.name] = {
            "tool": tool,
            "function": func
        }
        print(f"🔧 Registered tool: {tool.name} (risk: {tool.risk_level})")
    
    def execute(self, tool_name: str, **kwargs):
        """Execute a registered tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool_info = self.tools[tool_name]
        tool = tool_info["tool"]
        
        # Log tool execution
        print(f"🛠️ Executing tool: {tool.name}")
        print(f"   Parameters: {kwargs}")
        print(f"   Risk level: {tool.risk_level}")
        
        # Execute the function
        return tool_info["function"](**kwargs)
    
    def list_tools(self):
        """List all registered tools"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def _generate_question_tool(self, topic: str, difficulty: str = "medium", context: str = ""):
        """Tool implementation for question generation"""
        from rag import generate_rag_response
        
        prompt = f"""
        Generate a {difficulty} difficulty technical interview question about {topic}.
        Context: {context}
        Return ONLY the question.
        """
        
        return generate_rag_response("question_generation", prompt).strip()
    
    def _analyze_answer_tool(self, question: str, answer: str, topic: str):
        """Tool implementation for answer analysis"""
        from .analyzer import analyze_answer
        return analyze_answer(question, answer)
    
    def _select_topic_tool(self, covered_topics: List[str], strengths: List[str], weaknesses: List[str]):
        """Tool implementation for topic selection"""
        from .topic_selector import choose_topic
        # Use the existing topic selector
        return choose_topic(set(covered_topics))

# Global tool registry instance
tool_registry = ToolRegistry()