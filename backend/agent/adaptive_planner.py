# backend/agent/adaptive_planner.py

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class SessionGoal:
    """Strategic goals for a session"""
    target_mastery: float = 0.7
    min_questions_per_topic: int = 2
    max_questions_per_topic: int = 5
    target_topics: List[str] = field(default_factory=list)
    time_budget_minutes: int = 30

@dataclass
class TopicHeatmap:
    """Heatmap data for each topic"""
    topic: str
    mastery_level: float
    learning_velocity: float
    stagnation_count: int = 0
    last_improvement: float = 0
    question_count: int = 0
    consecutive_poor: int = 0
    priority_score: float = 1.0

class AdaptivePlanner:
    """
    Strategic planning engine for truly agentic interviews
    """
    
    def __init__(self):
        self.session_goals: Dict[str, SessionGoal] = {}
        self.heatmaps: Dict[str, Dict[str, TopicHeatmap]] = {}  # user_id -> topic -> heatmap
        self.session_history: Dict[str, List[Dict]] = {}  # user_id -> session summaries
    
    def create_session_plan(self, user_id: int, masteries: Dict, time_limit: int = 30) -> Dict:
        """
        Create a strategic session plan based on user's learning history
        """
        # Initialize heatmap for this user if needed
        if user_id not in self.heatmaps:
            self.heatmaps[user_id] = {}
        
        # Update heatmaps with current mastery data
        for topic, data in masteries.items():
            if topic not in self.heatmaps[user_id]:
                self.heatmaps[user_id][topic] = TopicHeatmap(
                    topic=topic,
                    mastery_level=data.get('mastery_level', 0.0),
                    learning_velocity=data.get('learning_velocity', 0.0)
                )
            else:
                heatmap = self.heatmaps[user_id][topic]
                old_mastery = heatmap.mastery_level
                heatmap.mastery_level = data.get('mastery_level', 0.0)
                heatmap.learning_velocity = data.get('learning_velocity', 0.0)
                
                # Detect stagnation
                if abs(heatmap.mastery_level - old_mastery) < 0.05:
                    heatmap.stagnation_count += 1
                else:
                    heatmap.stagnation_count = 0
                    heatmap.last_improvement = datetime.now().timestamp()
        
        # Calculate priority scores
        for heatmap in self.heatmaps[user_id].values():
            # Lower mastery = higher priority
            mastery_factor = 1.0 - heatmap.mastery_level
            
            # Stagnation penalty
            stagnation_penalty = min(heatmap.stagnation_count * 0.1, 0.5)
            
            # Learning velocity boost (if improving)
            velocity_boost = max(heatmap.learning_velocity * 2, 0)
            
            heatmap.priority_score = mastery_factor + stagnation_penalty - velocity_boost
        
        # Sort topics by priority
        sorted_topics = sorted(
            self.heatmaps[user_id].values(),
            key=lambda x: x.priority_score,
            reverse=True
        )
        
        # Create session plan
        plan = {
            'primary_focus': sorted_topics[0].topic if sorted_topics else None,
            'secondary_focus': sorted_topics[1].topic if len(sorted_topics) > 1 else None,
            'tertiary_focus': sorted_topics[2].topic if len(sorted_topics) > 2 else None,
            
            # 🔥 FIX: Handle empty sorted_topics safely
            'estimated_questions': (
                {
                    sorted_topics[0].topic: min(5, max(2, int(5 * sorted_topics[0].priority_score)))
                }
                if sorted_topics 
                else {"DBMS": 3}  # Default fallback
            ),
            
            'target_mastery_improvement': 0.1,
            'time_budget': time_limit,
            'strategy': self._determine_strategy(sorted_topics)
        }
        
        # Add secondary and tertiary estimates
        if len(sorted_topics) > 1:
            plan['estimated_questions'][sorted_topics[1].topic] = min(4, max(2, int(4 * sorted_topics[1].priority_score)))
        if len(sorted_topics) > 2:
            plan['estimated_questions'][sorted_topics[2].topic] = min(3, max(1, int(3 * sorted_topics[2].priority_score)))
        
        return plan
    
    def _determine_strategy(self, heatmaps: List[TopicHeatmap]) -> str:
        """Determine teaching strategy based on heatmap data"""
        if not heatmaps:
            return "balanced"
        
        # Check for struggling topics
        struggling = [h for h in heatmaps if h.mastery_level < 0.4]
        if struggling:
            return "remedial"
        
        # Check for stagnant topics
        stagnant = [h for h in heatmaps if h.stagnation_count > 3]
        if stagnant:
            return "breakthrough"
        
        # Check for advancing topics
        advancing = [h for h in heatmaps if h.learning_velocity > 0.1]
        if advancing:
            return "accelerated"
        
        return "balanced"
    
    def evaluate_session(self, user_id: int, session_data: Dict) -> Dict:
        """
        Evaluate session performance and update learning strategy
        """
        if user_id not in self.session_history:
            self.session_history[user_id] = []
        
        # Store session summary
        self.session_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'questions_asked': session_data.get('questions_asked', 0),
            'avg_semantic': session_data.get('avg_semantic', 0),
            'avg_keyword': session_data.get('avg_keyword', 0),
            'improvement': session_data.get('improvement', 0),
            'topics_covered': session_data.get('topics_covered', [])
        })
        
        # Analyze long-term trends
        if len(self.session_history[user_id]) >= 3:
            recent = self.session_history[user_id][-3:]
            improvements = [s['improvement'] for s in recent]
            
            # Detect learning plateaus
            if all(i < 0.05 for i in improvements):
                return {
                    'recommendation': 'increase_difficulty',
                    'reason': 'Learning plateau detected',
                    'action': 'Move to more challenging concepts'
                }
            
            # Detect rapid improvement
            if improvements[-1] > 0.2:
                return {
                    'recommendation': 'accelerate',
                    'reason': 'Rapid learning detected',
                    'action': 'Increase pace and complexity'
                }
        
        return {
            'recommendation': 'continue',
            'reason': 'Normal progression',
            'action': 'Maintain current strategy'
        }
    
    def get_next_topic_strategic(self, user_id: int, current_topic: str, 
                                  available_topics: List[str]) -> str:
        """
        Strategically choose next topic based on learning patterns
        """
        if user_id not in self.heatmaps:
            return available_topics[0] if available_topics else "DBMS"
        
        # Get heatmaps for available topics
        available_heatmaps = [
            h for h in self.heatmaps[user_id].values()
            if h.topic in available_topics
        ]
        
        if not available_heatmaps:
            return available_topics[0]
        
        # Sort by priority score
        available_heatmaps.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Don't choose the same topic twice in a row unless it's high priority
        if available_heatmaps[0].topic == current_topic and len(available_heatmaps) > 1:
            # Check if current topic still has very high priority
            if available_heatmaps[0].priority_score > available_heatmaps[1].priority_score * 1.5:
                return available_heatmaps[0].topic
            return available_heatmaps[1].topic
        
        return available_heatmaps[0].topic

# Global planner instance
adaptive_planner = AdaptivePlanner()