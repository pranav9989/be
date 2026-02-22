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
    Uses evidence-based stagnation detection (consecutive poor answers)
    """
    
    def __init__(self):
        self.session_goals: Dict[str, SessionGoal] = {}
        self.heatmaps: Dict[str, Dict[str, TopicHeatmap]] = {}  # user_id -> topic -> heatmap
        self.session_history: Dict[str, List[Dict]] = {}  # user_id -> session summaries
    
    def create_session_plan(self, user_id: int, masteries: Dict, time_limit: int = 30) -> Dict:
        """
        Create a strategic session plan based on user's learning history
        Uses evidence-based stagnation (consecutive poor answers) not EMA noise
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
                    learning_velocity=data.get('learning_velocity', 0.0),
                    consecutive_poor=data.get('consecutive_poor', 0)
                )
            else:
                heatmap = self.heatmaps[user_id][topic]
                old_mastery = heatmap.mastery_level
                heatmap.mastery_level = data.get('mastery_level', 0.0)
                heatmap.learning_velocity = data.get('learning_velocity', 0.0)
                heatmap.consecutive_poor = data.get('consecutive_poor', 0)
                
                # 🔥 FIXED: Stagnation based on CONSECUTIVE POOR, not EMA noise
                if heatmap.consecutive_poor >= 2:
                    # User is struggling with this topic (multiple poor answers in a row)
                    heatmap.stagnation_count += 1
                    heatmap.last_improvement = 0  # Reset improvement timestamp
                    print(f"📊 Topic {topic}: {heatmap.consecutive_poor} consecutive poor answers → stagnation +{heatmap.stagnation_count}")
                else:
                    # User is improving or stable - slowly reduce stagnation
                    if heatmap.stagnation_count > 0:
                        heatmap.stagnation_count = max(0, heatmap.stagnation_count - 1)
                        heatmap.last_improvement = datetime.now().timestamp()
                        print(f"📊 Topic {topic}: Improvement detected, stagnation -1 → {heatmap.stagnation_count}")
        
        # Calculate priority scores
        for heatmap in self.heatmaps[user_id].values():
            # Lower mastery = higher priority
            mastery_factor = 1.0 - heatmap.mastery_level
            
            # 🔥 REDUCED STAGNATION PENALTY - max 0.2 instead of 0.5
            stagnation_penalty = min(heatmap.stagnation_count * 0.05, 0.2)
            
            # Learning velocity boost (if improving)
            velocity_boost = max(heatmap.learning_velocity * 2, 0)
            
            # Add small boost if recently improved
            if heatmap.last_improvement > 0:
                days_since_improvement = (datetime.now().timestamp() - heatmap.last_improvement) / (24 * 3600)
                if days_since_improvement < 7:  # Improved in last week
                    velocity_boost += 0.1
            
            heatmap.priority_score = mastery_factor + stagnation_penalty - velocity_boost
            heatmap.priority_score = max(0.1, min(2.0, heatmap.priority_score))  # Clamp to reasonable range
        
        # Sort topics by priority (highest first)
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
        
        # Add stagnation info for debugging
        plan['stagnation_info'] = {
            heatmap.topic: heatmap.stagnation_count 
            for heatmap in sorted_topics[:3]
        }
        
        return plan
    
    def _determine_strategy(self, heatmaps: List[TopicHeatmap]) -> str:
        """Determine teaching strategy based on heatmap data"""
        if not heatmaps:
            return "balanced"
        
        # Check for struggling topics (low mastery)
        struggling = [h for h in heatmaps if h.mastery_level < 0.4]
        if struggling:
            return "remedial"
        
        # Check for stagnant topics (high stagnation count from repeated poor performance)
        stagnant = [h for h in heatmaps if h.stagnation_count > 2]  # Reduced threshold
        if stagnant:
            return "breakthrough"
        
        # Check for advancing topics (positive learning velocity)
        advancing = [h for h in heatmaps if h.learning_velocity > 0.1]
        if advancing:
            return "accelerated"
        
        return "balanced"
    
    def evaluate_session(self, user_id: int, session_data: Dict) -> Dict:
        """
        Evaluate session performance and update learning strategy
        Tracks long-term trends for strategic adjustments
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
        
        # Keep only last 10 sessions to prevent memory bloat
        if len(self.session_history[user_id]) > 10:
            self.session_history[user_id] = self.session_history[user_id][-10:]
        
        # Analyze long-term trends
        if len(self.session_history[user_id]) >= 3:
            recent = self.session_history[user_id][-3:]
            improvements = [s['improvement'] for s in recent]
            avg_semantic = sum(s['avg_semantic'] for s in recent) / 3
            
            # Detect learning plateaus (no improvement for 3 sessions)
            if all(i < 0.05 for i in improvements) and avg_semantic < 0.6:
                return {
                    'recommendation': 'increase_difficulty',
                    'reason': 'Learning plateau detected',
                    'action': 'Move to more challenging concepts or try different approach'
                }
            
            # Detect rapid improvement
            if improvements[-1] > 0.2 and avg_semantic > 0.7:
                return {
                    'recommendation': 'accelerate',
                    'reason': 'Rapid learning detected',
                    'action': 'Increase pace and complexity'
                }
            
            # Detect consistent struggle
            if all(i < 0.1 for i in improvements) and avg_semantic < 0.4:
                return {
                    'recommendation': 'remediate',
                    'reason': 'Consistent difficulty detected',
                    'action': 'Focus on fundamentals and simplify'
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
        Uses priority scores from heatmaps
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
            # Check if current topic still has very high priority (1.5x next)
            if available_heatmaps[0].priority_score > available_heatmaps[1].priority_score * 1.5:
                return available_heatmaps[0].topic
            return available_heatmaps[1].topic
        
        return available_heatmaps[0].topic
    
    def get_heatmap_summary(self, user_id: int) -> Dict:
        """Get a summary of current heatmaps for display/debugging"""
        if user_id not in self.heatmaps:
            return {}
        
        summary = {}
        for topic, heatmap in self.heatmaps[user_id].items():
            summary[topic] = {
                'mastery': round(heatmap.mastery_level, 3),
                'velocity': round(heatmap.learning_velocity, 3),
                'stagnation': heatmap.stagnation_count,
                'consecutive_poor': heatmap.consecutive_poor,
                'priority': round(heatmap.priority_score, 3)
            }
        
        return summary


# Global planner instance
adaptive_planner = AdaptivePlanner()