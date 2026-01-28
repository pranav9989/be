# backend/agent/state.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import time


@dataclass
class QARecord:
    question: str
    topic: Optional[str]
    difficulty: str
    answer: str
    analysis: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InterviewAgentState:
    # -------- Session --------
    session_id: str
    user_id: int
    status: str = "CREATED"
    started_at: float = field(default_factory=time.time)

    # -------- Current Context --------
    current_question: Optional[str] = None
    current_topic: Optional[str] = None
    difficulty: str = "medium"
    followup_count: int = 0

    # -------- History --------
    history: List[QARecord] = field(default_factory=list)

    # -------- Aggregate Signals --------
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    # -------- Limits --------
    max_followups_per_topic: int = 3
    max_questions_total: int = 12
    max_duration_sec: int = 30 * 60  # ⏱️ 30 minutes

    # -------- Helpers --------
    def total_questions_asked(self) -> int:
        return len(self.history)

    def time_remaining_sec(self) -> int:
        elapsed = time.time() - self.started_at
        return max(0, int(self.max_duration_sec - elapsed))

    def is_time_over(self) -> bool:
        return self.time_remaining_sec() <= 0

    def reset_for_new_topic(self, topic: str):
        self.current_topic = topic
        self.followup_count = 0
        self.difficulty = "medium"
