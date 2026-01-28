# backend/agent/topic_selector.py

import random

TOPIC_WEIGHTS = {
    "DBMS": 0.33,
    "OS": 0.33,
    "OOPS": 0.34
}


def choose_topic(already_asked: set) -> str:
    available = {k: v for k, v in TOPIC_WEIGHTS.items() if k not in already_asked}

    if not available:
        available = TOPIC_WEIGHTS

    topics = list(available.keys())
    weights = list(available.values())

    return random.choices(topics, weights=weights, k=1)[0]
