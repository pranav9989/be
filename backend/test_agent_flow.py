from agent.controller import InterviewAgentController

session_id = "smoke_test"
user_id = 42

print("\n--- START SESSION ---")
q1 = InterviewAgentController.start_session(session_id, user_id)
print("Q1:", q1)

analysis_low = {
    "coverage_score": 0.3,
    "depth": "shallow",
    "missing_concepts": ["indexing"]
}

print("\n--- ANSWER 1 (WEAK) ---")
r1 = InterviewAgentController.handle_answer(
    session_id,
    answer="It reduces redundancy",
    analysis=analysis_low
)
print(r1)

analysis_good = {
    "coverage_score": 0.85,
    "depth": "deep",
    "missing_concepts": []
}

print("\n--- ANSWER 2 (STRONG) ---")
r2 = InterviewAgentController.handle_answer(
    session_id,
    answer="Detailed explanation with examples",
    analysis=analysis_good
)
print(r2)
