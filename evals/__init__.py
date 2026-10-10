"""Evaluation studies answering the review.

Each module is a standalone study: it builds its own corpus, runs the system,
and writes a JSON record under ``outputs/evals/``. Nothing here is imported by
the library — these are experiments, and they are kept separate so that a
result can never be mistaken for a code path the product depends on.

    python -m evals.planner_validation
    python -m evals.integrity_robustness
    python -m evals.multitask
    python -m evals.adaptation_tradeoffs
"""
