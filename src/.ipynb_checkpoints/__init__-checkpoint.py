"""
upresearchLens: Automated Research Hypothesis Triage & Evaluation
"""
from .engine import (
    clean_text,
    ensure_live_link,
    build_corpus,
    load_and_vectorize,
    retrieve_similar_papers,
    novelty_score,
    feasibility_score,
    evaluate_idea,
    fetch_google_trends,
)

__all__ = [
    "clean_text",
    "ensure_live_link",
    "build_corpus",
    "load_and_vectorize",
    "retrieve_similar_papers",
    "novelty_score",
    "feasibility_score",
    "evaluate_idea",
]