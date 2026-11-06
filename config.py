"""Paths and settings for the Kuku Promo Finder application."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

VALIDATED_PATH = OUTPUTS / "validated_promos.json"
SUMMARY_PATH = OUTPUTS / "validation_summary.json"
QUEUE_PATH = OUTPUTS / "human_review_queue.json"
RANKED_PATH = OUTPUTS / "ranked_promos.json"
SCORED_PATH = OUTPUTS / "scored_candidates.json"
DECISIONS_PATH = OUTPUTS / "reviewer_decisions.json"

GENERATED_PATHS = (
    VALIDATED_PATH,
    SUMMARY_PATH,
    QUEUE_PATH,
    RANKED_PATH,
    SCORED_PATH,
)

APP_TITLE = "Kuku Promo Finder"
DECISION_CHOICES = ("APPROVED", "REJECTED", "HOLD")
PAGES = (
    "Dashboard",
    "Top Promos",
    "Human Review",
    "All Candidates",
    "Review Decisions",
)

PRIORITY_ORDER = ("high", "medium", "low", "none")
