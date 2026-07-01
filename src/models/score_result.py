"""Domain model for externally produced candidate score results."""

from dataclasses import dataclass


@dataclass(slots=True)
class ScoreResult:
    """Store a completed candidate scoring and ranking result."""

    candidate_id: str
    total_score: float
    ranking: int
    recommendation: str
    reasoning: str
