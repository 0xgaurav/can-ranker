"""Domain model for candidate-to-job matching results."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MatchResult:
    """Store component similarities from a completed match operation."""

    candidate_id: str
    overall_similarity: float
    skill_similarity: float
    experience_similarity: float
    education_similarity: float
    behavior_similarity: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
