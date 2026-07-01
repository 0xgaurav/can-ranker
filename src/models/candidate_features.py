"""Domain models for structured candidate features.

These dataclasses are passive data containers. Feature extraction, matching,
ranking, and scoring behavior belongs outside this module.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class SkillFeatures:
    """Normalized and categorized candidate skills."""

    normalized_skills: list[str] = field(default_factory=list)
    technical_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    skill_count: int = 0
    unique_skill_count: int = 0
    primary_skill: str | None = None


@dataclass(slots=True)
class CareerFeatures:
    """Structured measurements derived from candidate career history."""

    total_years_experience: float = 0.0
    number_of_companies: int = 0
    average_tenure_months: float = 0.0
    longest_tenure_months: int = 0
    shortest_tenure_months: int = 0
    leadership_experience: bool = False
    promotion_count: int = 0
    seniority_level: str = "unknown"
    career_growth_score: float = 0.0

    @property
    def longest_tenure(self) -> int:
        """Return the longest role tenure in months."""
        return self.longest_tenure_months

    @property
    def shortest_tenure(self) -> int:
        """Return the shortest role tenure in months."""
        return self.shortest_tenure_months


@dataclass(slots=True)
class TimelineFeatures:
    """Temporal characteristics of candidate employment history."""

    career_start_date: date | None = None
    career_end_date: date | None = None
    total_roles: int = 0
    career_span_months: int = 0
    employment_gap_count: int = 0
    total_gap_months: int = 0
    longest_gap_months: int = 0
    overlapping_role_count: int = 0
    has_current_role: bool = False


@dataclass(slots=True)
class BehaviorFeatures:
    """Observed behavioral signals without weights or scores."""

    positive_signals: list[str] = field(default_factory=list)
    risk_signals: list[str] = field(default_factory=list)
    leadership_signals: list[str] = field(default_factory=list)
    collaboration_signals: list[str] = field(default_factory=list)
    communication_signals: list[str] = field(default_factory=list)
    adaptability_signals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateFeatures:
    """Aggregate feature model for a single candidate."""

    candidate_id: str
    skill_features: SkillFeatures = field(default_factory=SkillFeatures)
    career_features: CareerFeatures = field(default_factory=CareerFeatures)
    timeline_features: TimelineFeatures = field(default_factory=TimelineFeatures)
    behavior_features: BehaviorFeatures = field(default_factory=BehaviorFeatures)
