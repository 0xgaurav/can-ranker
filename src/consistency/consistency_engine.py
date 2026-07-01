"""Orchestrate deterministic candidate consistency analysis."""

import logging

from src.consistency.behavior_consistency import (
    check_behavior_consistency,
)
from src.consistency.career_consistency import check_career_consistency
from src.consistency.education_consistency import (
    check_education_consistency,
)
from src.consistency.skill_consistency import check_skill_consistency
from src.consistency.timeline_consistency import check_timeline_consistency


logger = logging.getLogger(__name__)


def _malformed_candidate_result() -> dict[str, object]:
    """Create the safe result returned for malformed candidate input.

    Returns:
        A fresh consistency-analysis result with all sections consistent.
    """
    return {
        "skills": {
            "consistent": True,
            "supported_skills": [],
            "unsupported_skills": [],
        },
        "education": {
            "consistent": True,
            "valid_entries": 0,
            "invalid_entries": 0,
        },
        "career": {
            "consistent": True,
            "valid_entries": 0,
            "invalid_entries": 0,
            "unique_companies": 0,
        },
        "timeline": {
            "consistent": True,
            "valid_roles": 0,
            "invalid_roles": 0,
        },
        "behavior": {
            "consistent": True,
            "valid_signals": 0,
            "invalid_signals": 0,
        },
        "overall_consistent": True,
    }


def analyze_candidate_consistency(
    candidate: dict[str, object],
) -> dict[str, object]:
    """Run all configured candidate consistency checks.

    The candidate is passed without modification to the skill, education,
    career, timeline, and behavior validators. Overall consistency requires
    every validator to report a consistent result.

    Args:
        candidate: Candidate mapping to analyze.

    Returns:
        The five validator results and their combined consistency decision. A
        malformed candidate produces a safe result with all sections marked
        consistent.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot analyze malformed candidate for consistency")
        return _malformed_candidate_result()

    skill_result = check_skill_consistency(candidate)
    education_result = check_education_consistency(candidate)
    career_result = check_career_consistency(candidate)
    timeline_result = check_timeline_consistency(candidate)
    behavior_result = check_behavior_consistency(candidate)

    overall_consistent = bool(
        skill_result["consistent"]
        and education_result["consistent"]
        and career_result["consistent"]
        and timeline_result["consistent"]
        and behavior_result["consistent"]
    )

    if not overall_consistent:
        logger.warning("Overall candidate consistency check failed")

    logger.info(
        "Candidate consistency analysis completed: consistent=%s",
        overall_consistent,
    )
    return {
        "skills": skill_result,
        "education": education_result,
        "career": career_result,
        "timeline": timeline_result,
        "behavior": behavior_result,
        "overall_consistent": overall_consistent,
    }
