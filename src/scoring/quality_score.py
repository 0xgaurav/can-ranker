"""Calculate candidate quality scores from match results."""

from __future__ import annotations

import logging
import math
from typing import Any

from src.models.match_result import MatchResult


logger = logging.getLogger(__name__)

_HIGH_SIMILARITY_THRESHOLD = 0.80
_LOW_SKILL_SIMILARITY_THRESHOLD = 0.50
_EXPERIENCE_REWARD = 0.05
_EDUCATION_REWARD = 0.05
_LOW_SKILL_PENALTY = 0.10


def _get_finite_float(match: Any, attribute_name: str) -> float | None:
    """Return a finite float attribute, or ``None`` when it is malformed."""
    try:
        value = float(getattr(match, attribute_name))
    except (AttributeError, TypeError, ValueError, OverflowError):
        logger.warning(
            "Ignoring invalid %s while calculating quality score",
            attribute_name,
        )
        return None

    if not math.isfinite(value):
        logger.warning(
            "Ignoring non-finite %s while calculating quality score",
            attribute_name,
        )
        return None

    return value


def calculate_quality_score(match: MatchResult) -> float:
    """Calculate a normalized quality score for a candidate match.

    The score begins with ``overall_similarity``. Strong experience and
    education similarities each add ``0.05``, while skill similarity below
    ``0.50`` subtracts ``0.10``. The result is constrained to the inclusive
    range ``[0.0, 1.0]``.

    Malformed component similarities are ignored. If the match or its base
    overall similarity is invalid, the function returns ``0.0``.

    Args:
        match: Candidate-to-job match data used for scoring.

    Returns:
        The calculated quality score as a float.
    """
    logger.info("Starting quality score calculation")

    if match is None:
        logger.error("Cannot calculate quality score: match is None")
        return 0.0

    overall_similarity = _get_finite_float(match, "overall_similarity")
    if overall_similarity is None:
        logger.error(
            "Cannot calculate quality score without overall_similarity"
        )
        return 0.0

    score = overall_similarity

    experience_similarity = _get_finite_float(
        match, "experience_similarity"
    )
    if (
        experience_similarity is not None
        and experience_similarity >= _HIGH_SIMILARITY_THRESHOLD
    ):
        score += _EXPERIENCE_REWARD
        logger.info(
            "Applied %.2f reward for experience similarity %.3f",
            _EXPERIENCE_REWARD,
            experience_similarity,
        )

    education_similarity = _get_finite_float(match, "education_similarity")
    if (
        education_similarity is not None
        and education_similarity >= _HIGH_SIMILARITY_THRESHOLD
    ):
        score += _EDUCATION_REWARD
        logger.info(
            "Applied %.2f reward for education similarity %.3f",
            _EDUCATION_REWARD,
            education_similarity,
        )

    skill_similarity = _get_finite_float(match, "skill_similarity")
    if (
        skill_similarity is not None
        and skill_similarity < _LOW_SKILL_SIMILARITY_THRESHOLD
    ):
        score -= _LOW_SKILL_PENALTY
        logger.info(
            "Applied %.2f penalty for skill similarity %.3f",
            _LOW_SKILL_PENALTY,
            skill_similarity,
        )

    clamped_score = max(0.0, min(1.0, score))
    if clamped_score != score:
        logger.info(
            "Clamped quality score from %.3f to %.3f", score, clamped_score
        )

    logger.info("Quality score calculation complete: %.3f", clamped_score)
    return float(clamped_score)
