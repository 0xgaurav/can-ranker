"""Calculate a candidate's behavior score from match results."""

from __future__ import annotations

import logging
import math
from collections.abc import Sized
from typing import Any

from src.models.match_result import MatchResult


logger = logging.getLogger(__name__)

_MISSING_SKILLS_THRESHOLD = 5
_MATCHED_SKILLS_THRESHOLD = 10
_MISSING_SKILLS_PENALTY = 0.10
_MATCHED_SKILLS_REWARD = 0.10


def _safe_length(value: Any, attribute_name: str) -> int | None:
    """Return the length of a collection, or ``None`` when it is malformed."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sized):
        logger.warning(
            "Ignoring malformed %s while calculating behavior score",
            attribute_name,
        )
        return None

    try:
        return len(value)
    except (TypeError, OverflowError):
        logger.warning(
            "Unable to determine the length of %s; ignoring it",
            attribute_name,
        )
        return None


def calculate_behavior_score(match: MatchResult) -> float:
    """Calculate and return a normalized behavior score.

    The calculation starts with ``behavior_similarity``, subtracts ``0.10``
    when more than five skills are missing, and adds ``0.10`` when more than
    ten skills are matched. The final value is constrained to ``[0.0, 1.0]``.
    Invalid input produces a safe score of ``0.0``.

    Args:
        match: Candidate-to-job matching data.

    Returns:
        The calculated behavior score as a float.
    """
    logger.info("Starting behavior score calculation")

    if match is None:
        logger.error("Cannot calculate behavior score: match is None")
        return 0.0

    try:
        similarity = float(match.behavior_similarity)
    except (AttributeError, TypeError, ValueError, OverflowError):
        logger.error(
            "Cannot calculate behavior score: invalid behavior_similarity"
        )
        return 0.0

    if not math.isfinite(similarity):
        logger.error(
            "Cannot calculate behavior score: behavior_similarity is not finite"
        )
        return 0.0

    score = similarity

    missing_count = _safe_length(
        getattr(match, "missing_skills", None), "missing_skills"
    )
    if (
        missing_count is not None
        and missing_count > _MISSING_SKILLS_THRESHOLD
    ):
        score -= _MISSING_SKILLS_PENALTY
        logger.info(
            "Applied %.2f penalty for %d missing skills",
            _MISSING_SKILLS_PENALTY,
            missing_count,
        )

    matched_count = _safe_length(
        getattr(match, "matched_skills", None), "matched_skills"
    )
    if (
        matched_count is not None
        and matched_count > _MATCHED_SKILLS_THRESHOLD
    ):
        score += _MATCHED_SKILLS_REWARD
        logger.info(
            "Applied %.2f reward for %d matched skills",
            _MATCHED_SKILLS_REWARD,
            matched_count,
        )

    clamped_score = max(0.0, min(1.0, score))
    if clamped_score != score:
        logger.info(
            "Clamped behavior score from %.3f to %.3f", score, clamped_score
        )

    logger.info("Behavior score calculation complete: %.3f", clamped_score)
    return float(clamped_score)
