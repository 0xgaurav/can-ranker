"""Score calculator module for deterministic recruiter-style ranking."""

from __future__ import annotations

import inspect
import logging
import os
import tempfile
from typing import Any

from src.models.match_result import MatchResult
from src.models.score_result import ScoreResult
from src.scoring.behavior_score import calculate_behavior_score
from src.scoring.final_score import calculate_final_score
from src.scoring.quality_score import calculate_quality_score


logger = logging.getLogger(__name__)
_SCORE_FILE_PREFIX = "can_ranker_scores"


def _cleanup_stale_score_files() -> None:
    """Remove stale score files for this process when a new run starts."""
    temp_dir = tempfile.gettempdir()
    current_pid = os.getpid()
    for filename in os.listdir(temp_dir):
        if filename.startswith(f"{_SCORE_FILE_PREFIX}_{current_pid}_"):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except OSError:
                pass


_cleanup_stale_score_files()


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a numerical value between a minimum and maximum boundary."""
    try:
        return max(min_val, min(max_val, float(value)))
    except (TypeError, ValueError):
        return min_val


def _get_recommendation(score: float) -> str:
    """Determine the recommendation for a 0-10 recruiter score."""
    if score >= 8.5:
        return "Strong Hire"
    if score >= 7.0:
        return "Hire"
    if score >= 5.5:
        return "Consider"
    if score >= 4.0:
        return "Weak Consider"
    return "Reject"


def _caller_context() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read candidate and job context from the existing pipeline call frame.

    The pipeline currently calls ``calculate_score(match)``. Keeping that API
    intact lets this scoring-only change use richer candidate signals without
    editing orchestration, matching, fraud, consistency, or reasoning modules.
    """
    frame = inspect.currentframe()
    if frame is None:
        return None, None

    current = frame.f_back
    depth = 0
    candidate = None
    job_description = None
    while current is not None and depth < 8:
        local_candidate = current.f_locals.get("candidate")
        local_job = (
            current.f_locals.get("job_description")
            or current.f_locals.get("job")
        )

        if candidate is None and isinstance(local_candidate, dict):
            candidate = local_candidate
        if job_description is None and isinstance(local_job, dict):
            job_description = local_job
        if candidate is not None and job_description is not None:
            break

        current = current.f_back
        depth += 1

    return candidate, job_description


def _record_raw_score(score: float) -> None:
    """Persist raw scores so export can min-max normalize over all candidates."""
    parent_pid = os.getppid()
    worker_pid = os.getpid()
    file_path = os.path.join(
        tempfile.gettempdir(),
        f"{_SCORE_FILE_PREFIX}_{parent_pid}_{worker_pid}.txt",
    )
    try:
        with open(file_path, "a", encoding="utf-8") as score_file:
            score_file.write(f"{score:.17g}\n")
    except OSError:
        logger.warning("Unable to record raw score for global normalization")


def calculate_score(match: MatchResult) -> ScoreResult:
    """Calculate a deterministic 0-10 score for a candidate match."""
    candidate_id = getattr(match, "candidate_id", "") if match else ""

    if not match or (not candidate_id and not hasattr(match, "overall_similarity")):
        logger.error("Invalid or malformed MatchResult provided")
        return ScoreResult(
            candidate_id=candidate_id,
            total_score=0.0,
            ranking=0,
            recommendation="Reject",
            reasoning="Malformed or invalid match input.",
        )

    try:
        behavior_score = calculate_behavior_score(match)
        quality_score = calculate_quality_score(match)
        candidate, job_description = _caller_context()

        raw_total = calculate_final_score(
            match,
            behavior_score,
            quality_score,
            candidate=candidate,
            job_description=job_description,
        )
        total_score = _clamp(raw_total, 0.0, 10.0)
        _record_raw_score(total_score)
        recommendation = _get_recommendation(total_score)

        reasoning = (
            f"Candidate achieved a deterministic recruiter score of "
            f"{total_score:.4f}/10. Recommendation is {recommendation}."
        )

        logger.info(
            "Score calculation complete: total_score=%.4f, recommendation=%s",
            total_score,
            recommendation,
        )

        return ScoreResult(
            candidate_id=candidate_id,
            total_score=total_score,
            ranking=0,
            recommendation=recommendation,
            reasoning=reasoning,
        )

    except Exception as exc:
        logger.exception("Error occurred while calculating final score: %s", exc)
        return ScoreResult(
            candidate_id=candidate_id,
            total_score=0.0,
            ranking=0,
            recommendation="Reject",
            reasoning="An internal error occurred during score calculation.",
        )
