"""CSV exporter for ranked candidate score results."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

from src.models.score_result import ScoreResult


logger = logging.getLogger(__name__)

CSV_HEADER = ("candidate_id", "rank", "score", "reasoning")
_SCORE_FILE_PREFIX = "can_ranker_scores"


def _global_score_bounds() -> tuple[float, float] | None:
    """Read raw score bounds recorded by worker processes."""
    temp_dir = tempfile.gettempdir()
    current_pid = os.getpid()
    prefix = f"{_SCORE_FILE_PREFIX}_{current_pid}_"
    values: list[float] = []
    score_files: list[str] = []

    for filename in os.listdir(temp_dir):
        if filename.startswith(prefix) and filename.endswith(".txt"):
            score_files.append(os.path.join(temp_dir, filename))

    for file_path in score_files:
        try:
            with open(file_path, "r", encoding="utf-8") as score_file:
                for line in score_file:
                    try:
                        values.append(float(line.strip()))
                    except ValueError:
                        continue
        except OSError:
            logger.warning("Unable to read score normalization file: %s", file_path)
        finally:
            try:
                os.remove(file_path)
            except OSError:
                pass

    if not values:
        return None
    return min(values), max(values)


def _normalize_export_scores(score_results: list[ScoreResult]) -> list[ScoreResult]:
    """Normalize provided scores to 0-10 and assign deterministic ranks."""
    if not score_results:
        return []

    raw_scores = [float(result.total_score) for result in score_results]
    global_bounds = _global_score_bounds()
    if global_bounds is None:
        min_score = min(raw_scores)
        max_score = max(raw_scores)
    else:
        min_score, max_score = global_bounds

    if max_score == min_score:
        for result in score_results:
            result.total_score = 5.0
    else:
        score_range = max_score - min_score
        for result in score_results:
            result.total_score = 10.0 * (
                (float(result.total_score) - min_score) / score_range
            )

    ranked_results = sorted(
        score_results,
        key=lambda result: (-result.total_score, result.candidate_id),
    )
    for rank, result in enumerate(ranked_results, start=1):
        result.ranking = rank

    return ranked_results


def export_score_results(
    score_results: Iterable[ScoreResult],
    output_path: str | Path,
) -> None:
    """Export ranked score results to a UTF-8 CSV file.

    Args:
        score_results: Ranked score results to export.
        output_path: Destination CSV path.

    Raises:
        OSError: If the output file cannot be written.
    """
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    normalized_results = _normalize_export_scores(list(score_results))
    rows_written = 0
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(CSV_HEADER)

        for result in normalized_results:
            writer.writerow(
                (
                    result.candidate_id,
                    result.ranking,
                    f"{round(result.total_score, 4):.4f}",
                    result.reasoning,
                )
            )
            rows_written += 1

    logger.info("Exported %d ranked candidate(s) to %s", rows_written, path)
