"""Run the end-to-end candidate ranking pipeline."""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from src.consistency.consistency_engine import analyze_candidate_consistency
from src.exporter.csv_exporter import export_score_results
from src.features.feature_extractor import extract_candidate_features
from src.fraud.fraud_engine import analyze_candidate_fraud
from src.matcher.semantic_matcher import semantic_match
from src.models.score_result import ScoreResult
from src.parser.candidate_loader import validate_candidate
from src.parser.jd_loader import load_job_description
from src.reasoning.reason_generator import generate_reason
from src.scoring.score_calculator import calculate_score
from src.utils.normalizer import normalize_skill_list

try:
    import orjson
except ImportError:  # pragma: no cover - depends on optional runtime package
    orjson = None


logger = logging.getLogger(__name__)

DEFAULT_JOB_DESCRIPTION_PATH = Path("data/raw/job_description.docx")
DEFAULT_CANDIDATES_PATH = Path("data/raw/candidates.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/output/submission.csv")
TOP_CANDIDATE_LIMIT = 100
PROCESS_CHUNKS_PER_WORKER = 4

_WORKER_JOB_DESCRIPTION: dict[str, Any] | None = None


def _json_loads(raw_line: str) -> Any:
    """Decode one JSON line with orjson when it is installed."""
    if orjson is not None:
        return orjson.loads(raw_line)
    return json.loads(raw_line)


def prepare_job_description(job_description: dict[str, Any]) -> dict[str, Any]:
    """Attach reusable normalized JD data for hot-path matching."""
    required_skills = job_description.get("required_skills", [])
    if not isinstance(required_skills, list):
        required_skills = []

    normalized_required_skills = tuple(normalize_skill_list(required_skills))
    prepared_job = dict(job_description)
    prepared_job["_normalized_required_skills"] = normalized_required_skills
    prepared_job["_normalized_required_skill_set"] = frozenset(
        normalized_required_skills
    )
    prepared_job["_required_skill_names_lower"] = frozenset(
        skill.lower() for skill in required_skills if isinstance(skill, str)
    )
    return prepared_job


def _init_worker(job_description: dict[str, Any]) -> None:
    """Initialize process-local shared data."""
    global _WORKER_JOB_DESCRIPTION
    _WORKER_JOB_DESCRIPTION = job_description
    logging.disable(logging.WARNING)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence. When omitted, ``sys.argv`` is used.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run CAN-Ranker and export the top candidate submission CSV."
    )
    parser.add_argument(
        "--job-description",
        type=Path,
        default=DEFAULT_JOB_DESCRIPTION_PATH,
        help="Path to the job description JSON or DOCX file.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help="Path to the JSON Lines candidates file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the generated submission CSV.",
    )
    return parser.parse_args(argv)


def iter_candidate_jsonl(file_path: Path) -> Iterator[dict[str, Any]]:
    """Yield structurally valid candidates from a JSON Lines file.

    Blank lines are ignored. Malformed JSON lines and structurally invalid
    candidates are logged and skipped.

    Args:
        file_path: Path to the candidate JSONL file.

    Yields:
        Valid candidate dictionaries, one at a time.

    Raises:
        FileNotFoundError: If the candidates file does not exist.
        OSError: If the candidates file cannot be read.
    """
    if not file_path.is_file():
        message = f"Candidate JSONL file does not exist: {file_path}"
        logger.error(message)
        raise FileNotFoundError(message)

    with file_path.open("r", encoding="utf-8") as candidates_file:
        for line_number, line in enumerate(candidates_file, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                candidate = _json_loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON candidate at line %d: %s",
                    line_number,
                    exc.msg,
                )
                continue
            except ValueError as exc:
                logger.warning(
                    "Skipping malformed JSON candidate at line %d: %s",
                    line_number,
                    exc,
                )
                continue

            if not validate_candidate(candidate):
                logger.warning(
                    "Skipping invalid candidate at line %d", line_number
                )
                continue

            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                logger.warning(
                    "Skipping candidate at line %d: candidate_id is empty",
                    line_number,
                )
                continue

            yield candidate


def process_candidate(
    candidate: dict[str, Any],
    job_description: dict[str, Any] | None = None,
) -> ScoreResult:
    """Run one candidate through the ranking pipeline."""

    score_result, match_result = _score_candidate(candidate, job_description)

    score_result.reasoning = generate_reason(
        candidate,
        match_result,
        score_result,
        job_description or _WORKER_JOB_DESCRIPTION or {},
    )

    return score_result


def _score_candidate(
    candidate: dict[str, Any],
    job_description: dict[str, Any] | None = None,
) -> tuple[ScoreResult, Any]:
    """Score one candidate without generating export-only reasoning."""

    if job_description is None:
        job_description = _WORKER_JOB_DESCRIPTION
    if job_description is None:
        raise RuntimeError("Job description has not been initialized")

    candidate_id = candidate["candidate_id"]

    # Feature extraction
    candidate_features = extract_candidate_features(candidate)

    # Semantic matching
    match_result = semantic_match(candidate_features, job_description)

    # Scoring
    score_result = calculate_score(match_result)

    # Fraud detection
    fraud_result = analyze_candidate_fraud(candidate)
    if fraud_result.get("fraud_detected"):
        logger.warning("Fraud detected for candidate %s", candidate_id)

    # Consistency checks
    consistency_result = analyze_candidate_consistency(candidate)
    if not consistency_result.get("overall_consistent", True):
        logger.warning("Consistency failed for candidate %s", candidate_id)

    return score_result, match_result


def _process_candidate_task(
    task: tuple[int, dict[str, Any]],
) -> tuple[int, ScoreResult | None, str, str | None]:
    """Worker entry point that keeps candidate failures isolated."""
    sequence, candidate = task
    candidate_id = str(candidate.get("candidate_id", "<unknown>"))
    try:
        score_result, _ = _score_candidate(candidate)
        return sequence, score_result, candidate_id, None
    except Exception as exc:
        return sequence, None, candidate_id, str(exc)


def _push_top_result(
    top_heap: list[tuple[float, int, int, ScoreResult, dict[str, Any]]],
    sequence: int,
    result: ScoreResult,
    candidate: dict[str, Any],
) -> None:
    """Retain only the top-ranked results with stable input-order ties."""
    entry = (result.total_score, -sequence, sequence, result, candidate)
    if len(top_heap) < TOP_CANDIDATE_LIMIT:
        heapq.heappush(top_heap, entry)
    elif entry[:2] > top_heap[0][:2]:
        heapq.heapreplace(top_heap, entry)


def _rank_top_heap(
    top_heap: list[tuple[float, int, int, ScoreResult, dict[str, Any]]],
) -> list[tuple[ScoreResult, dict[str, Any]]]:
    """Sort heap entries by score descending, then original input order."""
    top_results = [
        (result, candidate)
        for _, _, _, result, candidate in sorted(
            top_heap,
            key=lambda entry: (-entry[0], entry[2]),
        )
    ]

    for rank, (result, _) in enumerate(top_results, start=1):
        result.ranking = rank

    return top_results


def _generate_ranked_reasoning(
    ranked_results: list[tuple[ScoreResult, dict[str, Any]]],
    job_description: dict[str, Any],
) -> list[ScoreResult]:
    """Generate reasoning only for candidates that will be exported."""
    results: list[ScoreResult] = []

    for result, candidate in ranked_results:
        candidate_features = extract_candidate_features(candidate)
        match_result = semantic_match(candidate_features, job_description)
        result.reasoning = generate_reason(
            candidate,
            match_result,
            result,
            job_description,
        )
        results.append(result)

    return results


def _process_candidates_parallel(
    candidates_path: Path,
    job_description: dict[str, Any],
) -> tuple[list[ScoreResult], int, int]:
    """Process candidates across CPU cores while retaining a bounded heap."""
    worker_count = os.cpu_count() or 1
    max_pending = max(worker_count * PROCESS_CHUNKS_PER_WORKER, worker_count)
    candidate_iter = enumerate(iter_candidate_jsonl(candidates_path))
    top_heap: list[tuple[float, int, int, ScoreResult, dict[str, Any]]] = []
    processed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(job_description,),
    ) as executor:
        pending: set[Future[tuple[int, ScoreResult | None, str, str | None]]] = set()
        pending_candidates: dict[
            Future[tuple[int, ScoreResult | None, str, str | None]],
            dict[str, Any],
        ] = {}

        def submit_next() -> bool:
            try:
                sequence, candidate = next(candidate_iter)
            except StopIteration:
                return False
            future = executor.submit(_process_candidate_task, (sequence, candidate))
            pending.add(future)
            pending_candidates[future] = candidate
            return True

        for _ in range(max_pending):
            if not submit_next():
                break

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)

            for future in completed:
                candidate = pending_candidates.pop(future, {})
                try:
                    sequence, result, candidate_id, error = future.result()
                except Exception:
                    skipped_count += 1
                    logger.exception("Skipping candidate after worker failure")
                    continue

                if result is None:
                    skipped_count += 1
                    logger.warning(
                        "Skipping candidate %r after pipeline error: %s",
                        candidate_id,
                        error,
                    )
                else:
                    processed_count += 1
                    _push_top_result(top_heap, sequence, result, candidate)

                if processed_count and processed_count % 1000 == 0:
                    logger.info("Candidate processed count: %d", processed_count)

            while len(pending) < max_pending and submit_next():
                pass

    ranked_results = _generate_ranked_reasoning(
        _rank_top_heap(top_heap),
        job_description,
    )
    return ranked_results, processed_count, skipped_count


def rank_candidates(score_results: list[ScoreResult]) -> list[ScoreResult]:
    """Sort score results, assign ranks, and retain the top candidates.

    Args:
        score_results: Unranked score results.

    Returns:
        The top ranked score results in descending score order.
    """
    sorted_results = sorted(
        enumerate(score_results),
        key=lambda item: (-item[1].total_score, item[0]),
    )
    top_results = [result for _, result in sorted_results[:TOP_CANDIDATE_LIMIT]]

    for rank, result in enumerate(top_results, start=1):
        result.ranking = rank

    return top_results


def run_pipeline(
    job_description_path: Path = DEFAULT_JOB_DESCRIPTION_PATH,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[ScoreResult]:
    """Execute the full recruitment ranking pipeline.

    Args:
        job_description_path: Path to the JSON job-description file.
        candidates_path: Path to the JSONL candidates file.
        output_path: Destination submission CSV path.

    Returns:
        The ranked top score results exported to CSV.
    """
    logger.info("Loading job description from %s", job_description_path)
    job_description = prepare_job_description(
        load_job_description(job_description_path)
    )

    logger.info("Loading candidates from %s", candidates_path)
    ranked_results, processed_count, skipped_count = _process_candidates_parallel(
        candidates_path,
        job_description,
    )
    export_score_results(ranked_results, output_path)
    logger.info("CSV exported: %s", output_path)

    logger.info(
        "Finished pipeline: processed=%d skipped=%d exported=%d",
        processed_count,
        skipped_count,
        len(ranked_results),
    )
    return ranked_results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline from the command line.

    Args:
        argv: Optional argument sequence. When omitted, ``sys.argv`` is used.

    Returns:
        Process exit code.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.setLevel(logging.INFO)
    args = parse_args(argv)
    run_pipeline(args.job_description, args.candidates, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
