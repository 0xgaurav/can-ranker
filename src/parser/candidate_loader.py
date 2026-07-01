"""Load and validate candidate records from JSON files.

This module provides the input-boundary utilities for candidate data.  It is
deliberately limited to structural validation and does not perform feature
extraction, scoring, or any other AI-related processing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Final


logger = logging.getLogger(__name__)

REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "candidate_id",
        "profile",
        "career_history",
        "education",
        "skills",
        "certifications",
        "languages",
        "redrob_signals",
    }
)


class CandidateSchemaError(ValueError):
    """Raised when a candidate file does not have the expected JSON schema."""


def load_candidates(file_path: str) -> list[dict]:
    """Load valid candidate records from a JSON file.

    The file must contain a JSON array.  Records that do not contain every
    required top-level field are excluded from the returned list.

    Args:
        file_path: Path to a UTF-8 encoded JSON file containing candidates.

    Returns:
        A list containing only structurally valid candidate dictionaries.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist or is not a file.
        json.JSONDecodeError: If the file does not contain valid JSON.
        CandidateSchemaError: If the JSON root is not an array.
        OSError: If the file cannot be read for another operating-system
            reason, such as insufficient permissions.
    """
    path = Path(file_path)

    if not path.is_file():
        message = f"Candidate file does not exist or is not a file: {path}"
        logger.error(message)
        raise FileNotFoundError(message)

    try:
        with path.open("r", encoding="utf-8") as candidate_file:
            raw_candidates: Any = json.load(candidate_file)
    except json.JSONDecodeError as exc:
        logger.error(
            "Invalid JSON in candidate file %s at line %d, column %d: %s",
            path,
            exc.lineno,
            exc.colno,
            exc.msg,
        )
        raise json.JSONDecodeError(
            f"Invalid JSON in candidate file '{path}': {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc
    except OSError:
        logger.exception("Unable to read candidate file: %s", path)
        raise

    if not isinstance(raw_candidates, list):
        message = (
            f"Invalid candidate schema in '{path}': "
            "the JSON root must be an array"
        )
        logger.error(message)
        raise CandidateSchemaError(message)

    valid_candidates = validate_candidates(raw_candidates)
    logger.info(
        "Loaded %d valid candidate(s) from %s; discarded %d invalid record(s)",
        len(valid_candidates),
        path,
        len(raw_candidates) - len(valid_candidates),
    )
    return valid_candidates


def validate_candidate(candidate: dict) -> bool:
    """Check whether a candidate has all required top-level fields.

    Additional fields are allowed.  This function performs structural
    validation only; it does not validate field values or nested objects.

    Args:
        candidate: Candidate record to validate.

    Returns:
        ``True`` when ``candidate`` is a dictionary containing every required
        field; otherwise ``False``.
    """
    if not isinstance(candidate, dict):
        logger.warning(
            "Rejected candidate record: expected a dictionary, received %s",
            type(candidate).__name__,
        )
        return False

    missing_fields = REQUIRED_FIELDS.difference(candidate)
    if missing_fields:
        logger.warning(
            "Rejected candidate record %r: missing required field(s): %s",
            candidate.get("candidate_id", "<unknown>"),
            ", ".join(sorted(missing_fields)),
        )
        return False

    return True


def validate_candidates(candidates: list[dict]) -> list[dict]:
    """Filter a collection to its structurally valid candidate records.

    Args:
        candidates: Candidate records to validate.

    Returns:
        A new list containing valid records in their original order.

    Raises:
        CandidateSchemaError: If ``candidates`` is not a list.
    """
    if not isinstance(candidates, list):
        message = "Invalid candidate schema: candidates must be provided as a list"
        logger.error(message)
        raise CandidateSchemaError(message)

    return [candidate for candidate in candidates if validate_candidate(candidate)]


def get_candidate_by_id(
    candidates: list[dict], candidate_id: str
) -> dict | None:
    """Return the first candidate with the requested identifier.

    Args:
        candidates: Candidate records to search.
        candidate_id: Exact candidate identifier to match.

    Returns:
        The first matching candidate dictionary, or ``None`` when no match is
        found.  Malformed records in ``candidates`` are ignored.
    """
    for candidate in candidates:
        is_match = (
            isinstance(candidate, dict)
            and candidate.get("candidate_id") == candidate_id
        )
        if is_match:
            return candidate

    return None
