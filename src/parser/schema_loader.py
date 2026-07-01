"""Reusable structural schemas for recruitment-platform records.

The validators in this module check only that a value is a dictionary and
contains the required top-level fields.  They intentionally do not validate
field values, nested data, or business rules.
"""

import logging
from typing import Final


logger = logging.getLogger(__name__)

_CANDIDATE_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
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

_JOB_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "job_id",
        "title",
        "description",
        "required_skills",
    }
)


def get_candidate_required_fields() -> frozenset[str]:
    """Return the required top-level fields for a candidate record.

    Returns:
        An immutable set containing every required candidate field name.
    """
    return _CANDIDATE_REQUIRED_FIELDS


def get_job_required_fields() -> frozenset[str]:
    """Return the required top-level fields for a job-description record.

    Returns:
        An immutable set containing every required job field name.
    """
    return _JOB_REQUIRED_FIELDS


def validate_candidate_schema(candidate: dict) -> bool:
    """Validate the top-level structure of a candidate record.

    Additional fields are allowed.  Values associated with required fields are
    not inspected.

    Args:
        candidate: Candidate record to validate.

    Returns:
        ``True`` if ``candidate`` is a dictionary containing every required
        candidate field; otherwise ``False``.
    """
    return _validate_required_fields(
        candidate,
        _CANDIDATE_REQUIRED_FIELDS,
        record_name="candidate",
    )


def validate_job_schema(job: dict) -> bool:
    """Validate the top-level structure of a job-description record.

    Additional fields are allowed.  Values associated with required fields are
    not inspected.

    Args:
        job: Job-description record to validate.

    Returns:
        ``True`` if ``job`` is a dictionary containing every required job
        field; otherwise ``False``.
    """
    return _validate_required_fields(
        job,
        _JOB_REQUIRED_FIELDS,
        record_name="job description",
    )


def _validate_required_fields(
    record: object,
    required_fields: frozenset[str],
    *,
    record_name: str,
) -> bool:
    """Check a record's type and required top-level fields.

    Args:
        record: Value to inspect.
        required_fields: Field names that must exist in the record.
        record_name: Human-readable record name used in log messages.

    Returns:
        ``True`` when ``record`` is a dictionary containing all required
        fields; otherwise ``False``.
    """
    if not isinstance(record, dict):
        logger.warning(
            "Invalid %s schema: expected a dictionary, received %s",
            record_name,
            type(record).__name__,
        )
        return False

    missing_fields = required_fields.difference(record)
    if missing_fields:
        logger.warning(
            "Invalid %s schema: missing required field(s): %s",
            record_name,
            ", ".join(sorted(missing_fields)),
        )
        return False

    return True
