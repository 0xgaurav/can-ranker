"""Validate the structural consistency of candidate education entries."""

import logging


logger = logging.getLogger(__name__)


def _empty_result() -> dict[str, bool | int]:
    """Create the safe result used for malformed education data.

    Returns:
        A fresh consistent result with zero education-entry counts.
    """
    return {
        "consistent": True,
        "valid_entries": 0,
        "invalid_entries": 0,
    }


def _is_valid_entry(entry: object) -> bool:
    """Determine whether an education entry contains required text fields.

    Args:
        entry: Potential education entry to validate.

    Returns:
        ``True`` when the entry is a dictionary containing non-empty string
        values for both ``degree`` and ``institution``.
    """
    if not isinstance(entry, dict):
        return False

    degree = entry.get("degree")
    institution = entry.get("institution")
    return (
        isinstance(degree, str)
        and bool(degree.strip())
        and isinstance(institution, str)
        and bool(institution.strip())
    )


def check_education_consistency(
    candidate: dict[str, object],
) -> dict[str, bool | int]:
    """Check candidate education entries for structural consistency.

    Only the ``education`` field is inspected. An entry is valid when it is a
    dictionary with non-empty string values for ``degree`` and ``institution``.
    Candidate data is never modified.

    Args:
        candidate: Candidate mapping containing education data.

    Returns:
        A mapping containing the consistency decision and valid and invalid
        entry counts. Missing or malformed education data produces a safe,
        consistent result with zero counts.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot check education for malformed candidate input")
        return _empty_result()

    education = candidate.get("education")
    if not isinstance(education, list):
        logger.warning("Candidate education is missing or malformed")
        return _empty_result()

    valid_entries = 0
    invalid_entries = 0

    for entry in education:
        if _is_valid_entry(entry):
            valid_entries += 1
        else:
            invalid_entries += 1
            logger.warning("Invalid education entry encountered")

    consistent = invalid_entries == 0
    logger.info(
        "Education consistency check completed: consistent=%s, valid=%d, "
        "invalid=%d",
        consistent,
        valid_entries,
        invalid_entries,
    )
    return {
        "consistent": consistent,
        "valid_entries": valid_entries,
        "invalid_entries": invalid_entries,
    }
