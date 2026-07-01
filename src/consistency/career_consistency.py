"""Validate the structural consistency of candidate career history."""

import logging


logger = logging.getLogger(__name__)


def _empty_result() -> dict[str, bool | int]:
    """Create the safe result used for malformed career history.

    Returns:
        A fresh consistent result containing zero entry and company counts.
    """
    return {
        "consistent": True,
        "valid_entries": 0,
        "invalid_entries": 0,
        "unique_companies": 0,
    }


def _is_valid_entry(entry: object) -> bool:
    """Determine whether a career entry contains its required fields.

    Args:
        entry: Potential career-history entry to validate.

    Returns:
        ``True`` when the entry is a dictionary containing non-empty string
        values for both ``company`` and ``title``.
    """
    if not isinstance(entry, dict):
        return False

    company = entry.get("company")
    title = entry.get("title")
    return (
        isinstance(company, str)
        and bool(company.strip())
        and isinstance(title, str)
        and bool(title.strip())
    )


def check_career_consistency(
    candidate: dict[str, object],
) -> dict[str, bool | int]:
    """Check candidate career-history entries for consistency.

    Only ``career_history`` is inspected. Valid entries require non-empty
    string values for ``company`` and ``title``. Optional date fields are
    ignored, and candidate data is never modified.

    Args:
        candidate: Candidate mapping containing career-history data.

    Returns:
        A mapping containing the consistency decision, entry counts, and
        case-insensitive unique-company count. Missing or malformed career
        history produces a safe, consistent result with zero counts.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot check career history for malformed candidate")
        return _empty_result()

    career_history = candidate.get("career_history")
    if not isinstance(career_history, list):
        logger.warning("Candidate career history is missing or malformed")
        return _empty_result()

    valid_entries = 0
    invalid_entries = 0
    companies: set[str] = set()

    for entry in career_history:
        if not _is_valid_entry(entry):
            invalid_entries += 1
            logger.warning("Invalid career entry encountered")
            continue

        valid_entries += 1
        company = entry["company"]
        companies.add(company.strip().casefold())

    consistent = invalid_entries == 0
    unique_companies = len(companies)

    logger.info(
        "Career consistency check completed: consistent=%s, valid=%d, "
        "invalid=%d, companies=%d",
        consistent,
        valid_entries,
        invalid_entries,
        unique_companies,
    )
    return {
        "consistent": consistent,
        "valid_entries": valid_entries,
        "invalid_entries": invalid_entries,
        "unique_companies": unique_companies,
    }
