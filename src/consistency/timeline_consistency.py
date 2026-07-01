"""Validate chronological consistency in candidate career history."""

import logging
from datetime import date, datetime


logger = logging.getLogger(__name__)

_DATE_FORMATS = (
    ("%Y-%m-%d", 10),
    ("%Y-%m", 7),
    ("%Y", 4),
)
_CURRENT_DATE_LABELS = frozenset({"present", "current"})


def _empty_result() -> dict[str, bool | int]:
    """Create the safe result used for malformed career history.

    Returns:
        A fresh consistent result containing zero role counts.
    """
    return {
        "consistent": True,
        "valid_roles": 0,
        "invalid_roles": 0,
    }


def _parse_date(value: object, *, allow_current: bool = False) -> date | None:
    """Parse one supported career date value.

    Partial dates resolve to the first day of their represented year or month.
    Current-role labels resolve to today's local date only when explicitly
    allowed.

    Args:
        value: Potential date string to parse.
        allow_current: Whether ``Present`` and ``Current`` are accepted.

    Returns:
        The parsed date, or ``None`` when the value is malformed.
    """
    if not isinstance(value, str):
        return None

    normalized_value = value.strip()
    if (
        allow_current
        and normalized_value.casefold() in _CURRENT_DATE_LABELS
    ):
        return date.today()

    for date_format, expected_length in _DATE_FORMATS:
        if len(normalized_value) != expected_length:
            continue
        try:
            return datetime.strptime(normalized_value, date_format).date()
        except ValueError:
            return None

    return None


def check_timeline_consistency(
    candidate: dict[str, object],
) -> dict[str, bool | int]:
    """Check candidate career roles for chronological consistency.

    Only ``career_history`` is inspected. Roles with missing or malformed
    dates are ignored. A parseable role is valid when its start date is not
    later than its end date. Candidate data is never modified.

    Args:
        candidate: Candidate mapping containing career-history data.

    Returns:
        A mapping containing the consistency decision and valid and invalid
        role counts. Missing or malformed career history produces a safe,
        consistent result with zero counts.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot check timeline for malformed candidate input")
        return _empty_result()

    career_history = candidate.get("career_history")
    if not isinstance(career_history, list):
        logger.warning("Candidate career history is missing or malformed")
        return _empty_result()

    valid_roles = 0
    invalid_roles = 0

    for entry in career_history:
        if not isinstance(entry, dict):
            logger.warning("Ignoring career entry with malformed dates")
            continue

        start_date = _parse_date(entry.get("start_date"))
        end_date = _parse_date(entry.get("end_date"), allow_current=True)
        if start_date is None or end_date is None:
            logger.warning("Ignoring career entry with malformed dates")
            continue

        if start_date <= end_date:
            valid_roles += 1
        else:
            invalid_roles += 1
            logger.warning(
                "Invalid career timeline: start date occurs after end date"
            )

    consistent = invalid_roles == 0
    logger.info(
        "Timeline consistency check completed: consistent=%s, valid=%d, "
        "invalid=%d",
        consistent,
        valid_roles,
        invalid_roles,
    )
    return {
        "consistent": consistent,
        "valid_roles": valid_roles,
        "invalid_roles": invalid_roles,
    }
