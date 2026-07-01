"""Detect reserved honeypot fields in candidate data."""

import logging


logger = logging.getLogger(__name__)

HONEYPOT_FIELDS = frozenset(
    {
        "honeypot",
        "test_field",
        "internal_notes",
        "debug",
    }
)


def detect_honeypot(
    candidate: dict[str, object],
) -> dict[str, bool | list[str]]:
    """Detect honeypot fields among a candidate's top-level keys.

    Field names are matched exactly and only at the top level. The candidate
    mapping is never modified.

    Args:
        candidate: Candidate mapping whose top-level keys should be inspected.

    Returns:
        A mapping containing the detection status and an alphabetically sorted
        list of triggered field names. Malformed input produces a safe result
        with no triggered fields.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot inspect malformed candidate for honeypot fields")
        return {
            "honeypot_detected": False,
            "triggered_fields": [],
        }

    triggered_fields = sorted(HONEYPOT_FIELDS.intersection(candidate.keys()))
    honeypot_detected = bool(triggered_fields)

    if honeypot_detected:
        logger.warning(
            "Honeypot fields detected: %s", ", ".join(triggered_fields)
        )

    logger.info(
        "Candidate honeypot inspection completed: detected=%s",
        honeypot_detected,
    )
    return {
        "honeypot_detected": honeypot_detected,
        "triggered_fields": triggered_fields,
    }
