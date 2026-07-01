"""Detect suspicious keyword repetition in candidate-provided text."""

import logging


logger = logging.getLogger(__name__)

KEYWORD_STUFFING_THRESHOLD = 3

SUSPICIOUS_KEYWORDS = frozenset(
    {
        "expert",
        "guru",
        "ninja",
        "rockstar",
        "genius",
        "world class",
        "best",
        "elite",
        "top",
    }
)

_CANDIDATE_TEXT_FIELDS = (
    "profile",
    "career_history",
    "redrob_signals",
)


def _count_keyword_hits(text: str) -> int:
    """Count all suspicious keyword occurrences in text.

    Args:
        text: Lowercase or mixed-case text to inspect.

    Returns:
        Total non-overlapping occurrences of all suspicious keywords.
    """
    normalized_text = text.lower()
    return sum(
        normalized_text.count(keyword) for keyword in SUSPICIOUS_KEYWORDS
    )


def detect_keyword_stuffing(text: str) -> bool:
    """Determine whether text contains excessive suspicious keywords.

    Args:
        text: Text to inspect. Values of any other type are considered safe.

    Returns:
        ``True`` when the total keyword count is greater than the configured
        threshold; otherwise, ``False``.
    """
    if not isinstance(text, str):
        logger.warning("Keyword stuffing check received non-string input")
        return False

    keyword_hits = _count_keyword_hits(text)
    stuffing_detected = keyword_hits > KEYWORD_STUFFING_THRESHOLD
    if stuffing_detected:
        logger.warning(
            "Keyword stuffing detected with %d suspicious occurrences",
            keyword_hits,
        )

    return stuffing_detected


def _extract_text(value: object) -> str:
    """Recursively extract lowercase text from supported containers.

    Strings are retained, list items and dictionary values are traversed, and
    all other value types are ignored. Cyclic containers are visited once to
    prevent infinite recursion.

    Args:
        value: String, list, dictionary, or unsupported value to process.

    Returns:
        A single lowercase string containing the extracted text.
    """
    visited: set[int] = set()

    def extract(current: object) -> list[str]:
        """Collect text fragments from one value recursively."""
        if isinstance(current, str):
            return [current.lower()]

        if isinstance(current, list):
            identity = id(current)
            if identity in visited:
                return []
            visited.add(identity)
            return [
                fragment
                for item in current
                for fragment in extract(item)
            ]

        if isinstance(current, dict):
            identity = id(current)
            if identity in visited:
                return []
            visited.add(identity)
            return [
                fragment
                for item in current.values()
                for fragment in extract(item)
            ]

        return []

    return " ".join(extract(value))


def analyze_candidate_keywords(
    candidate: dict[str, object],
) -> dict[str, bool | int]:
    """Analyze selected candidate fields for suspicious keyword repetition.

    Only ``profile``, ``career_history``, and ``redrob_signals`` are
    inspected. The candidate mapping is never modified.

    Args:
        candidate: Candidate data containing fields eligible for inspection.

    Returns:
        A mapping containing the keyword-stuffing decision and total number
        of suspicious keyword occurrences. Malformed candidates produce a
        safe result with no hits.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot analyze malformed candidate keyword data")
        return {
            "keyword_stuffing": False,
            "keyword_hits": 0,
        }

    candidate_text = " ".join(
        _extract_text(candidate.get(field, ""))
        for field in _CANDIDATE_TEXT_FIELDS
    )
    keyword_hits = _count_keyword_hits(candidate_text)
    stuffing_detected = keyword_hits > KEYWORD_STUFFING_THRESHOLD

    if stuffing_detected:
        logger.warning(
            "Candidate keyword stuffing detected with %d occurrences",
            keyword_hits,
        )

    logger.info(
        "Candidate keyword analysis completed with %d occurrences",
        keyword_hits,
    )
    return {
        "keyword_stuffing": stuffing_detected,
        "keyword_hits": keyword_hits,
    }
