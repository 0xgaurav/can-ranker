"""Extract career-history features from candidate records."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime

from src.models.candidate_features import CareerFeatures
from src.utils.normalizer import normalize_company


logger = logging.getLogger(__name__)

_PRESENT_MARKERS = {"present", "current", "now", "ongoing"}
_SENIORITY_LEVELS = [
    "unknown",
    "junior",
    "mid",
    "senior",
    "lead",
    "manager",
    "director",
    "executive",
]
_SENIORITY_RANKS = {
    level: index for index, level in enumerate(_SENIORITY_LEVELS)
}
_SENIORITY_KEYWORDS = {
    "executive": {"chief", "cto", "ceo", "cxo", "vp", "vice president"},
    "director": {"director", "head"},
    "manager": {"manager", "management"},
    "lead": {"lead", "principal", "architect", "staff"},
    "senior": {"senior", "sr"},
    "junior": {"junior", "jr", "intern", "trainee", "associate"},
}
_TITLE_PUNCTUATION_RE = re.compile(r"[^a-z0-9+#]+")


@dataclass(slots=True)
class _CareerEntry:
    """Normalized internal representation of one career-history entry."""

    company: str | None
    title: str | None
    start_date: date | None
    end_date: date | None
    duration_months: int
    is_current: bool
    explicit_promotions: int
    sequence: int


def extract_career_features(candidate: dict) -> CareerFeatures:
    """Extract structured career features from a candidate.

    Args:
        candidate: Candidate record containing ``career_history`` entries.

    Returns:
        Career features derived from valid career-history entries.
    """
    if not isinstance(candidate, dict):
        logger.warning(
            "Cannot extract career features: expected a dictionary, received %s",
            type(candidate).__name__,
        )
        return _empty_career_features()

    raw_history = candidate.get("career_history") or candidate.get("experience") or []
    if not isinstance(raw_history, list):
        logger.warning(
            "Cannot extract career features for candidate %r: history must be a "
            "list",
            candidate.get("candidate_id", "<unknown>"),
        )
        return _empty_career_features()

    entries = [
        entry
        for index, raw_entry in enumerate(raw_history)
        if (entry := _parse_career_entry(raw_entry, index)) is not None
    ]
    if not entries:
        return _empty_career_features()

    durations = [entry.duration_months for entry in entries]
    companies = {entry.company for entry in entries if entry.company}
    promotion_count = max(
        _get_explicit_promotion_count(candidate, entries),
        _infer_promotion_count(entries),
    )

    features = CareerFeatures(
        total_years_experience=round(sum(durations) / 12, 2),
        number_of_companies=len(companies),
        average_tenure_months=round(sum(durations) / len(durations), 2),
        longest_tenure_months=max(durations),
        shortest_tenure_months=min(durations),
        leadership_experience=any(
            entry.is_current and _is_leadership_title(entry.title)
            for entry in entries
        )
        or any(_is_leadership_title(entry.title) for entry in entries),
        promotion_count=promotion_count,
        seniority_level=_get_highest_seniority(entries),
        career_growth_score=_calculate_career_growth_score(
            entries,
            promotion_count,
        ),
    )
    logger.info(
        "Extracted career features for candidate %r from %d role(s)",
        candidate.get("candidate_id", "<unknown>"),
        len(entries),
    )
    return features


def _parse_career_entry(raw_entry: object, sequence: int) -> _CareerEntry | None:
    """Parse one raw career-history entry.

    Args:
        raw_entry: Raw career entry.
        sequence: Source order for stable sorting.

    Returns:
        Parsed career entry, or ``None`` when the entry is unusable.
    """
    if not isinstance(raw_entry, dict):
        logger.warning("Ignored malformed career-history entry")
        return None

    start_date = _parse_date(raw_entry.get("start_date"))
    end_date = _parse_date(
        raw_entry.get("end_date"),
        use_today=bool(raw_entry.get("is_current")),
    )
    duration_months = _extract_nonnegative_int(raw_entry.get("duration_months"))
    if duration_months == 0 and start_date and end_date:
        duration_months = _months_between(start_date, end_date)

    if duration_months == 0:
        logger.warning("Ignored career-history entry with no usable duration")
        return None

    company = _normalize_optional_company(raw_entry.get("company"))
    title = _normalize_optional_text(raw_entry.get("title"))
    return _CareerEntry(
        company=company,
        title=title,
        start_date=start_date,
        end_date=end_date,
        duration_months=duration_months,
        is_current=bool(raw_entry.get("is_current")),
        explicit_promotions=_extract_nonnegative_int(
            raw_entry.get("promotion_count")
        ),
        sequence=sequence,
    )


def _parse_date(value: object, *, use_today: bool = False) -> date | None:
    """Parse a date-like value into a date.

    Args:
        value: Raw date value.
        use_today: Whether ``None`` should be treated as today.

    Returns:
        Parsed date, or ``None`` when unavailable or invalid.
    """
    if value is None:
        return date.today() if use_today else None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if 1 <= value <= 9999:
            return date(value, 1, 1)
        return None
    if not isinstance(value, str):
        return None

    normalized_value = value.strip().casefold()
    if normalized_value in _PRESENT_MARKERS:
        return date.today()

    try:
        parts = normalized_value.split("-")
        if len(parts) == 3:
            year, month, day = (int(part) for part in parts)
            return date(year, month, day)
        if len(parts) == 2:
            year, month = (int(part) for part in parts)
            return date(year, month, 1)
        if len(parts) == 1 and parts[0]:
            return date(int(parts[0]), 1, 1)
    except ValueError:
        pass

    logger.warning("Ignored unsupported career date value %r", value)
    return None


def _months_between(start_date: date, end_date: date) -> int:
    """Calculate completed calendar months between two dates."""
    months = (
        (end_date.year - start_date.year) * 12
        + end_date.month
        - start_date.month
    )
    if end_date.day < start_date.day:
        months -= 1
    return max(months, 0)


def _normalize_optional_text(value: object) -> str | None:
    """Return normalized nonblank text or ``None``."""
    if not isinstance(value, str):
        return None
    normalized_value = " ".join(value.casefold().split())
    return normalized_value or None


def _normalize_optional_company(value: object) -> str | None:
    """Return a normalized company name or ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return normalize_company(value)
    except TypeError:
        return None


def _extract_nonnegative_number(value: object) -> float | None:
    """Return a finite nonnegative number or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if numeric_value < 0 or numeric_value == float("inf"):
        return None
    if numeric_value != numeric_value:
        return None
    return numeric_value


def _extract_nonnegative_int(value: object) -> int:
    """Return a nonnegative integer count, defaulting to zero."""
    numeric_value = _extract_nonnegative_number(value)
    return int(numeric_value) if numeric_value is not None else 0


def _normalize_title(title: str) -> str:
    """Normalize title punctuation for keyword matching."""
    return " ".join(_TITLE_PUNCTUATION_RE.sub(" ", title).split())


def _get_seniority_level(title: str | None) -> str:
    """Map a role title to a deterministic seniority level."""
    if not title:
        return "unknown"

    normalized_title = _normalize_title(title)
    padded_title = f" {normalized_title} "
    for level, keywords in _SENIORITY_KEYWORDS.items():
        if any(f" {keyword} " in padded_title for keyword in keywords):
            return level
    return "mid"


def _seniority_rank(title: str | None) -> int:
    """Return the ordinal rank of a role title's seniority level."""
    return _SENIORITY_RANKS[_get_seniority_level(title)]


def _get_highest_seniority(entries: list[_CareerEntry]) -> str:
    """Return the highest recognized seniority across parsed entries."""
    return max(
        (_get_seniority_level(entry.title) for entry in entries),
        key=_SENIORITY_RANKS.__getitem__,
        default="unknown",
    )


def _is_leadership_title(title: str | None) -> bool:
    """Return whether a role title indicates leadership responsibility."""
    return _get_seniority_level(title) in {
        "lead",
        "manager",
        "director",
        "executive",
    }


def _get_explicit_promotion_count(
    candidate: dict,
    entries: list[_CareerEntry],
) -> int:
    """Return the largest available explicit promotion count."""
    entry_count = sum(entry.explicit_promotions for entry in entries)
    candidate_count = _extract_nonnegative_int(candidate.get("promotion_count"))
    return max(entry_count, candidate_count)


def _infer_promotion_count(entries: list[_CareerEntry]) -> int:
    """Infer upward same-company seniority transitions."""
    entries_by_company: dict[str, list[_CareerEntry]] = {}
    for entry in entries:
        if entry.company:
            entries_by_company.setdefault(entry.company, []).append(entry)

    promotion_count = 0
    for company_entries in entries_by_company.values():
        ordered_entries = sorted(company_entries, key=_career_entry_sort_key)
        previous_rank: int | None = None
        for entry in ordered_entries:
            if not entry.title:
                continue
            current_rank = _seniority_rank(entry.title)
            if previous_rank is not None and current_rank > previous_rank:
                promotion_count += 1
            previous_rank = current_rank

    return promotion_count


def _career_entry_sort_key(entry: _CareerEntry) -> tuple[int, int]:
    """Sort dated entries chronologically and preserve undated source order."""
    if entry.start_date is not None:
        return entry.start_date.toordinal(), entry.sequence
    return date.max.toordinal(), entry.sequence


def _calculate_career_growth_score(
    entries: list[_CareerEntry],
    promotion_count: int,
) -> float:
    """Calculate a bounded, descriptive career-progression metric."""
    titled_entries = [entry for entry in entries if entry.title]
    if not titled_entries:
        return 0.0

    ordered_entries = sorted(titled_entries, key=_career_entry_sort_key)
    initial_rank = _seniority_rank(ordered_entries[0].title)
    highest_rank = max(_seniority_rank(entry.title) for entry in titled_entries)
    maximum_gain = max(len(_SENIORITY_LEVELS) - 1 - initial_rank, 1)
    seniority_growth = max(highest_rank - initial_rank, 0) / maximum_gain
    promotion_growth = min(promotion_count / 3, 1.0)
    return round(min((seniority_growth + promotion_growth) / 2, 1.0), 3)


def _empty_career_features() -> CareerFeatures:
    """Return the neutral feature value for missing or malformed history."""
    return CareerFeatures(
        total_years_experience=0.0,
        number_of_companies=0,
        average_tenure_months=0.0,
        longest_tenure_months=0,
        shortest_tenure_months=0,
        leadership_experience=False,
        promotion_count=0,
        seniority_level="unknown",
        career_growth_score=0.0,
    )
