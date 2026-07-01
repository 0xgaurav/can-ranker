"""Detect deterministic outliers in candidate-provided data."""

import logging
import math


logger = logging.getLogger(__name__)

MAX_EXPERIENCE_YEARS = 50
MAX_SKILL_COUNT = 150
MAX_COMPANY_COUNT = 40

_SAFE_RESULT = {
    "experience_outlier": False,
    "skill_outlier": False,
    "company_outlier": False,
}


def _valid_experience_years(value: object) -> float | None:
    """Return a valid experience duration or ``None``.

    Args:
        value: Potential experience duration to validate.

    Returns:
        A finite, non-negative numeric duration, or ``None`` when malformed.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    try:
        numeric_value = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        return None

    return numeric_value


def _analyze_career_history(
    career_history: object,
) -> tuple[float, int]:
    """Calculate total experience and unique company count.

    Args:
        career_history: Potential list of candidate career entries.

    Returns:
        A tuple containing total valid experience years and the number of
        unique valid company names.
    """
    if not isinstance(career_history, list):
        return 0.0, 0

    total_experience = 0.0
    companies: set[str] = set()

    for entry in career_history:
        if not isinstance(entry, dict):
            continue

        experience_years = _valid_experience_years(
            entry.get("experience_years")
        )
        if experience_years is not None:
            total_experience += experience_years

        company = entry.get("company")
        if isinstance(company, str) and company.strip():
            companies.add(company.strip().casefold())

    return total_experience, len(companies)


def detect_outliers(
    candidate: dict[str, object],
) -> dict[str, bool]:
    """Detect experience, skill, and company-count outliers.

    Only the candidate's ``career_history``, ``skills``, and ``education``
    fields are in scope. Education data has no configured outlier rule and is
    therefore not used in the returned decisions. Candidate data is never
    modified.

    Args:
        candidate: Candidate data to inspect.

    Returns:
        A mapping containing the three configured outlier decisions. A
        malformed candidate produces three ``False`` values.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot detect outliers in malformed candidate data")
        return _SAFE_RESULT.copy()

    total_experience, company_count = _analyze_career_history(
        candidate.get("career_history")
    )

    skills = candidate.get("skills")
    skill_count = len(skills) if isinstance(skills, list) else 0

    experience_outlier = total_experience > MAX_EXPERIENCE_YEARS
    skill_outlier = skill_count > MAX_SKILL_COUNT
    company_outlier = company_count > MAX_COMPANY_COUNT

    if experience_outlier:
        logger.warning(
            "Experience outlier detected: %.2f total years",
            total_experience,
        )
    if skill_outlier:
        logger.warning("Skill outlier detected: %d skills", skill_count)
    if company_outlier:
        logger.warning(
            "Company outlier detected: %d unique companies", company_count
        )

    logger.info(
        "Candidate outlier detection completed: experience=%s, skills=%s, "
        "companies=%s",
        experience_outlier,
        skill_outlier,
        company_outlier,
    )
    return {
        "experience_outlier": experience_outlier,
        "skill_outlier": skill_outlier,
        "company_outlier": company_outlier,
    }
