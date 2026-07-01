"""Professional deterministic scoring for candidate ranking."""

from __future__ import annotations

import math
from typing import Any, Iterable

from src.models.match_result import MatchResult


_SEMANTIC_WEIGHT = 0.35
_SKILL_WEIGHT = 0.15
_CAREER_WEIGHT = 0.10
_PRODUCTION_WEIGHT = 0.05
_RECRUITER_WEIGHT = 0.10
_QUALITY_WEIGHT = 0.05
_BEHAVIOR_WEIGHT = 0.05
_CONSISTENCY_WEIGHT = 0.05
_JD_PRIORITY_WEIGHT = 0.05
_FINE_GRAINED_WEIGHT = 0.05
_MAX_FRAUD_PENALTY = 0.05

_PRODUCTION_TERMS = frozenset(
    {
        "production ml",
        "deployment",
        "deploy",
        "large scale",
        "large-scale",
        "distributed systems",
        "search systems",
        "recommendation systems",
        "recommender",
        "ranking",
        "retrieval",
        "vector search",
        "mlops",
        "kubernetes",
        "docker",
        "serving",
    }
)

_JD_PRIORITY_TERMS = frozenset(
    {
        "retrieval",
        "ranking",
        "recommendation systems",
        "recommendation",
        "recommender",
        "llms",
        "llm",
        "embeddings",
        "embedding",
        "vector search",
        "vector databases",
        "production ml",
        "fine-tuning",
    }
)


def _clamp01(value: Any) -> float:
    """Return a finite float constrained to the inclusive range [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _finite_number(value: Any, default: float = 0.0) -> float:
    """Return a finite float without clamping."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _normalize_count(value: Any, cap: float) -> float:
    """Normalize a nonnegative count with an upper cap."""
    if cap <= 0:
        return 0.0
    return _clamp01(max(_finite_number(value), 0.0) / cap)


def _normalize_rate(value: Any) -> float:
    """Normalize rates supplied either as 0-1 values or 0-100 percentages."""
    numeric = _finite_number(value)
    if numeric > 1.0:
        numeric /= 100.0
    return _clamp01(numeric)


def _profile(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(candidate, dict) and isinstance(candidate.get("profile"), dict):
        return candidate["profile"]
    return {}


def _signals(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(candidate, dict) and isinstance(candidate.get("redrob_signals"), dict):
        return candidate["redrob_signals"]
    return {}


def _iter_strings(value: Any) -> Iterable[str]:
    """Yield string fragments from nested candidate structures."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_strings(nested)


def _text_blob(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    fields = (
        "profile",
        "skills",
        "career_history",
        "projects",
        "certifications",
        "achievements",
        "publications",
        "summary",
        "about",
    )
    return " ".join(
        fragment.casefold()
        for field in fields
        for fragment in _iter_strings(candidate.get(field))
    )


def _skill_names(candidate: dict[str, Any] | None) -> list[str]:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("skills"), list):
        return []

    names: list[str] = []
    for skill in candidate["skills"]:
        if isinstance(skill, str) and skill.strip():
            names.append(skill.strip())
        elif isinstance(skill, dict):
            name = skill.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def _skill_endorsements(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("skills"), list):
        return 0.0
    total = 0.0
    for skill in candidate["skills"]:
        if isinstance(skill, dict):
            total += max(_finite_number(skill.get("endorsements")), 0.0)
    return total


def _duplicate_ratio(values: Iterable[str]) -> float:
    normalized = [value.strip().casefold() for value in values if value.strip()]
    if not normalized:
        return 0.0
    return _clamp01((len(normalized) - len(set(normalized))) / len(normalized))


def _years_of_experience(candidate: dict[str, Any] | None) -> float:
    profile = _profile(candidate)
    direct = _finite_number(
        profile.get("years_of_experience", profile.get("experience_years")),
        default=-1.0,
    )
    if direct >= 0.0:
        return direct

    if not isinstance(candidate, dict) or not isinstance(candidate.get("career_history"), list):
        return 0.0

    total_months = 0.0
    total_years = 0.0
    for entry in candidate["career_history"]:
        if not isinstance(entry, dict):
            continue
        total_months += max(_finite_number(entry.get("duration_months")), 0.0)
        total_years += max(_finite_number(entry.get("experience_years")), 0.0)
    if total_months > 0:
        return total_months / 12.0
    return total_years


def _career_score(match: MatchResult, candidate: dict[str, Any] | None) -> float:
    years = _years_of_experience(candidate)
    experience = max(
        _clamp01(getattr(match, "experience_similarity", 0.0)),
        _normalize_count(years, 15.0),
    )

    blob = _text_blob(candidate)
    seniority = 0.0
    if any(term in blob for term in ("principal", "staff", "lead", "architect")):
        seniority = 0.85
    if any(term in blob for term in ("manager", "director", "head", "vp", "chief")):
        seniority = max(seniority, 1.0)
    if any(term in blob for term in ("senior", "sr.")):
        seniority = max(seniority, 0.7)

    leadership = 1.0 if any(
        term in blob for term in ("lead", "manager", "mentor", "architect", "director")
    ) else 0.0

    companies = _company_names(candidate)
    progression = _clamp01(0.5 + (0.1 * min(len(companies), 5)))
    return _clamp01(
        (experience * 0.45)
        + (seniority * 0.25)
        + (leadership * 0.15)
        + (progression * 0.15)
    )


def _company_names(candidate: dict[str, Any] | None) -> list[str]:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("career_history"), list):
        return []
    companies: list[str] = []
    for entry in candidate["career_history"]:
        if isinstance(entry, dict) and isinstance(entry.get("company"), str):
            companies.append(entry["company"])
    return companies


def _production_score(candidate: dict[str, Any] | None) -> float:
    blob = _text_blob(candidate)
    if not blob:
        return 0.0
    hits = sum(1 for term in _PRODUCTION_TERMS if term in blob)
    return _normalize_count(hits, 6.0)


def _recruiter_signal_score(candidate: dict[str, Any] | None) -> float:
    signals = _signals(candidate)
    profile = _profile(candidate)
    components = (
        _normalize_rate(signals.get("recruiter_response_rate")),
        _normalize_count(
            signals.get("saved_by_recruiters", signals.get("saved_by_recruiters_30d")),
            50.0,
        ),
        _normalize_count(signals.get("profile_views_received_30d"), 100.0),
        _normalize_rate(
            signals.get(
                "profile_completion",
                signals.get(
                    "profile_completeness_score",
                    profile.get("profile_completion"),
                ),
            )
        ),
        _normalize_rate(signals.get("interview_completion_rate")),
        1.0 if (
            signals.get("verified_profile")
            or signals.get("verified_email")
            or signals.get("verified_phone")
            or profile.get("verified_profile")
        ) else 0.0,
        1.0 if (
            signals.get("open_to_work")
            or signals.get("open_to_work_flag")
            or profile.get("open_to_work")
        ) else 0.0,
    )
    return sum(components) / len(components)


def _fraud_penalty(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return 0.0

    penalty = 0.0
    blob = _text_blob(candidate)
    suspicious_hits = sum(
        blob.count(term)
        for term in ("expert", "guru", "ninja", "rockstar", "genius", "world class")
    )
    if suspicious_hits > 3:
        penalty += 0.015

    penalty += _duplicate_ratio(_skill_names(candidate)) * 0.015
    penalty += _duplicate_ratio(_company_names(candidate)) * 0.010

    years = _years_of_experience(candidate)
    if years > 50.0:
        penalty += 0.020
    if len(_skill_names(candidate)) > 150:
        penalty += 0.015
    if len(set(name.casefold() for name in _company_names(candidate))) > 40:
        penalty += 0.015

    return min(penalty, _MAX_FRAUD_PENALTY)


def _consistency_score(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return 0.5

    checks = []
    skills = candidate.get("skills")
    education = candidate.get("education")
    career = candidate.get("career_history")
    signals = candidate.get("redrob_signals")

    checks.append(isinstance(skills, list) and len(_skill_names(candidate)) == len(skills))
    checks.append(isinstance(education, list))
    checks.append(isinstance(career, list) and all(isinstance(item, dict) for item in career))
    checks.append(isinstance(signals, dict))
    checks.append(_duplicate_ratio(_skill_names(candidate)) == 0.0)
    checks.append(_duplicate_ratio(_company_names(candidate)) <= 0.25)

    return sum(1.0 for passed in checks if passed) / len(checks)


def _jd_priority_score(
    match: MatchResult,
    candidate: dict[str, Any] | None,
    job_description: dict[str, Any] | None,
) -> float:
    candidate_blob = _text_blob(candidate)
    matched_blob = " ".join(
        str(skill).casefold() for skill in getattr(match, "matched_skills", [])
    )
    jd_blob = ""
    if isinstance(job_description, dict):
        jd_blob = " ".join(fragment.casefold() for fragment in _iter_strings(job_description))

    active_terms = [
        term for term in _JD_PRIORITY_TERMS if term in jd_blob or term in matched_blob
    ]
    if not active_terms:
        active_terms = list(_JD_PRIORITY_TERMS)

    hits = sum(
        1
        for term in active_terms
        if term in candidate_blob or term in matched_blob
    )
    return _normalize_count(hits, min(len(active_terms), 6))


def _fine_grained_score(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    signals = _signals(candidate)
    profile = _profile(candidate)

    components = (
        _normalize_count(_years_of_experience(candidate), 20.0),
        _normalize_rate(signals.get("recruiter_response_rate")),
        _normalize_rate(
            signals.get(
                "profile_completion",
                signals.get(
                    "profile_completeness_score",
                    profile.get("profile_completion"),
                ),
            )
        ),
        _normalize_rate(signals.get("interview_completion_rate")),
        _normalize_count(signals.get("github_activity_score"), 10.0),
        _normalize_count(_skill_endorsements(candidate), 300.0),
        _normalize_count(len(candidate.get("certifications", []) or []), 8.0),
        _normalize_count(len(candidate.get("projects", []) or []), 10.0),
        _normalize_count(len(candidate.get("publications", []) or []), 8.0),
        _normalize_count(signals.get("connection_count"), 500.0),
        _normalize_count(
            signals.get("saved_by_recruiters", signals.get("saved_by_recruiters_30d")),
            50.0,
        ),
        _normalize_count(signals.get("profile_views_received_30d"), 100.0),
        _normalize_count(
            signals.get("search_appearances_30d", signals.get("search_appearance_30d")),
            150.0,
        ),
        _normalize_rate(signals.get("resume_completeness")),
        _normalize_count(signals.get("recent_activity_score"), 10.0),
    )
    return sum(components) / len(components)


def calculate_final_score(
    match: MatchResult,
    behavior_score: float,
    quality_score: float,
    candidate: dict[str, Any] | None = None,
    job_description: dict[str, Any] | None = None,
) -> float:
    """Calculate a deterministic raw score on a 0-10 scale.

    The score is intentionally composed from many small weighted signals so
    candidates separate naturally without random noise or ID-based decimals.
    """
    semantic = _clamp01(getattr(match, "overall_similarity", 0.0))
    skill = _clamp01(getattr(match, "skill_similarity", 0.0))
    career = _career_score(match, candidate)
    production = _production_score(candidate)
    recruiter = _recruiter_signal_score(candidate)
    quality = _clamp01(quality_score)
    behavior = _clamp01(behavior_score)
    consistency = _consistency_score(candidate)
    jd_priority = _jd_priority_score(match, candidate, job_description)
    fine_grained = _fine_grained_score(candidate)
    fraud_penalty = _fraud_penalty(candidate)

    weighted = (
        (semantic * _SEMANTIC_WEIGHT)
        + (skill * _SKILL_WEIGHT)
        + (career * _CAREER_WEIGHT)
        + (production * _PRODUCTION_WEIGHT)
        + (recruiter * _RECRUITER_WEIGHT)
        + (quality * _QUALITY_WEIGHT)
        + (behavior * _BEHAVIOR_WEIGHT)
        + (consistency * _CONSISTENCY_WEIGHT)
        + (jd_priority * _JD_PRIORITY_WEIGHT)
        + (fine_grained * _FINE_GRAINED_WEIGHT)
        - fraud_penalty
    )
    return max(0.0, min(10.0, weighted * 10.0))
