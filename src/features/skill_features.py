"""Extract normalized skill features from candidate records."""

from __future__ import annotations

import logging
from typing import Final

from src.models.candidate_features import SkillFeatures
from src.utils.normalizer import normalize_skill_list


logger = logging.getLogger(__name__)

TECHNICAL_SKILLS: Final[dict[str, str]] = {
    "airflow": "data",
    "apache airflow": "data",
    "apache beam": "data",
    "aws": "cloud",
    "azure": "cloud",
    "bentoml": "machine_learning",
    "c++": "programming_language",
    "c#": "programming_language",
    "dbt": "data",
    "django": "framework",
    "docker": "devops",
    "fastapi": "framework",
    "flask": "framework",
    "gcp": "cloud",
    "git": "devops",
    "java": "programming_language",
    "javascript": "programming_language",
    "kafka": "data",
    "kubernetes": "devops",
    "machine learning": "machine_learning",
    "milvus": "database",
    "mongodb": "database",
    "mysql": "database",
    "nlp": "machine_learning",
    "node.js": "framework",
    "pandas": "library",
    "php": "programming_language",
    "postgresql": "database",
    "pytorch": "machine_learning",
    "python": "programming_language",
    "r": "programming_language",
    "react": "framework",
    "redis": "database",
    "ruby": "programming_language",
    "rust": "programming_language",
    "scala": "programming_language",
    "scikit-learn": "machine_learning",
    "spark": "data",
    "spring": "framework",
    "sql": "database",
    "swift": "programming_language",
    "tensorflow": "machine_learning",
    "terraform": "devops",
    "typescript": "programming_language",
    "vue.js": "framework",
}

SOFT_SKILLS: Final[dict[str, str]] = {
    "adaptability": "personal_effectiveness",
    "attention to detail": "personal_effectiveness",
    "collaboration": "interpersonal",
    "communication": "interpersonal",
    "conflict resolution": "interpersonal",
    "creativity": "cognitive",
    "critical thinking": "cognitive",
    "decision making": "leadership",
    "emotional intelligence": "interpersonal",
    "interpersonal skills": "interpersonal",
    "leadership": "leadership",
    "mentoring": "leadership",
    "negotiation": "interpersonal",
    "organization": "personal_effectiveness",
    "presentation": "communication",
    "problem solving": "cognitive",
    "teamwork": "interpersonal",
    "time management": "personal_effectiveness",
}


def extract_skill_features(candidate: dict) -> SkillFeatures:
    """Extract structured, normalized skill features from a candidate.

    Args:
        candidate: Candidate record containing a list-valued ``skills`` field.

    Returns:
        Normalized skill features. Empty features are returned for malformed
        input.
    """
    if not isinstance(candidate, dict):
        logger.warning(
            "Cannot extract skill features: expected a dictionary, received %s",
            type(candidate).__name__,
        )
        return SkillFeatures()

    raw_skills = candidate.get("skills", [])
    if not isinstance(raw_skills, list):
        logger.warning(
            "Cannot extract skill features for candidate %r: skills must be a "
            "list",
            candidate.get("candidate_id", "<unknown>"),
        )
        return SkillFeatures()

    valid_skills = _get_valid_skill_entries(raw_skills)
    normalized_skills = [
        skill for skill in normalize_skill_list(valid_skills) if skill
    ]
    technical_skills = [
        skill for skill in normalized_skills if skill in TECHNICAL_SKILLS
    ]
    soft_skills = [
        skill for skill in normalized_skills if skill in SOFT_SKILLS
    ]

    logger.info(
        "Extracted %d unique skill(s) from %d valid source entries for "
        "candidate %r",
        len(normalized_skills),
        len(valid_skills),
        candidate.get("candidate_id", "<unknown>"),
    )
    return SkillFeatures(
        normalized_skills=normalized_skills,
        technical_skills=technical_skills,
        soft_skills=soft_skills,
        skill_count=len(valid_skills),
        unique_skill_count=len(normalized_skills),
        primary_skill=normalized_skills[0] if normalized_skills else None,
    )


def _get_valid_skill_entries(skills: list[object]) -> list[str]:
    """Return nonblank skill names in their original order.

    Skill values may be plain strings or dictionaries with a ``name`` field,
    matching the raw candidate schema.

    Args:
        skills: Raw values from a candidate's skill collection.

    Returns:
        Skill names that contain at least one non-whitespace character.
    """
    valid_skills: list[str] = []
    ignored_count = 0

    for skill in skills:
        if isinstance(skill, str) and skill.strip():
            valid_skills.append(skill)
        elif (
            isinstance(skill, dict)
            and isinstance(skill.get("name"), str)
            and skill["name"].strip()
        ):
            valid_skills.append(skill["name"])
        else:
            ignored_count += 1

    if ignored_count:
        logger.warning("Ignored %d invalid or blank skill value(s)", ignored_count)

    return valid_skills
