"""Check whether candidate skills are supported by career history."""

import logging


logger = logging.getLogger(__name__)


def _empty_result() -> dict[str, bool | list[str]]:
    """Create the safe result used for malformed skill data.

    Returns:
        A fresh consistent result containing no classified skills.
    """
    return {
        "consistent": True,
        "supported_skills": [],
        "unsupported_skills": [],
    }


def check_skill_consistency(
    candidate: dict[str, object],
) -> dict[str, bool | list[str]]:
    """Check skills against the candidate's career history.

    Each string-valued skill is searched as a case-insensitive substring of
    the string representation of ``career_history``. Original skill order and
    duplicate entries are preserved, and the candidate is never modified.

    Args:
        candidate: Candidate mapping containing skills and career history.

    Returns:
        A mapping with the consistency decision and the ordered supported and
        unsupported skill lists. Malformed skill data is treated as
        consistent with empty lists.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot check skills for malformed candidate input")
        return _empty_result()

    skills = candidate.get("skills")
    if not isinstance(skills, list):
        logger.warning("Candidate skills are missing or malformed")
        return _empty_result()

    career_text = str(candidate.get("career_history", "")).casefold()
    supported_skills: list[str] = []
    unsupported_skills: list[str] = []

    for skill in skills:
        if not isinstance(skill, str):
            continue

        if skill.casefold() in career_text:
            supported_skills.append(skill)
        else:
            unsupported_skills.append(skill)

    consistent = len(unsupported_skills) == 0
    if not consistent:
        logger.warning(
            "Skill consistency check failed with %d unsupported skills",
            len(unsupported_skills),
        )

    logger.info(
        "Skill consistency check completed: consistent=%s, supported=%d, "
        "unsupported=%d",
        consistent,
        len(supported_skills),
        len(unsupported_skills),
    )
    return {
        "consistent": consistent,
        "supported_skills": supported_skills,
        "unsupported_skills": unsupported_skills,
    }
