"""Deterministic, profile-backed candidate reasoning generation."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

PROFICIENCY_SCORE = {
    "beginner": 1.0,
    "intermediate": 2.0,
    "advanced": 3.0,
    "expert": 4.0,
}

FOUNDATION_SKILLS = {
    "git",
    "java",
    "javascript",
    "python",
    "r",
    "sql",
    "typescript",
}

SKILL_GROUP_ALIASES = {
    "llm": (
        "fine-tuning llms",
        "generative ai",
        "gpt",
        "hugging face transformers",
        "large language model",
        "llama",
        "llm",
        "llms",
        "lora",
        "mistral",
        "peft",
        "prompt engineering",
        "qlora",
        "transformer",
        "transformers",
    ),
    "retrieval": (
        "bm25",
        "dense retrieval",
        "elasticsearch",
        "embeddings",
        "faiss",
        "haystack",
        "hybrid retrieval",
        "hybrid search",
        "information retrieval",
        "milvus",
        "opensearch",
        "pgvector",
        "pinecone",
        "qdrant",
        "retrieval",
        "semantic search",
        "sentence transformers",
        "vector database",
        "vector databases",
        "vector recall",
        "vector search",
        "weaviate",
    ),
    "rag": (
        "langchain",
        "rag",
        "retrieval augmented generation",
    ),
    "ranking": (
        "a/b testing",
        "learning to rank",
        "learning-to-rank",
        "ndcg",
        "ranking",
        "recommendation systems",
        "recommendations",
        "xgboost",
    ),
    "ml": (
        "computer vision",
        "deep learning",
        "machine learning",
        "mlops",
        "nlp",
        "pytorch",
        "scikit-learn",
        "tensorflow",
    ),
    "data": (
        "airflow",
        "apache beam",
        "apache flink",
        "data engineering",
        "etl",
        "kafka",
        "spark",
        "streaming",
    ),
    "cloud": (
        "aws",
        "azure",
        "docker",
        "gcp",
        "google cloud",
        "kubernetes",
        "terraform",
    ),
}

SPECIALIST_SKILL_GROUPS = {"llm", "retrieval", "rag", "ranking", "data", "cloud"}

GENERIC_REASONS = {
    "excellent communication",
    "good candidate",
    "good fit",
    "good technical background",
    "high github activity",
    "high recruiter visibility",
    "strong behavioral alignment",
    "strong profile",
}


def _profile(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    return profile if isinstance(profile, dict) else {}


def _records(candidate: dict[str, Any], key: str) -> list[Any]:
    records = candidate.get(key, [])
    return records if isinstance(records, list) else []


def _signals(candidate: dict[str, Any]) -> dict[str, Any]:
    signals = candidate.get("redrob_signals", {})
    return signals if isinstance(signals, dict) else {}


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _norm(value: Any) -> str:
    return _clean_text(value).casefold()


def _title(candidate: dict[str, Any]) -> str:
    profile = _profile(candidate)
    title = _clean_text(profile.get("current_title"))
    if title:
        return title

    headline = _clean_text(profile.get("headline"))
    if headline:
        return re.split(r"\s+[|:-]\s+", headline, maxsplit=1)[0]

    return "Professional"


def _years(candidate: dict[str, Any]) -> float:
    try:
        return float(_profile(candidate).get("years_of_experience", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _response_rate(candidate: dict[str, Any]) -> float:
    try:
        return float(_signals(candidate).get("recruiter_response_rate", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _iter_text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        text = _clean_text(value)
        if text:
            yield text
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_text_values(nested_value)
        return

    if isinstance(value, list):
        for nested_value in value:
            yield from _iter_text_values(nested_value)


def _candidate_text(candidate: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for key in (
        "profile",
        "career_history",
        "education",
        "skills",
        "certifications",
        "projects",
        "publications",
        "achievements",
        "patents",
    ):
        text_parts.extend(_iter_text_values(candidate.get(key)))
    return " ".join(text_parts).casefold()


def _job_terms(job: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for skill in job.get("required_skills", []):
        if isinstance(skill, str) and skill.strip():
            terms.add(_norm(skill))

    for value in _iter_text_values(job):
        for alias_group in SKILL_GROUP_ALIASES.values():
            for alias in alias_group:
                if alias in value.casefold():
                    terms.add(alias)
    return terms


def _job_text(job: dict[str, Any]) -> str:
    return " ".join(_iter_text_values(job)).casefold()


def _skill_groups(skill_name: str) -> set[str]:
    normalized = _norm(skill_name)
    groups: set[str] = set()
    for group, aliases in SKILL_GROUP_ALIASES.items():
        if any(alias == normalized or alias in normalized for alias in aliases):
            groups.add(group)
    return groups


def _job_groups(job_terms: set[str], job_text: str) -> set[str]:
    groups: set[str] = set()
    for group, aliases in SKILL_GROUP_ALIASES.items():
        if any(alias in job_terms or alias in job_text for alias in aliases):
            groups.add(group)
    return groups


def _skill_relevance(
    skill_name: str,
    job_terms: set[str],
    job_text: str,
    active_job_groups: set[str],
    matched_skills: set[str],
) -> float:
    normalized = _norm(skill_name)
    relevance = 0.0

    if normalized in job_terms or normalized in matched_skills:
        relevance = 8.0
    elif normalized and normalized in job_text:
        relevance = 6.0

    overlapping_groups = _skill_groups(skill_name).intersection(active_job_groups)
    if overlapping_groups:
        relevance = max(relevance, 5.0 + len(overlapping_groups))

    if normalized in FOUNDATION_SKILLS and relevance:
        relevance = min(relevance, 4.0)

    return relevance


def _skill_score(
    skill: dict[str, Any],
    job_terms: set[str],
    job_text: str,
    active_job_groups: set[str],
    matched_skills: set[str],
) -> tuple[float, float, float, float, float]:
    name = _clean_text(skill.get("name"))
    proficiency = PROFICIENCY_SCORE.get(_norm(skill.get("proficiency")), 0.0)
    endorsements = _safe_float(skill.get("endorsements"))
    duration = _safe_float(skill.get("duration_months"))
    relevance = _skill_relevance(
        name,
        job_terms,
        job_text,
        active_job_groups,
        matched_skills,
    )
    specialty_boost = 0.0
    if _skill_groups(name).intersection(SPECIALIST_SKILL_GROUPS):
        specialty_boost = 2.0

    score = (
        relevance * 20.0
        + proficiency * 8.0
        + min(endorsements, 80.0) * 0.20
        + min(duration, 120.0) * 0.08
        + specialty_boost
    )
    return score, relevance, proficiency, endorsements, duration


def _top_job_skills(
    candidate: dict[str, Any],
    match: Any,
    job: dict[str, Any],
) -> list[str]:
    skills = [
        skill
        for skill in _records(candidate, "skills")
        if isinstance(skill, dict) and _clean_text(skill.get("name"))
    ]
    if not skills:
        return ["Skills not listed", "Profile evidence unavailable"]

    job_terms = _job_terms(job)
    text = _job_text(job)
    active_job_groups = _job_groups(job_terms, text)
    matched_skills = {
        _norm(skill)
        for skill in getattr(match, "matched_skills", [])
        if isinstance(skill, str)
    }

    scored_skills: list[tuple[float, float, float, float, float, int, str]] = []
    for index, skill in enumerate(skills):
        score, relevance, proficiency, endorsements, duration = _skill_score(
            skill,
            job_terms,
            text,
            active_job_groups,
            matched_skills,
        )
        scored_skills.append(
            (
                score,
                relevance,
                proficiency,
                endorsements,
                duration,
                -index,
                _clean_text(skill.get("name")),
            )
        )

    scored_skills.sort(reverse=True)
    selected = [entry[-1] for entry in scored_skills[:2]]

    if len(selected) == 1:
        selected.append(selected[0])

    return selected


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_records(candidate: dict[str, Any], key: str) -> bool:
    return bool(_records(candidate, key))


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _certification_reason(candidate: dict[str, Any]) -> str | None:
    certifications = [
        cert
        for cert in _records(candidate, "certifications")
        if isinstance(cert, dict)
    ]
    if not certifications:
        return None

    names = " ".join(_iter_text_values(certifications)).casefold()
    if "aws" in names:
        return "AWS Certified"
    if "azure" in names:
        return "Azure Certified"
    if "google cloud" in names or re.search(r"\bgcp\b", names):
        return "Google Cloud Certified"
    if "tensorflow" in names:
        return "TensorFlow Developer Certificate"
    if "kubernetes" in names:
        return "Kubernetes certification"
    if "nlp" in names:
        return "NLP certification"
    if "deep learning" in names:
        return "Deep learning certification"
    if "machine learning" in names:
        return "Machine learning certification"
    return "Professional certification"


def _leadership_reason(candidate: dict[str, Any], text: str) -> str | None:
    titles = " ".join(
        _clean_text(value)
        for value in (
            _profile(candidate).get("current_title"),
            *(
                role.get("title")
                for role in _records(candidate, "career_history")
                if isinstance(role, dict)
            ),
        )
    ).casefold()

    if re.search(r"\b(managed|led)\b.{0,80}\b(engineers|engineering team)\b", text):
        return "Led engineering teams"
    if re.search(
        r"\b(managed|led)\b.{0,80}\b(team|teams|cross-functional|"
        r"platform|migration|rollout)\b",
        text,
    ):
        return "Managed cross-functional teams"
    if re.search(r"\b(tech lead|team lead|staff|principal|head of)\b", titles):
        return "Technical leadership experience"
    return None


def _production_reason(text: str) -> str | None:
    if "recommendation" in text and _has_any(text, ("production", "shipped", "serving")):
        return "Production recommendation systems"
    if _has_any(text, ("real-time", "real time", "streaming")):
        return "Real-time data pipelines"
    if _has_any(text, ("distributed", "millions", "50m+", "30m+", "qps", "latency", "p95")):
        return "Large-scale distributed systems"
    if _has_any(text, ("production", "shipped", "deployed")) and _has_any(
        text,
        ("ml", "machine learning", "model", "ranking", "retrieval"),
    ):
        return "Built production-scale ML systems"
    if _has_any(text, ("microservices", "scale")):
        return "Built scalable backend systems"
    return None


def _ai_project_reason(text: str) -> str | None:
    if _has_any(text, ("rag", "retrieval augmented generation")):
        return "Built RAG systems"
    if _has_any(text, ("fine-tuned", "fine tuning", "fine-tuning", "lora", "qlora", "peft")):
        return "Fine-tuning experience"
    if _has_any(text, ("llm", "llms", "generative ai")):
        return "Hands-on LLM projects"
    if _has_any(text, ("embedding", "embeddings", "vector search", "semantic search")):
        return "Embedding search projects"
    if _has_any(text, ("transformer", "transformers")):
        return "Transformer model experience"
    return None


def _cloud_reason(candidate: dict[str, Any], text: str) -> str | None:
    skill_text = " ".join(
        _norm(skill.get("name"))
        for skill in _records(candidate, "skills")
        if isinstance(skill, dict)
    )
    cloud_text = f"{skill_text} {text}"
    has_deployment_context = _has_any(
        text,
        ("deployed", "deployment", "production", "infrastructure"),
    )
    if (
        has_deployment_context
        and _has_any(cloud_text, ("kubernetes", "docker", "terraform"))
        and _has_any(
        text,
        ("ml", "machine learning", "model", "ai"),
        )
    ):
        return "Containerized ML deployment"
    if has_deployment_context and _has_any(cloud_text, ("kubernetes", "docker", "terraform")):
        return "Containerized deployment experience"
    if _has_any(cloud_text, ("aws", "azure", "gcp", "google cloud")):
        if has_deployment_context:
            return "Cloud-native deployment experience"
        return "Cloud infrastructure skills"
    return None


def _research_reason(candidate: dict[str, Any], text: str) -> str | None:
    education_text = " ".join(_iter_text_values(_records(candidate, "education"))).casefold()
    title = _norm(_profile(candidate).get("current_title"))
    if _has_any(title, ("research scientist", "applied scientist")):
        return "Applied AI research"
    if _has_any(education_text, ("phd", "ph.d", "doctorate")):
        return "Research background"
    if "research" in education_text and _has_any(education_text, ("m.s", "m.sc", "masters", "master")):
        return "Academic research experience"
    if "research" in text and _has_any(text, ("model", "ml", "ai", "ranking")):
        return "Applied AI research"
    return None


def _recruiter_signal_reason(candidate: dict[str, Any]) -> str:
    signals = _signals(candidate)
    saved = _safe_float(signals.get("saved_by_recruiters_30d"))
    appearances = _safe_float(signals.get("search_appearance_30d"))
    response = _safe_float(signals.get("recruiter_response_rate"))
    interviews = _safe_float(signals.get("interview_completion_rate"))

    if saved >= 20 or appearances >= 1000:
        return "Frequently contacted by recruiters"
    if response >= 0.75:
        return "High recruiter response"
    if interviews >= 0.85:
        return "Strong interview completion"
    if signals.get("verified_email") and signals.get("verified_phone"):
        return "Verified profile"
    if signals.get("open_to_work_flag"):
        return "Open to work"
    return "Profile data reviewed"


def _unique_strength(candidate: dict[str, Any]) -> str:
    text = _candidate_text(candidate)

    if _has_records(candidate, "publications") or re.search(
        r"\b(published (ml|machine learning|ai|research)|"
        r"journal publication|conference paper|research paper)\b",
        text,
    ):
        if _has_any(text, ("ml", "machine learning", "ai", "deep learning")):
            return "Published ML research"
        return "Research publications"

    if _has_records(candidate, "patents") or "patent" in text:
        if "filed" in text:
            return "Filed patents"
        return "Patent holder"

    if _has_any(text, ("kaggle", "competition project", "competition projects")):
        if "gold" in text and "kaggle" in text:
            return "Gold Kaggle profile"
        return "Kaggle competition experience"

    if re.search(
        r"\b(open-source|open source|oss|github)\b.{0,80}"
        r"\b(contributor|project|projects|repo|repository|library)\b",
        text,
    ):
        return "Open-source contributor"

    certification_reason = _certification_reason(candidate)
    if certification_reason:
        return certification_reason

    leadership_reason = _leadership_reason(candidate, text)
    if leadership_reason:
        return leadership_reason

    production_reason = _production_reason(text)
    if production_reason:
        return production_reason

    ai_project_reason = _ai_project_reason(text)
    if ai_project_reason:
        return ai_project_reason

    cloud_reason = _cloud_reason(candidate, text)
    if cloud_reason:
        return cloud_reason

    research_reason = _research_reason(candidate, text)
    if research_reason:
        return research_reason

    return _recruiter_signal_reason(candidate)


def _validate_reason(reason: str) -> str:
    normalized = reason.casefold()
    if any(generic in normalized for generic in GENERIC_REASONS):
        logger.warning("Generic reasoning phrase blocked: %s", reason)
        return "Profile-backed evidence reviewed"
    return reason


def generate_reason(
    candidate: dict[str, Any],
    match: Any,
    score: Any,
    job: dict[str, Any],
) -> str:
    """Generate exactly four evidence-backed reasoning parts.

    Args:
        candidate: Raw candidate record.
        match: Candidate/job match result. Used only for skill relevance hints.
        score: Score result, accepted for API compatibility.
        job: Parsed job description.

    Returns:
        Concise deterministic reasoning in the form:
        role + experience; two skills; one differentiator; response rate.
    """
    del score

    try:
        skills = _top_job_skills(candidate, match, job)
        evidence = _validate_reason(_unique_strength(candidate))

        return (
            f"{_title(candidate)} with {_years(candidate):.1f} years; "
            f"{skills[0]}, {skills[1]}; "
            f"{evidence}; "
            f"response rate {_response_rate(candidate):.2f}."
        )
    except Exception:
        logger.exception("Reason generation failed")
        return "Candidate profile reviewed; Skills unavailable; Evidence unavailable; response rate 0.00."
