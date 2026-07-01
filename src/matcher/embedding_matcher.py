"""
Embedding matcher module.
Provides semantic similarity matching between a candidate and a job.
Currently implements a deterministic heuristic placeholder until real
embedding models are integrated.
"""
import logging
from typing import Dict, Any, Set
from src.models.candidate_features import CandidateFeatures
from src.utils.normalizer import normalize_skill_list
logger = logging.getLogger(__name__)
def _clamp(value: float) -> float:
    """Clamp a value between 0.0 and 1.0."""
    return max(0.0, min(1.0, value))
def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute the Jaccard similarity between two sets of strings."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union)
class BaseEmbeddingModel:
    """
    Abstract base interface for embedding models.
    Future models (e.g., SentenceTransformers, OpenAI) should inherit from this.
    """
    def compute_similarity(
        self, candidate: CandidateFeatures, job: Dict[str, Any]
    ) -> float:
        """Compute similarity between candidate and job."""
        raise NotImplementedError
class DeterministicPlaceholderModel(BaseEmbeddingModel):
    """
    A placeholder model that computes semantic similarity deterministically
    using heuristics over normalized skills, technical skills, soft skills,
    career seniority level, and leadership experience.
    """
    def _extract_safe_set(self, source: Any, attribute: str) -> Set[str]:
        """Safely extract a list of strings from an attribute and return as a set."""
        if not source:
            return set()
        val = getattr(source, attribute, [])
        if isinstance(val, list):
            # Safe extraction assuming elements should be treated as strings
            return {str(v).strip().lower() for v in val if isinstance(v, str) and v}
        return set()
    def compute_similarity(
        self, candidate: CandidateFeatures, job: Dict[str, Any]
    ) -> float:
        """Compute the deterministic similarity."""
        logger.debug("Computing deterministic semantic similarity placeholder")
        # 1. Skills Matching
        job_skills = job.get("_normalized_required_skill_set")
        if job_skills is None:
            job_skills_raw = job.get("required_skills", [])
            if not isinstance(job_skills_raw, list):
                job_skills_raw = []
            job_skills = set(normalize_skill_list(job_skills_raw))
        cand_norm = set()
        cand_tech = set()
        cand_soft = set()
        if candidate.skill_features:
            cand_norm = self._extract_safe_set(
                candidate.skill_features, "normalized_skills"
            )
            cand_tech = self._extract_safe_set(
                candidate.skill_features, "technical_skills"
            )
            cand_soft = self._extract_safe_set(
                candidate.skill_features, "soft_skills"
            )
        all_cand_skills = cand_norm | cand_tech | cand_soft
        skill_sim = _jaccard_similarity(all_cand_skills, job_skills)
        # 2. Seniority & Leadership Matching
        job_seniority = str(job.get("required_seniority", "")).strip().lower()
        job_needs_leadership = bool(job.get("requires_leadership", False))
        cand_seniority = ""
        cand_has_leadership = False
        if candidate.career_features:
            cand_seniority = str(
                getattr(candidate.career_features, "seniority_level", "")
            ).strip().lower()
            cand_has_leadership = bool(
                getattr(candidate.career_features, "leadership_experience", False)
            )
        # Seniority similarity heuristic
        seniority_sim = 1.0
        if job_seniority:
            if not cand_seniority:
                seniority_sim = 0.5
            elif job_seniority == cand_seniority:
                seniority_sim = 1.0
            else:
                seniority_sim = 0.7
        # Leadership similarity heuristic
        leadership_sim = 1.0
        if job_needs_leadership:
            if cand_has_leadership:
                leadership_sim = 1.0
            else:
                leadership_sim = 0.0
        # Weighted combination for the final score
        final_score = (skill_sim * 0.6) + (seniority_sim * 0.2) + (leadership_sim * 0.2)
        return _clamp(final_score)
_MODEL = DeterministicPlaceholderModel()


def compute_embedding_similarity(candidate: CandidateFeatures, job: dict) -> float:
    """
    Computes the semantic similarity between a candidate and a job.
    This is the public API entry point. It utilizes a deterministic
    placeholder model. Real embedding models can be hot-swapped here internally
    without altering the external API.
    Args:
        candidate: CandidateFeatures object containing candidate data.
        job: Dictionary representing job requirements.
    Returns:
        Float value between 0.0 and 1.0 representing semantic similarity.
    """
    logger.info("Starting embedding similarity computation")
    if not isinstance(job, dict):
        logger.error("Job must be a dictionary. Got %s", type(job).__name__)
        job = {}
    try:
        similarity = _MODEL.compute_similarity(candidate, job)
    except Exception as e:
        logger.exception("Error during embedding similarity computation: %s", e)
        similarity = 0.0
    similarity = _clamp(similarity)
    logger.info("Completed embedding similarity computation: %.3f", similarity)
    return similarity
