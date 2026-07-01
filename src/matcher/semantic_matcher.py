"""
Semantic matcher module.
Provides a unified matching interface combining both keyword matching
and embedding similarity matching for a candidate and job profile.
"""
import logging
from src.matcher.keyword_matcher import match_keywords
from src.matcher.embedding_matcher import compute_embedding_similarity
from src.models.match_result import MatchResult
from src.models.candidate_features import CandidateFeatures
logger = logging.getLogger(__name__)
def _clamp(value: float) -> float:
    """Clamp a value between 0.0 and 1.0."""
    return max(0.0, min(1.0, value))
def semantic_match(candidate: CandidateFeatures, job: dict) -> MatchResult:
    """
    Combines keyword matching and embedding similarity to compute
    an overall semantic match result.
    
    Args:
        candidate: CandidateFeatures object containing extracted features.
        job: Dictionary representing job requirements.
        
    Returns:
        MatchResult populated with similarities and extracted skills.
    """
    logger.info("Starting semantic match evaluation")
    
    if not isinstance(job, dict):
        logger.error("Job must be a dictionary. Got %s", type(job).__name__)
        job = {}
        
    keyword_result = match_keywords(candidate, job)
    embedding_similarity = compute_embedding_similarity(candidate, job)
    
    keyword_sim = keyword_result.get("skill_similarity", 0.0)
    
    overall_similarity = (0.4 * keyword_sim) + (0.6 * embedding_similarity)
    
    candidate_id = getattr(candidate, "candidate_id", "")
    if candidate_id is None:
        candidate_id = ""
        
    logger.info(
        "Semantic match complete: keyword_sim=%.3f, embedding_sim=%.3f, overall_sim=%.3f",
        keyword_sim, embedding_similarity, overall_similarity
    )
    
    return MatchResult(
        candidate_id=str(candidate_id),
        overall_similarity=_clamp(float(overall_similarity)),
        skill_similarity=_clamp(float(keyword_sim)),
        experience_similarity=0.0,
        education_similarity=0.0,
        behavior_similarity=0.0,
        matched_skills=keyword_result.get("matched_skills", []),
        missing_skills=keyword_result.get("missing_skills", [])
    )