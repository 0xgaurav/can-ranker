"""
Keyword matcher module.
Provides functionality for matching candidate skills against job requirements.
"""
import logging
from typing import Dict, Any, List, Set
from src.models.candidate_features import CandidateFeatures
from src.utils.normalizer import normalize_skill_list
logger = logging.getLogger(__name__)
def _clamp(value: float) -> float:
    """Clamp a value between 0.0 and 1.0."""
    return max(0.0, min(1.0, value))
def match_keywords(candidate: CandidateFeatures, job: dict) -> Dict[str, Any]:
    """
    Match candidate skills against required job skills.
    
    Args:
        candidate: CandidateFeatures object containing extracted skill features.
        job: Dictionary containing job requirements, specifically 'required_skills'.
        
    Returns:
        Dictionary containing:
        - matched_skills: List of required skills the candidate possesses.
        - missing_skills: List of required skills the candidate lacks.
        - skill_similarity: Float between 0.0 and 1.0 representing the match ratio.
    """
    logger.info("Starting keyword matching for skills")
    
    if not isinstance(job, dict):
        logger.error("Job must be a dictionary. Got %s", type(job).__name__)
        job = {}
        
    # Extract required skills
    required_skills = job.get("_normalized_required_skills")
    if required_skills is None:
        raw_required = job.get("required_skills", [])
        if not isinstance(raw_required, list):
            raw_required = []
        required_skills = normalize_skill_list(raw_required)
            
    # Extract candidate skills from normalized_skills field
    candidate_skills_raw = []
    if candidate and candidate.skill_features and hasattr(candidate.skill_features, 'normalized_skills'):
        raw_skills = candidate.skill_features.normalized_skills
        if isinstance(raw_skills, list):
            candidate_skills_raw = raw_skills
            
    # Remove duplicates from candidate skills
    candidate_skills: Set[str] = {s for s in candidate_skills_raw if s}
    
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    
    for req in required_skills:
        if req in candidate_skills:
            matched_skills.append(req)
        else:
            missing_skills.append(req)
            
    if required_skills:
        skill_similarity = len(matched_skills) / len(required_skills)
    else:
        # If the job requires no specific skills, return 1.0 if the candidate has any skills,
        # or 0.0 if they have no skills at all.
        skill_similarity = 1.0 if candidate_skills else 0.0
        
    skill_similarity = _clamp(float(skill_similarity))
    
    logger.info(
        "Completed keyword matching: %d matched, %d missing, similarity=%.2f",
        len(matched_skills), len(missing_skills), skill_similarity
    )
    
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_similarity": skill_similarity
    }
