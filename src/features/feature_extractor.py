"""
Aggregates all feature extraction logic.
"""
import logging
from typing import Dict, Any
from src.models.candidate_features import CandidateFeatures
from src.features.skill_features import extract_skill_features
from src.features.career_features import extract_career_features
from src.features.timeline_features import extract_timeline_features
from src.features.behavior_features import extract_behavior_features
logger = logging.getLogger(__name__)
def extract_candidate_features(candidate: dict) -> CandidateFeatures:
    """
    Extracts all candidate features and aggregates them into a CandidateFeatures object.
    
    Pipeline:
    1. Extract skill features
    2. Extract career features
    3. Extract timeline features
    4. Extract behavior features
    5. Build CandidateFeatures object
    
    Args:
        candidate: A dictionary containing candidate information.
        
    Returns:
        A CandidateFeatures object containing all extracted features.
    """
    logger.info("Starting candidate feature extraction")
    
    # Handle malformed candidate dictionaries gracefully
    if not isinstance(candidate, dict):
        logger.error("Malformed candidate data: expected dictionary, got %s", type(candidate).__name__)
        candidate = {}
        
    skills = extract_skill_features(candidate)
    career = extract_career_features(candidate)
    timeline = extract_timeline_features(candidate)
    behavior = extract_behavior_features(candidate, timeline)
        
    logger.info("Completed candidate feature extraction")
    
    return CandidateFeatures(
        candidate_id=candidate.get("candidate_id", ""),
        skill_features=skills,
        career_features=career,
        timeline_features=timeline,
        behavior_features=behavior,
    )
