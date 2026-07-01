"""
Behavioral features extractor.
Provides extraction logic to identify behavioral signals from candidate profiles.
"""
import logging
import re
from functools import lru_cache
from typing import List, Any
# Assuming timeline_features is in the same package structure
from src.features.timeline_features import extract_timeline_features
from src.models.candidate_features import BehaviorFeatures
logger = logging.getLogger(__name__)
# Keywords maps
POSITIVE_KEYWORDS = [
    "certifications", "open source", "awards", "volunteering", 
    "hackathons", "publications", "certification", "award", 
    "hackathon", "publication"
]
LEADERSHIP_KEYWORDS = ["lead", "manager", "director", "mentor", "team lead", "architect"]
COLLABORATION_KEYWORDS = ["teamwork", "collaboration", "cross-functional", "agile", "scrum"]
COMMUNICATION_KEYWORDS = ["presentation", "communication", "stakeholder", "client", "public speaking"]
ADAPTABILITY_KEYWORDS = ["multiple domains", "career switch", "upskilling", "certifications", "learning", "certification"]


@lru_cache(maxsize=32)
def _keyword_patterns(keywords: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile keyword search patterns once per keyword group."""
    return tuple(
        (kw, re.compile(r'\b' + re.escape(kw) + r'\b'))
        for kw in keywords
    )


def _search_text(text: str, keywords: List[str]) -> List[str]:
    """Find which keywords appear in the text using word boundaries."""
    found = set()
    if not text:
        return []
    
    text_lower = text.lower()
    for kw, pattern in _keyword_patterns(tuple(keywords)):
        if pattern.search(text_lower):
            found.add(kw)
    return list(found)
def _extract_text_from_fields(candidate: dict) -> str:
    """Concatenate relevant fields into a single block of text for searching."""
    fields_to_search = [
        'profile',
        'certifications',
        'projects',
        'achievements',
        'languages',
        'redrob_signals',
        'career_history',
        # Also include standard experience keys just in case
        'experience',
        'work_history',
        'positions',
        'summary',
        'about',
        'awards',
        'volunteering',
        'hackathons',
        'publications'
    ]
    
    extracted_text = []
    
    def _recursively_extract(data: Any):
        if isinstance(data, str):
            extracted_text.append(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                _recursively_extract(v)
        elif isinstance(data, list):
            for item in data:
                _recursively_extract(item)
                
    for field in fields_to_search:
        if candidate.get(field):
            # Inject the field name itself as a signal in case its presence implies a signal
            extracted_text.append(field)
            _recursively_extract(candidate[field])
            
    return " ".join(extracted_text)
def extract_behavior_features(candidate: dict, timeline=None) -> BehaviorFeatures:
    """
    Extract behavioral signals from a candidate profile.
    
    Args:
        candidate: Dictionary containing candidate data.
        
    Returns:
        BehaviorFeatures dataclass with extracted signals.
    """
    logger.info("Extracting behavior features")
    
    text_corpus = _extract_text_from_fields(candidate)
    
    # Positive signals from text
    positive_signals_set = set()
    for sig in _search_text(text_corpus, POSITIVE_KEYWORDS):
        # Normalize plurals to match requested rules
        if sig in ("certification", "certifications"):
            positive_signals_set.add("certifications")
        elif sig in ("award", "awards"):
            positive_signals_set.add("awards")
        elif sig in ("hackathon", "hackathons"):
            positive_signals_set.add("hackathons")
        elif sig in ("publication", "publications"):
            positive_signals_set.add("publications")
        else:
            positive_signals_set.add(sig)
            
    # Risk signals
    risk_signals_set = set()
    
    # 1. employment gaps > 12 months
    # 2. frequent job changes
    if timeline is None:
        timeline = extract_timeline_features(candidate)
    if timeline.longest_gap_months > 12:
        risk_signals_set.add("employment gaps > 12 months")
        
    if timeline.total_roles > 0 and timeline.career_span_months > 0:
        avg_months_per_role = timeline.career_span_months / timeline.total_roles
        # arbitrary threshold for frequent job changes: < 18 months average
        if avg_months_per_role < 18 and timeline.total_roles >= 3:
            risk_signals_set.add("frequent job changes")
            
    # 3. missing profile
    # The prompt mentions "missing profile". So we strictly check "profile".
    if not candidate.get("profile"):
        risk_signals_set.add("missing profile")
        
    # 4. empty skills
    if not candidate.get("skills") or len(candidate.get("skills", [])) == 0:
        risk_signals_set.add("empty skills")
        
    # 5. incomplete education
    # Assume incomplete if education missing or degree not present
    edu = candidate.get("education", [])
    if not edu:
        risk_signals_set.add("incomplete education")
    else:
        has_degree = False
        for e in edu:
            if isinstance(e, dict) and (e.get('degree') or e.get('qualification') or e.get('graduated')):
                has_degree = True
                break
        if not has_degree:
            risk_signals_set.add("incomplete education")
            
    # Leadership signals
    leadership_signals_set = set(_search_text(text_corpus, LEADERSHIP_KEYWORDS))
    
    # Collaboration signals
    collaboration_signals_set = set(_search_text(text_corpus, COLLABORATION_KEYWORDS))
    
    # Communication signals
    communication_signals_set = set(_search_text(text_corpus, COMMUNICATION_KEYWORDS))
    
    # Adaptability signals
    adapt_raw = _search_text(text_corpus, ADAPTABILITY_KEYWORDS)
    adaptability_signals_set = set()
    for sig in adapt_raw:
        if sig == "certification":
            adaptability_signals_set.add("certifications")
        else:
            adaptability_signals_set.add(sig)
            
    return BehaviorFeatures(
        positive_signals=sorted(list(positive_signals_set)),
        risk_signals=sorted(list(risk_signals_set)),
        leadership_signals=sorted(list(leadership_signals_set)),
        collaboration_signals=sorted(list(collaboration_signals_set)),
        communication_signals=sorted(list(communication_signals_set)),
        adaptability_signals=sorted(list(adaptability_signals_set))
    )
