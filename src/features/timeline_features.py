"""
Timeline features extractor.
Provides the TimelineFeatures dataclass and extraction logic to process candidate timelines.
"""
import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
from datetime import date
logger = logging.getLogger(__name__)
_DATE_YMD_RE = re.compile(r'^(\d{4})-(\d{1,2})-(\d{1,2})$')
_DATE_YM_RE = re.compile(r'^(\d{4})-(\d{1,2})$')
_DATE_Y_RE = re.compile(r'^(\d{4})$')
@dataclass
class TimelineFeatures:
    """Dataclass holding extracted features from a candidate's employment timeline."""
    career_start_date: Optional[str]
    career_end_date: Optional[str]
    total_roles: int
    career_span_months: int
    employment_gap_count: int
    total_gap_months: int
    longest_gap_months: int
    overlapping_role_count: int
    has_current_role: bool
def _parse_date(date_str: Any) -> Optional[date]:
    """
    Parse date from string. Supports:
    - YYYY-MM-DD
    - YYYY-MM
    - YYYY
    - Present / Current
    """
    if not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    if not date_str:
        return None
        
    lower_date = date_str.lower()
    if lower_date in ('present', 'current', 'now'):
        return date.today()
        
    # YYYY-MM-DD
    match = _DATE_YMD_RE.match(date_str)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
            
    # YYYY-MM
    match = _DATE_YM_RE.match(date_str)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), 1)
        except ValueError:
            pass
            
    # YYYY
    match = _DATE_Y_RE.match(date_str)
    if match:
        try:
            return date(int(match.group(1)), 1, 1)
        except ValueError:
            pass
            
    logger.warning("Malformed date encountered: %s", date_str)
    return None
def _months_between(start: date, end: date) -> int:
    """Calculate the number of months between two dates."""
    if start > end:
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month)
def extract_timeline_features(candidate: dict) -> TimelineFeatures:
    """
    Extract timeline features from a candidate profile.
    
    Args:
        candidate: Dictionary containing candidate data. Should include an 
                   experience list with start and end dates.
                   
    Returns:
        TimelineFeatures dataclass with computed metrics.
    """
    exp_list = (candidate.get('experience') or 
                candidate.get('work_history') or 
                candidate.get('positions') or [])
                
    if not isinstance(exp_list, list):
        logger.warning("Candidate experience list is not a list or is missing.")
        exp_list = []
        
    valid_roles: List[Tuple[date, date]] = []
    has_current_role = False
    
    for role in exp_list:
        if not isinstance(role, dict):
            continue
            
        start_raw = role.get('start_date') or role.get('startDate') or role.get('start')
        end_raw = role.get('end_date') or role.get('endDate') or role.get('end')
        
        start_d = _parse_date(start_raw)
        if not start_d:
            logger.warning("Skipping role due to missing or invalid start date: %s", start_raw)
            continue
            
        if end_raw is None:
            # If end date is missing, we assume it's a current role
            end_d = date.today()
        else:
            end_d_parsed = _parse_date(end_raw)
            if not end_d_parsed:
                logger.warning("Skipping role due to unparseable end date: %s", end_raw)
                continue
            end_d = end_d_parsed
            
        if start_d > end_d:
            logger.warning("Start date %s is after end date %s. Skipping role.", start_d, end_d)
            continue
            
        valid_roles.append((start_d, end_d))
        
        if end_d >= date.today():
            has_current_role = True
    if not valid_roles:
        return TimelineFeatures(
            career_start_date=None,
            career_end_date=None,
            total_roles=len(exp_list),
            career_span_months=0,
            employment_gap_count=0,
            total_gap_months=0,
            longest_gap_months=0,
            overlapping_role_count=0,
            has_current_role=False
        )
    # Sort roles by start date
    valid_roles.sort(key=lambda x: x[0])
    
    career_start_date = valid_roles[0][0]
    career_end_date = max(r[1] for r in valid_roles)
    career_span_months = _months_between(career_start_date, career_end_date)
    
    # Overlapping roles count
    overlapping_indices = set()
    for i in range(len(valid_roles)):
        start_i, end_i = valid_roles[i]
        for j in range(i + 1, len(valid_roles)):
            start_j, end_j = valid_roles[j]
            # Next role starts before current role ends -> overlap
            if start_j <= end_i:
                overlapping_indices.add(i)
                overlapping_indices.add(j)
            else:
                # Since sorted by start date, no further roles will overlap with role i
                break
                
    overlapping_role_count = len(overlapping_indices)
    
    # Merge intervals to find gaps
    merged_intervals = [valid_roles[0]]
    for start, end in valid_roles[1:]:
        last_start, last_end = merged_intervals[-1]
        if start <= last_end:
            # Overlapping or adjacent, merge them
            merged_intervals[-1] = (last_start, max(last_end, end))
        else:
            # Disjoint, add new interval
            merged_intervals.append((start, end))
            
    # Calculate gaps
    employment_gap_count = 0
    total_gap_months = 0
    longest_gap_months = 0
    
    for i in range(1, len(merged_intervals)):
        prev_end = merged_intervals[i-1][1]
        curr_start = merged_intervals[i][0]
        
        gap_months = _months_between(prev_end, curr_start)
        # We only count gap if it's strictly > 0 months.
        if gap_months > 0:
            employment_gap_count += 1
            total_gap_months += gap_months
            if gap_months > longest_gap_months:
                longest_gap_months = gap_months
                
    return TimelineFeatures(
        career_start_date=career_start_date.isoformat(),
        career_end_date=career_end_date.isoformat(),
        total_roles=len(exp_list),
        career_span_months=career_span_months,
        employment_gap_count=employment_gap_count,
        total_gap_months=total_gap_months,
        longest_gap_months=longest_gap_months,
        overlapping_role_count=overlapping_role_count,
        has_current_role=has_current_role
    )
