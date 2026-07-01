"""Orchestrate deterministic candidate fraud-signal detection."""

import logging

from src.fraud.honeypot_detector import detect_honeypot
from src.fraud.keyword_detector import analyze_candidate_keywords
from src.fraud.outlier_detector import detect_outliers


logger = logging.getLogger(__name__)


def _malformed_candidate_result() -> dict[str, object]:
    """Create the safe result returned for malformed candidate input.

    Returns:
        A fresh fraud-analysis result with every detection flag disabled.
    """
    return {
        "keyword": {
            "keyword_stuffing": False,
            "keyword_hits": 0,
        },
        "outliers": {
            "experience_outlier": False,
            "skill_outlier": False,
            "company_outlier": False,
        },
        "honeypot": {
            "honeypot_detected": False,
            "triggered_fields": [],
        },
        "fraud_detected": False,
    }


def analyze_candidate_fraud(
    candidate: dict[str, object],
) -> dict[str, object]:
    """Analyze a candidate for configured fraud signals.

    The candidate is passed without modification to the keyword, outlier, and
    honeypot detectors. Fraud is detected when any detector reports one of its
    configured boolean signals.

    Args:
        candidate: Candidate mapping to analyze.

    Returns:
        The three detector results and their combined fraud decision. A
        malformed candidate produces a safe result with all flags disabled.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot analyze malformed candidate for fraud")
        return _malformed_candidate_result()

    keyword_result = analyze_candidate_keywords(candidate)
    outlier_result = detect_outliers(candidate)
    honeypot_result = detect_honeypot(candidate)

    fraud_detected = bool(
        keyword_result["keyword_stuffing"]
        or outlier_result["experience_outlier"]
        or outlier_result["skill_outlier"]
        or outlier_result["company_outlier"]
        or honeypot_result["honeypot_detected"]
    )

    if fraud_detected:
        logger.warning("Fraud signals detected for candidate")

    logger.info(
        "Candidate fraud analysis completed: detected=%s", fraud_detected
    )
    return {
        "keyword": keyword_result,
        "outliers": outlier_result,
        "honeypot": honeypot_result,
        "fraud_detected": fraud_detected,
    }
