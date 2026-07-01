"""Validate the structural consistency of candidate behavioral signals."""

import logging


logger = logging.getLogger(__name__)


def _empty_result() -> dict[str, bool | int]:
    """Create the safe result used for malformed behavioral signal data.

    Returns:
        A fresh consistent result containing zero signal counts.
    """
    return {
        "consistent": True,
        "valid_signals": 0,
        "invalid_signals": 0,
    }


def check_behavior_consistency(
    candidate: dict[str, object],
) -> dict[str, bool | int]:
    """Check candidate behavioral signals for consistency.

    Only ``redrob_signals`` is inspected. Each valid signal must be a
    non-empty string. Candidate data is never modified.

    Args:
        candidate: Candidate mapping containing behavioral signal data.

    Returns:
        A mapping containing the consistency decision and valid and invalid
        signal counts. Missing or malformed signal data produces a safe,
        consistent result with zero counts.
    """
    if not isinstance(candidate, dict):
        logger.warning("Cannot check behavior for malformed candidate input")
        return _empty_result()

    signals = candidate.get("redrob_signals")
    if not isinstance(signals, list):
        logger.warning("Candidate behavioral signals are missing or malformed")
        return _empty_result()

    valid_signals = 0
    invalid_signals = 0

    for signal in signals:
        if isinstance(signal, str) and signal.strip():
            valid_signals += 1
        else:
            invalid_signals += 1
            logger.warning("Invalid behavioral signal encountered")

    consistent = invalid_signals == 0
    logger.info(
        "Behavior consistency check completed: consistent=%s, valid=%d, "
        "invalid=%d",
        consistent,
        valid_signals,
        invalid_signals,
    )
    return {
        "consistent": consistent,
        "valid_signals": valid_signals,
        "invalid_signals": invalid_signals,
    }
