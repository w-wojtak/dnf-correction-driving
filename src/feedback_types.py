# src/feedback_types.py
"""
Feedback operation types for DNF correction system.
"""

from enum import Enum

class FeedbackType(Enum):
    SKIP  = "skip"
    LOCK  = "lock"
    EARLY = "early"
    LATE  = "late"
    SWAP  = "swap"
    # TODO: ADD = "add"