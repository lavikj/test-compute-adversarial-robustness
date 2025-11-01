"""Voting mechanisms for self-consistency."""
import os
import sys
from typing import List, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.inference_budget import extract_integer


def majority_vote(outputs: List[str], tie_break: str = "median") -> Optional[int]:
    """
    Perform majority voting on model outputs.
    
    Args:
        outputs: List of model output strings
        tie_break: How to break ties ("median", "first", "random")
        
    Returns:
        Voted integer answer, or None if no valid integers found
    """
    integers = []
    for output in outputs:
        val = extract_integer(output)
        if val is not None:
            integers.append(val)
    
    if not integers:
        return None
    
    # Count votes
    counts = {}
    for val in integers:
        counts[val] = counts.get(val, 0) + 1
    
    # Find max count
    max_count = max(counts.values())
    winners = [val for val, count in counts.items() if count == max_count]
    
    if len(winners) == 1:
        return winners[0]
    
    # Tie break
    if tie_break == "median":
        return int(np.median(winners))
    elif tie_break == "first":
        return winners[0]
    elif tie_break == "random":
        return np.random.choice(winners)
    else:
        return winners[0]


def vote_confidence(outputs: List[str]) -> float:
    """
    Compute confidence of the vote (fraction of outputs agreeing with majority).
    
    Args:
        outputs: List of model output strings
        
    Returns:
        Confidence score [0, 1]
    """
    voted = majority_vote(outputs)
    if voted is None:
        return 0.0
    
    integers = [extract_integer(out) for out in outputs]
    integers = [val for val in integers if val is not None]
    
    if not integers:
        return 0.0
    
    agreement = sum(1 for val in integers if val == voted)
    return agreement / len(integers) if integers else 0.0

