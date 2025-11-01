"""Evaluation metrics."""
from typing import List, Optional
import numpy as np

from ..defense.voting import majority_vote


def attack_success_rate(
    predictions: List[Optional[int]],
    true_answers: List[int],
    attacker_goals: List[int],
) -> float:
    """
    Compute attack success rate.
    
    Args:
        predictions: Model predictions (from voting)
        true_answers: Correct answers
        attacker_goals: Target values attacker wants (same length)
        
    Returns:
        Fraction of predictions that match attacker goal (not true answer)
    """
    if len(predictions) != len(true_answers) or len(predictions) != len(attacker_goals):
        raise ValueError("All lists must have same length")
    
    successes = 0
    total = 0
    
    for pred, true_val, goal in zip(predictions, true_answers, attacker_goals):
        if pred is None:
            continue  # Skip invalid predictions
        total += 1
        # Success: prediction matches attacker goal (and differs from true answer)
        if pred == goal and pred != true_val:
            successes += 1
    
    return successes / total if total > 0 else 0.0


def accuracy(predictions: List[Optional[int]], true_answers: List[int]) -> float:
    """
    Compute accuracy (fraction matching true answer).
    
    Args:
        predictions: Model predictions
        true_answers: Correct answers
        
    Returns:
        Accuracy score [0, 1]
    """
    if len(predictions) != len(true_answers):
        raise ValueError("Lists must have same length")
    
    correct = 0
    total = 0
    
    for pred, true_val in zip(predictions, true_answers):
        if pred is None:
            continue
        total += 1
        if pred == true_val:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def goal_satisfied(prediction: Optional[int], goal: int) -> bool:
    """
    Check if prediction satisfies attacker goal.
    
    Args:
        prediction: Model prediction
        goal: Attacker target value
        
    Returns:
        True if prediction matches goal
    """
    return prediction is not None and prediction == goal

