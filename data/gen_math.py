"""Generate math problems for evaluation."""
import random
from typing import List, Tuple, Optional


def sample_add(n: int, digits: int = 2, seed: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Generate n addition problems.
    
    Args:
        n: Number of problems to generate
        digits: Number of digits per operand (2 or 3)
        seed: Random seed for reproducibility
        
    Returns:
        List of (question_str, answer_int) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    problems = []
    min_val = 10 ** (digits - 1)
    max_val = 10 ** digits - 1
    
    for _ in range(n):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        answer = a + b
        question = f"{a} + {b} ="
        problems.append((question, answer))
    
    return problems


def sample_mul(n: int, digits: int = 2, seed: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Generate n multiplication problems.
    
    Args:
        n: Number of problems to generate
        digits: Number of digits per operand (2 or 3)
        seed: Optional random seed for reproducibility
        
    Returns:
        List of (question_str, answer_int) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    problems = []
    min_val = 10 ** (digits - 1)
    max_val = 10 ** digits - 1
    
    for _ in range(n):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        answer = a * b
        question = f"{a} × {b} ="
        problems.append((question, answer))
    
    return problems


def train_test_split(problems: List[Tuple[str, int]], train_ratio: float = 0.7, seed: Optional[int] = None):
    """
    Split problems into train and test sets.
    
    Args:
        problems: List of (question, answer) tuples
        train_ratio: Proportion for training set
        seed: Random seed
        
    Returns:
        (train_problems, test_problems)
    """
    if seed is not None:
        random.seed(seed)
    
    shuffled = problems.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]
    
    return train, test

