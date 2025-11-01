"""Inference budget management for self-consistency."""
import os
import sys
from typing import List, Optional
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import LLMClient


def run_with_budget(
    model: LLMClient,
    prompt: str,
    k: int,
    max_tokens: int = 100,
    deliberate_steps: Optional[int] = None,
    temperature: float = 0.7,
) -> List[str]:
    """
    Run model k times with inference budget, return all outputs.
    
    Args:
        model: LLM client
        prompt: Input prompt
        k: Number of self-consistency samples
        max_tokens: Maximum tokens per sample
        deliberate_steps: Optional deliberate reasoning steps
        temperature: Sampling temperature
        
    Returns:
        List of k generated outputs
    """
    outputs = []
    for i in range(k):
        output = model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None,
            deliberate_steps=deliberate_steps,
        )
        outputs.append(output)
    
    return outputs


def extract_integer(response: str) -> Optional[int]:
    """
    Extract the first integer from a response.
    
    Args:
        response: Model response string
        
    Returns:
        First integer found, or None
    """
    # Try to find integer in various formats
    # Look for standalone numbers
    patterns = [
        r'\b(\d+)\b',  # Word boundary number
        r'=\s*(\d+)',  # After equals sign
        r':\s*(\d+)',  # After colon
        r'(\d+)',      # Any integer
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    return None

