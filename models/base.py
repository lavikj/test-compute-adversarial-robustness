"""Abstract base class for LLM clients."""
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Abstract interface for LLM inference."""
    
    @property
    @abstractmethod
    def supports_deliberate(self) -> bool:
        """Whether this model supports deliberate reasoning steps."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        deliberate_steps: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: List of stop sequences
            deliberate_steps: Optional number of deliberate reasoning steps
            
        Returns:
            Generated text
        """
        pass

