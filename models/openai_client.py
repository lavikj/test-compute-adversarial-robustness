"""OpenAI API client."""
import os
from typing import Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Model identifier (e.g., "gpt-4o-mini", "gpt-4")
            api_key: API key (if None, reads from OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key.")
        self.client = OpenAI(api_key=api_key)
        self._supports_deliberate = model_name.startswith("o1") or "o3" in model_name
    
    @property
    def supports_deliberate(self) -> bool:
        """OpenAI o1/o3 models support deliberate reasoning."""
        return self._supports_deliberate
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        deliberate_steps: Optional[int] = None,
    ) -> str:
        """Generate using OpenAI API."""
        # For o1/o3 models, use reasoning API if deliberate_steps is specified
        if self._supports_deliberate and deliberate_steps:
            # Map steps to effort level
            if deliberate_steps <= 3:
                effort = "low"
            elif deliberate_steps <= 7:
                effort = "medium"
            else:
                effort = "high"
            
            # Note: o1 models use different API structure
            # For now, we'll use standard chat completion with extended max_tokens
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens * deliberate_steps if deliberate_steps else max_tokens,
                temperature=temperature,
                stop=stop,
            )
        else:
            # Standard chat completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        
        return response.choices[0].message.content

