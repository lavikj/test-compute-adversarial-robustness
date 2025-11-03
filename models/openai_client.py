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
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate using OpenAI API."""
        # o1/o3 models use different API parameters
        if self._supports_deliberate:
            # o1/o3 models:
            # - Don't support temperature or stop parameters
            # - Don't support system messages
            # - Use internal reasoning tokens
            # - Support reasoning_effort: "low", "medium", "high"
            
            reasoning_tokens = max(max_tokens * 2, 500)  # Reduced for speed
            
            # Build kwargs
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": reasoning_tokens,
            }
            
            # Add reasoning_effort if specified (o1 models only)
            if reasoning_effort and reasoning_effort in ["low", "medium", "high"]:
                kwargs["reasoning_effort"] = reasoning_effort
                print(f"  [Using reasoning_effort={reasoning_effort}]", flush=True)
            
            try:
                response = self.client.chat.completions.create(**kwargs)
            except TypeError:
                # Fallback: try without max_completion_tokens
                del kwargs["max_completion_tokens"]
                response = self.client.chat.completions.create(**kwargs)
        else:
            # Standard chat completion for non-o1 models
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        
        return response.choices[0].message.content

