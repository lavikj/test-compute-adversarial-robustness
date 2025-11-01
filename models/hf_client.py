"""HuggingFace model client."""
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient


class HuggingFaceClient(LLMClient):
    """HuggingFace model client implementation."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize HuggingFace client.
        
        Args:
            model_name: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            device: Device to use (if None, auto-detects)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully.")
    
    @property
    def supports_deliberate(self) -> bool:
        """HF models don't natively support deliberate steps (emulate with max_tokens)."""
        return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        deliberate_steps: Optional[int] = None,
    ) -> str:
        """Generate using HuggingFace model."""
        # For deliberate steps, increase max_tokens proportionally
        effective_max_tokens = max_tokens
        if deliberate_steps:
            effective_max_tokens = max_tokens * max(1, deliberate_steps)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=effective_max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        prompt_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_text = generated_text[prompt_length:]
        
        # Apply stop sequences if provided
        if stop:
            for stop_seq in stop:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
        
        return generated_text.strip()

