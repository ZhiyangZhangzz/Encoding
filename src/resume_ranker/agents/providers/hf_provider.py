# src/resume_ranker/agents/providers/hf_provider.py
from transformers import pipeline
from .base import BaseProvider

class HFProvider(BaseProvider):
    def __init__(self, model_name: str = "google/flan-t5-base"):
        super().__init__(model_name)
        self.client = pipeline("text-generation", model=model_name)

    def generate(self, prompt: str) -> str:
        result = self.client(prompt, max_new_tokens=500, do_sample=False)
        return result[0]["generated_text"]
