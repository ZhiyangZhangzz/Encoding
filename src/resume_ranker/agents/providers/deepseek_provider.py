# src/resume_ranker/agents/providers/deepseek_provider.py
import os
from openai import OpenAI
from .base import BaseProvider

class DeepSeekProvider(BaseProvider):
    def __init__(self, model_name: str = "deepseek-chat", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 DeepSeek API Key")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
