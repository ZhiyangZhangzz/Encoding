# src/resume_ranker/agents/resume_agent.py
import json
from typing import Dict, Any, Literal

from .prompts import RESUME_EXTRACTION_PROMPT
from .providers.openai_provider import OpenAIProvider
from .providers.hf_provider import HFProvider
from .providers.deepseek_provider import DeepSeekProvider

class ResumeAgent:
    def __init__(self, provider: Literal["openai", "hf", "deepseek"] = "openai", model_name: str = None):
        if provider == "openai":
            self.provider = OpenAIProvider(model_name or "gpt-4o-mini")
        elif provider == "hf":
            self.provider = HFProvider(model_name or "google/flan-t5-base")
        elif provider == "deepseek":
            self.provider = DeepSeekProvider(model_name or "deepseek-chat")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def extract_resume_info(self, resume_text: str) -> Dict[str, Any]:
        prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
        output = self.provider.generate(prompt)
        return self.provider.parse_json(output)

    def test_agent(self, resume_text: str):
        parsed = self.extract_resume_info(resume_text)
        print("=== LLM 输出 ===")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        return parsed
