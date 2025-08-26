import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseProvider
from ..prompts import RESUME_EXTRACTION_PROMPT

load_dotenv()  # ✅ 自动加载 .env 文件里的 OPENAI_API_KEY


class OpenAIProvider(BaseProvider):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 OpenAI API Key")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """普通文本生成"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def extract_resume_info(self, resume_text: str) -> dict:
        """调用 LLM 提取简历信息，返回 JSON"""
        prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
        raw_output = self.generate(prompt)

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {"raw_output": raw_output}
