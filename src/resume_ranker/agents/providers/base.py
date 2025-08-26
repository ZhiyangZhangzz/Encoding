# src/resume_ranker/agents/providers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import json
from ..prompts import RESUME_EXTRACTION_PROMPT


class BaseProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """给定 prompt，返回原始模型输出"""
        pass

    def parse_json(self, output: str) -> Dict[str, Any]:
        """默认 JSON 解析"""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw_output": output}

    def extract_resume_info(self, resume_text: str) -> Dict[str, Any]:
        """
        用统一的 Prompt 格式调用模型，提取简历关键信息
        所有 Provider 子类都可以直接使用
        """
        prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
        raw_output = self.generate(prompt)
        return self.parse_json(raw_output)
