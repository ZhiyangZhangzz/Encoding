# src/resume_ranker/agents/test/test_openai_provider.py
import os
import pdfplumber
import json
from src.resume_ranker.agents.providers.openai_provider import OpenAIProvider
from src.resume_ranker.agents.prompts import RESUME_EXTRACTION_PROMPT, JD_EXTRACTION_PROMPT


def read_first_resume_text(resumes_dir: str = "./data/resumes") -> str:
    """读取第一个 PDF 简历，返回文本"""
    if not os.path.isdir(resumes_dir):
        raise FileNotFoundError(f"❌ Resumes directory not found: {resumes_dir}")

    pdf_files = [f for f in os.listdir(resumes_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"❌ No PDF resumes found in {resumes_dir}")

    fpath = os.path.join(resumes_dir, pdf_files[0])
    print(f"📄 Using resume: {fpath}")

    text_chunks = []
    with pdfplumber.open(fpath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)

    return "\n".join(text_chunks).strip()


def read_first_jd_text(jd_dir: str = "./data/job_description") -> str:
    """读取第一个 JD txt 文件"""
    if not os.path.isdir(jd_dir):
        raise FileNotFoundError(f"❌ Job description directory not found: {jd_dir}")

    jd_files = [f for f in os.listdir(jd_dir) if f.lower().endswith(".txt")]
    if not jd_files:
        raise FileNotFoundError(f"❌ No job descriptions found in {jd_dir}")

    fpath = os.path.join(jd_dir, jd_files[0])
    print(f"📄 Using job description: {fpath}")

    with open(fpath, "r", encoding="utf-8") as f:
        return f.read().strip()


def test_openai_provider():
    provider = OpenAIProvider(model_name="gpt-4o-mini")

    # ===== Resume =====
    resume_text = read_first_resume_text()
    resume_prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
    resume_output = provider.generate(resume_prompt)

    try:
        resume_json = json.loads(resume_output)
    except json.JSONDecodeError:
        resume_json = {"raw_output": resume_output}

    print("\n✅ Extracted Resume JSON:")
    print(json.dumps(resume_json, indent=2, ensure_ascii=False))

    # ===== JD =====
    jd_text = read_first_jd_text()
    jd_prompt = JD_EXTRACTION_PROMPT.format(jd_text=jd_text)
    jd_output = provider.generate(jd_prompt)

    try:
        jd_json = json.loads(jd_output)
    except json.JSONDecodeError:
        jd_json = {"raw_output": jd_output}

    print("\n✅ Extracted JD JSON:")
    print(json.dumps(jd_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_openai_provider()
