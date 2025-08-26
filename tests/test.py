# src/resume_ranker/main.py
import os
import json
import pdfplumber
from typing import List, Tuple
from src.resume_ranker.interface import Ranker
from src.resume_ranker.agents.providers.openai_provider import OpenAIProvider
from src.resume_ranker.agents.prompts import RESUME_EXTRACTION_PROMPT, JD_EXTRACTION_PROMPT


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_pdf_file(path: str) -> Tuple[str, int]:
    text_chunks: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    text = "\n".join(text_chunks).strip()
    return text, len(pdf.pages)


def load_job_description(jd_file: str) -> str:
    if not os.path.isfile(jd_file):
        raise FileNotFoundError(f"未找到职位描述文件: {jd_file}")
    return read_text_file(jd_file)


def load_resumes(resumes_dir: str) -> Tuple[List[str], List[str], List[int]]:
    texts, names, pages = [], [], []
    if not os.path.isdir(resumes_dir):
        raise FileNotFoundError(f"简历目录不存在: {resumes_dir}")
    for fname in sorted(os.listdir(resumes_dir)):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(resumes_dir, fname)
            txt, pg = read_pdf_file(fpath)
            texts.append(txt)
            names.append(fname)
            pages.append(pg)
    if not texts:
        raise FileNotFoundError(f"在 {resumes_dir} 中未找到 PDF 简历")
    return texts, names, pages


def extract_with_openai(provider: OpenAIProvider, text: str, prompt_template: str, kind: str):
    prompt = prompt_template.format(resume_text=text) if kind == "resume" else prompt_template.format(jd_text=text)
    raw = provider.generate(prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_output": raw}

def serialize_resume_json(resume: dict) -> str:
    parts = []
    if resume.get("title"):
        parts.append(f"Title: {resume['title']}")
    if resume.get("skills"):
        parts.append(f"Skills: {resume['skills']}")
    if resume.get("experience"):
        exp_str = "; ".join([f"{e.get('position','')} at {e.get('company','')} ({e.get('years','')})" for e in resume["experience"]])
        parts.append(f"Experience: {exp_str}")
    if resume.get("education"):
        edu = resume["education"]
        parts.append(f"Education: {edu.get('degree','')} at {edu.get('school','')}")
    if resume.get("keywords"):
        parts.append(f"Keywords: {resume['keywords']}")
    return "\n".join(parts)


def serialize_jd_json(jd: dict) -> str:
    parts = []
    if jd.get("title"):
        parts.append(f"Title: {jd['title']}")
    if jd.get("skills"):
        parts.append(f"Skills: {jd['skills']}")
    if jd.get("responsibilities"):
        parts.append("Responsibilities: " + "; ".join(jd["responsibilities"]))
    if jd.get("requirements"):
        parts.append("Requirements: " + "; ".join(jd["requirements"]))
    if jd.get("keywords"):
        parts.append(f"Keywords: {jd['keywords']}")
    return "\n".join(parts)

def main():
    # ===== 配置路径 =====
    jd_file = "./data/job_description/jd1.txt"
    resumes_dir = "./data/resumes"
    cache_dir = os.environ.get("HF_HOME", None)

    print("🔍 Step 1: 加载职位描述和简历...")
    # ===== 加载原始数据 =====
    job_desc = load_job_description(jd_file)
    resumes, resume_names, resume_pages = load_resumes(resumes_dir)
    print(f"✅ 加载完成: 职位描述 1 份，简历 {len(resumes)} 份")

    # ===== 初始化 OpenAIProvider =====
    print("\n🔍 Step 2: 初始化 OpenAIProvider...")
    provider = OpenAIProvider(model_name="gpt-4o-mini")
    print("✅ OpenAIProvider 初始化完成")

    # ===== 提取结构化 JSON =====
    print("\n🔍 Step 3: 提取简历和 JD 的 JSON 信息...")
    resume_jsons = []
    for i, text in enumerate(resumes):
        print(f"  -> 解析简历 {i+1}/{len(resumes)}: {resume_names[i]}")
        rj = extract_with_openai(provider, text, RESUME_EXTRACTION_PROMPT, "resume")
        resume_jsons.append(rj)
    jd_json = extract_with_openai(provider, job_desc, JD_EXTRACTION_PROMPT, "jd")
    print("✅ 提取完成")

    # ===== 序列化 =====
    print("\n🔍 Step 4: 序列化 JSON 为文本...")
    jd_serialized = serialize_jd_json(jd_json)
    resumes_serialized = [serialize_resume_json(rj) for rj in resume_jsons]
    print("✅ 序列化完成")

    # 打印样例
    print("\n📄 JD JSON 示例:")
    print(json.dumps(jd_json, indent=2, ensure_ascii=False))
    print("\n📄 第一个 Resume JSON 示例:")
    print(json.dumps(resume_jsons[0], indent=2, ensure_ascii=False))

    # ===== 排序 =====
    print("\n🔍 Step 5: 排序候选简历...")
    ranker = Ranker(model_cache_dir=cache_dir, device="cpu")
    ranker.load_models()
    results = ranker.rank_with_fields(
        jd_json,
        resume_jsons,   # 所有简历 JSON
        weights={"skills": 0.4, "experience": 0.5, "education": 0.1},
        top_k=min(30, len(resumes)),
    )
    print("✅ 排序完成")

    # ===== 输出结果 =====
    print("\n📊 Step 6: 输出结果")
    payload = []
    for i, r in enumerate(results):
        idx = r.resume_index
        fname = resume_names[idx]
        payload.append({
            "rank": i + 1,
            "file": fname,
            "final_score": r.final_score,
            "field_scores": r.field_scores,
            "meta": r.meta
        })

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n🎉 全流程完成")



if __name__ == "__main__":
    main()
