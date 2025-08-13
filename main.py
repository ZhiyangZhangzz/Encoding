import os
import json
import pdfplumber
from typing import List, Tuple

from src.resume_ranker.interface import Ranker


def read_text_file(path: str) -> str:
    """读取 UTF-8 文本文件"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_pdf_file(path: str) -> Tuple[str, int]:
    """读取 PDF 文件文本，返回 (text, page_count)"""
    pages = 0
    text_chunks: List[str] = []
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    text = "\n".join(text_chunks).strip()
    return text, pages


def load_job_description(jd_file: str) -> str:
    """读取职位描述 txt 文件"""
    if not os.path.isfile(jd_file):
        raise FileNotFoundError(f"未找到职位描述文件: {jd_file}")
    return read_text_file(jd_file)


def load_resumes(resumes_dir: str) -> Tuple[List[str], List[str], List[int]]:
    """读取目录下所有 PDF 简历，返回 (texts, file_names, page_counts)"""
    texts: List[str] = []
    names: List[str] = []
    pages: List[int] = []
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


def _np_to_native(o):
    """将 numpy 标量/数组安全转换为原生 Python 类型，避免 json 序列化错误"""
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    return o


def main():
    # ===== 配置路径 =====
    jd_file = "./data/job_description/jd1.txt"  # 职位描述文件路径
    resumes_dir = "./data/resumes"              # 简历 PDF 文件夹路径
    cache_dir = os.environ.get("HF_HOME", None) # 模型缓存目录（可选）

    # ===== 加载数据 =====
    job_desc = load_job_description(jd_file)
    resumes, resume_names, resume_pages = load_resumes(resumes_dir)

    # ===== 排序 =====
    # 明确使用 CPU，避免 CPU 版 torch 误用 CUDA 报错
    ranker = Ranker(model_cache_dir=cache_dir, device="cpu")
    ranker.load_models()
    results = ranker.rank(
        job_desc,
        resumes,
        top_k=min(5, len(resumes)),
        cross_rerank_k=min(3, len(resumes)),
        blend_alpha=0.5
    )

    # ===== 输出结果（包含文件名；数值强制转原生） =====
    payload = []
    for i, r in enumerate(results):
        idx = int(r.resume_index)
        fname = resume_names[idx] if 0 <= idx < len(resume_names) else f"idx_{idx}"
        payload.append({
            "rank": int(i + 1),
            "file": fname,
            "score": float(r.score),               # 兼容字段（final_score）
            "final_score": float(r.final_score) if r.final_score is not None else None,
            "bi_score": float(r.bi_score) if r.bi_score is not None else None,
            "cross_score": float(r.cross_score) if r.cross_score is not None else None,
            "resume_index": int(r.resume_index),
            "pages": int(resume_pages[idx]) if 0 <= idx < len(resume_pages) else None,
            "meta": r.meta
        })

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_np_to_native))



if __name__ == "__main__":
    main()
