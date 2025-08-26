import os
import json
import pdfplumber
from transformers import pipeline
from src.resume_ranker.agents.prompts import RESUME_EXTRACTION_PROMPT


def test_resume_extraction_hf():
    """
    测试 HuggingFace 模型从 PDF 简历中提取 JSON
    """
    resumes_dir = "./data/resumes"
    if not os.path.isdir(resumes_dir):
        raise FileNotFoundError(f"❌ Resumes directory not found: {resumes_dir}")

    # 找到第一个 PDF 文件
    pdf_files = [f for f in os.listdir(resumes_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"❌ No PDF resumes found in {resumes_dir}")

    first_file = pdf_files[0]
    fpath = os.path.join(resumes_dir, first_file)

    print(f"\n📄 Processing first resume (PDF): {first_file}")

    # ===== 读取 PDF =====
    text_chunks = []
    with pdfplumber.open(fpath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    resume_text = "\n".join(text_chunks).strip()

    # ===== 初始化 HuggingFace pipeline =====
    model_name = "google/flan-t5-base"   # smaller modle
    generator = pipeline("text2text-generation", model=model_name)

    # ===== 构造 Prompt =====
    prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)

    # ===== 调用模型 =====
    result = generator(prompt, max_new_tokens=500, do_sample=False)
    raw_output = result[0]["generated_text"]

    print("\n🤖 Raw model output:")
    print(raw_output)

    # ===== 尝试解析 JSON =====
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        json_str = raw_output[start:end]
        extracted = json.loads(json_str)
    except Exception as e:
        print("⚠ Failed to parse JSON, showing raw output instead:", e)
        extracted = {"raw_output": raw_output}

    # ===== 打印最终 JSON =====
    print("\n✅ Extracted JSON:")
    print(json.dumps(extracted, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_resume_extraction_hf()
