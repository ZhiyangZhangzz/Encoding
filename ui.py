import os
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from src.resume_ranker.interface import Ranker

def run_ranking():
    jd_text = jd_input.get("1.0", tk.END).strip()
    resumes_texts = resumes_input.get("1.0", tk.END).strip().split("\n---\n")

    if not jd_text or not resumes_texts[0]:
        messagebox.showerror("错误", "请先输入职位描述和简历")
        return

    # 初始化 Ranker
    ranker = Ranker(model_cache_dir=os.environ.get("HF_HOME", None), device="cpu")
    ranker.load_models()

    results = ranker.rank(
        jd_text,
        resumes_texts,
        top_k=min(5, len(resumes_texts)),
        cross_rerank_k=min(3, len(resumes_texts)),
        blend_alpha=0.5
    )

    output_text.delete("1.0", tk.END)
    for i, r in enumerate(results):
        output_text.insert(tk.END, f"排名 {i+1}: 简历 #{r.resume_index}, 分数: {r.score:.4f}\n")
        output_text.insert(tk.END, f"  Bi-Encoder 分数: {r.bi_score}\n")
        output_text.insert(tk.END, f"  Cross-Encoder 分数: {r.cross_score}\n")
        output_text.insert(tk.END, f"  最终分数: {r.final_score}\n\n")

def load_resume_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            resumes_input.insert(tk.END, f.read() + "\n---\n")

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    root.title("Resume Ranker")

    # 职位描述输入
    tk.Label(root, text="职位描述（JD）:").pack()
    jd_input = scrolledtext.ScrolledText(root, width=80, height=5)
    jd_input.pack()

    # 简历输入
    tk.Label(root, text="简历内容（多份简历用 '---' 分隔）:").pack()
    resumes_input = scrolledtext.ScrolledText(root, width=80, height=10)
    resumes_input.pack()

    # 加载简历文件按钮
    tk.Button(root, text="加载简历文件", command=load_resume_file).pack()

    # 运行按钮
    tk.Button(root, text="运行匹配评分", command=run_ranking).pack()

    # 输出框
    tk.Label(root, text="结果:").pack()
    output_text = scrolledtext.ScrolledText(root, width=80, height=10)
    output_text.pack()

    root.mainloop()
