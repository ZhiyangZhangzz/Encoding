from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Any, Tuple
import numpy as np

from .providers.sbert_bi import SBERTBiEncoder
from .providers.sbert_cross import SbertCrossEncoder
from .utils.similarity import cosine_sim_matrix


@dataclass
class RankResult:
    # 兼容 main.py 的字段
    resume_index: int
    score: float  # 主输出分（与你 main.py 使用的 r.score 对应）

    # 额外可用的细节字段（便于调试/分析，不影响主流程）
    final_score: Optional[float] = None
    bi_score: Optional[float] = None
    cross_score: Optional[float] = None
    meta: Optional[dict] = None
    field_scores: Optional[Dict[str, float]] = None


class Ranker:
    """
    用法：
        ranker = Ranker()
        ranker.load_models()
        results = ranker.rank(job_desc, resumes, top_k=10, cross_rerank_k=5, blend_alpha=0.5)
    """
    def __init__(
        self,
        bi_model_name: str = "sentence-transformers/all-mpnet-base-v2",  # 换成你想要的模型
        cross_model_name: Optional[str] = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.bi = SBERTBiEncoder(
            model_name=bi_model_name,
            model_cache_dir=model_cache_dir,
            device=device,
        )
        self.cross = (
            SbertCrossEncoder(
                model_name=cross_model_name,
                model_cache_dir=model_cache_dir,
                device=device,
            )
            if cross_model_name
            else None
        )

    def load_models(self) -> None:
        self.bi.load()
        if self.cross:
            self.cross.load()
    
    def encode(self, job_desc: str, resumes: Iterable[str]) -> dict[str, np.ndarray]:
        """
        返回：{"jd": [1, d], "res": [N, d]}
        """
        jd_vec = np.array(self.bi.encode([job_desc], normalize=True), dtype=np.float32)  # [1, d]
        res_vecs = np.array(self.bi.encode(list(resumes), normalize=True), dtype=np.float32)  # [N, d]
        return {"jd": jd_vec, "res": res_vecs}

    def rank(
        self,
        job_desc: str,
        resumes: List[str],
        top_k: int = 10,
        cross_rerank_k: int = 0,
        return_scores: bool = True,      # 兼容参数（不影响行为）
        meta: Optional[List[Dict[str, Any]]] = None,
        blend_alpha: Optional[float] = None,  # None: 用 cross；否则 final=(1-a)*bi + a*cross
    ) -> List[RankResult]:
        """
        1) bi-encoder 粗排（cosine 相似度）
        2) 可选 cross-encoder 精排/融合
        3) 按最终分排序并返回 Top-K
        """
        if not resumes:
            return []

        # 1) bi 编码与分数
        enc = self.encode(job_desc, resumes)
        sims = cosine_sim_matrix(enc["jd"], enc["res"])[0]  # [N]  cosine similarity
        N = len(resumes)

        # 候选集合：覆盖 top_k / cross_rerank_k
        k = min(top_k if cross_rerank_k == 0 else max(top_k, cross_rerank_k), N)
        top_idx_np = np.argsort(-sims)[:k]
        top_idx: List[int] = [int(i) for i in top_idx_np]               # 统一为 python int
        bi_scores: List[float] = [float(sims[i]) for i in top_idx]      # 统一为 python float

        # 2) cross 重排/融合（仅前 rerank_k）
        # 保存为 (idx, bi, cross?, final)
        results_pairs: List[Tuple[int, float, Optional[float], float]] = []

        if self.cross and cross_rerank_k and cross_rerank_k > 0:
            rerank_k = min(cross_rerank_k, len(top_idx))
            pair_inputs = [(job_desc, resumes[i]) for i in top_idx[:rerank_k]]
            cross_scores_raw = self.cross.predict(pair_inputs)  # List[float]，可能为负
            cross_scores: List[float] = [float(x) for x in cross_scores_raw]

            # 前 rerank_k：用 cross 或融合
            for i in range(rerank_k):
                idx_i = top_idx[i]
                bi_i = bi_scores[i]
                cross_i = cross_scores[i]
                if blend_alpha is None:
                    final_i = cross_i
                else:
                    final_i = (1.0 - float(blend_alpha)) * bi_i + float(blend_alpha) * cross_i
                results_pairs.append((idx_i, bi_i, cross_i, float(final_i)))

            # 若需补齐到 top_k，剩余用 bi 分数作为最终分
            for i in range(rerank_k, min(top_k, len(top_idx))):
                idx_i = top_idx[i]
                bi_i = bi_scores[i]
                results_pairs.append((idx_i, bi_i, None, float(bi_i)))
        else:
            # 仅 bi：final=bi
            for i in range(min(top_k, len(top_idx))):
                idx_i = top_idx[i]
                bi_i = bi_scores[i]
                results_pairs.append((idx_i, bi_i, None, float(bi_i)))

        # 3) 按最终分排序并截断到 top_k
        results_pairs.sort(key=lambda x: -x[3])
        results_pairs = results_pairs[:min(top_k, len(results_pairs))]

        # 4) 打包结果（保证原生类型；兼容 main.py 的 r.score）
        results: List[RankResult] = []
        for idx_i, bi_i, cross_i, final_i in results_pairs:
            idx_py = int(idx_i)
            bi_py = float(bi_i)
            cross_py = float(cross_i) if cross_i is not None else None
            final_py = float(final_i)

            results.append(
                RankResult(
                    resume_index=idx_py,
                    score=final_py,            # ✅ 兼容 main.py 使用的 r.score
                    final_score=final_py,      # 额外字段（可选查看）
                    bi_score=bi_py,            # 额外字段
                    cross_score=cross_py,      # 额外字段
                    meta=(meta[idx_py] if meta and idx_py < len(meta) else None),
                )
            )
        return results
    

    def rank_with_fields(
        self,
        jd_json: dict,
        resumes_json: List[dict],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
    ) -> List[RankResult]:
        """
        基于结构化 JSON 做分字段打分 + 排序
        """
        if not weights:
            weights = {"skills": 0.4, "experience": 0.5, "education": 0.1}

        results: List[RankResult] = []

        for idx, res_json in enumerate(resumes_json):
            field_scores: Dict[str, float] = {}

            # ===== Skills 相似度 =====
            if jd_json.get("skills") and res_json.get("skills"):
                jd_vec = self.bi.encode([jd_json["skills"]], normalize=True)   # shape [1, d]
                res_vec = self.bi.encode([res_json["skills"]], normalize=True) # shape [1, d]
                sim = cosine_sim_matrix(jd_vec, res_vec)[0][0]
                field_scores["skills"] = float(sim)

            # ===== Requirements vs Experience 相似度 =====
            if jd_json.get("requirements") and res_json.get("experience"):
                jd_req = "; ".join(jd_json["requirements"])
                res_exp = "; ".join([
                    f"{e.get('position','')} at {e.get('company','')} ({e.get('years','')})"
                    for e in res_json["experience"]
                ])
                jd_vec = self.bi.encode([jd_req], normalize=True)
                res_vec = self.bi.encode([res_exp], normalize=True)
                sim = cosine_sim_matrix(jd_vec, res_vec)[0][0]
                field_scores["experience"] = float(sim)

            # ===== Education 相似度 =====
            if jd_json.get("education") and res_json.get("education"):
                jd_edu = f"{jd_json['education'].get('degree','')} at {jd_json['education'].get('school','')}"
                res_edu = f"{res_json['education'].get('degree','')} at {res_json['education'].get('school','')}"
                jd_vec = self.bi.encode([jd_edu], normalize=True)
                res_vec = self.bi.encode([res_edu], normalize=True)
                sim = cosine_sim_matrix(jd_vec, res_vec)[0][0]
                field_scores["education"] = float(sim)

            # ===== 加权求和 =====
            total = sum(weights.get(k, 0) * v for k, v in field_scores.items())

            results.append(
                RankResult(
                    resume_index=idx,
                    score=total,
                    final_score=total,
                    field_scores=field_scores,
                    meta=res_json,
                )
            )

        # 排序
        results.sort(key=lambda r: -r.score)
        return results[:top_k]
