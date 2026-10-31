"""融合关键词与稠密向量排名的本地检索器。"""

from copy import deepcopy


class HybridSearchError(ValueError):
    """表示融合参数或底层检索结果不符合约定。"""


def reciprocal_rank_fusion(
    keyword_results: list[dict], vector_results: list[dict], *,
    top_k: int, rrf_k: int = 60,
) -> list[dict]:
    """按 Reciprocal Rank Fusion 合并两路结果，保留首次出现顺序作为平局规则。"""
    if type(top_k) is not int or top_k < 1:
        raise HybridSearchError("top_k 必须是正整数")
    if type(rrf_k) is not int or rrf_k < 1:
        raise HybridSearchError("rrf_k 必须是正整数")
    if not isinstance(keyword_results, list) or not isinstance(vector_results, list):
        raise HybridSearchError("两路检索结果必须是列表")

    scores: dict[str, float] = {}
    records: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    for source_index, results in enumerate((keyword_results, vector_results)):
        seen_in_source: set[str] = set()
        for rank, result in enumerate(results, 1):
            if not isinstance(result, dict) or not isinstance(result.get("chunk_id"), str):
                raise HybridSearchError("底层结果必须包含字符串 chunk_id")
            chunk_id = result["chunk_id"]
            if chunk_id in seen_in_source:
                raise HybridSearchError("同一路结果不能重复包含同一 chunk")
            seen_in_source.add(chunk_id)
            if chunk_id in records and records[chunk_id] != {
                key: value for key, value in result.items() if key not in ("score", "rank")
            }:
                raise HybridSearchError("两路结果对同一 chunk 的元数据不一致")
            records.setdefault(
                chunk_id,
                {key: deepcopy(value) for key, value in result.items() if key not in ("score", "rank")},
            )
            first_seen.setdefault(chunk_id, source_index * 1_000_000 + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid], cid))
    fused = []
    for rank, chunk_id in enumerate(ordered[:top_k], 1):
        result = deepcopy(records[chunk_id])
        result["score"] = scores[chunk_id]
        result["rank"] = rank
        fused.append(result)
    return fused


class HybridRetriever:
    """并行调用关键词和向量检索器，再用 RRF 产生统一结果。"""

    def __init__(self, keyword_retriever, vector_retriever, *, rrf_k: int = 60) -> None:
        """保存两路检索器及 RRF 平滑常数，不修改其内部快照。"""
        if not callable(getattr(keyword_retriever, "search", None)):
            raise HybridSearchError("关键词检索器必须提供 search 方法")
        if not callable(getattr(vector_retriever, "search", None)):
            raise HybridSearchError("向量检索器必须提供 search 方法")
        if type(rrf_k) is not int or rrf_k < 1:
            raise HybridSearchError("rrf_k 必须是正整数")
        self._keyword = keyword_retriever
        self._vector = vector_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """获取两路 Top-K 结果并返回 RRF 融合后的 Top-K。"""
        if type(top_k) is not int or top_k < 1:
            raise HybridSearchError("top_k 必须是正整数")
        keyword = self._keyword.search(query, top_k)
        vector = self._vector.search(query, top_k)
        return reciprocal_rank_fusion(keyword, vector, top_k=top_k, rrf_k=self.rrf_k)
