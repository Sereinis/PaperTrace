"""按字段固定子问题检索并有界融合，不依赖论文名称、数值或人工证据编号。"""

import copy

from papertrace.hybrid_search import reciprocal_rank_fusion


RETRIEVAL_VERSION = "report-field-facets-v3-bounded-expansion"
SELECTION_POLICY = "adaptive_prefix_union_then_rrf_v2"
MAX_FIELD_CANDIDATES = 40
FIELD_QUERIES = {
    "research_question": ("research question model motivation problem limitations computational efficiency parallelization",),
    "datasets": ("training data dataset sentence pairs preprocessing vocabulary tokenization",
                 "evaluation test set development split"),
    "model": ("model encoder decoder architecture layers dimensions heads feed forward",
              "model positional encoding embeddings",
              "multi-head attention heads key value dimensions feed forward inner layer"),
    "training_settings": ("training hardware steps schedule batch tokens",
                          "optimizer learning rate warmup",
                          "regularization dropout label smoothing"),
    "evaluation_metrics": ("evaluation metrics test set BLEU perplexity",
                           "inference beam search length penalty checkpoint averaging decoding"),
}


def field_candidate_budget(top_k: int, query_count: int) -> int:
    """计算显式候选预算上界，不改变底层深度；最多40块。"""
    if (type(top_k) is not int or not 1 <= top_k <= 20
            or type(query_count) is not int or not 1 <= query_count <= 3):
        raise ValueError("候选预算需要合法深度和子问题数量")
    return min(MAX_FIELD_CANDIDATES, max(2 * top_k, query_count * top_k))


class FieldRetriever:
    """将固定子问题转发到底层，平等融合排名且限制最终候选规模。"""

    def __init__(self, retriever, queries: tuple[str, ...], *, adaptive_budget: bool = False) -> None:
        """保存有界子问题，不创建客户端或解析实验范围。"""
        if (not isinstance(queries, tuple) or not 1 <= len(queries) <= 3
                or any(not isinstance(q, str) or not q.strip() or len(q) > 4000 for q in queries)
                or len(set(queries)) != len(queries)):
            raise ValueError("字段检索需要1至3个互不重复的有界子问题")
        if not callable(getattr(retriever, "search", None)):
            raise ValueError("底层检索器必须提供search")
        if type(adaptive_budget) is not bool:
            raise ValueError("adaptive_budget必须是布尔值")
        self.retriever, self.queries, self.last_search = retriever, queries, None
        self.adaptive_budget = adaptive_budget

    def _search(self, top_k: int, union: bool) -> list[dict]:
        """检查各路元数据并按等权RRF融合，不以多轮融合分数代替原排名。"""
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k必须是1至20的整数")
        use_union = union and callable(getattr(self.retriever, "search_candidates", None))
        base_limit = 2 * top_k if union else top_k
        limit = base_limit
        scores, metadata, order, audit, rankings = {}, {}, {}, [], []
        for query in self.queries:
            values = (self.retriever.search_candidates(query, top_k) if use_union else
                      self.retriever.search(query, top_k))
            source_limit = 2 * top_k if use_union else top_k
            if not isinstance(values, list) or len(values) > source_limit:
                raise ValueError("子问题候选超过约定上限")
            # 用既有契约检查重复ID和结构；保留底层原始排名顺序。
            reciprocal_rank_fusion(values, [], top_k=source_limit)
            audit.append({"query": query, "candidate_count": len(values),
                          "candidate_ids": [v["chunk_id"] for v in values]})
            rankings.append([v["chunk_id"] for v in values])
            for rank, value in enumerate(values, 1):
                cid = value["chunk_id"]
                item = {k: v for k, v in value.items() if k not in ("score", "rank")}
                if cid in metadata and metadata[cid] != item:
                    raise ValueError("子问题对同一分块的元数据不一致")
                if cid not in metadata:
                    order[cid] = len(order)
                    metadata[cid] = copy.deepcopy(item)
                scores[cid] = scores.get(cid, 0.0) + 1 / (60 + rank)
        rrf_order = sorted(metadata, key=lambda cid: (-scores[cid], order[cid], cid))
        required_ids = set(cid for ranking in rankings for cid in ranking[:top_k])
        hard_limit = field_candidate_budget(top_k, len(self.queries)) if union and self.adaptive_budget else base_limit
        if union and self.adaptive_budget:
            limit = min(hard_limit, max(base_limit, len(required_ids)))
        # 为每个子问题预留同等深度的前缀，避免多路重合项挤掉单路证据。
        # 去重不消耗名额，剩余容量仍按原等权RRF填充；容量小于子问题数时不预留。
        depth = (top_k if union and self.adaptive_budget and len(required_ids) <= limit
                 else min(top_k, limit // len(rankings)))
        reserved, reserved_set = [], set()
        for rank in range(depth):
            for ranking in rankings:
                if rank < len(ranking) and ranking[rank] not in reserved_set:
                    reserved.append(ranking[rank])
                    reserved_set.add(ranking[rank])
        selected = reserved + [cid for cid in rrf_order if cid not in reserved_set][:limit - len(reserved)]
        # 最终仍按RRF排序；预留只影响成员选择，不伪造语义相关性或原分数。
        selected_set = set(selected)
        ordered = [cid for cid in rrf_order if cid in selected_set]
        for row in audit:
            row["retained_candidate_ids"] = [cid for cid in row["candidate_ids"] if cid in selected_set]
            row["dropped_candidate_ids"] = [cid for cid in row["candidate_ids"] if cid not in selected_set]
        result = [dict(copy.deepcopy(metadata[cid]), score=scores[cid], rank=i)
                  for i, cid in enumerate(ordered, 1)]
        self.last_search = {"version": RETRIEVAL_VERSION, "queries": audit, "rrf_k": 60,
                            "top_k_per_source_per_query": top_k, "source_union": use_union,
                            "candidate_limit": limit, "final_candidate_count": len(result),
                            "selection_policy": (SELECTION_POLICY if union and self.adaptive_budget
                                                 else "round_robin_prefix_then_rrf_v1"),
                            "base_candidate_limit": base_limit, "hard_candidate_limit": hard_limit,
                            "required_prefix_candidate_count": len(required_ids),
                            "budget_expanded": limit > base_limit,
                            "budget_reason": ("prefix_union_exceeds_hard_cap" if len(required_ids) > hard_limit
                                              else "prefix_union_exceeds_base_budget" if limit > base_limit
                                              else "base_budget_sufficient"),
                            "prefix_truncated_by_cap": len(required_ids) > hard_limit,
                            "dropped_prefix_candidate_ids": [cid for cid in rrf_order
                                                             if cid in required_ids and cid not in selected_set],
                            "reserved_prefix_depth": depth, "reserved_candidate_ids": reserved,
                            "unconstrained_rrf_candidate_ids": rrf_order[:limit],
                            "ranking_coverage_basis": "rank_prefix_retention_not_semantic_coverage"}
        return result

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """使用固定子问题而非生成任务，返回融合Top-K。"""
        return self._search(top_k, False)

    def search_candidates(self, query: str, top_k: int = 5) -> list[dict]:
        """融合底层候选并集；可显式启用前缀扩容，始终受硬上限约束。"""
        return self._search(top_k, True)
