"""计算指定论文内关键词检索的离线指标，不调用外部模型。"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence

from papertrace.keyword_search import KeywordRetriever


def _ids(values: list[str], field: str, allow_empty: bool = False) -> set[str]:
    """校验非空字符串 ID 列表，拒绝重复值以避免虚增指标。"""
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"{field} must be a non-empty list")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate IDs")
    return set(values)


def _cutoffs(ks: Sequence[int]) -> tuple[int, ...]:
    """校验截断位置并排序去重，不接受布尔值或小数。"""
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    return tuple(sorted(set(ks)))


def corpus_fingerprint(chunks: list[dict]) -> str:
    """为有序 chunk 快照生成指纹，文本、元数据或顺序变化都会改变指纹。"""
    payload = json.dumps(
        chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def score_ranking(
    retrieved_ids: list[str], relevant_ids: list[str], ks: Sequence[int] = (1, 3, 5)
) -> dict[str, float]:
    """计算单问题的 Recall@K、Hit@K 和截断倒数排名 RR@K。

    Recall 的分母为全部已标注相关 ID 数；RR 只考虑前 K 条内的首次命中。
    无命中时三个指标均为零。无相关证据的问题不适用本指标，明确报错。
    """
    relevant = _ids(relevant_ids, "relevant_ids")
    _ids(retrieved_ids, "retrieved_ids", allow_empty=True)
    metrics = {}
    for k in _cutoffs(ks):
        hits = [rank for rank, cid in enumerate(retrieved_ids[:k], 1)
                if cid in relevant]
        metrics[f"recall@{k}"] = len(hits) / len(relevant)
        metrics[f"hit@{k}"] = float(bool(hits))
        metrics[f"rr@{k}"] = 1 / hits[0] if hits else 0.0
    return metrics


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    """按问题做宏平均，将单问题 RR 汇总为 MRR。"""
    return {
        key.replace("rr@", "mrr@", 1): sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def evaluate_retrieval(
    chunks: list[dict], cases: list[dict], ks: Sequence[int] = (1, 3, 5)
) -> dict:
    """校验标注，在各论文内检索，返回总体、分论文和逐问题结果。

    每个问题必须包含 query_id、document_id、query、relevant_chunk_ids。
    可选 evidence 用 chunk_id、page_number、quote 核对标注与语料一致性。
    所有输入校验完成后才运行检索；不修改输入或根据检索结果调整标注。
    """
    cutoffs = _cutoffs(ks)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks must be a non-empty list")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    grouped = defaultdict(list)
    by_id = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("each chunk must be an object")
        for field in ("chunk_id", "document_id", "text"):
            if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                raise ValueError(f"chunk {field} must be a non-empty string")
        if type(chunk.get("page_number")) is not int or chunk["page_number"] < 1:
            raise ValueError("chunk page_number must be a positive integer")
        cid = chunk["chunk_id"]
        if cid in by_id:
            raise ValueError(f"duplicate chunk_id: {cid}")
        by_id[cid] = chunk
        grouped[chunk["document_id"]].append(chunk)

    query_ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        for field in ("query_id", "document_id", "query"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"case {field} must be a non-empty string")
        query_ids.append(case["query_id"])
        did = case["document_id"]
        if did not in grouped:
            raise ValueError(f"unknown document_id: {did}")
        relevant = _ids(case.get("relevant_chunk_ids"), "relevant_chunk_ids")
        for cid in relevant:
            if cid not in by_id or by_id[cid]["document_id"] != did:
                raise ValueError(f"relevant chunk not in document {did}: {cid}")
        if "evidence" in case:
            evidence = case["evidence"]
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("evidence must be a non-empty list")
            evidence_ids = []
            for entry in evidence:
                if (not isinstance(entry, dict)
                        or not isinstance(entry.get("chunk_id"), str)
                        or entry["chunk_id"] not in relevant):
                    raise ValueError("evidence chunk_id must be a relevant chunk")
                chunk = by_id[entry["chunk_id"]]
                quote = entry.get("quote")
                if (type(entry.get("page_number")) is not int
                        or entry["page_number"] != chunk["page_number"]):
                    raise ValueError("evidence page_number mismatch")
                if not isinstance(quote, str) or not quote.strip() or quote not in chunk["text"]:
                    raise ValueError("evidence quote not found in chunk text")
                evidence_ids.append(entry["chunk_id"])
            if _ids(evidence_ids, "evidence IDs") != relevant:
                raise ValueError("evidence must cover all relevant_chunk_ids")
    _ids(query_ids, "query_ids")

    retrievers = {did: KeywordRetriever(items) for did, items in grouped.items()}
    details = []
    per_document = defaultdict(list)
    for case in cases:
        results = retrievers[case["document_id"]].search(case["query"], max(cutoffs))
        metrics = score_ranking(
            [item["chunk_id"] for item in results], case["relevant_chunk_ids"], cutoffs
        )
        details.append({
            "query_id": case["query_id"], "document_id": case["document_id"],
            "query": case["query"], "relevant_chunk_ids": list(case["relevant_chunk_ids"]),
            "metrics": metrics, "results": results,
        })
        per_document[case["document_id"]].append(metrics)
    return {
        "retriever": {"name": "keyword_bm25_style", "k1": 1.5, "b": 0.75},
        "scope": "within_document", "ks": list(cutoffs),
        "query_count": len(cases), "chunk_count": len(chunks),
        "metrics": _mean_metrics([item["metrics"] for item in details]),
        "by_document": {
            did: {"query_count": len(rows), "metrics": _mean_metrics(rows)}
            for did, rows in per_document.items()
        },
        "queries": details,
    }
