"""在固定开发集上比较关键词与稠密向量检索，复用分块和查询向量。"""

import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

from papertrace.embedding_cache import embed_chunks, validate_chunks
from papertrace.embeddings import SiliconFlowEmbedder
from papertrace.hybrid_search import HybridRetriever
from papertrace.keyword_search import KeywordRetriever
from papertrace.retrieval_eval import corpus_fingerprint, evaluate_retrieval
from papertrace.vector_search import VectorRetriever, _validate_snapshot


def validate_comparison_inputs(chunks: list[dict], dataset: dict, ks=(1, 3, 5)) -> dict:
    """计费前核对数据版本、标注及每篇论文指纹，并返回相同输入的关键词报告。"""
    validate_chunks(chunks)
    if (not isinstance(dataset, dict) or type(dataset.get("schema_version")) is not int
            or dataset["schema_version"] != 1
            or not isinstance(dataset.get("dataset_id"), str)
            or not dataset["dataset_id"].strip()):
        raise ValueError("数据集版本或身份无效")
    # 连同未知元数据一起验证 UTF-8 和有限 JSON 数值，避免请求后才失败。
    json.dumps(dataset, ensure_ascii=False, allow_nan=False).encode("utf-8")
    baseline = evaluate_retrieval(chunks, dataset.get("cases"), ks)
    for case in dataset["cases"]:
        case["query"].encode("utf-8")
    documents = dataset.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("缺少论文指纹清单")
    expected = {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("论文清单条目无效")
        did, digest = document.get("document_id"), document.get("chunks_sha256")
        if (not isinstance(did, str) or not did.strip() or did in expected
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("论文清单身份或指纹无效")
        expected[did] = digest
    actual = {did: corpus_fingerprint([c for c in chunks if c["document_id"] == did])
              for did in dict.fromkeys(c["document_id"] for c in chunks)}
    if expected != actual:
        raise ValueError("论文清单或语料指纹不匹配")
    return baseline


class _CachedQueryEmbedder:
    """仅从本次已经校验的查询向量映射读取，不创建网络客户端。"""

    def __init__(self, model: str, vectors: dict) -> None:
        """保存模型名称和按原始问题文本索引的向量。"""
        self.model = model
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        """返回向量副本，未缓存查询明确失败而不触发额外计费。"""
        return [list(self._vectors[text]) for text in texts]


def compare_retrieval(
    snapshot: dict, dataset: dict, query_cache_dir: str | Path, *,
    ks=(1, 3, 5), embedder_factory: Callable[[], SiliconFlowEmbedder] | None = None,
) -> dict:
    """固定语料与金标，缓存查询后在各论文内比较两种方法，不修改输入。"""
    snapshot, dataset = copy.deepcopy(snapshot), copy.deepcopy(dataset)
    _validate_snapshot(snapshot)
    chunks = [row["chunk"] for row in snapshot["records"]]
    started = time.perf_counter()
    keyword = validate_comparison_inputs(chunks, dataset, ks)
    keyword_seconds = time.perf_counter() - started
    query_chunks = [
        {"chunk_id": case["query_id"], "document_id": "evaluation_queries",
         "page_number": 1, "text": case["query"]}
        for case in dataset["cases"]
    ]
    # 人工页码仅满足缓存输入契约，不是论文证据，绝不进入候选语料。
    started = time.perf_counter()
    queries = embed_chunks(
        query_chunks, query_cache_dir, model=snapshot["embedding_config"]["model"],
        embedder_factory=embedder_factory,
    )
    query_seconds = time.perf_counter() - started
    _validate_snapshot(queries)
    client = _CachedQueryEmbedder(
        snapshot["embedding_config"]["model"],
        {row["chunk"]["text"]: row["embedding"] for row in queries["records"]},
    )
    by_id = {row["chunk"]["chunk_id"]: row for row in snapshot["records"]}

    def factory(items):
        """从已完整校验的快照创建单篇论文子快照，不能先全库检索再过滤。"""
        subset = {
            "schema_version": 1, "embedding_config": snapshot["embedding_config"],
            "chunk_count": len(items), "chunks_sha256": corpus_fingerprint(items),
            "records": [by_id[chunk["chunk_id"]] for chunk in items],
        }
        return VectorRetriever(subset, client)

    started = time.perf_counter()
    vector = evaluate_retrieval(
        chunks, dataset["cases"], ks, retriever_factory=factory,
        retriever_info={"name": "bge_m3_dense_cosine", **snapshot["embedding_config"]},
    )
    vector_seconds = time.perf_counter() - started
    started = time.perf_counter()

    def hybrid_factory(items):
        """为单篇论文同时创建关键词和向量检索器，保持候选范围一致。"""
        subset = {
            "schema_version": 1, "embedding_config": snapshot["embedding_config"],
            "chunk_count": len(items), "chunks_sha256": corpus_fingerprint(items),
            "records": [by_id[chunk["chunk_id"]] for chunk in items],
        }
        return HybridRetriever(KeywordRetriever(items), VectorRetriever(subset, client))

    hybrid = evaluate_retrieval(
        chunks, dataset["cases"], ks, retriever_factory=hybrid_factory,
        retriever_info={"name": "rrf_keyword_bge_m3", "rrf_k": 60},
    )
    hybrid_seconds = time.perf_counter() - started
    deltas = {key: vector["metrics"][key] - value for key, value in keyword["metrics"].items()}
    hybrid_deltas = {key: hybrid["metrics"][key] - value for key, value in keyword["metrics"].items()}
    if not all(math.isfinite(value) for value in (*deltas.values(), *hybrid_deltas.values())):
        raise ValueError("指标必须为有限数值")
    return {
        "schema_version": 1, "dataset_id": dataset["dataset_id"],
        "annotation_status": dataset.get("annotation_status", "unspecified"),
        "split": dataset.get("split", "unspecified"), "scope": "within_document",
        "ks": keyword["ks"], "chunk_count": len(chunks), "query_count": len(query_chunks),
        "corpus_fingerprints": {doc["document_id"]: doc["chunks_sha256"]
                                for doc in dataset["documents"]},
        "query_fingerprint": queries["chunks_sha256"],
        "embedding_config": snapshot["embedding_config"],
        "query_cache_stats": queries["cache_stats"],
        "timing_seconds": {"keyword_validation_and_search": keyword_seconds,
                           "query_embedding_and_cache": query_seconds,
                           "vector_validation_and_search": vector_seconds,
                           "hybrid_search": hybrid_seconds},
        "keyword": keyword, "vector": vector, "hybrid": hybrid,
        "delta_vector_minus_keyword": deltas,
        "delta_hybrid_minus_keyword": hybrid_deltas,
        "limitations": ["Development set, not an independent test set.",
                        "AI-assisted annotations are not independently validated.",
                        "Observed timings are not a controlled benchmark; exact billing unavailable."],
    }
