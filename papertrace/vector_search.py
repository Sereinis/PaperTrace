"""使用 BGE-M3 向量对论文分块执行本地余弦相似度检索。"""

import json
import math
from pathlib import Path
from typing import Any

from papertrace.embeddings import DEFAULT_MODEL, EMBEDDING_DIMENSION


class VectorSearchError(ValueError):
    """表示向量快照、查询文本或检索参数不符合契约。"""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个有限且非零向量的余弦相似度。"""
    first = _validate_vector(left, "left")
    second = _validate_vector(right, "right")
    if len(first) != len(second):
        raise VectorSearchError("向量维度不一致")
    dot = math.fsum(a * b for a, b in zip(first, second))
    left_norm = math.sqrt(math.fsum(value * value for value in first))
    right_norm = math.sqrt(math.fsum(value * value for value in second))
    score = dot / (left_norm * right_norm)
    if not math.isfinite(score):
        raise VectorSearchError("余弦相似度必须是有限数值")
    return score


def load_vector_snapshot(path: str | Path) -> dict:
    """读取并校验由 embedding_cache 生成的向量快照。"""
    try:
        payload = json.loads(Path(path).read_bytes())
    except FileNotFoundError:
        raise VectorSearchError("向量快照文件不存在") from None
    except (OSError, ValueError, UnicodeError, RecursionError):
        raise VectorSearchError("向量快照不是可读取的有效 JSON") from None
    _validate_snapshot(payload)
    return payload


class VectorRetriever:
    """保存论文分块向量，并将查询文本转换后执行 Top-K 检索。"""

    def __init__(self, snapshot: dict | list[dict], embedder: Any) -> None:
        """校验快照和向量化客户端，初始化不可变的本地检索数据。"""
        _validate_embedder(embedder)
        records, model, expected_dimension = _extract_records(snapshot)
        if model is not None and getattr(embedder, "model", model) != model:
            raise VectorSearchError("向量化客户端模型与快照配置不一致")

        self._records: list[tuple[dict, list[float]]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise VectorSearchError(f"records[{index}] 必须是对象")
            chunk = record.get("chunk")
            if not isinstance(chunk, dict):
                raise VectorSearchError(f"records[{index}].chunk 必须是对象")
            if not isinstance(chunk.get("chunk_id"), str) or not chunk["chunk_id"].strip():
                raise VectorSearchError(f"records[{index}].chunk.chunk_id 必须是非空字符串")
            vector = _validate_vector(record.get("embedding"), f"records[{index}].embedding")
            self._records.append((dict(chunk), vector))
        self._embedder = embedder
        self._dimension = len(self._records[0][1]) if self._records else EMBEDDING_DIMENSION
        if expected_dimension is not None and expected_dimension != self._dimension:
            raise VectorSearchError("向量维度与快照配置不一致")
        for index, (_, vector) in enumerate(self._records):
            if len(vector) != self._dimension:
                raise VectorSearchError(f"records[{index}] 的向量维度不一致")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """向量化查询文本并返回带分数、排名和原始分块元数据的结果。"""
        if not isinstance(query, str) or not query.strip():
            raise VectorSearchError("query 必须是非空字符串")
        if type(top_k) is not int or top_k < 1:
            raise VectorSearchError("top_k 必须是正整数")
        if not self._records:
            return []

        generated = self._embedder.embed([query])
        if not isinstance(generated, list) or len(generated) != 1:
            raise VectorSearchError("查询向量化必须返回一个向量")
        query_vector = _validate_vector(generated[0], "query embedding")
        if len(query_vector) != self._dimension:
            raise VectorSearchError("查询向量维度与分块向量不一致")

        scored = [
            (cosine_similarity(query_vector, vector), index)
            for index, (_, vector) in enumerate(self._records)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict] = []
        for rank, (score, index) in enumerate(scored[:top_k], start=1):
            chunk, _ = self._records[index]
            result = dict(chunk)
            result["score"] = score
            result["rank"] = rank
            results.append(result)
        return results


def _validate_embedder(embedder: Any) -> None:
    """确认向量化客户端提供可调用的 embed 方法。"""
    if embedder is None or not callable(getattr(embedder, "embed", None)):
        raise VectorSearchError("embedder 必须提供可调用的 embed 方法")


def _extract_records(
    snapshot: dict | list[dict],
) -> tuple[list[dict], str | None, int | None]:
    """从完整快照提取记录、模型名称和声明的向量维度。"""
    if isinstance(snapshot, list):
        return snapshot, None, None
    if not isinstance(snapshot, dict):
        raise VectorSearchError("向量快照必须是对象或记录列表")
    records = snapshot.get("records")
    if not isinstance(records, list):
        raise VectorSearchError("向量快照必须包含 records 列表")
    config = snapshot.get("embedding_config")
    expected_dimension = None
    if config is not None:
        if not isinstance(config, dict) or not isinstance(config.get("model"), str):
            raise VectorSearchError("向量快照的 embedding_config 无效")
        model = config["model"]
        expected_dimension = config.get("dimension")
        if type(expected_dimension) is not int or expected_dimension < 1:
            raise VectorSearchError("向量快照的 dimension 必须是正整数")
    else:
        model = None
    return records, model, expected_dimension


def _validate_snapshot(snapshot: object) -> None:
    """校验从磁盘读取的快照外形，具体向量由检索器再次校验。"""
    if not isinstance(snapshot, dict):
        raise VectorSearchError("向量快照必须是对象")
    _extract_records(snapshot)


def _validate_vector(value: object, name: str) -> list[float]:
    """校验向量维度、元素类型、有限性和非零约束。"""
    if not isinstance(value, list) or not value:
        raise VectorSearchError(f"{name} 必须是非空向量列表")
    result: list[float] = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise VectorSearchError(f"{name} 必须由有限数值组成")
        converted = float(number)
        if not math.isfinite(converted):
            raise VectorSearchError(f"{name} 必须由有限数值组成")
        result.append(converted)
    if not any(result):
        raise VectorSearchError(f"{name} 不能是零向量")
    return result

