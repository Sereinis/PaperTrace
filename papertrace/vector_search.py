"""使用 BGE-M3 向量对论文分块执行本地余弦相似度检索。"""

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from papertrace.embedding_cache import embedding_config, validate_chunks
from papertrace.embeddings import EMBEDDING_DIMENSION, EmbeddingError


class VectorSearchError(ValueError):
    """表示向量快照、查询文本或检索参数不符合契约。"""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个有限且非零向量的余弦相似度。"""
    first = _validate_vector(left, "left")
    second = _validate_vector(right, "right")
    if len(first) != len(second):
        raise VectorSearchError("向量维度不一致")
    return _unit_similarity(_normalize(first), _normalize(second))


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

    def __init__(self, snapshot: dict, embedder: Any) -> None:
        """仅接收标准 BGE-M3 快照，完整校验后隔离保存分块和单位向量。"""
        _validate_snapshot(snapshot)
        self._model = snapshot["embedding_config"]["model"]
        _validate_embedder(embedder, self._model)
        self._records = [
            (copy.deepcopy(record["chunk"]),
             _normalize(_validate_vector(record["embedding"], "embedding")))
            for record in snapshot["records"]
        ]
        self._embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """向量化查询文本并返回带分数、排名和原始分块元数据的结果。"""
        if not isinstance(query, str) or not query.strip():
            raise VectorSearchError("query 必须是非空字符串")
        try:
            query.encode("utf-8")
        except UnicodeError:
            raise VectorSearchError("query 必须是有效的 UTF-8 文本") from None
        if type(top_k) is not int or top_k < 1:
            raise VectorSearchError("top_k 必须是正整数")
        _validate_embedder(self._embedder, self._model)
        if not self._records:
            return []

        try:
            generated = self._embedder.embed([query])
        except EmbeddingError:
            # 适配层已脱敏的服务错误保持原有状态码和重试信息。
            raise
        except Exception:
            # 外部客户端边界不回显任意异常正文，也不自动重试。
            raise VectorSearchError("查询向量化客户端执行失败") from None
        _validate_embedder(self._embedder, self._model)
        if not isinstance(generated, list) or len(generated) != 1:
            raise VectorSearchError("查询向量化必须返回一个向量")
        query_vector = _validate_vector(generated[0], "query embedding")
        if len(query_vector) != EMBEDDING_DIMENSION:
            raise VectorSearchError("查询向量维度与分块向量不一致")
        query_unit = _normalize(query_vector)

        scored = [
            (_unit_similarity(query_unit, vector), index)
            for index, (_, vector) in enumerate(self._records)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict] = []
        for rank, (score, index) in enumerate(scored[:top_k], start=1):
            chunk, _ = self._records[index]
            result = copy.deepcopy(chunk)
            result["score"] = score
            result["rank"] = rank
            results.append(result)
        return results


def _validate_embedder(embedder: Any, model: str) -> None:
    """要求客户端提供查询方法及与快照一致的显式模型身份。"""
    if embedder is None or not callable(getattr(embedder, "embed", None)):
        raise VectorSearchError("embedder 必须提供可调用的 embed 方法")
    if not isinstance(getattr(embedder, "model", None), str) or embedder.model != model:
        raise VectorSearchError("向量化客户端模型与快照配置不一致")


def _validate_snapshot(snapshot: object) -> None:
    """联网前校验版本、完整配置、记录数、分块指纹与全部分块向量。"""
    if not isinstance(snapshot, dict):
        raise VectorSearchError("向量快照必须是对象")
    if type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
        raise VectorSearchError("向量快照 schema_version 必须为 1")
    config = snapshot.get("embedding_config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), str):
        raise VectorSearchError("向量快照的 embedding_config 无效")
    try:
        expected_config = embedding_config(config["model"])
    except ValueError:
        raise VectorSearchError("向量快照模型不受支持") from None
    if type(config.get("dimension")) is not int or config != expected_config:
        raise VectorSearchError("向量快照模型配置不匹配")
    records = snapshot.get("records")
    if not isinstance(records, list):
        raise VectorSearchError("向量快照必须包含 records 列表")
    if type(snapshot.get("chunk_count")) is not int or snapshot["chunk_count"] != len(records):
        raise VectorSearchError("向量快照 chunk_count 与记录数不一致")
    chunks = []
    for record in records:
        if not isinstance(record, dict):
            raise VectorSearchError("向量快照记录必须是对象")
        chunks.append(record.get("chunk"))
        vector = _validate_vector(record.get("embedding"), "embedding")
        if len(vector) != EMBEDDING_DIMENSION:
            raise VectorSearchError("分块向量维度必须为 1024")
    try:
        validate_chunks(chunks)
        # 与 embedding_cache 的规范化 JSON 哈希保持完全一致，包括记录顺序。
        encoded = json.dumps(
            chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise VectorSearchError("向量快照分块字段重复、缺失或含无效 JSON 元数据") from None
    if snapshot.get("chunks_sha256") != hashlib.sha256(encoded).hexdigest():
        raise VectorSearchError("向量快照语料指纹不匹配")


def _validate_vector(value: object, name: str) -> list[float]:
    """校验向量维度、元素类型、有限性和非零约束。"""
    if not isinstance(value, list) or not value:
        raise VectorSearchError(f"{name} 必须是非空向量列表")
    result: list[float] = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise VectorSearchError(f"{name} 必须由有限数值组成")
        try:
            converted = float(number)
        except (OverflowError, ValueError):
            raise VectorSearchError(f"{name} 必须由有限数值组成") from None
        if not math.isfinite(converted):
            raise VectorSearchError(f"{name} 必须由有限数值组成")
        result.append(converted)
    if not any(result):
        raise VectorSearchError(f"{name} 不能是零向量")
    return result


def _normalize(vector: list[float]) -> list[float]:
    """先按最大绝对值缩放，再归一化，避免极端非零数值的平方溢出。"""
    scale = max(abs(value) for value in vector)
    scaled = [value / scale for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return [value / norm for value in scaled]


def _unit_similarity(left: list[float], right: list[float]) -> float:
    """对已校验的单位向量求点积，并裁剪浮点舍入产生的越界。"""
    score = math.fsum(a * b for a, b in zip(left, right))
    return max(-1.0, min(1.0, score))

