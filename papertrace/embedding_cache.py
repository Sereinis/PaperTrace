"""为论文分块生成可追溯的向量快照，按文本与模型配置复用本地缓存。"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Callable

from papertrace.embeddings import (
    DEFAULT_MODEL, EMBEDDING_DIMENSION, EMBEDDINGS_URL, SiliconFlowEmbedder,
)


_MAX_CACHE_BYTES = 1024 * 1024


class EmbeddingCacheError(ValueError):
    """表示缓存内容损坏或与期望契约不一致，不自动重新调用计费接口。"""


def embedding_config(model: str = DEFAULT_MODEL) -> dict:
    """返回影响向量语义的固定配置；批次和超时不影响缓存身份。"""
    if model not in (DEFAULT_MODEL, "Pro/BAAI/bge-m3"):
        raise ValueError("不支持的 BGE-M3 模型配置")
    return {
        "provider": "siliconflow", "endpoint": EMBEDDINGS_URL,
        "model": model, "dimension": EMBEDDING_DIMENSION,
        "encoding_format": "float", "input_transform": "identity-v1",
    }


def _json_bytes(value: object) -> bytes:
    """生成稳定 UTF-8 JSON，拒绝非有限数值和不可序列化元数据。"""
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("分块或缓存必须是有效的 UTF-8 JSON 数据") from None


def _digest(value: object) -> str:
    """对规范化 JSON 计算 SHA-256，用于身份校验而非安全认证。"""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _text_hash(text: str) -> str:
    """对未改写的原始 chunk 文本计算 UTF-8 摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_chunks(chunks: list[dict]) -> None:
    """联网前检查全部分块身份、页码和文本，保留未知的合法 JSON 字段。"""
    if not isinstance(chunks, list):
        raise ValueError("chunks 必须是列表")
    ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunk[{index}] 必须是对象")
        for name in ("chunk_id", "document_id", "text"):
            if not isinstance(chunk.get(name), str) or not chunk[name].strip():
                raise ValueError(f"chunk[{index}] 的 {name} 必须是非空字符串")
        if chunk["chunk_id"] in ids:
            raise ValueError("chunk_id 不能重复，包括跨输入文件的重复")
        ids.add(chunk["chunk_id"])
        if type(chunk.get("page_number")) is not int or chunk["page_number"] < 1:
            raise ValueError(f"chunk[{index}] 的 page_number 必须是正整数")
    _json_bytes(chunks)


def _validate_vector(value: object) -> list[float]:
    """拒绝数量、类型或数值异常的向量，不把零向量作为有效缓存。"""
    if not isinstance(value, list) or len(value) != EMBEDDING_DIMENSION:
        raise EmbeddingCacheError("向量维度必须为 1024")
    result = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise EmbeddingCacheError("向量必须由有限数值组成")
        try:
            converted = float(number)
        except (ValueError, OverflowError):
            raise EmbeddingCacheError("向量必须由有限数值组成") from None
        if not math.isfinite(converted):
            raise EmbeddingCacheError("向量必须由有限数值组成")
        result.append(converted)
    if not any(result):
        raise EmbeddingCacheError("不能缓存零向量")
    return result


def atomic_write_json(value: object, path: str | Path) -> None:
    """在同目录写临时文件并原子替换，避免中断留下半个 JSON 文件。"""
    target = Path(path)
    data = _json_bytes(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".papertrace-", suffix=".tmp",
            dir=target.parent, delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cache_identity(text: str, config: dict) -> tuple[str, str]:
    """将文本哈希与完整模型配置组合成固定长度的安全文件名。"""
    text_sha256 = _text_hash(text)
    key = _digest({"schema_version": 1, "config": config, "text_sha256": text_sha256})
    return key, text_sha256


def _read_cached_vector(path: Path, key: str, text_sha256: str, config: dict):
    """不存在时返回 None；存在但损坏时中止，不把损坏视为未命中。"""
    try:
        with path.open("rb") as file:
            raw = file.read(_MAX_CACHE_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(raw) > _MAX_CACHE_BYTES:
        raise EmbeddingCacheError("缓存文件过大，请检查指定缓存目录")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise EmbeddingCacheError("缓存 JSON 损坏，请检查指定缓存目录") from None
    if not isinstance(payload, dict):
        raise EmbeddingCacheError("缓存结构无效")
    checksum = payload.pop("checksum", None)
    if checksum != _digest(payload):
        raise EmbeddingCacheError("缓存内容校验失败")
    if (
        type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
        or payload.get("cache_key") != key or payload.get("config") != config
        or payload.get("text_sha256") != text_sha256
    ):
        raise EmbeddingCacheError("缓存身份或模型配置不匹配")
    return _validate_vector(payload.get("embedding"))


def embed_chunks(
    chunks: list[dict], cache_dir: str | Path, *, model: str = DEFAULT_MODEL,
    batch_size: int = 8,
    embedder_factory: Callable[[], SiliconFlowEmbedder] | None = None,
) -> dict:
    """复用内容缓存并批量生成缺失向量，返回保留全部 chunk 元数据的快照。

    先检查所有输入和已有缓存，再创建客户端；全部命中时无需密钥。
    每批校验成功后独立落盘；异常不返回部分快照，但保留已完成缓存。
    本地缓存不包含密钥、原文或页码，只包含模型身份、摘要及向量。
    """
    validate_chunks(chunks)
    config = embedding_config(model)
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch_size 必须是 1 到 32 的整数")
    # 脱离调用方对象，避免在网络回调期间发生元数据变化。
    snapshot = json.loads(_json_bytes(chunks))
    directory = Path(cache_dir)
    identities = [_cache_identity(chunk["text"], config) for chunk in snapshot]
    unique = {}
    for chunk, (key, text_sha256) in zip(snapshot, identities):
        unique.setdefault(key, (chunk["text"], text_sha256))
    vectors = {}
    missing = []
    for key, (text, text_sha256) in unique.items():
        vector = _read_cached_vector(directory / f"{key}.json", key, text_sha256, config)
        if vector is None:
            missing.append(key)
        else:
            vectors[key] = vector
    hit_count = len(vectors)
    if missing:
        client = (embedder_factory() if embedder_factory else
                  SiliconFlowEmbedder(model=model, batch_size=batch_size))
        if client.model != model:
            raise ValueError("向量化客户端模型与缓存配置不一致")
        for start in range(0, len(missing), batch_size):
            keys = missing[start:start + batch_size]
            generated = client.embed([unique[key][0] for key in keys])
            if not isinstance(generated, list) or len(generated) != len(keys):
                raise EmbeddingCacheError("生成向量数量与当前批次不一致")
            validated = [_validate_vector(vector) for vector in generated]
            for key, vector in zip(keys, validated):
                payload = {
                    "schema_version": 1, "cache_key": key, "config": config,
                    "text_sha256": unique[key][1], "embedding": vector,
                }
                payload["checksum"] = _digest(payload)
                atomic_write_json(payload, directory / f"{key}.json")
                vectors[key] = vector
    return {
        "schema_version": 1, "embedding_config": config,
        "chunks_sha256": _digest(snapshot), "chunk_count": len(snapshot),
        "cache_stats": {
            "unique_text_count": len(unique), "cache_hit_texts": hit_count,
            "embedded_texts": len(missing),
        },
        "records": [
            {"chunk": chunk, "embedding": list(vectors[key])}
            for chunk, (key, _) in zip(snapshot, identities)
        ],
    }
