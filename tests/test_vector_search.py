"""验证查询向量化和本地余弦相似度检索。"""

import copy
import hashlib
import json

import pytest

from papertrace.embedding_cache import embed_chunks, embedding_config
from papertrace.embeddings import DEFAULT_MODEL, EmbeddingError
from papertrace.vector_search import (
    VectorRetriever,
    VectorSearchError,
    cosine_similarity,
    load_vector_snapshot,
)


class FakeEmbedder:
    """用零填充后的固定向量替代真实 API，记录每次查询。"""

    model = DEFAULT_MODEL

    def __init__(self, vector=(1.0, 0.0)):
        """保存查询向量和调用记录。"""
        self.vector = _vector(vector)
        self.calls = []

    def embed(self, texts):
        """为每个输入返回同一个可控向量。"""
        self.calls.append(list(texts))
        return [list(self.vector) for _ in texts]


def _vector(values):
    """将可手算的短向量零填充为 BGE-M3 的 1024 维。"""
    return list(values) + [0.0] * (1024 - len(values))


def _seal(snapshot):
    """按正式缓存格式重新计算分块数量与规范化 JSON 指纹。"""
    chunks = [row["chunk"] for row in snapshot["records"]]
    snapshot["chunk_count"] = len(chunks)
    snapshot["chunks_sha256"] = hashlib.sha256(json.dumps(
        chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return snapshot


def _snapshot():
    """构造带完整配置、论文身份和章节元数据的标准快照。"""
    return _seal({
        "schema_version": 1,
        "embedding_config": embedding_config(),
        "records": [
            {"chunk": {"chunk_id": "first", "document_id": "paper", "page_number": 1,
                       "text": "one", "section_title": "Methods", "extra": {"tags": ["a"]}},
             "embedding": _vector([1.0, 0.0])},
            {"chunk": {"chunk_id": "second", "document_id": "paper", "page_number": 2,
                       "text": "two"}, "embedding": _vector([0.0, 1.0])},
            {"chunk": {"chunk_id": "tie", "document_id": "paper", "page_number": 3,
                       "text": "three"}, "embedding": _vector([1.0, 0.0])},
        ],
    })


def test_search_vectorizes_once_and_returns_stable_top_k():
    """查询只发送一次，并按分数降序、输入顺序处理并列。"""
    client = FakeEmbedder()
    retriever = VectorRetriever(_snapshot(), client)
    results = retriever.search("query", top_k=2)
    assert client.calls == [["query"]]
    assert [item["chunk_id"] for item in results] == ["first", "tie"]
    assert [item["rank"] for item in results] == [1, 2]
    assert results[0]["page_number"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_does_not_mutate_snapshot_or_chunk_metadata():
    """结果附加排名字段时不修改调用方传入的快照。"""
    snapshot = _snapshot()
    client = FakeEmbedder((0.0, 1.0))
    result = VectorRetriever(snapshot, client).search("query")
    assert "score" not in snapshot["records"][0]["chunk"]
    result[0]["text"] = "changed"
    assert snapshot["records"][1]["chunk"]["text"] == "two"


def test_empty_records_do_not_call_embedder():
    """空快照返回空结果，不因没有查询向量而调用接口。"""
    client = FakeEmbedder()
    snapshot = _snapshot()
    snapshot["records"] = []
    assert VectorRetriever(_seal(snapshot), client).search("query") == []
    assert client.calls == []


def test_cosine_similarity_and_invalid_vectors():
    """余弦相似度符合定义，并拒绝零向量和不一致维度。"""
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(2 ** -0.5)
    with pytest.raises(VectorSearchError):
        cosine_similarity([0, 0], [1, 0])
    with pytest.raises(VectorSearchError, match="维度"):
        cosine_similarity([1, 0], [1])


@pytest.mark.parametrize("query", ["", "  ", None, "\ud800"])
def test_invalid_query_is_rejected_before_network(query):
    """空查询不应触发向量化请求。"""
    client = FakeEmbedder()
    with pytest.raises(VectorSearchError):
        VectorRetriever(_snapshot(), client).search(query)
    assert client.calls == []


@pytest.mark.parametrize("top_k", [0, -1, True, "2"])
def test_invalid_top_k_is_rejected(top_k):
    """top_k 必须是正整数。"""
    with pytest.raises(VectorSearchError, match="top_k"):
        VectorRetriever(_snapshot(), FakeEmbedder()).search("query", top_k)


def test_model_mismatch_and_query_dimension_are_rejected():
    """模型配置或查询向量维度不一致时明确失败。"""
    client = FakeEmbedder()
    client.model = "other"
    with pytest.raises(VectorSearchError, match="模型"):
        VectorRetriever(_snapshot(), client)
    client = FakeEmbedder()
    client.vector = [1.0, 0.0]
    with pytest.raises(VectorSearchError, match="维度"):
        VectorRetriever(_snapshot(), client).search("query")


@pytest.mark.parametrize("bad", [None, [], {}, {"records": "bad"}, {"records": []}])
def test_invalid_snapshot_is_rejected(bad):
    """不再接受无配置记录列表或不完整快照。"""
    with pytest.raises(VectorSearchError):
        VectorRetriever(bad, FakeEmbedder())


def test_load_vector_snapshot(tmp_path):
    """磁盘快照可读取，非法 JSON 提供脱敏错误。"""
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    assert load_vector_snapshot(path)["records"][0]["chunk"]["chunk_id"] == "first"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(VectorSearchError):
        load_vector_snapshot(path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("schema_version", None),
    ("chunk_count", True), ("chunk_count", 0), ("chunk_count", None),
    ("chunks_sha256", None), ("chunks_sha256", "0" * 64),
    ("embedding_config", None), ("records", None), ("records", [None]),
])
def test_invalid_snapshot_envelope(field, value):
    """版本、计数、配置和语料指纹异常在查询前拒绝。"""
    snapshot = _snapshot()
    snapshot[field] = value
    client = FakeEmbedder()
    with pytest.raises(VectorSearchError):
        VectorRetriever(snapshot, client)
    assert not client.calls


@pytest.mark.parametrize("field,value", [
    ("provider", "other"), ("endpoint", "https://example.com"),
    ("model", "unknown"), ("model", []), ("dimension", 2),
    ("dimension", 1024.0), ("encoding_format", "base64"),
    ("input_transform", "changed"), ("unexpected", 1),
])
def test_incompatible_embedding_configuration(field, value):
    """不能混用模型、提供方、端点、维度或输入变换不同的快照。"""
    snapshot = _snapshot()
    snapshot["embedding_config"][field] = value
    with pytest.raises(VectorSearchError):
        VectorRetriever(snapshot, FakeEmbedder())


@pytest.mark.parametrize("field", list(embedding_config()))
def test_missing_configuration_field(field):
    """完整配置每个字段都必须存在，不能走默认值绕过检查。"""
    snapshot = _snapshot()
    del snapshot["embedding_config"][field]
    with pytest.raises(VectorSearchError):
        VectorRetriever(snapshot, FakeEmbedder())


@pytest.mark.parametrize("field,value", [
    ("chunk_id", ""), ("chunk_id", "second"), ("document_id", " "),
    ("document_id", None), ("text", ""), ("text", None), ("text", "\ud800"),
    ("page_number", True), ("page_number", 0), ("page_number", "1"),
    ("extra", {"private-marker": float("nan")}), ("extra", {1, 2}),
])
def test_invalid_chunk_metadata(field, value):
    """分块身份、文本、页码和未知元数据必须满足缓存层契约。"""
    snapshot = _snapshot()
    snapshot["records"][0]["chunk"][field] = value
    with pytest.raises(VectorSearchError):
        VectorRetriever(snapshot, FakeEmbedder())


@pytest.mark.parametrize("field", ["chunk_id", "document_id", "text", "page_number"])
def test_missing_chunk_fields(field):
    """必要的来源字段缺失时不返回伪完整证据。"""
    snapshot = _snapshot()
    del snapshot["records"][0]["chunk"][field]
    with pytest.raises(VectorSearchError):
        VectorRetriever(snapshot, FakeEmbedder())


@pytest.mark.parametrize("change", ["text", "page", "order"])
def test_corpus_fingerprint_detects_changed_metadata_and_order(change):
    """语料指纹覆盖完整分块元数据及顺序，而不只是文本。"""
    snapshot = _snapshot()
    if change == "order":
        snapshot["records"].reverse()
    elif change == "text":
        snapshot["records"][0]["chunk"]["text"] = "changed"
    else:
        snapshot["records"][0]["chunk"]["page_number"] = 9
    with pytest.raises(VectorSearchError, match="指纹"):
        VectorRetriever(snapshot, FakeEmbedder())


@pytest.mark.parametrize("bad", [
    None, (), [], [0] * 1024, [True] * 1024, ["1"] * 1024,
    [float("nan")] * 1024, [float("inf")] * 1024, [10 ** 400] * 1024,
    [1, 0],
])
def test_invalid_record_and_query_vectors(bad):
    """语料和客户端返回向量采用相同的有限数值与维度检查。"""
    snapshot = _snapshot()
    snapshot["records"][0]["embedding"] = bad
    with pytest.raises(VectorSearchError):
        VectorRetriever(snapshot, FakeEmbedder())
    client = FakeEmbedder()
    client.embed = lambda _: [bad]
    with pytest.raises(VectorSearchError):
        VectorRetriever(_snapshot(), client).search("query")


@pytest.mark.parametrize("bad", [None, {}, (), [], [[1] * 1024, [1] * 1024]])
def test_wrong_client_response_shape(bad):
    """客户端必须恰好返回一个列表向量，异常返回不能形成部分排名。"""
    client = FakeEmbedder()
    client.embed = lambda _: bad
    with pytest.raises(VectorSearchError):
        VectorRetriever(_snapshot(), client).search("query")


def test_client_model_is_required_and_rechecked():
    """客户端不能缺少模型身份，也不能初始化后切换为其他模型。"""
    client = FakeEmbedder()
    client.model = None
    with pytest.raises(VectorSearchError, match="模型"):
        VectorRetriever(_snapshot(), client)
    client.model = DEFAULT_MODEL
    retriever = VectorRetriever(_snapshot(), client)
    client.model = "Pro/BAAI/bge-m3"
    with pytest.raises(VectorSearchError, match="模型"):
        retriever.search("query")
    assert not client.calls


def test_client_failures_are_safe_and_not_retried():
    """保留适配层安全错误，未知客户端异常不泄露正文且不自动重试。"""
    client = FakeEmbedder()
    calls = []
    safe_error = EmbeddingError("safe error", status_code=429, retryable=True)

    def fail(_):
        """记录一次失败后抛出测试指定的异常。"""
        calls.append(1)
        raise failure

    client.embed = fail
    retriever = VectorRetriever(_snapshot(), client)
    failure = safe_error
    with pytest.raises(EmbeddingError) as exc:
        retriever.search("query")
    assert exc.value is safe_error
    failure = RuntimeError("private-marker")
    with pytest.raises(VectorSearchError) as exc:
        retriever.search("query")
    assert "private-marker" not in str(exc.value)
    assert len(calls) == 2


def test_nested_metadata_and_vectors_are_isolated():
    """初始化和返回结果均隔离嵌套对象，避免调用方污染后续检索。"""
    snapshot = _snapshot()
    original = copy.deepcopy(snapshot)
    retriever = VectorRetriever(snapshot, FakeEmbedder())
    assert snapshot == original
    snapshot["records"][0]["chunk"]["extra"]["tags"].append("changed")
    snapshot["records"][0]["embedding"][0] = -1
    first = retriever.search("query", top_k=20)
    assert len(first) == 3
    assert first[0]["section_title"] == "Methods"
    assert first[0]["extra"]["tags"] == ["a"]
    first[0]["extra"]["tags"].append("changed again")
    assert retriever.search("query")[0]["extra"]["tags"] == ["a"]


@pytest.mark.parametrize("scale", [1e308, 1e-308, 5e-324])
def test_extreme_finite_vectors_have_stable_similarity(scale):
    """极大和极小非零有限值不能因平方溢出或下溢破坏相似度。"""
    assert cosine_similarity([scale, scale], [scale, scale]) == pytest.approx(1)
    assert cosine_similarity([scale, scale], [-scale, -scale]) == pytest.approx(-1)
    assert cosine_similarity([scale, scale], [-scale, scale]) == pytest.approx(0)
    assert -1 <= cosine_similarity([scale, scale], [scale, scale]) <= 1


def test_zero_and_negative_scores_remain_in_top_k():
    """向量相似度与关键词非零命中不同，零分和负分仍参与排名。"""
    snapshot = _snapshot()
    snapshot["records"][2]["embedding"] = _vector([-1, 0])
    results = VectorRetriever(snapshot, FakeEmbedder()).search("query")
    assert [row["score"] for row in results] == pytest.approx([1, 0, -1])


def test_formal_cache_output_is_accepted_without_network(tmp_path):
    """缓存模块生成的标准快照可直接读取并执行离线检索。"""
    chunks = [record["chunk"] for record in _snapshot()["records"]]
    snapshot = embed_chunks(chunks, tmp_path / "cache", embedder_factory=FakeEmbedder)
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    loaded = load_vector_snapshot(path)
    assert loaded == snapshot
    client = FakeEmbedder()
    assert len(VectorRetriever(loaded, client).search("query")) == 3
    assert client.calls == [["query"]]


def test_loader_validates_records_not_only_envelope(tmp_path):
    """磁盘读取时已完整校验，不能等到查询时才发现坏快照。"""
    snapshot = _snapshot()
    snapshot["records"][0]["embedding"] = [0] * 1024
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(VectorSearchError):
        load_vector_snapshot(path)
    with pytest.raises(VectorSearchError, match="不存在"):
        load_vector_snapshot(tmp_path / "missing.json")


def test_pro_model_snapshot_and_client_match():
    """Pro 模型与普通模型配置分离，相同模型客户端仍可正常检索。"""
    snapshot = _snapshot()
    snapshot["embedding_config"] = embedding_config("Pro/BAAI/bge-m3")
    client = FakeEmbedder()
    client.model = "Pro/BAAI/bge-m3"
    assert VectorRetriever(snapshot, client).search("query")[0]["chunk_id"] == "first"


@pytest.mark.parametrize("client", [None, object(), type("Bad", (), {"embed": 1})()])
def test_invalid_client_interface(client):
    """客户端没有可调用的向量化方法时立即拒绝。"""
    with pytest.raises(VectorSearchError, match="embed"):
        VectorRetriever(_snapshot(), client)


def test_client_cannot_change_model_during_request():
    """客户端在请求内切换模型时不能返回错误配置的排名。"""
    client = FakeEmbedder()

    def switch(_):
        """模拟客户端内部模型发生变化。"""
        client.model = "Pro/BAAI/bge-m3"
        return [client.vector]

    client.embed = switch
    with pytest.raises(VectorSearchError, match="模型"):
        VectorRetriever(_snapshot(), client).search("query")


@pytest.mark.parametrize("bad", [None, [], "bad"])
def test_invalid_chunk_object(bad):
    """记录中的 chunk 本身也必须是对象。"""
    snapshot = _snapshot()
    snapshot["records"][0]["chunk"] = bad
    with pytest.raises(VectorSearchError):
        VectorRetriever(snapshot, FakeEmbedder())

