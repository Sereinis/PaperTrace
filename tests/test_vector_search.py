"""验证查询向量化和本地余弦相似度检索。"""

import json

import pytest

from papertrace.embeddings import DEFAULT_MODEL
from papertrace.vector_search import (
    VectorRetriever,
    VectorSearchError,
    cosine_similarity,
    load_vector_snapshot,
)


class FakeEmbedder:
    """用固定二维向量替代真实 API，记录每次查询。"""

    model = DEFAULT_MODEL

    def __init__(self, vector=(1.0, 0.0)):
        """保存查询向量和调用记录。"""
        self.vector = list(vector)
        self.calls = []

    def embed(self, texts):
        """为每个输入返回同一个可控向量。"""
        self.calls.append(list(texts))
        return [list(self.vector) for _ in texts]


def _snapshot():
    """构造包含章节和页码元数据的最小向量快照。"""
    return {
        "embedding_config": {"model": DEFAULT_MODEL, "dimension": 2},
        "records": [
            {"chunk": {"chunk_id": "first", "page_number": 1, "text": "one"},
             "embedding": [1.0, 0.0]},
            {"chunk": {"chunk_id": "second", "page_number": 2, "text": "two"},
             "embedding": [0.0, 1.0]},
            {"chunk": {"chunk_id": "tie", "page_number": 3, "text": "three"},
             "embedding": [1.0, 0.0]},
        ],
    }


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
    assert VectorRetriever({"records": []}, client).search("query") == []
    assert client.calls == []


def test_cosine_similarity_and_invalid_vectors():
    """余弦相似度符合定义，并拒绝零向量和不一致维度。"""
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(2 ** -0.5)
    with pytest.raises(VectorSearchError):
        cosine_similarity([0, 0], [1, 0])
    with pytest.raises(VectorSearchError, match="维度"):
        cosine_similarity([1, 0], [1])


@pytest.mark.parametrize("query", ["", "  ", None])
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
    with pytest.raises(VectorSearchError, match="维度"):
        VectorRetriever(_snapshot(), FakeEmbedder((1.0, 0.0, 0.0))).search("query")


@pytest.mark.parametrize("bad", [None, {}, {"records": "bad"}, {"records": []}])
def test_invalid_snapshot_is_rejected(bad):
    """缺少记录列表或快照类型错误时拒绝初始化。"""
    if bad == {"records": []}:
        # 空记录列表本身是合法快照，单独保留此边界。
        VectorRetriever(bad, FakeEmbedder())
    else:
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

