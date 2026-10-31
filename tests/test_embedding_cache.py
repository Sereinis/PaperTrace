"""在临时目录用模拟客户端验证内容缓存，不读取真实配置或访问网络。"""

import copy
import json

import pytest

from papertrace import embedding_cache as cache
from papertrace.embeddings import DEFAULT_MODEL, EmbeddingError


@pytest.fixture
def chunks():
    """提供含章节、页码与偏移的最小论文分块。"""
    return [
        {"chunk_id": "paper_p001_c001", "document_id": "paper", "page_number": 1,
         "text": "first text", "section_key": "abstract", "start_char": 0, "end_char": 10},
        {"chunk_id": "paper_p002_c001", "document_id": "paper", "page_number": 2,
         "text": "second text", "section_key": "results", "start_char": 4, "end_char": 15},
    ]


class FakeEmbedder:
    """用可识别的确定性数值替代真实语义向量。"""

    def __init__(self, model=DEFAULT_MODEL):
        """保存模型与请求记录。"""
        self.model = model
        self.calls = []

    def embed(self, texts):
        """按文本生成不同的非零向量并记录批次。"""
        self.calls.append(list(texts))
        return [[float(len(text))] * 1024 for text in texts]


def test_round_trip_keeps_metadata_and_hits_without_client(tmp_path, chunks):
    """第二次完全命中时无需创建客户端，返回当前元数据和原顺序。"""
    original = copy.deepcopy(chunks)
    client = FakeEmbedder()
    first = cache.embed_chunks(chunks, tmp_path, batch_size=1, embedder_factory=lambda: client)
    assert chunks == original
    assert [record["chunk"] for record in first["records"]] == chunks
    assert client.calls == [["first text"], ["second text"]]
    assert first["cache_stats"] == {
        "unique_text_count": 2, "cache_hit_texts": 0, "embedded_texts": 2,
    }

    def unexpected():
        """全命中不应读取密钥或创建网络客户端。"""
        pytest.fail("unexpected client creation")

    second = cache.embed_chunks(chunks, tmp_path, embedder_factory=unexpected)
    assert first["records"] == second["records"]
    assert second["chunks_sha256"] == first["chunks_sha256"]
    assert second["cache_stats"]["cache_hit_texts"] == 2
    assert second["cache_stats"]["embedded_texts"] == 0
    for file in tmp_path.glob("*.json"):
        payload = json.loads(file.read_bytes())
        assert "text" not in payload and "api_key" not in payload
        assert len(file.stem) == 64
        assert ".." not in file.name


def test_duplicate_text_embedded_once_and_results_do_not_alias(tmp_path, chunks):
    """重复文本只调用一次，分块元数据和返回向量仍分别独立。"""
    chunks[1]["text"] = chunks[0]["text"]
    client = FakeEmbedder()
    result = cache.embed_chunks(chunks, tmp_path, embedder_factory=lambda: client)
    assert client.calls == [["first text"]]
    assert result["cache_stats"]["unique_text_count"] == 1
    assert len(result["records"]) == 2
    result["records"][0]["embedding"][0] = -1
    assert result["records"][1]["embedding"][0] > 0


def test_text_change_only_embeds_missing_but_metadata_is_fresh(tmp_path, chunks):
    """元数据改变可复用向量，文本改变必须重新请求并改变语料指纹。"""
    client = FakeEmbedder()
    first = cache.embed_chunks(chunks, tmp_path, embedder_factory=lambda: client)
    chunks[0]["page_number"] = 8
    chunks[0]["section_key"] = "methods"
    chunks[1]["text"] = "changed text"
    client.calls.clear()
    second = cache.embed_chunks(list(reversed(chunks)), tmp_path, embedder_factory=lambda: client)
    assert client.calls == [["changed text"]]
    assert second["cache_stats"]["cache_hit_texts"] == 1
    assert second["chunks_sha256"] != first["chunks_sha256"]
    assert [row["chunk"] for row in second["records"]] == list(reversed(chunks))


def test_models_use_separate_cache_identities(tmp_path, chunks):
    """普通模型与Pro模型即使文本相同也不能混用缓存。"""
    cache.embed_chunks(chunks, tmp_path, embedder_factory=FakeEmbedder)
    client = FakeEmbedder("Pro/BAAI/bge-m3")
    result = cache.embed_chunks(chunks, tmp_path, model=client.model, embedder_factory=lambda: client)
    assert result["cache_stats"]["cache_hit_texts"] == 0
    assert len(list(tmp_path.glob("*.json"))) == 4


@pytest.mark.parametrize("mode", ["json", "checksum", "model", "key", "schema", "vector"])
def test_corruption_fails_before_any_new_request(tmp_path, chunks, mode):
    """损坏、伪装身份或非法向量应明确失败，不静默重新计费。"""
    cache.embed_chunks(chunks, tmp_path, embedder_factory=FakeEmbedder)
    file = next(tmp_path.glob("*.json"))
    payload = json.loads(file.read_bytes())
    if mode == "json":
        file.write_text("{", encoding="utf-8")
    else:
        if mode == "checksum":
            payload["embedding"][0] = 999
        else:
            if mode == "model":
                payload["config"]["model"] = "other"
            elif mode == "key":
                payload["cache_key"] = "wrong"
            elif mode == "schema":
                payload["schema_version"] = True
            else:
                payload["embedding"] = [0] * 1024
            payload.pop("checksum")
            payload["checksum"] = cache._digest(payload)
        file.write_text(json.dumps(payload), encoding="utf-8")
    chunks.append({"chunk_id": "new", "document_id": "paper", "page_number": 3, "text": "new"})

    def unexpected():
        """发现损坏前不能发起缺失向量请求。"""
        pytest.fail("unexpected client creation")

    with pytest.raises(cache.EmbeddingCacheError):
        cache.embed_chunks(chunks, tmp_path, embedder_factory=unexpected)


@pytest.mark.parametrize("mode", ["shape", "element", "id", "duplicate", "text", "page", "nan", "unicode"])
def test_invalid_chunks_fail_without_client(tmp_path, chunks, mode):
    """无效身份、文本、页码或JSON元数据在联网前拒绝。"""
    if mode == "shape":
        chunks = {}
    elif mode == "element":
        chunks.append(None)
    elif mode == "id":
        chunks[1]["document_id"] = ""
    elif mode == "duplicate":
        chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
    elif mode == "text":
        chunks[1]["text"] = " "
    elif mode == "page":
        chunks[1]["page_number"] = True
    elif mode == "nan":
        chunks[1]["extra"] = float("nan")
    else:
        chunks[1]["text"] = "\ud800"

    def unexpected():
        """非法输入不应创建客户端。"""
        pytest.fail("unexpected client creation")

    with pytest.raises(ValueError):
        cache.embed_chunks(chunks, tmp_path, embedder_factory=unexpected)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("batch_size", [0, 33, True, "8"])
def test_invalid_batch_size(tmp_path, chunks, batch_size):
    """批次参数必须是客户端保护范围内的整数。"""
    with pytest.raises(ValueError, match="batch_size"):
        cache.embed_chunks(chunks, tmp_path, batch_size=batch_size)


def test_empty_chunks_do_not_create_client_or_cache(tmp_path):
    """空列表生成合法空快照，不调用API，也不创建缓存文件。"""
    result = cache.embed_chunks([], tmp_path, embedder_factory=lambda: pytest.fail("network"))
    assert result["records"] == []
    assert result["cache_stats"]["embedded_texts"] == 0
    assert not list(tmp_path.iterdir())


def test_failed_later_batch_can_resume(tmp_path, chunks):
    """后续批次失败保留先前缓存，重跑只请求剩余文本。"""
    client = FakeEmbedder()
    original = client.embed

    def fail_later(texts):
        """第一批成功，第二批模拟安全接口错误。"""
        if client.calls:
            raise EmbeddingError("simulated failure")
        return original(texts)

    client.embed = fail_later
    with pytest.raises(EmbeddingError):
        cache.embed_chunks(chunks, tmp_path, batch_size=1, embedder_factory=lambda: client)
    assert len(list(tmp_path.glob("*.json"))) == 1
    retry = FakeEmbedder()
    result = cache.embed_chunks(chunks, tmp_path, embedder_factory=lambda: retry)
    assert retry.calls == [["second text"]]
    assert result["cache_stats"]["cache_hit_texts"] == 1


@pytest.mark.parametrize("bad", [[], [[1]], [[0] * 1024], [[True] * 1024], [[float("nan")] * 1024]])
def test_bad_generated_vectors_are_never_cached(tmp_path, chunks, bad):
    """防御替换客户端的异常返回，整批校验通过前不落盘。"""
    client = FakeEmbedder()
    client.embed = lambda _: bad
    with pytest.raises(cache.EmbeddingCacheError):
        cache.embed_chunks(chunks[:1], tmp_path, embedder_factory=lambda: client)
    assert not list(tmp_path.iterdir())


def test_client_model_mismatch_fails_before_embed(tmp_path, chunks):
    """客户端模型与缓存配置必须一致。"""
    with pytest.raises(ValueError, match="模型"):
        cache.embed_chunks(chunks, tmp_path, embedder_factory=lambda: FakeEmbedder("other"))


def test_atomic_write_failure_preserves_previous_file(tmp_path, monkeypatch):
    """替换失败时保留旧文件并清理临时文件，不留下半成品。"""
    path = tmp_path / "cache.json"
    cache.atomic_write_json({"old": 1}, path)
    original = path.read_bytes()

    def fail(*args):
        """模拟原子替换失败。"""
        raise OSError("cannot replace")

    monkeypatch.setattr(cache.os, "replace", fail)
    with pytest.raises(OSError):
        cache.atomic_write_json({"new": 2}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_cache_size_limit(tmp_path, chunks, monkeypatch):
    """过大缓存不被读取或作为未命中处理。"""
    cache.embed_chunks(chunks, tmp_path, embedder_factory=FakeEmbedder)
    monkeypatch.setattr(cache, "_MAX_CACHE_BYTES", 10)
    with pytest.raises(cache.EmbeddingCacheError, match="过大"):
        cache.embed_chunks(chunks, tmp_path)
