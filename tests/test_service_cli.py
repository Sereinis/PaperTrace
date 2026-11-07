"""隔离真实配置验证服务启动、查询缓存和检索组装。"""

import copy
import json

import pytest

from papertrace import service_cli
from papertrace.embedding_cache import atomic_write_json, embed_chunks
from papertrace.generation import GenerationResult


class Embedder:
    """生成固定非零向量，不调用网络。"""
    model = "BAAI/bge-m3"

    def embed(self, texts):
        """为每条文本生成合法的1024维向量。"""
        return [[1.0] + [0.0] * 1023 for _ in texts]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """构造合法本地快照，并禁止创建任何真实模型适配器。"""
    def blocked(*args, **kwargs):
        """阻断所有真实配置读取和请求。"""
        pytest.fail("unexpected model construction")

    monkeypatch.setattr(service_cli, "DeepSeekGenerator", blocked)
    monkeypatch.setattr(service_cli, "SiliconFlowEmbedder", blocked)
    chunks = [{"chunk_id": "a1", "document_id": "a", "page_number": 1,
               "text": "The model uses attention."}]
    chunk_path, vector_path = tmp_path / "chunks.json", tmp_path / "vectors.json"
    atomic_write_json({"chunks": chunks}, chunk_path)
    snapshot = embed_chunks(chunks, tmp_path / "chunks-cache", embedder_factory=Embedder)
    atomic_write_json(snapshot, vector_path)
    return chunk_path, vector_path, tmp_path / "queries", chunks


def test_startup_is_lazy_and_unknown_document_does_not_generate(inputs):
    """启动校验不加载密钥，未知论文在检索前拒绝。"""
    cp, vp, cache, _ = inputs
    service = service_cli.create_service([cp], vectors=vp, query_cache=cache)
    with pytest.raises(ValueError, match="document_not_found"):
        service.answer("attention", "unknown")


def test_cached_hybrid_uses_existing_query_without_client(inputs, monkeypatch):
    """精确命中旧缓存时混合检索不创建向量客户端。"""
    cp, vp, cache, _ = inputs
    query = [{"chunk_id": "old-id", "document_id": "evaluation_queries",
              "page_number": 1, "text": "attention"}]
    embed_chunks(query, cache, embedder_factory=Embedder)
    service = service_cli.create_service([cp], vectors=vp, query_cache=cache, offline_query_cache=True)

    class Generator:
        """模拟结构合法的中文回答。"""

        def generate(self, messages, **kwargs):
            """返回绑定首片段的答案。"""
            return GenerationResult(json.dumps({"status": "answered", "claims": [{"text": "使用注意力。",
                "citations": [{"segment_id": "E1.S1"}]}], "missing_info": []}), "test", "test", "id", None)

    service.generator = Generator()
    answer = service.answer("attention", "a")
    assert answer["claims"][0]["citations"][0]["chunk_id"] == "a1"
    assert service.query_embedder.hits == 1 and service.query_embedder.generated == 0


def test_cache_miss_only_generates_once_and_then_reuses(inputs):
    """新查询仅生成一次，后续原样查询命中缓存。"""
    _, _, cache, _ = inputs
    calls = []

    def factory():
        """记录一次延迟客户端创建。"""
        calls.append(1)
        return Embedder()

    client = service_cli.CachedQueryEmbedder(Embedder.model, cache, factory)
    first = client.embed(["new query"])
    assert client.embed(["new query"]) == first
    assert len(calls) == 1 and client.hits == client.generated == 1


def test_offline_cache_miss_fails_without_generation(inputs):
    """离线缓存未命中不默默联网或进入生成器。"""
    cp, vp, cache, _ = inputs
    service = service_cli.create_service([cp], vectors=vp, query_cache=cache, offline_query_cache=True)
    with pytest.raises(ValueError) as error:
        service.answer("not cached", "a")
    assert error.value.code == "retrieval_failed"


def test_metadata_mismatch_precedes_network(inputs):
    """分块与快照不一致时启动失败，不读模型配置。"""
    cp, vp, cache, chunks = inputs
    changed = copy.deepcopy(chunks)
    changed[0]["page_number"] = 2
    atomic_write_json({"chunks": changed}, cp)
    with pytest.raises(ValueError, match="元数据"):
        service_cli.create_service([cp], vectors=vp, query_cache=cache)


@pytest.mark.parametrize("options", [
    {"generation_timeout": 0}, {"generation_timeout": float("nan")},
    {"embedding_timeout": 61}, {"top_k": True}, {"max_tokens": 9000}, {"model": "other"},
])
def test_invalid_options_do_not_create_client(inputs, options):
    """非法预算或超时在创建客户端前拒绝。"""
    cp, _, _, _ = inputs
    with pytest.raises(ValueError):
        service_cli.create_service([cp], retrieval="keyword", **options)


def test_keyword_empty_retrieval_skips_generation(inputs):
    """关键词无候选直接拒答，不加载DeepSeek配置。"""
    cp, _, _, _ = inputs
    service = service_cli.create_service([cp], retrieval="keyword")
    result = service.answer("xyznotpresent", "a")
    assert result["status"] == "insufficient" and result["generation"] is None


def test_lazy_generator_uses_configured_timeout(monkeypatch):
    """模拟官方适配器校验模型和超时只在生成时注入。"""
    calls = []

    class Fake:
        """只记录参数，不接触真实配置。"""

        def __init__(self, **kwargs):
            """记录服务端模型选项。"""
            calls.append(kwargs)

        def generate(self, messages, **kwargs):
            """返回固定模拟结果。"""
            return "done"

    monkeypatch.setattr(service_cli, "DeepSeekGenerator", Fake)
    generator = service_cli.LazyGenerator("deepseek-flash", 30)
    assert not calls
    assert generator.generate([]) == "done"
    assert calls == [{"model": "deepseek-flash", "timeout": 30}]


def test_main_startup_failure_is_sanitized(tmp_path, capsys):
    """启动失败只返回固定消息，不展示路径及原异常。"""
    assert service_cli.main(["--chunks", str(tmp_path / "private-path.json"), "--retrieval", "keyword"]) == 1
    assert "private-path" not in capsys.readouterr().out
