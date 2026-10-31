"""用隔离模拟客户端验证固定开发集的对比口径及缓存复用。"""

import copy
import pytest

from papertrace.embedding_cache import embed_chunks
from papertrace.embeddings import DEFAULT_MODEL, EmbeddingError
from papertrace.retrieval_eval import corpus_fingerprint, evaluate_retrieval
from papertrace.vector_eval import compare_retrieval


class Client:
    """按文本首字母生成两种正交向量，不读取配置或联网。"""

    model = DEFAULT_MODEL

    def __init__(self):
        """记录各批输入，用来确认没有重复请求。"""
        self.calls = []

    def embed(self, texts):
        """返回可手算的 1024 维测试向量。"""
        self.calls.append(list(texts))
        return [[float(text.startswith("apple")), float(not text.startswith("apple"))]
                + [0.0] * 1022 for text in texts]


@pytest.fixture
def sample(tmp_path):
    """准备两篇含相似文本的论文，便于检查候选集不串论文。"""
    chunks = [
        {"chunk_id": "a1", "document_id": "a", "page_number": 1, "text": "apple"},
        {"chunk_id": "a2", "document_id": "a", "page_number": 2, "text": "pear"},
        {"chunk_id": "b1", "document_id": "b", "page_number": 1, "text": "apple"},
    ]
    snapshot = embed_chunks(chunks, tmp_path / "corpus", embedder_factory=Client)
    dataset = {"schema_version": 1, "dataset_id": "dev", "split": "development",
               "annotation_status": "unverified", "documents": [
                   {"document_id": did, "chunks_sha256": corpus_fingerprint(
                       [c for c in chunks if c["document_id"] == did])} for did in ("a", "b")],
               "cases": [
                   {"query_id": "q1", "document_id": "a", "query": "apple?",
                    "relevant_chunk_ids": ["a1", "a2"]},
                   {"query_id": "q2", "document_id": "b", "query": "apple?",
                    "relevant_chunk_ids": ["b1"]},
               ]}
    return snapshot, dataset


def test_within_document_macro_metrics_cache_and_immutability(sample, tmp_path):
    """同论文检索、按问题平均，重复查询仅嵌入一次且复跑无需客户端。"""
    snapshot, dataset = sample
    before = copy.deepcopy(sample)
    client = Client()
    first = compare_retrieval(snapshot, dataset, tmp_path / "queries", ks=[2, 1, 2],
                              embedder_factory=lambda: client)
    assert sample == before
    assert client.calls == [["apple?"]]
    assert first["ks"] == [1, 2]
    assert first["vector"]["metrics"]["recall@1"] == pytest.approx(0.75)
    assert first["vector"]["metrics"]["recall@2"] == 1
    assert [r["chunk_id"] for r in first["vector"]["queries"][1]["results"]] == ["b1"]
    assert first["query_cache_stats"]["embedded_texts"] == 1
    second = compare_retrieval(snapshot, dataset, tmp_path / "queries", ks=[1, 2],
                               embedder_factory=lambda: pytest.fail("network"))
    assert second["query_cache_stats"]["cache_hit_texts"] == 1
    assert first["vector"] == second["vector"]
    assert first["keyword"] == second["keyword"]
    for key in first["keyword"]["metrics"]:
        assert first["delta_vector_minus_keyword"][key] == pytest.approx(
            first["vector"]["metrics"][key] - first["keyword"]["metrics"][key])
    assert first["annotation_status"] == "unverified"


@pytest.mark.parametrize("mode", ["schema", "id", "manifest", "hash", "duplicate", "unicode",
                                  "gold", "evidence", "nan", "ks", "snapshot"])
def test_invalid_inputs_fail_before_query_client(sample, tmp_path, mode):
    """不兼容语料、标注和快照必须在潜在计费前失败。"""
    snapshot, dataset = sample
    ks = [1, 3, 5]
    if mode == "schema":
        dataset["schema_version"] = True
    elif mode == "id":
        dataset["dataset_id"] = " "
    elif mode == "manifest":
        dataset["documents"] = []
    elif mode == "hash":
        dataset["documents"][0]["chunks_sha256"] = "0" * 64
    elif mode == "duplicate":
        dataset["documents"].append(dataset["documents"][0])
    elif mode == "unicode":
        dataset["cases"][-1]["query"] = "\ud800"
    elif mode == "gold":
        dataset["cases"][0]["relevant_chunk_ids"] = ["b1"]
    elif mode == "evidence":
        dataset["cases"][0]["evidence"] = [{"chunk_id": "a1", "page_number": 1, "quote": "wrong"}]
    elif mode == "nan":
        dataset["extra"] = float("nan")
    elif mode == "ks":
        ks = [True]
    else:
        snapshot["records"][0]["embedding"] = [0] * 1024
    with pytest.raises((ValueError, UnicodeError)):
        compare_retrieval(snapshot, dataset, tmp_path / "queries", ks=ks,
                          embedder_factory=lambda: pytest.fail("network"))
    assert not (tmp_path / "queries").exists()


def test_safe_api_failure_propagates(sample, tmp_path):
    """查询请求失败不生成假对比结果，也不隐式重试。"""
    snapshot, dataset = sample
    client = Client()

    def fail(_):
        """模拟已脱敏的额度或网络错误。"""
        raise EmbeddingError("safe")

    client.embed = fail
    with pytest.raises(EmbeddingError):
        compare_retrieval(snapshot, dataset, tmp_path / "queries", embedder_factory=lambda: client)


def test_custom_retriever_does_not_run_before_all_annotations_valid(sample):
    """通用评测器完成全部标注检查后才调用工厂。"""
    snapshot, dataset = sample
    chunks = [r["chunk"] for r in snapshot["records"]]
    dataset["cases"][-1]["query_id"] = dataset["cases"][0]["query_id"]
    with pytest.raises(ValueError):
        evaluate_retrieval(chunks, dataset["cases"], retriever_factory=lambda _: pytest.fail("factory"),
                           retriever_info={"name": "test"})


@pytest.mark.parametrize("bad", [[], [{"chunk_id": []}], [{"chunk_id": "b1", "rank": 1, "score": 1}],
                                 [{"chunk_id": "a1", "rank": True, "score": 1}]])
def test_custom_result_validation(sample, bad):
    """异常返回和跨论文返回不能虚增指标，合法空返回按零计分。"""
    snapshot, dataset = sample

    class Retriever:
        """返回指定异常结果。"""

        def search(self, query, top_k):
            """模拟检索实现的契约偏差。"""
            return bad

    args = ([r["chunk"] for r in snapshot["records"]], dataset["cases"])
    if bad:
        with pytest.raises(ValueError):
            evaluate_retrieval(*args, retriever_factory=lambda _: Retriever(), retriever_info={"name": "bad"})
    else:
        assert evaluate_retrieval(*args, retriever_factory=lambda _: Retriever(),
                                  retriever_info={"name": "empty"})["metrics"]["hit@1"] == 0


@pytest.mark.parametrize("mode", ["huge", "nan", "inf", "bool", "metadata", "duplicate", "rank", "too-many"])
def test_custom_result_contract(sample, mode):
    """分数溢出、伪造元数据、重复条目及数量排名偏差统一拒绝。"""
    snapshot, dataset = sample
    chunks = [r["chunk"] for r in snapshot["records"]]
    row = {**chunks[0], "rank": 1, "score": 1.0}
    if mode in ("huge", "nan", "inf", "bool"):
        row["score"] = {"huge": 10 ** 1000, "nan": float("nan"), "inf": float("inf"), "bool": True}[mode]
    elif mode == "metadata":
        row["text"] = "not the corpus text"
    elif mode == "rank":
        row["rank"] = 2
    results = [row]
    if mode == "duplicate":
        results.append({**row, "rank": 2})
    elif mode == "too-many":
        results = [{**row, "rank": index} for index in range(1, 7)]

    class Retriever:
        """返回完整但违反指定契约的结果。"""

        def search(self, query, top_k):
            """不联网，仅返回测试条目。"""
            return results

    with pytest.raises(ValueError):
        evaluate_retrieval(chunks, dataset["cases"], retriever_factory=lambda _: Retriever(),
                           retriever_info={"name": "bad"})
