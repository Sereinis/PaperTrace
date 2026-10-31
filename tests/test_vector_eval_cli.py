"""使用隔离临时目录检查对比命令路径保护和离线查询缓存。"""

import json

import pytest

from papertrace import embedding_cache, vector_eval_cli
from papertrace.embedding_cache import embed_chunks
from papertrace.retrieval_eval import corpus_fingerprint
from papertrace.embeddings import DEFAULT_MODEL, EmbeddingError


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """生成合法向量快照与开发集，替换真实客户端，禁止读取密钥。"""
    calls = []

    class Client:
        """返回固定模拟向量。"""

        def __init__(self, **kwargs):
            """声明请求模型。"""
            self.model = kwargs.get("model", DEFAULT_MODEL)

        def embed(self, texts):
            """记录模拟请求。"""
            calls.append(list(texts))
            return [[1.0] * 1024 for _ in texts]

    monkeypatch.setattr(embedding_cache, "SiliconFlowEmbedder", Client)
    chunks = [{"chunk_id": "a1", "document_id": "a", "page_number": 1, "text": "apple"}]
    snapshot = embed_chunks(chunks, tmp_path / "corpus")
    calls.clear()
    vectors, data, output = tmp_path / "vectors.json", tmp_path / "dev.json", tmp_path / "out" / "comparison.json"
    vectors.write_text(json.dumps(snapshot), encoding="utf-8")
    dataset = {"schema_version": 1, "dataset_id": "dev", "documents": [
        {"document_id": "a", "chunks_sha256": corpus_fingerprint(chunks)}], "cases": [
        {"query_id": "q1", "document_id": "a", "query": "apple?", "relevant_chunk_ids": ["a1"]}]}
    data.write_text(json.dumps(dataset), encoding="utf-8")
    cache = tmp_path / "queries"
    argv = ["--vectors", str(vectors), "--dataset", str(data), "--query-cache-dir", str(cache),
            "--output", str(output)]
    return argv, vectors, data, output, calls


def test_cli_report_and_replay(inputs):
    """报告记录输入哈希，复跑新输出只读查询缓存。"""
    argv, vectors, data, output, calls = inputs
    originals = [p.read_bytes() for p in (vectors, data)]
    assert vector_eval_cli.main(argv) == 0
    report = json.loads(output.read_bytes())
    assert report["vector"]["metrics"]["hit@1"] == 1
    assert len(report["dataset_sha256"]) == 64
    assert len(report["vector_snapshot_sha256"]) == 64
    assert len(calls) == 1
    argv[-1] = str(output.with_name("second.json"))
    assert vector_eval_cli.main(argv) == 0
    assert len(calls) == 1
    assert [p.read_bytes() for p in (vectors, data)] == originals


@pytest.mark.parametrize("mode", ["exists", "input", "cache-output", "cache-input", "missing", "bad-json",
                                  "fingerprint", "bad-vector", "ks", "parent-file"])
def test_cli_failure_before_api(inputs, mode):
    """路径或输入错误时不计费、不覆盖已有数据。"""
    argv, vectors, data, output, calls = inputs
    if mode == "exists":
        output.parent.mkdir()
        output.write_text("keep", encoding="utf-8")
    elif mode == "input":
        argv[-1] = str(data)
    elif mode == "cache-output":
        argv[-1] = str(output.parent.parent / "queries" / "bad.json")
    elif mode == "cache-input":
        argv[5] = str(data.parent)
    elif mode == "missing":
        vectors.unlink()
    elif mode == "bad-json":
        data.write_text("{", encoding="utf-8")
    elif mode == "fingerprint":
        d = json.loads(data.read_bytes())
        d["documents"][0]["chunks_sha256"] = "0" * 64
        data.write_text(json.dumps(d), encoding="utf-8")
    elif mode == "bad-vector":
        d = json.loads(vectors.read_bytes())
        d["records"][0]["embedding"] = []
        vectors.write_text(json.dumps(d), encoding="utf-8")
    elif mode == "ks":
        argv += ["--ks", "0"]
    else:
        output.parent.write_text("keep", encoding="utf-8")
    assert vector_eval_cli.main(argv) == 1
    assert not calls
    if mode == "exists":
        assert output.read_text(encoding="utf-8") == "keep"


def test_cli_safe_api_failure(inputs, monkeypatch, capsys):
    """接口失败保留安全消息，输出报告不存在。"""
    argv, _, _, output, _ = inputs

    def fail(*args, **kwargs):
        """模拟安全接口错误。"""
        raise EmbeddingError("safe failure")

    monkeypatch.setattr(vector_eval_cli, "compare_retrieval", fail)
    assert vector_eval_cli.main(argv) == 1
    assert "safe failure" in capsys.readouterr().err
    assert not output.exists()
