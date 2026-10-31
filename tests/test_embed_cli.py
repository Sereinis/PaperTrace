"""用模拟向量化验证命令行输入、结果追踪和路径保护。"""

import json

import pytest

from papertrace import embedding_cache, embed_cli
from papertrace.embeddings import DEFAULT_MODEL, EmbeddingError


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """隔离真实客户端，准备现有分块格式的输入文件。"""
    calls = []

    class Client:
        """替换默认客户端，绝不读取.env。"""

        def __init__(self, **kwargs):
            """记录请求模型配置。"""
            self.model = kwargs.get("model", DEFAULT_MODEL)

        def embed(self, texts):
            """返回固定非零向量，不访问网络。"""
            calls.append(texts)
            return [[1.0] * 1024 for _ in texts]

    monkeypatch.setattr(embedding_cache, "SiliconFlowEmbedder", Client)
    source = tmp_path / "chunks.json"
    source.write_text(json.dumps({
        "document_id": "paper", "max_chars": 1200, "overlap_chars": 100,
        "chunks": [{"chunk_id": "p1", "document_id": "paper", "page_number": 1,
                    "text": "paper text", "section_title": "Methods"}], "error": None,
    }), encoding="utf-8")
    output, directory = tmp_path / "results" / "vectors.json", tmp_path / "cache"
    argv = ["--chunks", str(source), "--cache-dir", str(directory), "--output", str(output)]
    return argv, source, output, directory, calls


def test_cli_snapshot_and_second_run_reuses_cache(inputs, capsys):
    """输出保留语料配置、原分块与来源哈希，新输出可离线复用。"""
    argv, source, output, _, calls = inputs
    assert embed_cli.main(argv) == 0
    payload = json.loads(output.read_bytes())
    assert payload["records"][0]["chunk"]["section_title"] == "Methods"
    assert payload["source_files"][0]["metadata"]["max_chars"] == 1200
    assert len(payload["source_files"][0]["sha256"]) == 64
    assert payload["embedding_config"]["model"] == DEFAULT_MODEL
    assert payload["cache_stats"]["embedded_texts"] == 1
    argv[-1] = str(output.with_name("second.json"))
    assert embed_cli.main(argv) == 0
    assert len(calls) == 1
    assert "新生成 0" in capsys.readouterr().out
    assert source.exists()


@pytest.mark.parametrize("mode", ["missing", "json", "shape", "error", "duplicate", "nan"])
def test_invalid_input_never_calls_client(inputs, mode, capsys):
    """输入错误时非零退出，不产生快照或计费请求。"""
    argv, source, output, _, calls = inputs
    if mode == "missing":
        source.unlink()
    elif mode == "json":
        source.write_text("{", encoding="utf-8")
    elif mode == "shape":
        source.write_text("[]", encoding="utf-8")
    else:
        data = json.loads(source.read_bytes())
        if mode == "error":
            data["error"] = "private-marker"
        elif mode == "duplicate":
            data["chunks"] *= 2
        else:
            data["extra"] = float("nan")
        source.write_text(json.dumps(data), encoding="utf-8")
    assert embed_cli.main(argv) == 1
    assert not output.exists() and not calls
    assert "private-marker" not in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["input", "exists", "cache-output", "cache-input", "cache-file", "repeated-input"])
def test_cli_path_guards(inputs, mode):
    """拒绝覆盖输入、已有结果或混放缓存，不在检查前调用接口。"""
    argv, source, output, directory, calls = inputs
    if mode == "input":
        argv[-1] = str(source)
    elif mode == "exists":
        output.parent.mkdir()
        output.write_text("keep", encoding="utf-8")
    elif mode == "cache-output":
        argv[-1] = str(directory / "snapshot.json")
    elif mode == "cache-input":
        argv[3] = str(source.parent)
    elif mode == "cache-file":
        directory.write_text("keep", encoding="utf-8")
    else:
        argv.insert(2, str(source))
    original = source.read_bytes()
    assert embed_cli.main(argv) == 1
    assert not calls and source.read_bytes() == original


def test_cli_api_failure_does_not_write_snapshot(inputs, monkeypatch, capsys):
    """接口失败时保留安全错误提示，不能生成假成功结果。"""
    argv, _, output, _, _ = inputs

    def fail(*args, **kwargs):
        """模拟已经脱敏的适配层错误。"""
        raise EmbeddingError("safe failure", status_code=429, retryable=True)

    monkeypatch.setattr(embed_cli, "embed_chunks", fail)
    assert embed_cli.main(argv) == 1
    assert not output.exists()
    assert "safe failure" in capsys.readouterr().err
